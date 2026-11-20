"""Saved-scalar and CAL-only Fine value audit. No model imports or forward.

Physical CAL labels serve only a fixed affine-span diagnostic. Its in-sample
R2 is a finite-CAL ceiling for pointwise affine readers, not confirmation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'outputs/HCF_STABLE_ID_RUN_20261028_V2'
OUT = Path(__file__).resolve().parent
FILES: dict[str, str] = {}
ARRAY_LOG: list[dict] = []


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jread(p: Path | str | dict) -> dict:
    ref = p if isinstance(p, dict) else None
    path = Path(ref['path'] if ref else p)
    actual = sha(path)
    if ref and ref.get('sha256'):
        assert actual == ref['sha256'], f'Sealed JSON changed: {path}'
    FILES[str(path)] = actual
    return json.loads(path.read_text())


def arr(spec: dict) -> np.ndarray:
    ref, member = spec['artifact'], spec['member']
    p = Path(ref['path'])
    assert not any(x in str(p).lower() for x in ['/test', '/private_test', '/final'])
    assert member in {'cal__x', 'cal__physical', 'cal', 'Fine4', 'G4'}
    assert member.startswith('cal') or 'CAL' in p.name
    actual = sha(p)
    assert actual == ref['sha256'], f'Sealed CAL archive changed: {p}'
    FILES[str(p)] = actual
    with np.load(p, allow_pickle=False) as archive:
        value = np.asarray(archive[member]).copy()
    assert value.shape == (32, 129, 4) and value.dtype == np.float32
    assert np.isfinite(value).all()
    ident = hashlib.sha256()
    ident.update(str(value.dtype).encode())
    ident.update(str(value.shape).encode())
    ident.update(np.ascontiguousarray(value).tobytes())
    typed = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=ident.hexdigest())
    if spec.get('identity') is not None:
        assert typed == spec['identity']
    ARRAY_LOG.append(dict(path=str(p), member=member, identity=typed))
    return value.astype(np.float64).reshape(-1, 4)


def cal_span(x: np.ndarray, target: np.ndarray) -> dict:
    """Exact finite-array affine least-squares projection; no thresholds tuned."""
    mean, std = x.mean(0), x.std(0, ddof=0)
    assert np.all(std > 0)
    design = np.column_stack((np.ones(len(x)), (x-mean)/std))
    coef, _, rank, sv = np.linalg.lstsq(design, target, rcond=1e-12)
    assert rank == 5
    error = np.mean((target-design@coef)**2, axis=0)
    var = target.var(0, ddof=0)
    assert np.all(var > 0)
    r2 = 1-error/var
    normal_residual = float(np.max(abs(design.T@(target-design@coef))/len(x)))
    return dict(R2_by_coordinate=r2.tolist(), mean_R2=float(r2.mean()),
                worst_R2=float(r2.min()), normalized_error_by_coordinate=(error/var).tolist(),
                design_rank=int(rank), design_singular_values=sv.tolist(),
                normal_equation_residual_max=normal_residual,
                sample_count=len(x), fit_and_score='same existing CAL; supervised oracle only',
                mathematical_scope='empirical upper R2 among pointwise affine functions on this CAL',
                population_or_TEST_upper_bound=False)


def key(x: dict) -> tuple[str, str]:
    return x['instance'], x['condition']


def compact_metric(x: dict) -> dict:
    return {k: x[k] for k in ['mean_R2', 'worst_R2', 'R2_by_coordinate',
                              'target_variance_by_coordinate', 'mapping_kind']}


def main() -> dict:
    dev_raw = jread(BASE/'conditional_dmd_raw_obs_v1/RESULTS.json')
    dev_fine = jread(BASE/'conditional_dmd_eigenframe_v1/RESULTS.json')
    conf_raw = jread(BASE/'dynamic_eigenframe_fresh_raw_control_v1/RESULTS.json')
    conf_fine = jread(BASE/'dynamic_eigenframe_confirmation_eval_v1/RESULTS.json')
    strong = jread(BASE/'weak_AR_observation_intervention_eval_v1/RESULTS.json')
    saved = []
    for panel, strength, raws, fines, source in [
        ('development_AR', .25, dev_raw['rows'], dev_fine['rows'], 'candidate'),
        ('new_AR_confirmation', .25, conf_raw['rows'], conf_fine['rows'], 'G4'),
        ('stronger_observation_same_weak_roots', 1., strong['raw_rows'], strong['Fine_rows'], 'G4')]:
        for rr in raws:
            cohort = rr.get('cohort', {'instance': rr.get('instance'), 'condition': 'weak'})
            raw = rr['candidate'] if 'candidate' in rr else rr['G4']
            fr = [r for r in fines if key(r['profile']) == key(cohort)]
            assert len(fr) == 3
            fine = [r[source] for r in fr]
            for f in fine:
                assert f['axis4']['target_variance_by_coordinate'] == raw['axis4']['target_variance_by_coordinate']
            vals = np.array([f['axis4']['mean_R2'] for f in fine])
            rawval = raw['axis4']['mean_R2']
            saved.append(dict(panel=panel, observation_strength=strength, **cohort,
                raw_parent_count=1, Fine_optimization_parent_count=3,
                Fine_G4_mean_R2=float(vals.mean()), Fine_G4_seed_R2=vals.tolist(),
                Fine_G4_min_R2=float(vals.min()), Fine_G4_max_R2=float(vals.max()),
                raw_G4_axis=compact_metric(raw['axis4']), raw_G4_dense=compact_metric(raw['dense4']),
                delta_R2_Fine_minus_raw=float(vals.mean()-rawval),
                descriptive_error_reduction=float(1-(1-vals.mean())/(1-rawval)),
                Fine_seed_values_above_same_raw=int(np.sum(vals > rawval)),
                inferential_status='saved exposed TEST scalar comparison; no new TEST/FINAL arrays',
                paired_statistical_test_or_CI=False))

    # Build exact CAL-only registries; do not follow any TEST references.
    raw_specs: dict[tuple[str, str, float], tuple[dict, dict]] = {}
    for folder, strength in [('conditional_dmd_raw_obs_v1', .25),
                             ('dynamic_eigenframe_fresh_raw_control_v1', .25)]:
        inp = jread(BASE/folder/'INPUTS.json')
        for r in inp['rows']:
            raw_specs[(*key(r['cohort']), strength)] = (r['CAL'], r['CAL_physical'])
    inter_inp = jread(BASE/'weak_AR_observation_intervention_eval_v1/INPUTS.json')
    for r in inter_inp['data_rows']:
        orig = next(x for x in inter_inp['weak_cohorts'] if x['instance'] == r['instance'])
        raw_specs[(r['instance'], 'weak', 1.)] = (
            dict(artifact=r['artifact'], member='cal__x', identity=r['identities']['cal']['x']),
            orig['original_CAL_target'])

    fine_specs: dict[tuple[str, str, float], list[tuple[dict, dict]]] = {k: [] for k in raw_specs}
    dev_inp = jread(BASE/'conditional_dmd_eigenframe_v1/INPUTS.json')
    for r in dev_inp['parents']:
        fine_specs[(*key(r['profile']), .25)].append((r['profile'], r['CAL']))
    conf_views = jread(BASE/'dynamic_eigenframe_confirmation_eval_v1/ALL12_MODELS_AND24_CAL_VIEWS_LOCK.json')
    for r in conf_views['rows']:
        fine_specs[(*key(r['profile']), .25)].append((r['profile'],
            dict(artifact=r['CAL_views'], member='Fine4', identity=r['CAL_view_identities']['Fine4'])))
    strong_views = jread(BASE/'weak_AR_observation_intervention_eval_v1/CAL_VIEWS_LOCK.json')
    for r in strong_views['Fine_rows']:
        fine_specs[(*key(r['profile']), 1.)].append((r['profile'],
            dict(artifact=r['artifact'], member='Fine4', identity=r['identities']['Fine4'])))

    ceiling = []
    for k, (xs, ys) in raw_specs.items():
        x, y = arr(xs), arr(ys)
        raw_oracle = cal_span(x, y)
        forward_oracle = cal_span(y, x)
        fs = []
        assert len(fine_specs[k]) == 3
        for p, spec in fine_specs[k]:
            f = cal_span(arr(spec), y)
            fs.append(dict(profile=p, Fine_CAL_dense_oracle=f,
                           delta_mean_R2_over_raw_CAL_ceiling=f['mean_R2']-raw_oracle['mean_R2']))
        vals = np.array([x['Fine_CAL_dense_oracle']['mean_R2'] for x in fs])
        ceiling.append(dict(instance=k[0], condition=k[1], observation_strength=k[2],
            raw_CAL_affine_ceiling=raw_oracle, physical_to_raw_CAL_affine_explained=forward_oracle,
            Fine_CAL_affine_oracle_mean=float(vals.mean()),
            Fine_CAL_affine_oracle_seed_range=[float(vals.min()), float(vals.max())],
            Fine_parent_rows=fs,
            difference_mean=float(vals.mean()-raw_oracle['mean_R2']),
            confirms_population_Fine_necessity=False))

    assert len(saved) == len(ceiling) == 10
    assert len(ARRAY_LOG) == 50  # 10 raw CAL, 10 physical CAL, 30 Fine CAL.
    return dict(date='2026-11-04', status='completed_saved_scalar_and_CAL_affine_span_audit',
        saved_paired_cohort_comparisons=saved, CAL_span_diagnostics=ceiling,
        unique_underlying_trajectory_mixing_roots=4,
        strong_weak_conditions_paired_not_independent_roots=True,
        stronger_observation_reuses_two_confirmation_weak_roots=True,
        raw_deterministic_parents=10, Fine_optimization_parents=30,
        new_neural_training=0, model_forward=0, checkpoint_load=0,
        new_physical_data=0, PRNG_calls=0, GPU_operations=0,
        TEST_FINAL_array_members_decoded=0,
        CAL_array_members_decoded=len(ARRAY_LOG), CAL_array_access=ARRAY_LOG,
        original_sources_and_inputs_sha256=FILES,
        limitations=['CAL oracle fits use physical labels after model locking; no training feedback.',
                     'CAL in-sample upper bounds are only for those finite arrays and affine readers.',
                     'No learned-model or raw-rule hyperparameter search, gate modification, or source edits.',
                     'Saved exposed TEST scalars are reused without opening their arrays.'])


if __name__ == '__main__':
    target = OUT/'FINE_VALUE_AUDIT.json'
    if target.exists():
        raise SystemExit('Preserve existing new audit output; do not overwrite.')
    target.write_text(json.dumps(main(), ensure_ascii=False, indent=2) + '\n')
    print(target)
