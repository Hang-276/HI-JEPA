"""Post-reveal factorial diagnostic on the existing 384 Fine windows only."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'fine_local_dose_window_20261110'
sys.path.insert(0, str(PRIOR))
spec = importlib.util.spec_from_file_location('frozen_local_dose', PRIOR / 'run.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
CELLS = ('estimated_affine', 'estimated_physical', 'physical_affine', 'physical_physical')
BOOTSTRAP_SEED = 2026336001


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, data):
    with (HERE / name).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def freeze():
    old_sha = old.verify_lock()
    previous = old.read('LOCK.json')['sources']
    own = [HERE / 'PROTOCOL.md', HERE / 'run.py']
    evidence = [PRIOR / name for name in ('LOCK.json', 'FORECAST.json', 'SUMMARY.json')]
    sources = previous | {str(path): sha(path) for path in own + evidence}
    save('LOCK.json', dict(utc=utc(), previous_lock_sha=old_sha, sources=sources,
                           interpretation='post_reveal_diagnostic_only'))


def verify():
    locked = json.loads((HERE / 'LOCK.json').read_text())
    for path, digest in locked['sources'].items():
        assert sha(path) == digest, f'Frozen input changed: {path}'
    assert old.verify_lock() == locked['previous_lock_sha']
    return sha(HERE / 'LOCK.json')


def reconstructed(native, plus, minus, beta, dose):
    z = native @ beta
    jacobian = ((plus - minus) @ beta) / (2 * dose)
    inverse = np.linalg.inv(jacobian)
    midpoint = (inverse[:, :-1] + inverse[:, 1:]) / 2
    step = np.diff(z, axis=1)
    integrated = (step[..., None, :] @ midpoint)[..., 0, :]
    residual = np.zeros_like(z)
    residual[:, 1:] = np.cumsum(integrated - step, axis=1)
    return z + residual


def run():
    started = time.perf_counter()
    lock_sha = verify()
    forecast, previous = old.read('FORECAST.json'), old.read('SUMMARY.json')
    assert sha(PRIOR / 'FORECAST.json') == previous['forecast_sha']
    encode, fine_sd, beta, affine_sd, reference = old.fit_reference(256, old.REFERENCE_SEED)
    assert reference == forecast['reference']
    physical_ref = old.shift.replay_physical(reference).astype(np.float64)
    physical_sd = physical_ref.std((0, 1))
    np.testing.assert_array_equal(physical_sd, previous['physical_reference_sd'])
    base = old.prior.ac.base
    panels, old_hits = [], {key: 0 for key in (*old.RULES, 'oracle')}
    errors = {lag: dict(sse=np.zeros(4), energy=np.zeros(4), count=0) for lag in (1, 4, 16, 32)}
    parity_max = 0.

    for row, previous_row in zip(forecast['panels'], previous['panels'], strict=True):
        index = row['index']
        public, receipt = old.public_panel(old.N, old.PANEL_SEED + index)
        assert receipt == row['evaluation']
        y = encode(public['x'])
        native_scores = old.lag_scores(y / fine_sd)
        native_set = base.point_sets(native_scores)
        np.testing.assert_array_equal(native_set, row['native_set'])
        selected = np.asarray(row['candidates'])
        assert len(selected) == 24 and len(np.unique(selected)) == 24
        plus, minus, branch_audit = old.endpoint_branches(public, receipt, selected, old.DOSE)
        for name in ('public_parity', 'query_shape', 'branch_sha256'):
            assert branch_audit[name] == row['audit'][name]
        plus_y, minus_y = encode(np.stack([plus, minus]))
        local = old.evaluate(y[selected], plus_y, minus_y, fine_sd, beta, affine_sd, old.DOSE)
        assert local['safe'].all()
        original = old.score_panel(row, physical_sd)
        assert original == previous_row
        for name in old_hits:
            old_hits[name] += original['hit'][name]
        for rule in old.RULES:
            delta = np.max(np.abs(local[rule + '_gap'] - row['gaps'][rule]))
            parity_max = max(parity_max, float(delta))
            np.testing.assert_allclose(local[rule + '_gap'], row['gaps'][rule], rtol=0, atol=1e-10)
            flags = selected[np.lexsort((selected, local[rule + '_gap']))[:old.ALARMS]].tolist()
            assert flags == row['flags'][rule]

        physical = old.shift.replay_physical(receipt).astype(np.float64)[selected]
        estimate = reconstructed(y[selected], plus_y, minus_y, beta, old.DOSE)
        scores = {
            'estimated_affine': old.lag_scores(estimate / affine_sd),
            'estimated_physical': old.lag_scores(estimate / physical_sd),
            'physical_affine': old.lag_scores(physical / affine_sd),
            'physical_physical': old.lag_scores(physical / physical_sd),
        }
        np.testing.assert_allclose(scores['estimated_affine'], local['signed_scores'], rtol=0, atol=1e-10)
        truth = base.point_sets(scores['physical_physical'])
        assert (truth >= 0).all() and (native_set[selected] >= 0).all()
        wrong = np.any(truth != native_set[selected], axis=1)
        assert int(wrong.sum()) == original['wrong_queried']
        cells = {}
        for name, values in scores.items():
            gap = base.margin(values, native_set[selected])
            ranked = np.lexsort((selected, gap))[:old.ALARMS]
            cells[name] = dict(gap=gap.tolist(), flags=selected[ranked].tolist(),
                               hit_ids=selected[ranked][wrong[ranked]].tolist(),
                               hit=int(wrong[ranked].sum()),
                               top2_agreement=int(np.all(base.point_sets(values) == truth, axis=1).sum()))
        assert cells['estimated_affine']['flags'] == row['flags']['signed']
        assert cells['physical_physical']['hit_ids'] == original['hit_ids']['oracle']
        np.testing.assert_allclose(cells['physical_physical']['gap'], original['truth_gap'], rtol=0, atol=1e-10)
        for lag, tally in errors.items():
            real = physical[:, lag:] - physical[:, :-lag]
            estimated = estimate[:, lag:] - estimate[:, :-lag]
            tally['sse'] += ((estimated - real) ** 2).sum((0, 1))
            tally['energy'] += (real ** 2).sum((0, 1))
            tally['count'] += real.shape[0] * real.shape[1]
        panels.append(dict(index=index, candidates=selected.tolist(), wrong=wrong.tolist(),
                           wrong_count=int(wrong.sum()), cells=cells))
        if (index + 1) % 4 == 0:
            print(f'Replayed {index + 1}/{old.PANELS} original panels', flush=True)

    assert old_hits == {key: value['count'] for key, value in previous['hits'].items()}
    assert old_hits['signed'] == 56 and old_hits['affine'] == 76 and old_hits['oracle'] == 88
    assert sum(row['wrong_count'] for row in panels) == 118
    results = {}
    hits = {}
    for name in CELLS:
        hits[name] = np.asarray([row['cells'][name]['hit'] for row in panels])
        gap = np.concatenate([row['cells'][name]['gap'] for row in panels])
        true_gap = np.concatenate([row['cells']['physical_physical']['gap'] for row in panels])
        results[name] = dict(hits=int(hits[name].sum()), alarms=96,
                            queried_recall=float(hits[name].sum() / 118),
                            gap_mae=float(np.abs(gap - true_gap).mean()),
                            top2_agreement=sum(row['cells'][name]['top2_agreement'] for row in panels) / 384)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(16, size=(4096, 16))
    support = np.asarray([row['wrong_count'] for row in panels])
    boot_support = support[indices].sum(1)
    assert (boot_support > 0).all()
    differences = {
        'physical_denominator_given_estimated_numerator': hits['estimated_physical'] - hits['estimated_affine'],
        'physical_denominator_given_physical_numerator': hits['physical_physical'] - hits['physical_affine'],
        'physical_numerator_given_affine_denominator': hits['physical_affine'] - hits['estimated_affine'],
        'physical_numerator_given_physical_denominator': hits['physical_physical'] - hits['estimated_physical'],
    }
    differences['interaction'] = (differences['physical_denominator_given_physical_numerator']
                                  - differences['physical_denominator_given_estimated_numerator'])
    contrasts = {name: dict(delta_hits=int(delta.sum()), delta_recall=float(delta.sum() / 118),
                            descriptive_panel_bootstrap_ci95=np.quantile(
                                delta[indices].sum(1) / boot_support, [.025, .975]).tolist())
                 for name, delta in differences.items()}
    increments = {str(lag): dict(relative_rmse_by_axis=np.sqrt(t['sse'] / t['energy']).tolist(),
                                  pooled_relative_rmse=float(np.sqrt(t['sse'].sum() / t['energy'].sum())),
                                  physical_standardized_rmse_by_axis=(np.sqrt(t['sse'] / t['count']) / physical_sd).tolist(),
                                  increments_per_axis=t['count']) for lag, t in errors.items()}
    estimated_gain = results['estimated_physical']['hits'] - results['estimated_affine']['hits']
    mae_aff = results['estimated_affine']['gap_mae']
    summary = dict(utc=utc(), lock_sha=lock_sha, forecast_sha=sha(PRIOR / 'FORECAST.json'),
                   status='post_reveal_diagnostic_only', elapsed_seconds=time.perf_counter() - started,
                   original_reproduction=dict(exact_old_panel_scores=True, max_gap_error=parity_max,
                                              original_hits=old_hits, queried=384, wrong_queried=118),
                   affine_reference_sd=affine_sd.tolist(), physical_reference_sd=physical_sd.tolist(),
                   sd_relative_error=(affine_sd / physical_sd - 1).tolist(),
                   physical_score_multiplier_with_affine_denominator=((physical_sd / affine_sd) ** 2).tolist(),
                   results=results, contrasts=contrasts, increment_reconstruction=increments,
                   attribution=dict(denominator_swap_additional_hits=estimated_gain,
                                    fraction_of_oracle_hit_headroom_closed=estimated_gain / (88 - 56),
                                    fraction_of_signed_gap_mae_removed=(mae_aff - results['estimated_physical']['gap_mae']) / mae_aff),
                   panels=panels)
    save('SUMMARY.json', summary)
    print(json.dumps({key: summary[key] for key in ('results', 'contrasts', 'attribution', 'increment_reconstruction')},
                     indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('lock', 'audit'))
    phase = parser.parse_args().phase
    {'lock': freeze, 'audit': run}[phase]()
