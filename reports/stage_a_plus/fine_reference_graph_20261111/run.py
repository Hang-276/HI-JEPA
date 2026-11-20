"""Prospective Fine reference-scale repair; observable forecast precedes label scoring."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'fine_local_dose_window_20261110'
sys.path.insert(0, str(PRIOR))
spec = importlib.util.spec_from_file_location('frozen_fine_dose', PRIOR / 'run.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
RULES = ('native', 'affine', 'signed_oldscale', 'signed_graphscale')
REFERENCE_SEED, PANEL_SEED = 2026343000, 2026343100
BOOTSTRAP_SEED = 2026343201
PANELS, N, SCREEN, ALARMS = 16, 128, 24, 6
GRAPH_PYTHON = '/opt/miniconda3/envs/evg-agent/bin/python'


def reconstruct_reference(z, jacobian):
    # Keep the existing torch and scipy environments in separate processes.
    with tempfile.TemporaryDirectory(prefix='fine_reference_graph_') as directory:
        root = Path(directory)
        np.savez(root / 'input.npz', z=z, jacobian=jacobian)
        subprocess.run([GRAPH_PYTHON, str(HERE / 'graph_worker.py'),
                        str(root / 'input.npz'), str(root / 'output.json')],
                       check=True, capture_output=True, text=True)
        result = json.loads((root / 'output.json').read_text())
    sd = result['sd']
    return None if sd is None else np.asarray(sd), result['diagnostics']


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def lock():
    prior_sha = old.verify_lock()
    local = [HERE / name for name in ('PROTOCOL.md', 'run.py', 'reference_graph.py', 'graph_worker.py',
                                     'GRAPH_SELF_CHECK.json')]
    if (HERE / 'GRAPH_ASSUMPTIONS.md').exists():
        local.append(HERE / 'GRAPH_ASSUMPTIONS.md')
    prior = old.read('LOCK.json')['sources']
    prior[str(PRIOR / 'LOCK.json')] = prior_sha
    save('LOCK.json', dict(utc=utc(), prior_lock_sha=prior_sha,
                           runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__,
                                        torch=version('torch'), graph_executable=GRAPH_PYTHON),
                           sources=prior | {str(path): sha(path) for path in local}))


def verify():
    frozen = read('LOCK.json')
    for path, digest in frozen['sources'].items():
        assert sha(path) == digest, f'Source changed: {path}'
    assert old.verify_lock() == frozen['prior_lock_sha']
    return sha(HERE / 'LOCK.json')


def fit_reference(count, seed):
    encode, fine_sd, beta, affine_sd, receipt = old.fit_reference(count, seed)
    public, replay = old.public_panel(count, seed)
    assert replay == receipt
    y = encode(public['x'])
    ids = np.arange(count)
    plus, minus, query_audit = old.endpoint_branches(public, receipt, ids, old.DOSE)
    plus_y, minus_y = encode(np.stack([plus, minus]))
    z = y @ beta
    jacobian = ((plus_y - minus_y) @ beta) / (2 * old.DOSE)
    graph_sd, diagnostic = reconstruct_reference(z, jacobian)
    views, _, provenance = old.prior.open_views()
    pilot_source = json.loads(old.PILOT.read_text())
    assert provenance == pilot_source['provenance']
    raw_control = pilot_source['controls']['raw']
    assert raw_control['available']
    raw_beta = np.asarray(raw_control['beta'])

    def raw_encode(values):
        return np.asarray(views['raw'](values), dtype=np.float64)

    raw_y = raw_encode(public['x'])
    raw_plus, raw_minus = raw_encode(np.stack([plus, minus]))
    raw_jacobian = ((raw_plus - raw_minus) @ raw_beta) / (2 * old.DOSE)
    raw_graph_sd, raw_diagnostic = reconstruct_reference(raw_y @ raw_beta, raw_jacobian)
    raw_state = dict(encode=raw_encode, sd=raw_y.std((0, 1)), beta=raw_beta, graph_sd=raw_graph_sd)
    record = dict(receipt=receipt, query=query_audit, graph=diagnostic,
                  fine_sd=fine_sd.tolist(), affine_sd=affine_sd.tolist(),
                  graph_sd=None if graph_sd is None else graph_sd.tolist(),
                  raw_graph=raw_diagnostic,
                  raw_graph_sd=None if raw_graph_sd is None else raw_graph_sd.tolist())
    return (encode, fine_sd, beta, affine_sd, graph_sd, raw_state), record


def evaluate(index, seed, count, screen, alarms, reference, engineering=False):
    encode, fine_sd, beta, affine_sd, graph_sd, raw_state = reference
    assert graph_sd is not None
    public, receipt = old.public_panel(count, seed)
    y = encode(public['x'])
    base = old.prior.ac.base
    native_scores = old.lag_scores(y / fine_sd)
    native_set = base.point_sets(native_scores)
    strict = (native_set >= 0).all(1)
    sets = np.where(strict[:, None], native_set, [0, 1])
    native_gap = np.where(strict, base.margin(native_scores, sets), 0.)
    selected = np.argsort(native_gap, kind='stable')[:screen]
    plus, minus, audit = old.endpoint_branches(public, receipt, selected, old.DOSE)
    plus_y, minus_y = encode(np.stack([plus, minus]))
    baseline = old.evaluate(y[selected], plus_y, minus_y, fine_sd, beta, affine_sd, old.DOSE)
    repaired = old.evaluate(y[selected], plus_y, minus_y, fine_sd, beta, graph_sd, old.DOSE)
    np.testing.assert_array_equal(baseline['native_set'], native_set[selected])
    np.testing.assert_allclose(baseline['native_gap'], native_gap[selected], rtol=0, atol=1e-10)
    gaps = dict(native=baseline['native_gap'], affine=baseline['affine_gap'],
                signed_oldscale=baseline['signed_gap'], signed_graphscale=repaired['signed_gap'])
    if raw_state['graph_sd'] is not None:
        raw_encode = raw_state['encode']
        raw_y = raw_encode(public['x'][selected])
        raw_plus, raw_minus = raw_encode(np.stack([plus, minus]))
        raw_scores = old.evaluate(raw_y, raw_plus, raw_minus, raw_state['sd'], raw_state['beta'],
                                  raw_state['graph_sd'], old.DOSE)
        gaps['raw_graph'] = base.margin(raw_scores['signed_scores'], sets[selected])
        audit.update(raw_safe_count=int(raw_scores['safe'].sum()),
                     raw_max_condition=float(raw_scores['max_condition'].max()))
    flags = {name: selected[np.lexsort((selected, gap))[:alarms]].tolist() for name, gap in gaps.items()}
    audit.update(safe_count=int(baseline['safe'].sum()),
                 max_condition=float(baseline['max_condition'].max()))
    if engineering:
        historical = np.diagonal(base.score_matrix(y / fine_sd), axis1=1, axis2=2)
        duplicates = encode(np.stack([public['x'][selected], public['x'][selected]]))
        z = y[selected] @ beta
        jacobian = ((plus_y - minus_y) @ beta) / (2 * old.DOSE)
        safe = baseline['safe']
        residual = np.zeros_like(z)
        inverse = np.linalg.inv(jacobian[safe])
        midpoint = (inverse[:, :-1] + inverse[:, 1:]) / 2
        dz = np.diff(z[safe], axis=1)
        dx = (dz[..., None, :] @ midpoint)[..., 0, :]
        residual[safe, 1:] = np.cumsum(dx - dz, axis=1)
        direct_scores = old.lag_scores((z + residual) / graph_sd)
        audit.update(historical_score_error=float(np.max(np.abs(native_scores - historical))),
                     duplicate_error=float(np.max(np.abs(duplicates - y[selected]))),
                     reconstruction_algebra_error=float(np.max(np.abs(direct_scores - repaired['signed_scores']))))
    return dict(index=index, evaluation=receipt, candidates=selected.tolist(),
                native_set=native_set.tolist(), gaps={name: value.tolist() for name, value in gaps.items()},
                flags=flags, audit=audit)


def pilot(lock_sha):
    start = time.perf_counter()
    reference, record = fit_reference(8, 2026342900)
    panel = None
    passed = reference[4] is not None
    if passed:
        panel = evaluate(-1, 2026342901, 8, 2, 1, reference, engineering=True)
        check = panel['audit']
        passed = (check['public_parity'] and check['safe_count'] == 2
                  and check['historical_score_error'] <= 1e-10
                  and check['duplicate_error'] <= 1e-4
                  and check['reconstruction_algebra_error'] <= 1e-10)
    elapsed = time.perf_counter() - start
    passed = bool(passed and elapsed <= 120.)
    save('ENGINEERING.json', dict(utc=utc(), lock_sha=lock_sha, passed=passed,
                                  elapsed_seconds=elapsed, reference=record, panel=panel))
    print(json.dumps(dict(passed=passed, elapsed_seconds=elapsed, graph=record['graph']), indent=2), flush=True)


def forecast(lock_sha):
    engineering = read('ENGINEERING.json')
    assert engineering['passed'] and engineering['lock_sha'] == lock_sha
    start = time.perf_counter()
    reference, record = fit_reference(256, REFERENCE_SEED)
    panels = []
    if reference[4] is not None:
        for index in range(PANELS):
            panels.append(evaluate(index, PANEL_SEED + index, N, SCREEN, ALARMS, reference))
            if (index + 1) % 4 == 0:
                print(f'Sealed observable predictions {index + 1}/{PANELS}', flush=True)
    save('FORECAST.json', dict(utc=utc(), lock_sha=lock_sha, reference=record,
                               graph_gate_passed=reference[4] is not None,
                               elapsed_seconds=time.perf_counter() - start, panels=panels))
    print('FORECAST SHA256 ' + sha(HERE / 'FORECAST.json'), flush=True)


def score(lock_sha):
    start = time.perf_counter()
    source = read('FORECAST.json')
    assert source['lock_sha'] == lock_sha
    if not source['graph_gate_passed']:
        save('SUMMARY.json', dict(utc=utc(), lock_sha=lock_sha, forecast_sha=sha(HERE / 'FORECAST.json'),
                                  decision='graph_gate_failed', labels_revealed=False))
        return
    assert len(source['panels']) == PANELS
    rules = RULES + (('raw_graph',) if source['reference']['raw_graph_sd'] is not None else ())
    physical_ref = old.shift.replay_physical(source['reference']['receipt']).astype(np.float64)
    physical_sd = physical_ref.std((0, 1))
    graph_sd = np.asarray(source['reference']['graph_sd'])
    base = old.prior.ac.base
    panels = []
    for row in source['panels']:
        physical = old.shift.replay_physical(row['evaluation']).astype(np.float64)
        truth_scores = old.lag_scores(physical / physical_sd)
        truth = base.point_sets(truth_scores)
        native = np.asarray(row['native_set'])
        valid = (native >= 0).all(1) & (truth >= 0).all(1)
        wrong = valid & np.any(native != truth, axis=1)
        selected = np.asarray(row['candidates'])
        sets = np.where((native[selected] >= 0).all(1)[:, None], native[selected], [0, 1])
        true_gap = base.margin(truth_scores[selected], sets)
        oracle = selected[np.lexsort((selected, np.where(valid[selected], true_gap, np.inf)))[:ALARMS]]
        flags = {name: np.asarray(row['flags'][name]) for name in rules} | {'oracle': oracle}
        assert all(len(np.unique(ids)) == ALARMS and np.isin(ids, selected).all() for ids in flags.values())
        hits = {name: ids[wrong[ids]] for name, ids in flags.items()}
        exchange = {name: dict(repaired_only=len(np.setdiff1d(hits['signed_graphscale'], hits[name])),
                               baseline_only=len(np.setdiff1d(hits[name], hits['signed_graphscale'])))
                    for name in rules if name != 'signed_graphscale'}
        panels.append(dict(index=row['index'], wrong_total=int(wrong.sum()),
                            wrong_queried=int(wrong[selected].sum()), invalid=int((~valid).sum()),
                            wrong_ids=np.flatnonzero(wrong).tolist(), truth_gap=true_gap.tolist(),
                            hit={name: len(ids) for name, ids in hits.items()},
                            hit_ids={name: ids.tolist() for name, ids in hits.items()}, exchange=exchange,
                            valid_queried=int(valid[selected].sum()),
                            abs_gap_error={name: float(np.abs(np.asarray(row['gaps'][name]) - true_gap)[valid[selected]].sum())
                                           for name in rules}))
    support = np.asarray([row['wrong_queried'] for row in panels])
    hits = {name: np.asarray([row['hit'][name] for row in panels]) for name in (*rules, 'oracle')}
    bootstrap = np.random.default_rng(BOOTSTRAP_SEED).integers(PANELS, size=(4096, PANELS))
    boot_support = support[bootstrap].sum(1)
    keep = boot_support > 0
    enough = support.sum() >= 40 and (support > 0).sum() >= 8
    contrasts = {}
    for name in rules:
        if name == 'signed_graphscale':
            continue
        delta = hits['signed_graphscale'] - hits[name]
        change = float(delta.sum() / support.sum()) if support.sum() else None
        ci = (np.quantile(delta[bootstrap].sum(1)[keep] / boot_support[keep], [.025, .975]).tolist()
              if keep.any() else [None, None])
        contrasts[name] = dict(delta_hits=int(delta.sum()), delta_queried_recall=change, ci95=ci,
                               exchange={direction: sum(row['exchange'][name][direction] for row in panels)
                                         for direction in ('repaired_only', 'baseline_only')},
                               passed=change is not None and change >= .05 and ci[0] > 0.)
    safe = sum(row['audit']['safe_count'] for row in source['panels'])
    decision = ('engineering_insufficient' if safe < .95 * PANELS * SCREEN else
                'insufficient_wrong_windows' if not enough else
                'passed' if all(contrasts[name]['passed'] for name in ('affine', 'signed_oldscale')) else
                'no_separation')
    total, valid = sum(row['wrong_total'] for row in panels), sum(row['valid_queried'] for row in panels)
    oracle_headroom = int(hits['oracle'].sum() - hits['affine'].sum())
    gained = int(hits['signed_graphscale'].sum() - hits['affine'].sum())
    summary = dict(utc=utc(), lock_sha=lock_sha, forecast_sha=sha(HERE / 'FORECAST.json'),
                   decision=decision, labels_revealed=True, elapsed_seconds=time.perf_counter() - start,
                   wrong_total=total, wrong_queried=int(support.sum()), queried=PANELS * SCREEN,
                   panels_with_wrong_queried=int((support > 0).sum()), safe_count=safe,
                   raw_secondary_available='raw_graph' in rules,
                   raw_safe_count=sum(row['audit'].get('raw_safe_count', 0) for row in source['panels']),
                   scored=PANELS * N, alarm_count_per_rule=PANELS * ALARMS,
                   invalid=sum(row['invalid'] for row in panels),
                   hits={name: dict(count=int(value.sum()),
                                    queried_recall=float(value.sum() / support.sum()) if support.sum() else None,
                                    full_recall=float(value.sum() / total) if total else None)
                         for name, value in hits.items()}, contrasts=contrasts,
                   post_reveal=dict(physical_reference_sd=physical_sd.tolist(), graph_reference_sd=graph_sd.tolist(),
                                     graph_sd_relative_error=(graph_sd / physical_sd - 1).tolist(),
                                     affine_sd_relative_error=(np.asarray(source['reference']['affine_sd']) / physical_sd - 1).tolist(),
                                     raw_graph_sd_relative_error=(None if 'raw_graph' not in rules else
                                         (np.asarray(source['reference']['raw_graph_sd']) / physical_sd - 1).tolist()),
                                     gap_mae={name: sum(row['abs_gap_error'][name] for row in panels) / valid if valid else None
                                              for name in rules},
                                     oracle_headroom_over_affine=oracle_headroom,
                                     repaired_gain_over_affine=gained,
                                     fraction_oracle_headroom_closed=gained / oracle_headroom if oracle_headroom else None),
                   panels=panels)
    save('SUMMARY.json', summary)
    print(json.dumps({key: summary[key] for key in ('decision', 'wrong_queried', 'hits', 'contrasts', 'post_reveal')},
                     indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('lock', 'pilot', 'forecast', 'score'))
    phase = parser.parse_args().phase
    if phase == 'lock':
        lock()
    else:
        {'pilot': pilot, 'forecast': forecast, 'score': score}[phase](verify())
