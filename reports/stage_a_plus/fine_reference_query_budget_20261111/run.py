"""Frozen reference-query budgets with independent reference/evaluation panel pairs."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
from node_query import node_branches

HERE = Path(__file__).resolve().parent
DENSE = HERE.parent / 'fine_reference_graph_20261111'
spec = importlib.util.spec_from_file_location('frozen_dense_reference', DENSE / 'run.py')
dense = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dense)
old = dense.old
GRAPH_PYTHON = '/opt/miniconda3/envs/evg-agent/bin/python'
BUDGETS = ('1', '64', '512', 'all')
RULES = ('native', 'affine_fine', 'affine_raw') + tuple(f'{rep}_{budget}' for rep in ('fine', 'raw') for budget in BUDGETS)
PANELS, REFERENCE, EVALUATION, SCREEN, ALARMS = 16, 96, 128, 24, 6
BOOTSTRAP_SEED = 2026345301


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
    dense_sha = dense.verify()
    sources = dense.read('LOCK.json')['sources']
    files = [HERE / name for name in ('PROTOCOL.md', 'run.py', 'node_query.py', 'sparse_worker.py', 'WORKER_SELF_CHECK.json')]
    files += [DENSE / 'LOCK.json']
    save('LOCK.json', dict(utc=utc(), dense_lock_sha=dense_sha,
                           runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__,
                                        torch=version('torch'), graph_executable=GRAPH_PYTHON),
                           sources=sources | {str(path): sha(path) for path in files}))


def verify():
    frozen = read('LOCK.json')
    for path, digest in frozen['sources'].items():
        assert sha(path) == digest, f'Frozen source changed: {path}'
    assert dense.verify() == frozen['dense_lock_sha']
    return sha(HERE / 'LOCK.json')


def solve(z, anchor_jacobian, anchor_ids):
    with tempfile.TemporaryDirectory(prefix='fine_reference_budget_') as directory:
        root = Path(directory)
        np.savez(root / 'input.npz', z=z, anchor_jacobian=anchor_jacobian, anchor_ids=anchor_ids)
        subprocess.run([GRAPH_PYTHON, str(HERE / 'sparse_worker.py'), '--input', str(root / 'input.npz'),
                        '--output', str(root / 'output.json')], check=True, capture_output=True, text=True)
        result = json.loads((root / 'output.json').read_text())
    return None if result['scale'] is None else np.asarray(result['scale']), result['diagnostics']


def fit_reference(count, seed, anchor_seed):
    fine_encode, fine_sd, fine_beta, fine_affine_sd, receipt = old.fit_reference(count, seed)
    public, replay = old.public_panel(count, seed)
    assert replay == receipt
    views, _, provenance = old.prior.open_views()
    pilot_source = json.loads(old.PILOT.read_text())
    assert provenance == pilot_source['provenance'] and pilot_source['controls']['raw']['available']
    raw_beta = np.asarray(pilot_source['controls']['raw']['beta'])

    def raw_encode(values):
        return np.asarray(views['raw'](values), dtype=np.float64)

    encoders = dict(fine=fine_encode, raw=raw_encode)
    features = {rep: encode(public['x']) for rep, encode in encoders.items()}
    betas = dict(fine=fine_beta, raw=raw_beta)
    reference = {rep: dict(encode=encoders[rep], beta=betas[rep], sd=features[rep].std((0, 1)),
                           affine_sd=(features[rep] @ betas[rep]).std((0, 1)), scales={})
                 for rep in ('fine', 'raw')}
    np.testing.assert_array_equal(reference['fine']['sd'], fine_sd)
    np.testing.assert_array_equal(reference['fine']['affine_sd'], fine_affine_sd)
    node_count = features['fine'].shape[0] * features['fine'].shape[1]
    permutation = np.random.default_rng(anchor_seed).permutation(node_count).astype(np.int64)
    graphs, queries = {rep: {} for rep in ('fine', 'raw')}, {}
    for budget in BUDGETS:
        count_used = node_count if budget == 'all' else int(budget)
        assert count_used <= node_count
        anchors = permutation[:count_used]
        plus, minus, query = node_branches(public, receipt, anchors, old.DOSE)
        queries[budget] = query
        for rep in ('fine', 'raw'):
            plus_y, minus_y = encoders[rep](np.stack([plus, minus]))
            jacobian = ((plus_y - minus_y) @ betas[rep]) / (2 * old.DOSE)
            scale, diagnostic = solve(features[rep] @ betas[rep], jacobian, anchors)
            reference[rep]['scales'][budget] = scale
            graphs[rep][budget] = diagnostic
    record = dict(receipt=receipt, anchor_seed=anchor_seed, node_count=node_count,
                  permutation_sha256=hashlib.sha256(permutation.tobytes()).hexdigest(), queries=queries,
                  graphs=graphs,
                  scales={rep: {budget: None if scale is None else scale.tolist()
                                 for budget, scale in state['scales'].items()} for rep, state in reference.items()},
                  native_sd={rep: state['sd'].tolist() for rep, state in reference.items()},
                  affine_sd={rep: state['affine_sd'].tolist() for rep, state in reference.items()})
    return reference, record


def evaluate(index, seed, count, screen, alarms, reference, engineering=False):
    public, receipt = old.public_panel(count, seed)
    base = old.prior.ac.base
    fine = reference['fine']
    all_fine_y = fine['encode'](public['x'])
    native_scores = old.lag_scores(all_fine_y / fine['sd'])
    native_set = base.point_sets(native_scores)
    strict = (native_set >= 0).all(1)
    safe_sets = np.where(strict[:, None], native_set, [0, 1])
    native_gap = np.where(strict, base.margin(native_scores, safe_sets), 0.)
    selected = np.argsort(native_gap, kind='stable')[:screen]
    plus, minus, audit = old.endpoint_branches(public, receipt, selected, old.DOSE)
    gaps = dict(native=native_gap[selected])
    fallback = {rep: {} for rep in ('fine', 'raw')}
    for rep, state in reference.items():
        y = all_fine_y[selected] if rep == 'fine' else state['encode'](public['x'][selected])
        plus_y, minus_y = state['encode'](np.stack([plus, minus]))
        local = old.evaluate(y, plus_y, minus_y, state['sd'], state['beta'], np.ones(4), old.DOSE)
        affine_scores = old.lag_scores((y @ state['beta']) / state['affine_sd'])
        gaps[f'affine_{rep}'] = base.margin(affine_scores, safe_sets[selected])
        audit[f'{rep}_safe_count'] = int(local['safe'].sum())
        audit[f'{rep}_max_condition'] = float(local['max_condition'].max())
        for budget in BUDGETS:
            scale = state['scales'][budget]
            fallback[rep][budget] = scale is None
            values = affine_scores if scale is None else local['signed_scores'] / scale**2
            gaps[f'{rep}_{budget}'] = base.margin(values, safe_sets[selected])
        if engineering and state['scales']['all'] is not None:
            explicit = old.evaluate(y, plus_y, minus_y, state['sd'], state['beta'],
                                    state['scales']['all'], old.DOSE)
            audit[f'{rep}_rescaling_algebra_error'] = float(np.max(np.abs(
                local['signed_scores'] / state['scales']['all']**2 - explicit['signed_scores'])))
    if engineering:
        historical = np.diagonal(base.score_matrix(all_fine_y / fine['sd']), axis1=1, axis2=2)
        duplicates = fine['encode'](np.stack([public['x'][selected], public['x'][selected]]))
        audit.update(historical_score_error=float(np.max(np.abs(native_scores - historical))),
                     duplicate_error=float(np.max(np.abs(duplicates - all_fine_y[selected]))))
    assert set(gaps) == set(RULES)
    flags = {name: selected[np.lexsort((selected, values))[:alarms]].tolist() for name, values in gaps.items()}
    return dict(index=index, evaluation=receipt, candidates=selected.tolist(), native_set=native_set.tolist(),
                gaps={name: values.tolist() for name, values in gaps.items()}, flags=flags,
                audit=audit, fallback=fallback)


def pilot(lock_sha):
    start = time.perf_counter()
    reference, record = fit_reference(8, 2026344900, 2026344902)
    full = all(reference[rep]['scales']['all'] is not None for rep in ('fine', 'raw'))
    panel = evaluate(-1, 2026344901, 8, 2, 1, reference, engineering=True)
    audit = panel['audit']
    passed = (full and audit['public_parity'] and audit['historical_score_error'] <= 1e-10
              and audit['duplicate_error'] <= 1e-4
              and all(audit[f'{rep}_safe_count'] == 2 and audit.get(f'{rep}_rescaling_algebra_error', 1.) <= 1e-10
                      for rep in ('fine', 'raw')))
    elapsed = time.perf_counter() - start
    passed = bool(passed and elapsed <= 120.)
    save('ENGINEERING.json', dict(utc=utc(), lock_sha=lock_sha, passed=passed,
                                  elapsed_seconds=elapsed, reference=record, panel=panel))
    print(json.dumps(dict(passed=passed, elapsed_seconds=elapsed,
                          graph_available={rep: {budget: scale is not None for budget, scale in state['scales'].items()}
                                           for rep, state in reference.items()}), indent=2), flush=True)


def forecast(lock_sha):
    engineering = read('ENGINEERING.json')
    assert engineering['passed'] and engineering['lock_sha'] == lock_sha
    start = time.perf_counter()
    panels = []
    for index in range(PANELS):
        reference, record = fit_reference(REFERENCE, 2026345000 + index, 2026345200 + index)
        panel = evaluate(index, 2026345100 + index, EVALUATION, SCREEN, ALARMS, reference)
        panel['reference'] = record
        panels.append(panel)
        print(f'Sealed observable paired panel {index + 1}/{PANELS}', flush=True)
    save('FORECAST.json', dict(utc=utc(), lock_sha=lock_sha, elapsed_seconds=time.perf_counter() - start,
                               budgets=list(BUDGETS), panels=panels))
    print('FORECAST SHA256 ' + sha(HERE / 'FORECAST.json'), flush=True)


def score(lock_sha):
    start = time.perf_counter()
    source = read('FORECAST.json')
    assert source['lock_sha'] == lock_sha and len(source['panels']) == PANELS
    base = old.prior.ac.base
    panels = []
    for row in source['panels']:
        physical_ref = old.shift.replay_physical(row['reference']['receipt']).astype(np.float64)
        physical_sd = physical_ref.std((0, 1))
        physical = old.shift.replay_physical(row['evaluation']).astype(np.float64)
        truth_scores = old.lag_scores(physical / physical_sd)
        truth = base.point_sets(truth_scores)
        native = np.asarray(row['native_set'])
        valid = (truth >= 0).all(1) & (native >= 0).all(1)
        wrong = valid & np.any(truth != native, axis=1)
        selected = np.asarray(row['candidates'])
        sets = np.where((native[selected] >= 0).all(1)[:, None], native[selected], [0, 1])
        true_gap = base.margin(truth_scores[selected], sets)
        oracle = selected[np.lexsort((selected, np.where(valid[selected], true_gap, np.inf)))[:ALARMS]]
        flags = {name: np.asarray(row['flags'][name]) for name in RULES} | {'oracle': oracle}
        assert all(len(np.unique(ids)) == ALARMS and np.isin(ids, selected).all() for ids in flags.values())
        hit_ids = {name: ids[wrong[ids]] for name, ids in flags.items()}
        scale_error = {rep: {budget: None if scale is None else (np.asarray(scale) / physical_sd - 1).tolist()
                             for budget, scale in row['reference']['scales'][rep].items()}
                       for rep in ('fine', 'raw')}
        panels.append(dict(index=row['index'], wrong_total=int(wrong.sum()), wrong_queried=int(wrong[selected].sum()),
                            wrong_ids=np.flatnonzero(wrong).tolist(), invalid=int((~valid).sum()),
                            valid_queried=int(valid[selected].sum()), truth_gap=true_gap.tolist(),
                            hit={name: len(ids) for name, ids in hit_ids.items()},
                            hit_ids={name: ids.tolist() for name, ids in hit_ids.items()},
                            abs_gap_error={name: float(np.abs(np.asarray(row['gaps'][name]) - true_gap)[valid[selected]].sum())
                                           for name in RULES},
                            physical_reference_sd=physical_sd.tolist(), scale_relative_error=scale_error))
    support = np.asarray([row['wrong_queried'] for row in panels])
    hits = {name: np.asarray([row['hit'][name] for row in panels]) for name in (*RULES, 'oracle')}
    bootstrap = np.random.default_rng(BOOTSTRAP_SEED).integers(PANELS, size=(4096, PANELS))
    boot_support = support[bootstrap].sum(1)
    keep = boot_support > 0

    def contrast(left, right):
        difference = hits[left] - hits[right]
        point = float(difference.sum() / support.sum()) if support.sum() else None
        ci = (np.quantile(difference[bootstrap].sum(1)[keep] / boot_support[keep], [.025, .975]).tolist()
              if keep.any() else [None, None])
        return dict(left=left, right=right, delta_hits=int(difference.sum()), delta_recall=point, ci95=ci)

    paired = {budget: contrast(f'fine_{budget}', f'raw_{budget}') for budget in BUDGETS}
    retention = {rep: {budget: contrast(f'{rep}_{budget}', f'{rep}_all') for budget in BUDGETS if budget != 'all'}
                 for rep in ('fine', 'raw')}
    main = paired['64']
    noninferior = retention['fine']['64']
    gain_pass = main['delta_recall'] is not None and main['delta_recall'] >= .05 and main['ci95'][0] > 0
    retention_pass = noninferior['ci95'][0] is not None and noninferior['ci95'][0] > -.05
    failures = {rep: {budget: sum(row['fallback'][rep][budget] for row in source['panels']) for budget in BUDGETS}
                for rep in ('fine', 'raw')}
    safe = {rep: sum(row['audit'][f'{rep}_safe_count'] for row in source['panels']) for rep in ('fine', 'raw')}
    usable = all(failures[rep]['all'] / PANELS <= .05 and safe[rep] >= .95 * PANELS * SCREEN for rep in ('fine', 'raw'))
    enough = support.sum() >= 40 and (support > 0).sum() >= 8
    decision = ('engineering_insufficient' if not usable else 'insufficient_wrong_windows' if not enough else
                'passed' if gain_pass and retention_pass else 'no_independent_fine_value_at_primary_budget')
    total, valid = sum(row['wrong_total'] for row in panels), sum(row['valid_queried'] for row in panels)
    cost = {budget: dict(reference_nodes=REFERENCE * 193 if budget == 'all' else int(budget),
                         reference_branches=8 * (REFERENCE * 193 if budget == 'all' else int(budget)),
                         evaluation_branches=8 * SCREEN * 193,
                         total_branches=8 * ((REFERENCE * 193 if budget == 'all' else int(budget)) + SCREEN * 193))
            for budget in BUDGETS}
    scale_mae = {rep: {budget: (None if all(row['scale_relative_error'][rep][budget] is None for row in panels) else
                                float(np.abs(np.asarray([row['scale_relative_error'][rep][budget] for row in panels
                                                         if row['scale_relative_error'][rep][budget] is not None])).mean()))
                       for budget in BUDGETS} for rep in ('fine', 'raw')}
    summary = dict(utc=utc(), lock_sha=lock_sha, forecast_sha=sha(HERE / 'FORECAST.json'), decision=decision,
                   elapsed_seconds=time.perf_counter() - start, wrong_total=total, wrong_queried=int(support.sum()),
                   queried=PANELS * SCREEN, scored=PANELS * EVALUATION, alarm_count_per_rule=PANELS * ALARMS,
                   panels_with_wrong_queried=int((support > 0).sum()), invalid=sum(row['invalid'] for row in panels),
                   safe_count=safe, graph_failures=failures, query_cost_per_panel=cost,
                   primary=dict(gain_pass=bool(gain_pass), retention_pass=bool(retention_pass),
                                fine64_vs_raw64=main, fine64_vs_fine_all=noninferior),
                   paired_fine_vs_raw=paired, sparse_vs_full=retention,
                   hits={name: dict(count=int(value.sum()), queried_recall=float(value.sum() / support.sum()) if support.sum() else None,
                                    full_recall=float(value.sum() / total) if total else None) for name, value in hits.items()},
                   post_reveal=dict(gap_mae={name: sum(row['abs_gap_error'][name] for row in panels) / valid if valid else None
                                             for name in RULES}, scale_relative_mae=scale_mae), panels=panels)
    save('SUMMARY.json', summary)
    print(json.dumps({key: summary[key] for key in ('decision', 'wrong_queried', 'primary', 'hits', 'graph_failures', 'post_reveal')},
                     indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('lock', 'pilot', 'forecast', 'score'))
    phase = parser.parse_args().phase
    if phase == 'lock':
        lock()
    else:
        {'pilot': pilot, 'forecast': forecast, 'score': score}[phase](verify())
