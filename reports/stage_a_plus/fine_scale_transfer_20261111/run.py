"""Evaluate denominator repair on two frozen encoders, with sealed predictions."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
sys.dont_write_bytecode = True

import numpy as np

import adapter

HERE = Path(__file__).resolve().parent
DENSE = HERE.parent / 'fine_reference_graph_20261111'
spec = importlib.util.spec_from_file_location('prior_dense_scale', DENSE / 'run.py')
dense = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dense)
base = dense.old.prior.ac.base
lag_scores = dense.old.lag_scores
KEYS = ('eps05_seed1_to008', 'eps008_seed0_to05')
PANELS, REFERENCES, EVALUATIONS = 16, 96, 128
DOSE = .0175
GAP_CUTOFFS = (.01, .03)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def write(name, data):
    payload = dict(utc=datetime.now(timezone.utc).isoformat(), **data)
    with (HERE / name).open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def lock():
    dense_lock = dense.verify()
    paths = [HERE / name for name in ('PROTOCOL.md', 'run.py', 'adapter.py', 'ADAPTER_ENGINEERING.json')]
    inherited = dense.read('LOCK.json')['sources']
    sources = inherited | adapter.sources() | {str(path): sha(path) for path in paths}
    write('LOCK.json', dict(sources=sources, dense_lock=dense_lock, python=sys.executable,
                            numpy=np.__version__, keys=list(KEYS)))


def verify():
    record = read('LOCK.json')
    for path, expected in record['sources'].items():
        if sha(path) != expected:
            raise ValueError(f'Frozen source changed: {path}')
    assert record['dense_lock'] == dense.verify()
    return sha(HERE / 'LOCK.json')


def responses(key, count, seed):
    public, receipt = adapter.public_panel(key, count, seed)
    plus, minus, audit = adapter.endpoint_branches(key, public, receipt, np.arange(count), DOSE)
    assert audit['public_parity']
    assert plus.shape == minus.shape == (count, 193, 4, 4)
    return public['x'], np.stack([plus, minus]), receipt, audit


def reference(state, x, branches):
    y = state['encode'](x)
    beta = state['beta']
    plus, minus = state['encode'](branches)
    z = y @ beta
    jacobian = ((plus - minus) @ beta) / (2 * DOSE)
    graph_sd, audit = dense.reconstruct_reference(z, jacobian)
    native_sd = y.std((0, 1))
    affine_sd = z.std((0, 1))
    assert (native_sd > 0.).all() and (affine_sd > 0.).all()
    return native_sd, affine_sd, graph_sd, audit


def estimate(state, x, branches, ref, engineering):
    native_sd, affine_sd, graph_sd, _ = ref
    y = state['encode'](x)
    plus, minus = state['encode'](branches)
    local = dense.old.evaluate(y, plus, minus, native_sd, state['beta'], np.ones(4), DOSE)
    numerator, safe = local['signed_scores'], local['safe']
    affine = local['affine_scores'] / affine_sd**2
    old = np.where(safe[:, None], numerator / affine_sd**2, affine)
    graph = old if graph_sd is None else np.where(safe[:, None], numerator / graph_sd**2, affine)
    audit = dict(safe=safe.tolist(), max_condition=float(local['max_condition'].max()),
                 algebra_error=float(local['algebra_error'].max()))
    if engineering:
        historical = np.diagonal(base.score_matrix(y / native_sd), axis1=1, axis2=2)
        audit['historical_score_error'] = float(np.abs(historical - local['native_scores']).max())
        direct = dense.old.evaluate(y, plus, minus, native_sd, state['beta'], affine_sd, DOSE)
        audit['rescaling_error'] = float(np.abs(direct['signed_scores'] - old).max())
        if graph_sd is not None:
            direct_graph = dense.old.evaluate(y, plus, minus, native_sd, state['beta'], graph_sd, DOSE)
            expected = np.where(safe[:, None], direct_graph['signed_scores'], affine)
            audit['rescaling_error'] = max(audit['rescaling_error'], float(np.abs(expected - graph).max()))
    scores = dict(native=local['native_scores'], affine=affine, signed_oldscale=old, signed_graphscale=graph)
    return scores, numerator, audit


def panel(key, model, index, reference_count, evaluation_count, ref_seed, eval_seed):
    xr, branches, ref_receipt, ref_audit = responses(key, reference_count, ref_seed)
    representations = {'fine': model, 'raw': model['raw']}
    refs = {name: reference(state, xr, branches) for name, state in representations.items()}
    x, branches, eval_receipt, eval_audit = responses(key, evaluation_count, eval_seed)
    fine, numerator, audit = estimate(model, x, branches, refs['fine'], engineering=index < 0)
    raw, _, raw_audit = estimate(model['raw'], x, branches, refs['raw'], engineering=index < 0)
    ordered = np.sort(fine['signed_oldscale'], axis=1)
    stable = ((ordered[:, 2] - ordered[:, 1] > np.ptp(fine['signed_graphscale'] - fine['signed_oldscale'], axis=1))
              & np.asarray(audit['safe']) & (refs['fine'][2] is not None))
    scores = fine | {'raw_affine': raw['affine'], 'raw_graph': raw['signed_graphscale']}
    assert np.array_equal(base.point_sets(scores['signed_oldscale'])[stable],
                          base.point_sets(scores['signed_graphscale'])[stable])
    return dict(index=index, reference=ref_receipt, evaluation=eval_receipt,
                reference_query=ref_audit, evaluation_query=eval_audit,
                refs={name: dict(native_sd=ref[0].tolist(), affine_sd=ref[1].tolist(),
                                 graph_sd=None if ref[2] is None else ref[2].tolist(), graph=ref[3])
                      for name, ref in refs.items()}, scores={name: value.tolist() for name, value in scores.items()},
                numerator=numerator.tolist(), stable=stable.tolist(), audit=audit, raw_audit=raw_audit)


def pilot(lock_sha):
    start = time.perf_counter()
    records = {}
    for m, key in enumerate(KEYS):
        model = adapter.open_model(key)
        seed = 2026345900 + 10*m
        trial = panel(key, model, -1, 8, 8, seed, seed + 1)
        passed = all(trial[name][check] <= 1e-9 for name in ('audit', 'raw_audit')
                     for check in ('algebra_error', 'historical_score_error', 'rescaling_error'))
        records[key] = dict(passed=passed, calibration=model['provenance'], trial=trial)
        print(key, 'engineering passed:', passed, flush=True)
    write('ENGINEERING.json', dict(lock_sha=lock_sha, records=records, seconds=time.perf_counter()-start))


def forecast(lock_sha):
    engineering = read('ENGINEERING.json')
    assert engineering['lock_sha'] == lock_sha
    start = time.perf_counter()
    records = {}
    for m, key in enumerate(KEYS):
        if not engineering['records'][key]['passed']:
            continue
        model = adapter.open_model(key)
        assert model['provenance'] == engineering['records'][key]['calibration']
        panels = []
        for g in range(PANELS):
            panels.append(panel(key, model, g, REFERENCES, EVALUATIONS,
                                2026346000 + 1000*m + g, 2026346200 + 1000*m + g))
            if (g + 1) % 4 == 0:
                print(f'{key}: {g + 1}/{PANELS} observable panels', flush=True)
        records[key] = dict(calibration=model['provenance'], panels=panels)
    write('FORECAST.json', dict(lock_sha=lock_sha, records=records, seconds=time.perf_counter()-start))
    print('FORECAST SHA256', sha(HERE / 'FORECAST.json'), flush=True)


def score_panel(row):
    physical_ref = adapter.replay_physical(row['reference']).astype(np.float64)
    physical = adapter.replay_physical(row['evaluation']).astype(np.float64)
    physical_sd = physical_ref.std((0, 1))
    physical_num = lag_scores(physical)
    truth_scores = physical_num / physical_sd**2
    truth = base.point_sets(truth_scores)
    valid = (truth >= 0).all(1)
    fine_ref = row['refs']['fine']
    graph_available = fine_ref['graph_sd'] is not None
    affine_sd = np.asarray(fine_ref['affine_sd'])
    eligible = valid & np.asarray(row['audit']['safe']) & graph_available
    scores = {name: np.asarray(x) for name, x in row['scores'].items()}
    native_set = np.argsort(scores['native'], axis=1, kind='stable')[:, :2]
    gap = base.margin(truth_scores, native_set)
    errors = {name: np.any(base.point_sets(x) != truth, axis=1) for name, x in scores.items()}
    abs_gap_errors = {name: np.abs(base.margin(x, native_set) - gap) for name, x in scores.items()}
    oracle_scores = dict(physical_numerator=physical_num / affine_sd**2,
                         physical_denominator=np.asarray(row['numerator']) / physical_sd**2,
                         physical_both=truth_scores)

    def summarize(mask, include_oracle=False):
        old, new = errors['signed_oldscale'], errors['signed_graphscale']
        error_count = {name: int(x[mask].sum()) for name, x in errors.items()}
        gap_error_sum = {name: float(x[mask].sum()) for name, x in abs_gap_errors.items()}
        if include_oracle:
            for name, x in oracle_scores.items():
                error_count[name] = int(np.any(base.point_sets(x) != truth, axis=1)[mask].sum())
                gap_error_sum[name] = float(np.abs(base.margin(x, native_set) - gap)[mask].sum())
        count = int(mask.sum())
        return dict(count=count, errors=error_count, gap_error_sum=gap_error_sum,
                    graph_available_count=count if graph_available else 0,
                    fixed=int((mask & old & ~new).sum()), introduced=int((mask & ~old & new).sum()))

    sorted_truth = np.sort(truth_scores, axis=1)
    truth_boundary = sorted_truth[:, 2] - sorted_truth[:, 1]
    bins = np.digitize(truth_boundary, GAP_CUTOFFS, right=True)
    stable = np.asarray(row['stable'])
    return dict(index=row['index'], physical_sd=physical_sd.tolist(),
                relative_sd_error={rep: {name: None if state[name] is None else
                                         (np.asarray(state[name]) / physical_sd - 1.).tolist()
                                         for name in ('affine_sd', 'graph_sd')} for rep, state in row['refs'].items()},
                valid=summarize(valid), eligible=summarize(eligible, include_oracle=True),
                boundary=[summarize(valid & (bins == k)) for k in range(3)],
                stable=summarize(valid & stable), other=summarize(valid & ~stable),
                graph_pass={rep: state['graph_sd'] is not None for rep, state in row['refs'].items()},
                safe={rep: sum(row[name]['safe']) for rep, name in (('fine', 'audit'), ('raw', 'raw_audit'))})


def aggregate(parts):
    n = sum(row['count'] for row in parts)
    errors = {k: sum(row['errors'][k] for row in parts) for k in parts[0]['errors']}
    return dict(count=n, errors=errors, risk={k: v / n if n else None for k, v in errors.items()},
                graph_available_count=sum(row['graph_available_count'] for row in parts),
                gap_mae={k: sum(row['gap_error_sum'][k] for row in parts) / n if n else None
                         for k in parts[0]['gap_error_sum']},
                fixed=sum(row['fixed'] for row in parts), introduced=sum(row['introduced'] for row in parts))


def paired(rows, sample, scope, key, a, b):
    n = np.asarray([row[scope]['count'] for row in rows])
    delta = np.asarray([row[scope][key][a] - row[scope][key][b] for row in rows])
    boot_n = n[sample].sum(1)
    keep = boot_n > 0
    return dict(delta=float(delta.sum() / n.sum()) if n.sum() else None,
                ci95=np.quantile(delta[sample].sum(1)[keep] / boot_n[keep], [.025, .975]).tolist()
                if keep.any() else [None, None])


def summarize_model(rows, model_index):
    sample = np.random.default_rng(2026346800 + model_index).integers(PANELS, size=(4096, PANELS))
    valid = aggregate([row['valid'] for row in rows])
    eligible = aggregate([row['eligible'] for row in rows])
    main = paired(rows, sample, 'valid', 'errors', 'signed_oldscale', 'signed_graphscale')
    mechanism = paired(rows, sample, 'eligible', 'gap_error_sum', 'physical_numerator', 'physical_denominator')
    old_mae = eligible['gap_mae']['signed_oldscale']
    rel = (1 - eligible['gap_mae']['physical_denominator'] / old_mae) if old_mae else None
    graph_pass = {rep: sum(row['graph_pass'][rep] for row in rows) for rep in ('fine', 'raw')}
    safe = {rep: sum(row['safe'][rep] for row in rows) for rep in ('fine', 'raw')}
    main_pass = (graph_pass['fine'] >= 15 and safe['fine'] >= .95*PANELS*EVALUATIONS
                 and main['delta'] is not None and main['delta'] >= .01 and main['ci95'][0] > 0.)
    mechanism_pass = rel is not None and rel >= .50 and mechanism['ci95'][0] > 0.
    return dict(main=main, main_pass=main_pass, mechanism=mechanism, mechanism_pass=mechanism_pass,
                oracle_denominator_relative_gap_mae_reduction=rel, graph_pass=graph_pass, safe=safe,
                valid=valid, eligible=eligible,
                boundary=[aggregate([row['boundary'][k] for row in rows]) for k in range(3)],
                stable=aggregate([row['stable'] for row in rows]), other=aggregate([row['other'] for row in rows]),
                panels=rows)


def score(lock_sha):
    forecast = read('FORECAST.json')
    assert forecast['lock_sha'] == lock_sha
    start = time.perf_counter()
    records = {}
    for m, key in enumerate(KEYS):
        if key not in forecast['records']:
            continue
        rows = [score_panel(row) for row in forecast['records'][key]['panels']]
        assert len(rows) == PANELS
        records[key] = summarize_model(rows, m)
        print(key, records[key]['main'], records[key]['mechanism'], flush=True)
    write('SUMMARY.json', dict(lock_sha=lock_sha, forecast_sha=sha(HERE / 'FORECAST.json'),
                               records=records, seconds=time.perf_counter()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('lock', 'pilot', 'forecast', 'score'))
    args = parser.parse_args()
    if args.phase == 'lock':
        lock()
    else:
        {'pilot': pilot, 'forecast': forecast, 'score': score}[args.phase](verify())
