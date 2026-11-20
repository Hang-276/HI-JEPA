"""Isolate common branch noise and estimate noisy shift moments by replication."""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'fine_active_shift_20261109'
sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location('active_shift', OLD / 'run.py')
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)
tr, src, base, bias, rep, mt, pr = ac.tr, ac.src, ac.base, ac.bias, ac.rep, ac.mt, ac.pr
rep.HERE = HERE

from noise_observations import replicate_pairs

SLOTS, PANELS = tr.SLOTS, 32
ESTIMATORS = ('exact', 'noisy', 'cross')
METHODS = ('native', 'passive') + tuple(f'{view}_{m}' for view in ('fine', 'raw') for m in ESTIMATORS)


def sources():
    return ac.sources() | {str(p): base.sha(p) for p in (
        Path(__file__), HERE / 'noise_observations.py', HERE / 'PROTOCOL.md', HERE / 'THEORY.md',
        OLD / 'LOCK.json', OLD / 'SUMMARY.json')}


def seeds(slot):
    i = SLOTS.index(slot)
    start = 2026176000 + 1000 * i
    return dict(reference=2026175000 + i, panels=[
        (start + 2 * k, start + 2 * k + 1, 2026181000 + 1000 * i + k) for k in range(PANELS)])


def stacked_difference(encode, data, center):
    # Returned design is [replicate, branch × episode × context, feature].
    b = pr.polynomial(encode(data['baseline']) - center, 3)[..., 1:]
    y = pr.polynomial(encode(data['branches']) - center, 3)[..., 1:]
    d = np.moveaxis(y - b[None], 1, 0)
    target = np.broadcast_to(data['shifts'][:, None, None], d.shape[1:-1] + (4,)).reshape(-1, 4)
    return d.reshape(2, -1, d.shape[-1]), target


def solve(x, target, cross):
    columns = np.sqrt(np.mean(x ** 2, axis=(0, 1)))
    if np.any(columns <= 0):
        return None, None
    x = x / columns
    if cross:
        product = x[0].T @ x[1]
        gram = (product + product.T) * .5
        eigen = np.linalg.eigvalsh(gram)
        condition = float(eigen[-1] / eigen[0]) if eigen[0] > 0 else None
        if condition is None or condition >= 1e12:
            return None, condition
        beta = np.linalg.solve(gram, x.mean(0).T @ target)
    else:
        beta, _, rank, singular = np.linalg.lstsq(x.reshape(-1, x.shape[-1]), np.tile(target, (2, 1)), rcond=None)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
        if rank < x.shape[-1] or condition is None or condition >= 1e6:
            return None, condition
    return beta / columns[:, None], condition


def readouts(encode, pairs, reference, evaluation):
    center = encode(pairs['exact']['baseline']).reshape(-1, 4).mean(0)
    exact, target = stacked_difference(encode, pairs['exact'], center)
    noisy, target_noisy = stacked_difference(encode, pairs['noisy'], center)
    np.testing.assert_array_equal(target, target_noisy)
    check, check_target = ac.feature_difference(encode, pairs['check'], center, 3)
    r = pr.polynomial(reference - center, 3)[..., 1:]
    y = pr.polynomial(evaluation - center, 3)[..., 1:]
    result, audit = {}, {}
    for method, x in zip(ESTIMATORS, (exact, noisy, noisy)):
        beta, cond = solve(x, target, method == 'cross')
        audit[method] = dict(condition=cond, check=None)
        if beta is None:
            result[method] = None
            continue
        scale = (r @ beta).std((0, 1))
        assert np.all(scale > 0)
        result[method] = mt.scores_on((y @ beta) / scale)
        audit[method]['check'] = np.linalg.norm(check @ beta - check_target) / np.linalg.norm(check_target)
    return result, audit


def forecast_slot(slot):
    _, views, actions, _ = src.reference.historical(slot)
    encode = {name: views[name][1] for name in ('fine', 'raw')}
    eps, roots = tr.target(slot), seeds(slot)
    public, reference = src.shift_data.fresh_observations(slot, eps, tr.REFERENCE_N, roots['reference'], 'evaluation')
    ref = {name: read(public['x']) for name, read in encode.items()}
    cov = bias.all_cov(ref['fine'])
    native = bias.unit_rows(np.eye(4)[base.historical_match(slot, 0)['perm']], cov)
    fold = np.arange(len(actions)) % 4
    fit, check = fold == 0, fold == 1
    panels = []
    for i, (cs, es, ss) in enumerate(roots['panels']):
        c, cr = src.shift_data.fresh_observations(slot, eps, len(actions), cs, 'calibration')
        e, er = src.shift_data.fresh_observations(slot, eps, src.previous.EVAL, es, 'evaluation')
        pairs, receipt = replicate_pairs(cr, fit, check, ac.TIMES, ss)
        yc, ye = encode['fine'](c['x']), {name: read(e['x']) for name, read in encode.items()}
        drift = src.eq.fit_drift(yc[fit], c['actions'][fit], c['regimes'][fit])
        rows = src.eq.rows(yc, c['actions'], c['regimes'], drift, cov, ~fit)
        ns = mt.scores_on(ye['fine'] @ native.T)
        own = base.point_sets(ns)
        scores = dict(native=ns, passive=ns if rows is None else mt.scores_on(ye['fine'] @ rows.T))
        audit = {}
        for name, read in encode.items():
            estimates, debug = readouts(read, pairs, ref[name], ye[name])
            for m in ESTIMATORS:
                scores[f'{name}_{m}'] = ns if estimates[m] is None else estimates[m]
                audit[f'{name}_{m}'] = debug[m]
        panels.append(dict(index=i, calibration=cr, evaluation=er, query=receipt, audit=audit,
                           missing_passive=rows is None,
                           predictions={name: mt.flags(values, own) for name, values in scores.items()}))
    return dict(reference=reference, panels=panels)


def score_slot(slot, frozen):
    scale = src.shift_data.replay_physical(frozen['reference']).astype(np.float64).std((0, 1))
    panels = []
    for row in frozen['panels']:
        values = row['predictions']
        panels.append(tr.measure(dict(evaluation=row['evaluation'], point={m: v['point'] for m, v in values.items()},
                                      alerts={m: v['alarm'] for m, v in values.items()}), scale))
    rng = np.random.default_rng(2026176901 + SLOTS.index(slot))
    draws = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(d, minlength=PANELS) for d in draws])
    point = {m: {k: rep.bootstrap([p['point'][m][k] for p in panels], weights) for k in ('near', 'all')}
             for m in METHODS}
    alarm = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in panels], weights) for k in ('hit', 'fpr')}
             for m in METHODS}
    contrasts, diagnostic = {}, {}
    for name in ('fine', 'raw'):
        for variant, reference in (('noisy', 'exact'), ('cross', 'noisy'), ('cross', 'exact')):
            m, r = f'{name}_{variant}', f'{name}_{reference}'
            near = rep.paired_interval(point[m]['near'], point[r]['near'])
            near['lower_family'] = rep.upper(point[m]['near'][1] - point[r]['near'][1], .0125)
            threshold = near['gain'] >= .03 and near['lower_family'] > 0 if reference == 'noisy' else near['lower_family'] > -.03
            contrasts[f'{m}_vs_{r}'] = dict(near=near, all=rep.paired_interval(point[m]['all'], point[r]['all']),
                                           hit=rep.paired_interval(alarm[m]['hit'], alarm[r]['hit']), passed=threshold)
        for variant in ESTIMATORS:
            m = f'{name}_{variant}'
            audit = [p['audit'][m] for p in frozen['panels']]
            valid = [a for a in audit if a['check'] is not None]
            diagnostic[m] = dict(unavailable=len(audit) - len(valid), summary={
                k: [float(fn([v[k] for v in valid])) for fn in (np.min, np.median, np.max)] if valid else None
                for k in ('condition', 'check')})
    return dict(point={m: {k: v[0] for k, v in d.items()} for m, d in point.items()},
                alarm={m: {k: v[0] for k, v in d.items()} for m, d in alarm.items()},
                contrasts=contrasts, diagnostics=diagnostic,
                missing_passive=sum(p['missing_passive'] for p in frozen['panels']), panels=panels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS}))
        return
    lock_sha = base.sha(lock)
    assert base.read(lock)['sources'] == sources()
    if phase == 'forecast':
        slots = {s: forecast_slot(s) for s in SLOTS}
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=lock_sha, slots=slots))
        return
    frozen = base.read(forecast)
    assert frozen['lock_sha'] == lock_sha
    slots = {s: score_slot(s, frozen['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=lock_sha, forecast_sha=base.sha(forecast), slots=slots))


if __name__ == '__main__':
    main()
