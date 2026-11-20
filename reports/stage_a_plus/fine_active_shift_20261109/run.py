"""Learn readout differences from active interventions, then score fresh paths."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

from active_observations import observed_pairs, sources as active_sources

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'fine_paired_readout_20261109'
spec = importlib.util.spec_from_file_location('paired_readout', OLD / 'run.py')
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)
tr, src, base, bias, rep, mt = pr.tr, pr.src, pr.base, pr.bias, pr.rep, pr.mt
rep.HERE = HERE

SLOTS, PANELS = tr.SLOTS, 32
TIMES = (8, 24, 40, 56, 72, 88, 104, 120)
METHODS = ('native', 'passive', 'fine_linear', 'fine_cubic', 'raw_linear', 'raw_cubic')
CONTRASTS = (('fine_cubic', 'passive'), ('fine_cubic', 'raw_cubic'),
             ('fine_cubic', 'fine_linear'), ('raw_cubic', 'raw_linear'))


def seeds(slot):
    i = SLOTS.index(slot)
    start = 2026159000 + 1000 * i
    return dict(reference=2026158000 + i, panels=[
        (start + 2 * k, start + 2 * k + 1, 2026174000 + 1000 * i + k) for k in range(PANELS)])


def sources():
    return pr.sources() | active_sources() | {str(p): base.sha(p) for p in (
        Path(__file__), HERE / 'PROTOCOL.md', HERE / 'THEORY.md', OLD / 'LOCK.json', OLD / 'SUMMARY.json')}


def feature_difference(encode, pairs, center, degree):
    y0 = pr.polynomial(encode(pairs['baseline']) - center, degree)[..., 1:]
    ys = pr.polynomial(encode(pairs['branches']) - center, degree)[..., 1:]
    diff = ys - y0[None]
    target = np.broadcast_to(pairs['shifts'][:, None, None], diff.shape[:-1] + (4,))
    return diff.reshape(-1, diff.shape[-1]), target.reshape(-1, 4)


def fit_scores(encode, pairs, ref, y, degree):
    center = encode(pairs['fit']['baseline']).reshape(-1, 4).mean(0)
    x, target = feature_difference(encode, pairs['fit'], center, degree)
    xscale = np.sqrt(np.mean(x ** 2, axis=0))
    if np.any(xscale <= 0):
        return None, dict(rank=0, condition=None, fit=None, check=None)
    beta, _, rank, singular = np.linalg.lstsq(x / xscale, target, rcond=None)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    if rank < x.shape[1] or condition is None or condition >= 1e6:
        return None, dict(rank=int(rank), condition=condition, fit=None, check=None)
    beta /= xscale[:, None]
    chk, target_chk = feature_difference(encode, pairs['check'], center, degree)
    fit_error = np.linalg.norm(x @ beta - target) / np.linalg.norm(target)
    check_error = np.linalg.norm(chk @ beta - target_chk) / np.linalg.norm(target_chk)
    estimated = pr.polynomial(y - center, degree)[..., 1:] @ beta
    reference = pr.polynomial(ref - center, degree)[..., 1:] @ beta
    sd = reference.std((0, 1))
    assert np.all(sd > 0) and np.all(np.isfinite(estimated))
    return mt.scores_on(estimated / sd), dict(rank=int(rank), condition=condition,
                                             fit=fit_error, check=check_error)


def forecast_slot(slot):
    _, views, a, _ = src.reference.historical(slot)
    encode = {name: views[name][1] for name in ('fine', 'raw')}
    eps, roots = tr.target(slot), seeds(slot)
    public, reference = src.shift_data.fresh_observations(slot, eps, tr.REFERENCE_N, roots['reference'], 'evaluation')
    ref = {name: read(public['x']) for name, read in encode.items()}
    cov = bias.all_cov(ref['fine'])
    native = bias.unit_rows(np.eye(4)[base.historical_match(slot, 0)['perm']], cov)
    fold = np.arange(len(a)) % 4
    fit, check = fold == 0, fold == 1
    panels = []
    for k, (cs, es, ss) in enumerate(roots['panels']):
        c, cr = src.shift_data.fresh_observations(slot, eps, len(a), cs, 'calibration')
        e, er = src.shift_data.fresh_observations(slot, eps, src.previous.EVAL, es, 'evaluation')
        pairs, receipt = observed_pairs(cr, fit, check, TIMES, ss)
        yc, ye = encode['fine'](c['x']), {name: read(e['x']) for name, read in encode.items()}
        drift = src.eq.fit_drift(yc[fit], c['actions'][fit], c['regimes'][fit])
        rows = src.eq.rows(yc, c['actions'], c['regimes'], drift, cov, ~fit)
        ns = mt.scores_on(ye['fine'] @ native.T)
        own = base.point_sets(ns)
        scores = dict(native=ns, passive=ns if rows is None else mt.scores_on(ye['fine'] @ rows.T))
        audit = {}
        for name, read in encode.items():
            for shape, degree in (('linear', 1), ('cubic', 3)):
                method = f'{name}_{shape}'
                result, audit[method] = fit_scores(read, pairs, ref[name], ye[name], degree)
                scores[method] = ns if result is None else result
        panels.append(dict(index=k, calibration=cr, evaluation=er, active=receipt, audit=audit,
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
    rng = np.random.default_rng(2026159901 + SLOTS.index(slot))
    draws = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(d, minlength=PANELS) for d in draws])
    point = {m: {k: rep.bootstrap([p['point'][m][k] for p in panels], weights) for k in ('near', 'all')}
             for m in METHODS}
    alarm = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in panels], weights) for k in ('hit', 'fpr')}
             for m in METHODS}
    contrasts = {}
    for m, r in CONTRASTS:
        near = rep.paired_interval(point[m]['near'], point[r]['near'])
        near['lower_family'] = rep.upper(point[m]['near'][1] - point[r]['near'][1], .0125)
        all_windows = rep.paired_interval(point[m]['all'], point[r]['all'])
        contrasts[f'{m}_vs_{r}'] = dict(near=near, all=all_windows, hit=rep.paired_interval(alarm[m]['hit'], alarm[r]['hit']),
                                      passed=near['gain'] >= .03 and near['lower_family'] > 0 and all_windows['gain'] > -.01)
    diagnostic = {}
    for m in METHODS[2:]:
        values = [p['audit'][m] for p in frozen['panels']]
        valid = [x for x in values if x['fit'] is not None]
        diagnostic[m] = dict(unavailable=len(values) - len(valid), summary={
            k: [float(fn([v[k] for v in valid])) for fn in (np.min, np.median, np.max)] if valid else None
            for k in ('condition', 'fit', 'check')})
    return dict(point={m: {k: v[0] for k, v in d.items()} for m, d in point.items()},
                alarm={m: {k: v[0] for k, v in d.items()} for m, d in alarm.items()}, contrasts=contrasts,
                diagnostics=diagnostic, missing_passive=sum(p['missing_passive'] for p in frozen['panels']),
                panels=panels)


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
