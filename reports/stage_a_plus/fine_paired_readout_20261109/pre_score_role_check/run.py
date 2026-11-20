"""Use paired simulator coordinates to isolate readout errors on fresh panels."""
import argparse
import importlib.util
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'fine_marginal_transport_20261109'
spec = importlib.util.spec_from_file_location('marginal_transport', OLD / 'run.py')
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)
tr, src, base, bias, rep = mt.tr, mt.src, mt.base, mt.bias, mt.rep
rep.HERE = HERE

SLOTS, PANELS, PAIRED_N = tr.SLOTS, 32, 24
PUBLIC = ('native', 'affine')
PAIRED = ('own_linear', 'own_cubic', 'full_linear', 'full_cubic')
CONTRASTS = (('own_cubic', 'own_linear'), ('full_linear', 'own_linear'),
             ('full_cubic', 'full_linear'), ('full_cubic', 'own_cubic'))


def seeds(slot):
    i = SLOTS.index(slot)
    start = 2026157000 + 1000 * i
    return dict(reference=2026156000 + i,
                panels=[tuple(range(start + 3 * k, start + 3 * k + 3)) for k in range(PANELS)])


def sources():
    return mt.sources() | {str(p): base.sha(p) for p in (
        Path(__file__), HERE / 'PROTOCOL.md', OLD / 'LOCK.json', OLD / 'SUMMARY.json')}


def forecast_slot(slot):
    _, views, a, _ = src.reference.historical(slot)
    _, encode = views['fine']
    eps, roots = tr.target(slot), seeds(slot)
    ref, receipt = src.shift_data.fresh_observations(slot, eps, tr.REFERENCE_N, roots['reference'], 'evaluation')
    cov = bias.all_cov(encode(ref['x']))
    native = bias.unit_rows(np.eye(4)[base.historical_match(slot, 0)['perm']], cov)
    fit = np.arange(len(a)) % 4 == 0
    panels = []
    for k, (cs, ps, es) in enumerate(roots['panels']):
        cal, cr = src.shift_data.fresh_observations(slot, eps, len(a), cs, 'calibration')
        _, pr = src.shift_data.fresh_observations(slot, eps, PAIRED_N, ps, 'calibration')
        ev, er = src.shift_data.fresh_observations(slot, eps, src.previous.EVAL, es, 'evaluation')
        yc, ye = encode(cal['x']), encode(ev['x'])
        drift = src.eq.fit_drift(yc[fit], cal['actions'][fit], cal['regimes'][fit])
        rows = src.eq.rows(yc, cal['actions'], cal['regimes'], drift, cov, ~fit)
        ns = mt.scores_on(ye @ native.T)
        own = base.point_sets(ns)
        scores = {'native': ns, 'affine': ns if rows is None else mt.scores_on(ye @ rows.T)}
        panels.append(dict(index=k, calibration=cr, paired=pr, evaluation=er, rows=rows,
                           predictions={name: mt.flags(values, own) for name, values in scores.items()}))
    return dict(reference=receipt, panels=panels)


def polynomial(values, degree):
    terms = [np.ones(values.shape[:-1])]
    for d in range(1, degree + 1):
        for indices in combinations_with_replacement(range(values.shape[-1]), d):
            terms.append(np.prod(values[..., indices], axis=-1))
    return np.stack(terms, axis=-1)


def fit_and_predict(x, z, reference, evaluation, degree, own):
    if not own:
        beta, _, rank, _ = np.linalg.lstsq(polynomial(x, degree), z, rcond=None)
        assert rank == beta.shape[0]
        return polynomial(reference, degree) @ beta, polynomial(evaluation, degree) @ beta

    outputs = [[], []]
    for j in range(4):
        axis = slice(j, j + 1)
        beta, _, rank, _ = np.linalg.lstsq(polynomial(x[:, axis], degree), z[:, j], rcond=None)
        assert rank == len(beta)
        for out, values in zip(outputs, (reference, evaluation)):
            out.append(polynomial(values[..., axis], degree) @ beta)
    return tuple(np.stack(out, axis=-1) for out in outputs)


def paired_slot(slot, frozen):
    _, views, _, _ = src.reference.historical(slot)
    _, encode = views['fine']
    reference = encode(mt.public(slot, frozen['reference'], 'evaluation')['x'])
    panels = []
    checked = False
    for row in frozen['panels']:
        if row['rows'] is None:
            panels.append({m: row['predictions']['native'] for m in PAIRED})
            continue
        w = np.asarray(row['rows'])
        xp = encode(mt.public(slot, row['paired'], 'calibration')['x']).reshape(-1, 4)
        yp = encode(mt.public(slot, row['evaluation'], 'evaluation')['x'])
        x = xp @ w.T
        z = src.shift_data.replay_physical(row['paired']).astype(np.float64).reshape(-1, 4)
        r = reference @ w.T
        y = yp @ w.T
        center = x.mean(0)
        own = np.asarray(row['predictions']['native']['point'])
        result = {}
        for method, degree, own_axis in zip(PAIRED, (1, 3, 1, 3), (True, True, False, False)):
            rhat, yhat = fit_and_predict(x - center, z, r - center, y - center, degree, own_axis)
            scale = rhat.std((0, 1))
            assert np.all(scale > 0)
            scores = mt.scores_on(yhat / scale)
            if method == 'own_linear':
                np.testing.assert_allclose(scores, mt.scores_on(y / r.std((0, 1))), rtol=0, atol=1e-11)
            if not checked and not own_axis:
                xp0 = xp.mean(0)
                raw_r, raw_y = fit_and_predict(xp - xp0, z, reference - xp0, yp - xp0, degree, False)
                np.testing.assert_allclose(scores, mt.scores_on(raw_y / raw_r.std((0, 1))), rtol=0, atol=1e-10)
            result[method] = mt.flags(scores, own)
        checked = True
        panels.append(result)
    return panels


def score_slot(slot, frozen, paired):
    scale = src.shift_data.replay_physical(frozen['reference']).astype(np.float64).std((0, 1))
    panels = []
    for row, oracle in zip(frozen['panels'], paired, strict=True):
        values = row['predictions'] | oracle
        data = dict(evaluation=row['evaluation'], point={m: v['point'] for m, v in values.items()},
                    alerts={m: v['alarm'] for m, v in values.items()})
        panels.append(tr.measure(data, scale))
    rng = np.random.default_rng(2026157901 + SLOTS.index(slot))
    draws = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(d, minlength=PANELS) for d in draws])
    point = {m: {k: rep.bootstrap([p['point'][m][k] for p in panels], weights) for k in ('near', 'all')}
             for m in PUBLIC + PAIRED}
    alarm = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in panels], weights) for k in ('hit', 'fpr')}
             for m in PUBLIC + PAIRED}
    contrasts = {}
    for m, r in CONTRASTS:
        near = rep.paired_interval(point[m]['near'], point[r]['near'])
        near['lower_family'] = rep.upper(point[m]['near'][1] - point[r]['near'][1], .05 / 6)
        all_windows = rep.paired_interval(point[m]['all'], point[r]['all'])
        contrasts[f'{m}_vs_{r}'] = dict(near=near, all=all_windows, hit=rep.paired_interval(alarm[m]['hit'], alarm[r]['hit']),
                                      passed=near['gain'] >= .03 and near['lower_family'] > 0 and all_windows['gain'] > -.01)
    return dict(point={m: {k: v[0] for k, v in d.items()} for m, d in point.items()},
                alarm={m: {k: v[0] for k, v in d.items()} for m, d in alarm.items()}, contrasts=contrasts,
                unavailable=sum(p['rows'] is None for p in frozen['panels']), panels=panels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'forecast', 'paired', 'score'))
    phase = parser.parse_args().phase
    lock, forecast, oracle = HERE / 'LOCK.json', HERE / 'FORECAST.json', HERE / 'PAIRED.json'
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
    forecast_sha = base.sha(forecast)
    assert frozen['lock_sha'] == lock_sha
    if phase == 'paired':
        slots = {s: paired_slot(s, frozen['slots'][s]) for s in SLOTS}
        rep.write('PAIRED.json', dict(utc=base.now(), lock_sha=lock_sha, forecast_sha=forecast_sha, slots=slots))
        return
    paired = base.read(oracle)
    assert paired['lock_sha'] == lock_sha and paired['forecast_sha'] == forecast_sha
    slots = {s: score_slot(s, frozen['slots'][s], paired['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=lock_sha, forecast_sha=forecast_sha,
                                   paired_sha=base.sha(oracle), slots=slots))


if __name__ == '__main__':
    main()
