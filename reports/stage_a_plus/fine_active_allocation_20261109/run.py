"""Reallocate the same number of active queries between repetition and coverage."""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'fine_active_dose_20261109'
sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location('active_dose', OLD / 'run.py')
ad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ad)
tr, src, base, bias, rep, mt, pr, ac = ad.tr, ad.src, ad.base, ad.bias, ad.rep, ad.mt, ad.pr, ad.ac
rep.HERE = HERE

from query import ARMS, queries

SLOTS, PANELS = tr.SLOTS, 32
METHODS = ('native', 'passive') + tuple(f'{view}_{arm}' for view in ('fine', 'raw') for arm in ARMS)
COMPARISONS = (('mean32', 'pooled2'), ('mean32', 'exact2'), ('mean2', 'pooled2'),
               ('mean8', 'mean2'), ('mean32', 'mean8'), ('exact32', 'exact2'), ('mean32', 'exact32'))


def seeds(slot):
    i = SLOTS.index(slot)
    start = 2026212000 + 1000 * i
    return dict(reference=2026210000 + i, panels=[
        (start + 2 * k, start + 2 * k + 1, 2026215000 + 1000 * i + k) for k in range(PANELS)])


def sources():
    return ad.sources() | {str(p): base.sha(p) for p in (
        Path(__file__), HERE / 'query.py', HERE / 'PROTOCOL.md', OLD / 'LOCK.json', OLD / 'SUMMARY.json')}


def check_roots():
    old = ad.archived_roots()
    new, q = [], []
    for s in SLOTS:
        plan = seeds(s)
        new.append(src.shift_data._stream_seed(s, plan['reference'], 'evaluation'))
        for c, e, p in plan['panels']:
            new += [src.shift_data._stream_seed(s, c, 'calibration'), src.shift_data._stream_seed(s, e, 'evaluation')]
            q.append(p)
    assert len(new) == len(set(new)) and not set(new) & old and len(q) == len(set(q))
    return dict(new_roots=len(new), checked_archived_roots=len(old), query_seeds=len(q))


def regress(data, target):
    scale = np.sqrt(np.mean(data ** 2, axis=0))
    if np.any(scale <= 0):
        return None, None
    beta, _, rank, s = np.linalg.lstsq(data / scale, target, rcond=None)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else None
    if rank < data.shape[1] or cond is None or cond >= 1e6:
        return None, cond
    return beta / scale[:, None], cond


def readouts(encode, pairs, held, reference, evaluation):
    center = reference.mean((0, 1))
    check, target = ac.feature_difference(encode, held, center, 3)
    reference = pr.polynomial(reference - center, 3)[..., 1:]
    evaluation = pr.polynomial(evaluation - center, 3)[..., 1:]
    scores, audit = {}, {}
    pooled_design = None
    for arm in ARMS:
        if arm == 'mean2':
            d = pooled_design.mean(1)
        else:
            pair = pairs[arm]
            b = pr.polynomial(encode(pair['baseline']) - center, 3)[..., 1:]
            y = pr.polynomial(encode(pair['branches']) - center, 3)[..., 1:]
            difference = y - b[None]
            if arm == 'pooled2':
                pooled_design, d = difference, difference
            else:
                d = difference.mean(1)
        truth = np.broadcast_to(pairs[arm]['shifts'].reshape((8,) + (1,) * (d.ndim - 2) + (4,)),
                                d.shape[:-1] + (4,))
        beta, cond = regress(d.reshape(-1, d.shape[-1]), truth.reshape(-1, 4))
        audit[arm] = dict(condition=cond, check=None, adjusted=None)
        if beta is None:
            scores[arm] = None
            continue
        sd = (reference @ beta).std((0, 1))
        assert np.all(sd > 0)
        scores[arm] = mt.scores_on((evaluation @ beta) / sd)
        pred = check @ beta
        alpha = (pred * target).sum(0) / (target ** 2).sum(0)
        relative = np.linalg.norm(pred - target) / np.linalg.norm(target)
        adjusted = np.linalg.norm(pred / alpha - target) / np.linalg.norm(target) if np.all(abs(alpha) > 1e-8) else None
        audit[arm].update(check=relative, adjusted=adjusted)
    return scores, audit


def forecast_slot(slot):
    _, views, actions, _ = src.reference.historical(slot)
    encode = {name: views[name][1] for name in ('fine', 'raw')}
    eps, plan = tr.target(slot), seeds(slot)
    public, rr = src.shift_data.fresh_observations(slot, eps, tr.REFERENCE_N, plan['reference'], 'evaluation')
    ref = {name: read(public['x']) for name, read in encode.items()}
    cov = bias.all_cov(ref['fine'])
    native = bias.unit_rows(np.eye(4)[base.historical_match(slot, 0)['perm']], cov)
    fold = np.arange(len(actions)) % 4
    fit, check = fold == 0, fold == 1
    panels = []
    for i, (cs, es, ss) in enumerate(plan['panels']):
        c, cr = src.shift_data.fresh_observations(slot, eps, len(actions), cs, 'calibration')
        e, er = src.shift_data.fresh_observations(slot, eps, src.previous.EVAL, es, 'evaluation')
        pairs, held, receipt = queries(cr, fit, check, ac.TIMES, ss)
        yc, ye = encode['fine'](c['x']), {name: read(e['x']) for name, read in encode.items()}
        drift = src.eq.fit_drift(yc[fit], c['actions'][fit], c['regimes'][fit])
        w = src.eq.rows(yc, c['actions'], c['regimes'], drift, cov, ~fit)
        ns = mt.scores_on(ye['fine'] @ native.T)
        own = base.point_sets(ns)
        scores = dict(native=ns, passive=ns if w is None else mt.scores_on(ye['fine'] @ w.T))
        audit = {}
        for name, read in encode.items():
            values, diagnostics = readouts(read, pairs, held, ref[name], ye[name])
            for arm in ARMS:
                method = f'{name}_{arm}'
                scores[method] = ns if values[arm] is None else values[arm]
                audit[method] = diagnostics[arm]
        panels.append(dict(index=i, calibration=cr, evaluation=er, query=receipt, audit=audit,
                           missing_passive=w is None,
                           predictions={name: mt.flags(v, own) for name, v in scores.items()}))
    return dict(reference=rr, panels=panels)


def score_slot(slot, frozen):
    scale = src.shift_data.replay_physical(frozen['reference']).astype(np.float64).std((0, 1))
    panels = []
    for row in frozen['panels']:
        v = row['predictions']
        panels.append(tr.measure(dict(evaluation=row['evaluation'], point={m: x['point'] for m, x in v.items()},
                                      alerts={m: x['alarm'] for m, x in v.items()}), scale))
    rng = np.random.default_rng(2026212901 + SLOTS.index(slot))
    draws = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(d, minlength=PANELS) for d in draws])
    point = {m: {k: rep.bootstrap([p['point'][m][k] for p in panels], weights) for k in ('near', 'all')} for m in METHODS}
    alarm = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in panels], weights) for k in ('hit', 'fpr')} for m in METHODS}
    contrasts = {}
    for name in ('fine', 'raw'):
        for first, second in COMPARISONS:
            m, r = f'{name}_{first}', f'{name}_{second}'
            near = rep.paired_interval(point[m]['near'], point[r]['near'])
            near['lower_family'] = rep.upper(point[m]['near'][1] - point[r]['near'][1], .0125)
            gate = None
            if name == 'fine' and (first, second) == ('mean32', 'pooled2'):
                gate = near['gain'] >= .03 and near['lower_family'] > 0
            if name == 'fine' and (first, second) == ('mean32', 'exact2'):
                gate = near['lower_family'] > -.03
            contrasts[f'{m}_vs_{r}'] = dict(near=near, hit=rep.paired_interval(alarm[m]['hit'], alarm[r]['hit']), passed=gate)
    audits = {}
    for m in METHODS[2:]:
        all_rows = [p['audit'][m] for p in frozen['panels']]
        usable = [x for x in all_rows if x['check'] is not None]
        audits[m] = dict(unavailable=len(all_rows) - len(usable),
                         medians={k: np.median([r[k] for r in usable if r[k] is not None])
                                  if any(r[k] is not None for r in usable) else None for k in ('check', 'adjusted', 'condition')})
    return dict(point={m: {k: v[0] for k, v in d.items()} for m, d in point.items()},
                alarm={m: {k: v[0] for k, v in d.items()} for m, d in alarm.items()}, contrasts=contrasts,
                diagnostics=audits, missing_passive=sum(p['missing_passive'] for p in frozen['panels']), panels=panels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS},
                                    stream_audit=check_roots()))
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
