"""Cross readout capacity with shared versus independent FHN branch noise."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'fine_active_noise_20261109'
ALLOC = HERE.parent / 'fine_active_allocation_20261109'
sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location('active_noise', OLD / 'run.py')
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)
tr, src, base, bias, rep, mt, pr, ac = an.tr, an.src, an.base, an.bias, an.rep, an.mt, an.pr, an.ac
rep.HERE = HERE

SLOTS, PANELS = tr.SLOTS, 32
ARMS = ('exact_linear', 'exact_cubic', 'noisy_linear', 'noisy_cubic')
METHODS = ('native', 'passive') + tuple(f'{v}_{a}' for v in ('fine', 'raw') for a in ARMS)


def seeds(slot):
    i = SLOTS.index(slot)
    start = 2026222000 + 1000 * i
    return dict(reference=2026220000 + i, panels=[
        (start + 2 * k, start + 2 * k + 1, 2026225000 + 1000 * i + k) for k in range(PANELS)])


def stream_audit():
    def archived_roots(item):
        if isinstance(item, dict):
            if item.get('namespace') == src.shift_data.FRESH_NAMESPACE and 'trajectory_seed' in item:
                yield item['trajectory_seed']
            for child in item.values():
                yield from archived_roots(child)
        if isinstance(item, list):
            for child in item:
                yield from archived_roots(child)

    paths = [p for p in HERE.parent.glob('fine_*_20261109/FORECAST.json') if p.parent != HERE]
    assert {OLD.name, ALLOC.name}.issubset({p.parent.name for p in paths})
    old = {r for p in paths for r in archived_roots(json.loads(p.read_text()))}
    active, query = [], []
    for s in SLOTS:
        p = seeds(s)
        active.append(src.shift_data._stream_seed(s, p['reference'], 'evaluation'))
        for c, e, q in p['panels']:
            active.extend((src.shift_data._stream_seed(s, c, 'calibration'),
                           src.shift_data._stream_seed(s, e, 'evaluation')))
            query.append(q)
    assert len(active) == len(set(active)) and not set(active) & old and len(query) == len(set(query))
    return dict(new_roots=len(active), archived_roots=len(old), queried_roots=len(query),
                checked_directories=sorted(p.parent.name for p in paths))


def sources():
    return an.sources() | {str(p): base.sha(p) for p in (
        Path(__file__), HERE / 'PROTOCOL.md', HERE / 'THEORY.md', OLD / 'LOCK.json', OLD / 'SUMMARY.json',
        ALLOC / 'STREAM_AUDIT.json')}


def readouts(encode, pairs, reference, evaluation):
    center = reference.mean((0, 1))
    check, target_check = ac.feature_difference(encode, pairs['check'], center, 3)
    rp = pr.polynomial(reference - center, 3)[..., 1:]
    yp = pr.polynomial(evaluation - center, 3)[..., 1:]
    scores, audit = {}, {}
    for condition in ('exact', 'noisy'):
        d, target = an.stacked_difference(encode, pairs[condition], center)
        for degree, col in (('linear', slice(0, 4)), ('cubic', slice(None))):
            name = f'{condition}_{degree}'
            beta, cond = an.solve(d[..., col], target, False)
            audit[name] = dict(condition=cond, check=None, adjusted=None)
            if beta is None:
                scores[name] = None
                continue
            scale = (rp[..., col] @ beta).std((0, 1))
            assert np.all(scale > 0)
            scores[name] = mt.scores_on((yp[..., col] @ beta) / scale)
            check_prediction = check[:, col] @ beta
            alpha = (check_prediction * target_check).sum(0) / (target_check ** 2).sum(0)
            audit[name]['check'] = np.linalg.norm(check_prediction - target_check) / np.linalg.norm(target_check)
            if np.all(abs(alpha) > 1e-8):
                audit[name]['adjusted'] = np.linalg.norm(check_prediction / alpha - target_check) / np.linalg.norm(target_check)
    return scores, audit


def forecast_slot(slot):
    _, views, actions, _ = src.reference.historical(slot)
    enc = {name: views[name][1] for name in ('fine', 'raw')}
    eps, plan = tr.target(slot), seeds(slot)
    ref, rr = src.shift_data.fresh_observations(slot, eps, tr.REFERENCE_N, plan['reference'], 'evaluation')
    rf = {name: encode(ref['x']) for name, encode in enc.items()}
    cov = bias.all_cov(rf['fine'])
    native = bias.unit_rows(np.eye(4)[base.historical_match(slot, 0)['perm']], cov)
    fold = np.arange(len(actions)) % 4
    fit, check = fold == 0, fold == 1
    panels = []
    for i, (cs, es, qs) in enumerate(plan['panels']):
        c, cr = src.shift_data.fresh_observations(slot, eps, len(actions), cs, 'calibration')
        e, er = src.shift_data.fresh_observations(slot, eps, src.previous.EVAL, es, 'evaluation')
        pairs, query = an.replicate_pairs(cr, fit, check, ac.TIMES, qs)
        yc, ye = enc['fine'](c['x']), {name: encode(e['x']) for name, encode in enc.items()}
        drift = src.eq.fit_drift(yc[fit], c['actions'][fit], c['regimes'][fit])
        w = src.eq.rows(yc, c['actions'], c['regimes'], drift, cov, ~fit)
        ns = mt.scores_on(ye['fine'] @ native.T)
        own = base.point_sets(ns)
        scores = dict(native=ns, passive=ns if w is None else mt.scores_on(ye['fine'] @ w.T))
        audits = {}
        for name, encode in enc.items():
            estimates, diagnostics = readouts(encode, pairs, rf[name], ye[name])
            for arm in ARMS:
                method = f'{name}_{arm}'
                scores[method] = ns if estimates[arm] is None else estimates[arm]
                audits[method] = diagnostics[arm]
        panels.append(dict(index=i, calibration=cr, evaluation=er, query=query, audit=audits,
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
    rng = np.random.default_rng(2026222901 + SLOTS.index(slot))
    draws = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(d, minlength=PANELS) for d in draws])
    point = {m: {k: rep.bootstrap([p['point'][m][k] for p in panels], weights) for k in ('near', 'all')} for m in METHODS}
    alarm = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in panels], weights) for k in ('hit', 'fpr')} for m in METHODS}
    contrast_pairs = [('fine_noisy_linear', 'passive')]
    for v in ('fine', 'raw'):
        contrast_pairs.extend((f'{v}_{a}', f'{v}_{b}') for a, b in (
            ('noisy_linear', 'exact_linear'), ('noisy_linear', 'noisy_cubic'),
            ('exact_cubic', 'exact_linear'), ('noisy_cubic', 'exact_cubic')))
    contrasts = {}
    for m, r in contrast_pairs:
        near = rep.paired_interval(point[m]['near'], point[r]['near'])
        near['lower_family'] = rep.upper(point[m]['near'][1] - point[r]['near'][1], .0125)
        all_windows = rep.paired_interval(point[m]['all'], point[r]['all'])
        gate = None
        if m == 'fine_noisy_linear' and r == 'passive':
            gate = near['gain'] >= .03 and near['lower_family'] > 0 and all_windows['gain'] > -.01
        if m == 'fine_noisy_linear' and r == 'fine_exact_linear':
            gate = near['lower_family'] > -.03
        contrasts[f'{m}_vs_{r}'] = dict(near=near, all=all_windows,
                                       hit=rep.paired_interval(alarm[m]['hit'], alarm[r]['hit']), passed=gate)
    diagnostics = {}
    for m in METHODS[2:]:
        rows = [p['audit'][m] for p in frozen['panels']]
        used = [r for r in rows if r['check'] is not None]
        diagnostics[m] = dict(unavailable=len(rows) - len(used),
                              medians={k: np.median([x[k] for x in used if x[k] is not None])
                                       if any(x[k] is not None for x in used) else None
                                       for k in ('check', 'adjusted', 'condition')})
    return dict(point={m: {k: v[0] for k, v in d.items()} for m, d in point.items()},
                alarm={m: {k: v[0] for k, v in d.items()} for m, d in alarm.items()}, contrasts=contrasts,
                diagnostics=diagnostics, missing_passive=sum(p['missing_passive'] for p in frozen['panels']), panels=panels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS},
                                    stream_audit=stream_audit()))
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
