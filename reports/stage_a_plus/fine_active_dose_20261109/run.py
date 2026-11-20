"""Test stronger known shifts against independent physical branch noise."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'fine_active_noise_20261109'
sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location('active_noise', OLD / 'run.py')
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)
tr, src, base, bias, rep, mt, pr, ac = an.tr, an.src, an.base, an.bias, an.rep, an.mt, an.pr, an.ac
rep.HERE = HERE

from dose_observations import LEVELS, dose_pairs

SLOTS, PANELS = tr.SLOTS, 32
ARMS = tuple(f'{kind}{level}' for level in LEVELS for kind in ('exact', 'noisy'))
METHODS = ('native', 'passive') + tuple(f'{view}_{arm}' for view in ('fine', 'raw') for arm in ARMS)
CONTRASTS = (('noisy4', 'noisy1'), ('noisy4', 'exact1'), ('noisy2', 'noisy1'),
             ('noisy4', 'exact4'), ('exact4', 'exact1'))


def sources():
    return an.sources() | {str(p): base.sha(p) for p in (
        Path(__file__), HERE / 'dose_observations.py', HERE / 'PROTOCOL.md', OLD / 'LOCK.json', OLD / 'SUMMARY.json')}


def seeds(slot):
    i = SLOTS.index(slot)
    start = 2026197000 + 1000 * i
    return dict(reference=2026196000 + i, panels=[
        (start + 2 * k, start + 2 * k + 1, 2026202000 + 1000 * i + k) for k in range(PANELS)])


def archived_roots():
    seen = set()

    def record(value):
        if isinstance(value, dict):
            if value.get('namespace') == src.shift_data.FRESH_NAMESPACE and 'trajectory_seed' in value:
                seen.add(value['trajectory_seed'])
            for child in value.values():
                record(child)
        if isinstance(value, list):
            for child in value:
                record(child)

    for path in HERE.parent.glob('fine_*_20261109/FORECAST.json'):
        if path.parent != HERE:
            record(json.loads(path.read_text()))
    return seen


def check_seed_plan():
    roots, queries = [], []
    for s in SLOTS:
        plan = seeds(s)
        roles = [(plan['reference'], 'evaluation')]
        for c, e, q in plan['panels']:
            roles.extend(((c, 'calibration'), (e, 'evaluation')))
            queries.append(q)
        roots.extend(src.shift_data._stream_seed(s, root, role) for root, role in roles)
    archived = archived_roots()
    assert len(roots) == len(set(roots)) and not set(roots) & archived
    assert len(queries) == len(set(queries))
    return dict(new_roots=len(roots), checked_archived_roots=len(archived), query_seeds=len(queries))


def readouts(encode, pairs, check_data, reference, evaluation):
    center = encode(pairs['exact1']['baseline']).reshape(-1, 4).mean(0)
    check, check_target = ac.feature_difference(encode, check_data, center, 3)
    r = pr.polynomial(reference - center, 3)[..., 1:]
    y = pr.polynomial(evaluation - center, 3)[..., 1:]
    result, audit = {}, {}
    for arm in ARMS:
        design, target = an.stacked_difference(encode, pairs[arm], center)
        beta, condition = an.solve(design, target, cross=False)
        audit[arm] = dict(condition=condition, check=None)
        if beta is None:
            result[arm] = None
            continue
        sd = (r @ beta).std((0, 1))
        assert np.all(sd > 0)
        result[arm] = mt.scores_on((y @ beta) / sd)
        audit[arm]['check'] = np.linalg.norm(check @ beta - check_target) / np.linalg.norm(check_target)
    return result, audit


def forecast_slot(slot):
    _, views, actions, _ = src.reference.historical(slot)
    encode = {name: views[name][1] for name in ('fine', 'raw')}
    eps, roots = tr.target(slot), seeds(slot)
    pub, ref_receipt = src.shift_data.fresh_observations(slot, eps, tr.REFERENCE_N, roots['reference'], 'evaluation')
    ref = {name: read(pub['x']) for name, read in encode.items()}
    cov = bias.all_cov(ref['fine'])
    native = bias.unit_rows(np.eye(4)[base.historical_match(slot, 0)['perm']], cov)
    fold = np.arange(len(actions)) % 4
    fit, check = fold == 0, fold == 1
    panels = []
    for i, (cs, es, ss) in enumerate(roots['panels']):
        c, cr = src.shift_data.fresh_observations(slot, eps, len(actions), cs, 'calibration')
        e, er = src.shift_data.fresh_observations(slot, eps, src.previous.EVAL, es, 'evaluation')
        pairs, check_data, receipt = dose_pairs(cr, fit, check, ac.TIMES, ss)
        yc, ye = encode['fine'](c['x']), {name: read(e['x']) for name, read in encode.items()}
        drift = src.eq.fit_drift(yc[fit], c['actions'][fit], c['regimes'][fit])
        w = src.eq.rows(yc, c['actions'], c['regimes'], drift, cov, ~fit)
        ns = mt.scores_on(ye['fine'] @ native.T)
        own = base.point_sets(ns)
        scores = dict(native=ns, passive=ns if w is None else mt.scores_on(ye['fine'] @ w.T))
        audit = {}
        for name, read in encode.items():
            estimates, diagnostics = readouts(read, pairs, check_data, ref[name], ye[name])
            for arm in ARMS:
                key = f'{name}_{arm}'
                scores[key] = ns if estimates[arm] is None else estimates[arm]
                audit[key] = diagnostics[arm]
        panels.append(dict(index=i, calibration=cr, evaluation=er, query=receipt, audit=audit,
                           missing_passive=w is None,
                           predictions={name: mt.flags(v, own) for name, v in scores.items()}))
    return dict(reference=ref_receipt, panels=panels)


def score_slot(slot, frozen):
    scale = src.shift_data.replay_physical(frozen['reference']).astype(np.float64).std((0, 1))
    panels = []
    for row in frozen['panels']:
        predictions = row['predictions']
        panels.append(tr.measure(dict(evaluation=row['evaluation'], point={m: v['point'] for m, v in predictions.items()},
                                      alerts={m: v['alarm'] for m, v in predictions.items()}), scale))
    rng = np.random.default_rng(2026197901 + SLOTS.index(slot))
    draws = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(d, minlength=PANELS) for d in draws])
    point = {m: {k: rep.bootstrap([p['point'][m][k] for p in panels], weights) for k in ('near', 'all')}
             for m in METHODS}
    alarm = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in panels], weights) for k in ('hit', 'fpr')}
             for m in METHODS}
    contrasts, diagnostics = {}, {}
    for view in ('fine', 'raw'):
        for first, second in CONTRASTS:
            m, r = f'{view}_{first}', f'{view}_{second}'
            near = rep.paired_interval(point[m]['near'], point[r]['near'])
            near['lower_family'] = rep.upper(point[m]['near'][1] - point[r]['near'][1], .0125)
            passed = None
            if view == 'fine' and (first, second) == ('noisy4', 'noisy1'):
                passed = near['gain'] >= .03 and near['lower_family'] > 0
            if view == 'fine' and (first, second) == ('noisy4', 'exact1'):
                passed = near['lower_family'] > -.03
            contrasts[f'{m}_vs_{r}'] = dict(near=near, all=rep.paired_interval(point[m]['all'], point[r]['all']),
                                           hit=rep.paired_interval(alarm[m]['hit'], alarm[r]['hit']), passed=passed)
        for arm in ARMS:
            m = f'{view}_{arm}'
            audit = [p['audit'][m] for p in frozen['panels']]
            valid = [a for a in audit if a['check'] is not None]
            diagnostics[m] = dict(unavailable=len(audit) - len(valid),
                                  check_median=np.median([a['check'] for a in valid]) if valid else None)
    return dict(point={m: {k: v[0] for k, v in d.items()} for m, d in point.items()},
                alarm={m: {k: v[0] for k, v in d.items()} for m, d in alarm.items()}, contrasts=contrasts,
                diagnostics=diagnostics, missing_passive=sum(p['missing_passive'] for p in frozen['panels']),
                panels=panels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS},
                                    stream_audit=check_seed_plan()))
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
