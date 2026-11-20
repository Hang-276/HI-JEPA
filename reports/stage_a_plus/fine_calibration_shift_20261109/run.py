"""Test matched versus mismatched calibration under a paired FHN dynamic shift."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
AFFINE = HERE.parent / 'fine_affine_control_20261109'
sys.path.insert(0, str(AFFINE))
spec = importlib.util.spec_from_file_location('affine_reference', AFFINE / 'run.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
base, bias, rep, eq, previous = reference.base, reference.bias, reference.rep, reference.eq, reference.prev
rep.HERE = HERE

import numpy as np
import shift_data

SOURCE_EPS = {'fhn_fresh_eps008': .08, 'fhn_fresh_eps05': .5}
EPS = (.08, .5)
METHODS = ('native', 'matched', 'mismatched', 'matched_refit', 'matched_raw')


def seeds(slot):
    start = 2026119000 + 1000 * tuple(SOURCE_EPS).index(slot)
    return [(start + 2 * i, start + 2 * i + 1) for i in range(previous.PANELS)]


def sources():
    paths = (Path(__file__), HERE / 'PROTOCOL.md', HERE / 'shift_data.py', HERE / 'ENGINEERING.json',
             AFFINE / 'run.py', AFFINE / 'LOCK.json', AFFINE / 'SUMMARY.json')
    return reference.sources() | shift_data.source_hashes() | {str(p): base.sha(p) for p in paths}


def projected(scores, mm, name, rows, missing):
    missing[name] = rows is None
    scores[name] = bias.project(mm, rows) if rows is not None else scores['native']


def candidate_rows(public, encode, drift, cov, mask):
    y = encode(public['x'])
    return eq.rows(y, public['actions'], public['regimes'], drift, cov, mask)


def forecast_slot(slot):
    _, representations, actions, regimes = reference.historical(slot)
    fy, fine = representations['fine']
    ry, raw = representations['raw']
    n = len(actions)
    group0 = np.arange(n) % 4 == 0
    est = ~group0
    hist_drift = {name: eq.fit_drift(train[group0], actions[group0], regimes[group0])
                  for name, (train, _) in representations.items()}
    fine_cov, raw_cov = bias.all_cov(fy), bias.all_cov(ry)
    native_rows = np.eye(4)[base.historical_match(slot, 0)['perm']]
    panels = []
    for i, (cs, es) in enumerate(seeds(slot)):
        cals = [shift_data.fresh_observations(slot, epsilon, n, cs, 'calibration') for epsilon in EPS]
        evals = [shift_data.fresh_observations(slot, epsilon, previous.EVAL, es, 'evaluation') for epsilon in EPS]
        assert cals[0][1]['physical_rng_stream_hashes'] == cals[1][1]['physical_rng_stream_hashes']
        assert evals[0][1]['physical_rng_stream_hashes'] == evals[1][1]['physical_rng_stream_hashes']
        fine_rows = [candidate_rows(data, fine, hist_drift['fine'], fine_cov, est) for data, _ in cals]
        results = {}
        for j, epsilon in enumerate(EPS):
            cal, _ = cals[j]
            evaluation, receipt = evals[j]
            mf = base.score_matrix(fine(evaluation['x']))
            mr = base.score_matrix(raw(evaluation['x']))
            new_fine = fine(cal['x'])
            new_drift = eq.fit_drift(new_fine[group0], cal['actions'][group0], cal['regimes'][group0])
            refit = eq.rows(new_fine, cal['actions'], cal['regimes'], new_drift, fine_cov, est)
            raw_rows = candidate_rows(cal, raw, hist_drift['raw'], raw_cov, est)
            scores = {'native': bias.project(mf, native_rows)}
            missing = {}
            for name, rows, mm in (('matched', fine_rows[j], mf), ('mismatched', fine_rows[1 - j], mf),
                                   ('matched_refit', refit, mf), ('matched_raw', raw_rows, mr)):
                projected(scores, mm, name, rows, missing)
            points = {key: base.point_sets(values) for key, values in scores.items()}
            native = points['native']
            safe = np.where((native >= 0).all(1)[:, None], native, [0, 1])
            alerts = {key: previous.alarm_mask(base.margin(values, safe)) for key, values in scores.items()}
            results[str(epsilon)] = dict(evaluation=receipt, point=points, alerts=alerts, missing=missing)
        panels.append(dict(index=i, calibration=[c[1] for c in cals], by_epsilon=results))
    return panels


def panel_score(slot, panel):
    z = shift_data.replay_physical(panel['evaluation']).astype(np.float64)
    z /= base.historical_match(slot, 0)['train_sd_z']
    scores = np.diagonal(base.score_matrix(z), axis1=1, axis2=2)
    truth = base.point_sets(scores)
    strict = (truth >= 0).all(1)
    ordered = np.sort(scores, axis=1)
    near = strict & ((ordered[:, 2] - ordered[:, 1]) <= .05)
    ok = {key: (np.asarray(p) == truth).all(1) & strict for key, p in panel['point'].items()}
    native_defined = (np.asarray(panel['point']['native']) >= 0).all(1)
    correct = native_defined & strict & ok['native']
    wrong = native_defined & strict & ~ok['native']
    alerts = {key: np.asarray(flag) for key, flag in panel['alerts'].items()}
    assert all(int(flag.sum()) == previous.ALARMS for flag in alerts.values())
    return dict(points={m: dict(all=rep.count(ok[m], strict), near=rep.count(ok[m], near)) for m in METHODS},
                alerts={m: dict(detection=rep.count(alerts[m], wrong), fpr=rep.count(alerts[m], correct))
                        for m in METHODS})


def score_slot(slot, panels):
    rng = np.random.default_rng(2026120901 + tuple(SOURCE_EPS).index(slot))
    draws = rng.integers(previous.PANELS, size=(rep.RESAMPLES, previous.PANELS))
    weights = np.stack([np.bincount(row, minlength=previous.PANELS) for row in draws])
    results = {}
    for epsilon in EPS:
        items = [p['by_epsilon'][str(epsilon)] for p in panels]
        rows = [panel_score(slot, p) for p in items]
        points = {m: {k: rep.bootstrap([r['points'][m][k] for r in rows], weights)
                      for k in ('all', 'near')} for m in METHODS}
        alerts = {m: {k: rep.bootstrap([r['alerts'][m][k] for r in rows], weights)
                      for k in ('detection', 'fpr')} for m in METHODS}
        false_alarm = {m: rep.upper(a['fpr'][1], .95) for m, a in alerts.items()}
        comparisons = {}
        for f, r in (('matched', 'mismatched'), ('matched_refit', 'matched'),
                     ('matched', 'matched_raw'), ('matched', 'native')):
            near = rep.paired_interval(points[f]['near'], points[r]['near'])
            near['lower9875'] = rep.upper(points[f]['near'][1] - points[r]['near'][1], .0125)
            near['passed'] = near['gain'] >= .03 and near['lower9875'] > 0
            detection = rep.paired_interval(alerts[f]['detection'], alerts[r]['detection'])
            detection['lower9875'] = rep.upper(alerts[f]['detection'][1] - alerts[r]['detection'][1], .0125)
            detection['passed'] = (detection['gain'] >= .10 and detection['lower9875'] > 0
                                    and max(false_alarm[f], false_alarm[r]) < .05)
            comparisons[f'{f}_vs_{r}'] = dict(near=near, detection=detection,
                                              all=rep.paired_interval(points[f]['all'], points[r]['all']))
        results[str(epsilon)] = dict(
            domain='in' if epsilon == SOURCE_EPS[slot] else 'shifted',
            points={m: {k: v[0] for k, v in p.items()} for m, p in points.items()},
            alerts={m: {**{k: v[0] for k, v in a.items()}, 'fpr_upper95': false_alarm[m]} for m, a in alerts.items()},
            comparisons=comparisons,
            unavailable={m: sum(p['missing'][m] for p in items) for m in METHODS[1:]}, panels=rows)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('verify', 'lock', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'verify':
        rep.write('ENGINEERING.json', dict(utc=base.now(),
                  historical=[shift_data.verify_historical_train(s) for s in SOURCE_EPS]))
        return
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SOURCE_EPS},
                                    panels=previous.PANELS, epsilon=EPS, source_eps=SOURCE_EPS))
        return
    sha = base.sha(lock)
    assert base.read(lock)['sources'] == sources()
    if phase == 'forecast':
        slots = {s: forecast_slot(s) for s in SOURCE_EPS}
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=sha, slots=slots))
        return
    data = base.read(forecast)
    assert data['lock_sha'] == sha
    slots = {s: score_slot(s, data['slots'][s]) for s in SOURCE_EPS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=sha, forecast_sha=base.sha(forecast), slots=slots))


if __name__ == '__main__':
    main()
