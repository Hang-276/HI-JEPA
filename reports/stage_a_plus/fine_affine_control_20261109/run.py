"""Compare both inputs under the same affine-equivariant reader."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'fine_equal_alarm_20261109'
spec = importlib.util.spec_from_file_location('equal_alarm', OLD / 'run.py')
prev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prev)
base, bias, rep = prev.base, prev.bias, prev.rep
rep.HERE = HERE

import numpy as np
import equivariant as eq

SLOTS = prev.SLOTS
METHODS = ('native', 'legacy_fine', 'legacy_raw', 'affine_fine', 'affine_raw')
MIX = np.array([[2., .4, 0., 0.], [0., .5, .7, 0.],
                [.2, 0., 1.5, .2], [.1, 0., 0., .75]])
OFFSET = np.array([.3, -.2, .4, -.1])


def sources():
    paths = (Path(__file__), HERE / 'PROTOCOL.md', HERE / 'THEORY.md', HERE / 'equivariant.py',
             HERE / 'ENGINEERING.json', OLD / 'LOCK.json', OLD / 'SUMMARY.json', OLD / 'run.py')
    return prev.sources() | {str(p): base.sha(p) for p in paths}


def seeds(slot):
    root = 2026117000 + 1000 * tuple(SLOTS).index(slot)
    return [(root + 2 * k, root + 2 * k + 1) for k in range(prev.PANELS)]


def historical(slot):
    data, fine = prev.implementations(slot)
    n, t, _ = SLOTS[slot]
    raw, actions, regimes, owner = base.inputs(slot)
    features, _ = base.old.fine_train(slot, 0, n, t, owner)
    encoder = fine.load_fine(slot, 0)
    mu, scale = base.scale_fit(raw)

    def xencode(values):
        return (values - mu) / scale

    def yencode(values):
        return (encoder.encode(values) - encoder.mean) / encoder.std

    return data, {'fine': (features, yencode), 'raw': (xencode(raw), xencode)}, actions, regimes


def parity_slot(slot):
    _, y, a, u = historical(slot)
    groups = np.arange(len(a)) % 4 != 0
    result = {}
    for name, (train, _) in y.items():
        rotated = train @ MIX.T + OFFSET
        values = []
        for x in (train, rotated):
            row = eq.rows(x, a, u, eq.fit_drift(x[~groups], a[~groups], u[~groups]), bias.all_cov(x), groups)
            assert row is not None
            values.append(bias.project(base.score_matrix(x[:1]), row))
        error = float(abs(values[0] - values[1]).max())
        assert error < 1e-8
        result[name] = error
    return result


def forecasts(slot):
    data, representations, a, u = historical(slot)
    cal_mask = np.arange(len(a)) % 4 != 0
    prep = {name: (encode, bias.all_cov(train), bias.fit_drift(train, a, u), eq.fit_drift(train[~cal_mask], a[~cal_mask], u[~cal_mask]))
            for name, (train, encode) in representations.items()}
    perm = base.historical_match(slot, 0)['perm']
    native_rows = np.eye(4)[perm]
    result = []
    for k, (cs, es) in enumerate(seeds(slot)):
        cal, cr = data.fresh_observations(slot, len(a), cs)
        evaluation, er = data.fresh_observations(slot, prev.EVAL, es)
        scores, missing = {}, {}
        for name, (encode, cov, old_drift, new_drift) in prep.items():
            yc, ye = encode(cal['x']), encode(evaluation['x'])
            mm = base.score_matrix(ye)
            r = bias.innovation(yc, cal['actions'], old_drift)
            legacy = prev.direction_rows(bias.stats(r, cal['regimes']), cal_mask, cov)
            affine = eq.rows(yc, cal['actions'], cal['regimes'], new_drift, cov, cal_mask)
            if name == 'fine':
                scores['native'] = bias.project(mm, native_rows)
            for method, rows in (('legacy', legacy), ('affine', affine)):
                key = f'{method}_{name}'
                missing[key] = rows is None
                scores[key] = bias.project(mm, rows) if rows is not None else scores['native']
        native = base.point_sets(scores['native'])
        safe = np.where((native >= 0).all(1)[:, None], native, [0, 1])
        alerts = {name: prev.alarm_mask(base.margin(s, safe)) for name, s in scores.items()}
        points = {name: base.point_sets(s) for name, s in scores.items()}
        result.append(dict(index=k, calibration=cr, evaluation=er, alerts=alerts, point=points, missing=missing))
    return result


def summarize(slot, panels):
    measured = [prev.panel_score(slot, p) for p in panels]
    rng = np.random.default_rng(2026117901 + tuple(SLOTS).index(slot))
    draws = rng.integers(prev.PANELS, size=(rep.RESAMPLES, prev.PANELS))
    weights = np.stack([np.bincount(row, minlength=prev.PANELS) for row in draws])
    point = {m: {k: rep.bootstrap([p['point'][m][k] for p in measured], weights)
                 for k in ('all', 'near')} for m in METHODS}
    alarm = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in measured], weights)
                 for k in ('hit', 'fpr')} for m in METHODS}
    alarm95 = {m: rep.upper(v['fpr'][1], .95) for m, v in alarm.items()}
    comparisons = {}
    for f, r in (('affine_fine', 'affine_raw'), ('legacy_fine', 'legacy_raw'),
                 ('affine_fine', 'legacy_fine'), ('affine_raw', 'legacy_raw'),
                 ('affine_fine', 'native')):
        near = rep.paired_interval(point[f]['near'], point[r]['near'])
        near['lower9875'] = rep.upper(point[f]['near'][1] - point[r]['near'][1], .0125)
        near['passed'] = (near['gain'] >= .03 and near['lower9875'] > 0)
        detection = rep.paired_interval(alarm[f]['hit'], alarm[r]['hit'])
        detection['lower9875'] = rep.upper(alarm[f]['hit'][1] - alarm[r]['hit'][1], .0125)
        detection['passed'] = (detection['gain'] >= .10 and detection['lower9875'] > 0
                                and max(alarm95[f], alarm95[r]) < .05)
        comparisons[f'{f}_vs_{r}'] = dict(near=near, detection=detection,
                         all=rep.paired_interval(point[f]['all'], point[r]['all']))
    return dict(comparisons=comparisons, point={m: {k: v[0] for k, v in o.items()} for m, o in point.items()},
                alarm={m: dict(detection=v['hit'][0], false_alarm=v['fpr'][0], fpr_upper95=alarm95[m])
                       for m, v in alarm.items()},
                unavailable={m: sum(p['missing'][m] for p in panels) for m in METHODS[1:]},
                panels=measured)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('phase', choices=('verify', 'lock', 'forecast', 'score'))
    phase = cli.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'verify':
        checks = {slot: parity_slot(slot) for slot in SLOTS}
        rep.write('ENGINEERING.json', dict(utc=base.now(), maxabs=checks, mix=MIX, offset=OFFSET))
        return
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS},
                                    panels=prev.PANELS, evaluation=prev.EVAL, alarms=prev.ALARMS))
        return
    assert base.read(lock)['sources'] == sources()
    sha = base.sha(lock)
    if phase == 'forecast':
        slots = {slot: forecasts(slot) for slot in SLOTS}
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=sha, slots=slots))
        return
    data = base.read(forecast)
    assert data['lock_sha'] == sha
    slots = {s: summarize(s, data['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=sha, forecast_sha=base.sha(forecast), slots=slots))


if __name__ == '__main__':
    main()
