"""Random replacement of calibration regime labels with held-out scoring."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
AFFINE = HERE.parent / 'fine_affine_control_20261109'
MOMENT = HERE.parent / 'fine_drift_moment_20261109'
sys.path.insert(0, str(AFFINE))
sys.path.insert(0, str(MOMENT))
spec = importlib.util.spec_from_file_location('noise_moments', MOMENT / 'run.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
src, base, bias, rep = m.src, m.base, m.bias, m.rep
control, prev = src.reference, src.previous
rep.HERE = HERE

import numpy as np

SLOTS, PANELS = m.SLOTS, 32
RATES = (0., .25, .5, .75, 1.)
METHODS = ('native', *(f'{name}_{eta}' for name in ('fine', 'raw', 'oracle_b') for eta in RATES))


def seeds(slot):
    root = 2026152000 + 1000 * SLOTS.index(slot)
    return [(root + 2 * k, root + 2 * k + 1) for k in range(PANELS)]


def sources():
    paths = (Path(__file__), HERE / 'PROTOCOL.md', MOMENT / 'run.py', MOMENT / 'LOCK.json',
             AFFINE / 'LOCK.json', AFFINE / 'SUMMARY.json', AFFINE / 'run.py')
    return m.source_paths() | {str(p): base.sha(p) for p in paths}


def contrast(r, u):
    masks = [u == k for k in base.old.CODES]
    assert all(np.any(mask) for mask in masks)
    b, *d = [m.moment(r[mask]) for mask in masks]
    return b, np.stack(d) - b


def forecast_slot(slot):
    data, views, a, u = control.historical(slot)
    fit = np.arange(len(a)) % 4 == 0
    candidates = {name: (encode, bias.all_cov(train), src.eq.fit_drift(train[fit], a[fit], u[fit]))
                  for name, (train, encode) in views.items()}
    native = np.eye(4)[base.historical_match(slot, 0)['perm']]
    panels = []
    for i, (cs, es) in enumerate(seeds(slot)):
        cal, cr = data.fresh_observations(slot, len(a), cs)
        ev, er = data.fresh_observations(slot, prev.EVAL, es)
        rng = np.random.default_rng(cs + 700000)
        u0 = cal['regimes'][~fit]
        replace = rng.random(u0.shape)
        label = rng.integers(0, 9, u0.shape)
        assert all(drift is not None for _, _, drift in candidates.values())
        scores, missing, signal = {}, {}, {}
        for name, (encode, cov, drift) in candidates.items():
            yc = encode(cal['x'][~fit])
            mm = base.score_matrix(encode(ev['x']))
            delta = np.diff(yc, axis=1)
            r = delta - src.eq._design(yc[:, :-1], cal['actions'][~fit]) @ drift
            b0, d0 = contrast(r, u0)
            if name == 'fine':
                scores['native'] = bias.project(mm, native)
            for eta in RATES:
                ut = np.where(replace < eta, label, u0)
                b, d = contrast(r, ut)
                w = m.readout(d, b, cov)
                src.projected(scores, mm, f'{name}_{eta}', w, missing)
                if name == 'fine':
                    diagnostic = m.readout(d, b0, cov)
                    src.projected(scores, mm, f'oracle_b_{eta}', diagnostic, missing)
                    signal[str(eta)] = dict(changed_fraction=float((ut != u0).mean()),
                        slope=float((d * d0).sum() / (d0 * d0).sum()))
        point = {k: base.point_sets(v) for k, v in scores.items()}
        own = point['native']
        safe = np.where((own >= 0).all(1)[:, None], own, [0, 1])
        alarms = {k: prev.alarm_mask(base.margin(v, safe)) for k, v in scores.items()}
        panels.append(dict(index=i, calibration=cr, evaluation=er, point=point,
                           alerts=alarms, missing=missing, signal=signal))
    return panels


def evaluate(slot, panels):
    measured = [prev.panel_score(slot, p) for p in panels]
    rng = np.random.default_rng(2026152901 + SLOTS.index(slot))
    draws = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(r, minlength=PANELS) for r in draws])
    points = {m: {k: rep.bootstrap([x['point'][m][k] for x in measured], weights)
                  for k in ('near', 'all')} for m in METHODS}
    alarms = {m: {k: rep.bootstrap([x['alerts'][m][k] for x in measured], weights)
                  for k in ('hit', 'fpr')} for m in METHODS}
    differences = {}
    for eta in RATES[1:]:
        rule = f'fine_{eta}'
        near = rep.paired_interval(points[rule]['near'], points['fine_0.0']['near'])
        hit = rep.paired_interval(alarms[rule]['hit'], alarms['fine_0.0']['hit'])
        near['low9875'] = rep.upper(points[rule]['near'][1] - points['fine_0.0']['near'][1], .0125)
        hit['low9875'] = rep.upper(alarms[rule]['hit'][1] - alarms['fine_0.0']['hit'][1], .0125)
        differences[str(eta)] = dict(near=near, hit=hit, passes_near=near['low9875'] > -.03,
                                     passes_alarm=hit['low9875'] > -.10)
    return dict(differences=differences, unavailable={m: sum(p['missing'][m] for p in panels)
                for m in METHODS[1:]}, point={m: {k: v[0] for k, v in d.items()} for m, d in points.items()},
                alarm={m: {k: v[0] for k, v in d.items()} for m, d in alarms.items()},
                median_signal={str(eta): {k: float(np.median([p['signal'][str(eta)][k] for p in panels]))
                    for k in ('changed_fraction', 'slope')} for eta in RATES},
                panels=measured)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = p.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS},
                                   panels=PANELS, replacement_rates=RATES))
        return
    sha = base.sha(lock)
    assert base.read(lock)['sources'] == sources()
    if phase == 'forecast':
        slots = {s: forecast_slot(s) for s in SLOTS}
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=sha, slots=slots))
        return
    data = base.read(forecast)
    assert data['lock_sha'] == sha
    summary = {s: evaluate(s, data['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=sha, forecast_sha=base.sha(forecast),
                                   slots=summary))


if __name__ == '__main__':
    main()
