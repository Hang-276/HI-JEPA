"""Evaluate nested extra calibration budgets on independent evaluation panels."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
AFFINE = HERE.parent / 'fine_affine_control_20261109'
sys.path.insert(0, str(AFFINE))
spec = importlib.util.spec_from_file_location('affine_reference', AFFINE / 'run.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
base, bias, rep, eq, prev = control.base, control.bias, control.rep, control.eq, control.prev
rep.HERE = HERE

import numpy as np

SLOTS = ('fhn_fresh_eps008', 'fhn_fresh_eps05')
BUDGETS = (24, 48, 96, 192)
METHODS = ('native', *(f'{name}{n}' for name in ('fine', 'raw') for n in BUDGETS))


def sources():
    return control.sources() | {str(p): base.sha(p) for p in (
        Path(__file__), HERE / 'PROTOCOL.md', AFFINE / 'LOCK.json', AFFINE / 'SUMMARY.json', AFFINE / 'run.py')}


def seeds(slot):
    start = 2026122000 + 1000 * SLOTS.index(slot)
    return [(start + 2 * k, start + 2 * k + 1) for k in range(prev.PANELS)]


def forecast(slot):
    data, reps, a, u = control.historical(slot)
    h0 = np.arange(len(a)) % 4 == 0
    prep = {name: (encode, bias.all_cov(train), eq.fit_drift(train[h0], a[h0], u[h0]))
            for name, (train, encode) in reps.items()}
    native_rows = np.eye(4)[base.historical_match(slot, 0)['perm']]
    panels = []
    for i, (cs, es) in enumerate(seeds(slot)):
        cal, cr = data.fresh_observations(slot, max(BUDGETS), cs)
        evaluation, er = data.fresh_observations(slot, prev.EVAL, es)
        scores, missing = {}, {}
        for name, (encode, covariance, drift) in prep.items():
            cal_y = encode(cal['x'])
            mm = base.score_matrix(encode(evaluation['x']))
            if name == 'fine':
                scores['native'] = bias.project(mm, native_rows)
            for n in BUDGETS:
                row = eq.rows(cal_y[:n], cal['actions'][:n], cal['regimes'][:n], drift, covariance,
                              np.arange(n) % 4 != 0)
                key = f'{name}{n}'
                scores[key] = bias.project(mm, row) if row is not None else scores['native']
                missing[key] = row is None
        points = {k: base.point_sets(s) for k, s in scores.items()}
        native = points['native']
        safe = np.where((native >= 0).all(1)[:, None], native, [0, 1])
        alerts = {k: prev.alarm_mask(base.margin(s, safe)) for k, s in scores.items()}
        panels.append(dict(index=i, calibration=cr, evaluation=er, point=points, alerts=alerts,
                           missing=missing))
    return panels


def evaluate(slot, panels):
    grouped = [prev.panel_score(slot, panel) for panel in panels]
    rng = np.random.default_rng(2026122901 + SLOTS.index(slot))
    draws = rng.integers(prev.PANELS, size=(rep.RESAMPLES, prev.PANELS))
    weights = np.stack([np.bincount(row, minlength=prev.PANELS) for row in draws])
    point = {m: {k: rep.bootstrap([p['point'][m][k] for p in grouped], weights)
                 for k in ('all', 'near')} for m in METHODS}
    alerts = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in grouped], weights)
                  for k in ('hit', 'fpr')} for m in METHODS}
    fpr_upper = {m: rep.upper(v['fpr'][1], .95) for m, v in alerts.items()}
    comparisons = {}
    ordered = [(f'fine{n}', 'native') for n in BUDGETS]
    ordered += [(f'fine{n}', f'raw{n}') for n in BUDGETS]
    ordered += [('fine96', 'fine24'), ('fine48', 'fine24'), ('fine96', 'fine48'), ('fine192', 'fine96')]
    for m, r in ordered:
        near = rep.paired_interval(point[m]['near'], point[r]['near'])
        alarm = rep.paired_interval(alerts[m]['hit'], alerts[r]['hit'])
        near['lower9875'] = rep.upper(point[m]['near'][1] - point[r]['near'][1], .0125)
        alarm['lower9875'] = rep.upper(alerts[m]['hit'][1] - alerts[r]['hit'][1], .0125)
        near['passed'] = near['gain'] >= .03 and near['lower9875'] > 0
        alarm['passed'] = (alarm['gain'] >= .10 and alarm['lower9875'] > 0
                           and max(fpr_upper[m], fpr_upper[r]) < .05)
        comparisons[f'{m}_vs_{r}'] = dict(near=near, alarm=alarm,
                                            all=rep.paired_interval(point[m]['all'], point[r]['all']))
    return dict(
        point={m: {k: v[0] for k, v in d.items()} for m, d in point.items()},
        alerts={m: dict(detection=d['hit'][0], fpr=d['fpr'][0], fpr_upper95=fpr_upper[m])
                for m, d in alerts.items()}, comparisons=comparisons,
        unavailable={m: sum(p['missing'][m] for p in panels) for m in METHODS[1:]},
        panels=grouped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock, prediction = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS},
                                   budgets=BUDGETS, panels=prev.PANELS, n_eval=prev.EVAL))
        return
    sha = base.sha(lock)
    assert base.read(lock)['sources'] == sources()
    if phase == 'forecast':
        slots = {s: forecast(s) for s in SLOTS}
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=sha, slots=slots))
        return
    data = base.read(prediction)
    assert data['lock_sha'] == sha
    summary = {s: evaluate(s, data['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=sha, forecast_sha=base.sha(prediction),
                                   slots=summary))


if __name__ == '__main__':
    main()
