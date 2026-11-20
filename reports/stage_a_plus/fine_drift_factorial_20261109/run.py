"""Balanced factorial of drift and direction data sources under FHN shift."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / 'fine_target_reference_20261109'
sys.path.insert(0, str(TARGET))
spec = importlib.util.spec_from_file_location('target_reference', TARGET / 'run.py')
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)
src, base, bias, rep = target.src, target.base, target.bias, target.rep
rep.HERE = HERE

import numpy as np

SLOTS = target.SLOTS
METHODS = ('native', 'ss', 'st', 'ts', 'tt', 'sh', 'th')
PAIRS = (('tt', 'ts'), ('tt', 'ss'), ('st', 'ss'), ('tt', 'th'), ('th', 'sh'), ('tt', 'native'))


def seeds(slot):
    i = SLOTS.index(slot)
    root = 2026127000 + 1000 * i
    return dict(reference=2026126000 + i,
                panels=[(root + 2 * k, root + 2 * k + 1) for k in range(src.previous.PANELS)])


def sources():
    paths = (Path(__file__), HERE / 'PROTOCOL.md', TARGET / 'LOCK.json',
             TARGET / 'SUMMARY.json', TARGET / 'run.py')
    return target.sources() | {str(p): base.sha(p) for p in paths}


def covariance_diagnostic(y, public, keep, source_drift, target_drift, first, second):
    if first is None or second is None or source_drift is None or target_drift is None:
        return None
    control = (public['regimes'] == 0) & keep[:, None]
    x = src.eq._design(y[:, :-1][control], public['actions'][control])
    delta = np.diff(y, axis=1)[control]
    baseline = []
    for d in (source_drift, target_drift):
        e = delta - x @ d
        baseline.append(e.T @ e / len(e))
    a, b = baseline
    dots = abs(np.sum(first * second, axis=1)) / (np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1))
    return dict(relative_frobenius=float(np.linalg.norm(b - a) / np.linalg.norm(a)),
                target_to_source_trace=float(np.trace(b) / np.trace(a)),
                axis_angle_deg=np.degrees(np.arccos(dots.clip(0, 1))))


def forecast_slot(slot):
    _, views, a, u = src.reference.historical(slot)
    train, encode = views['fine']
    e1, e2 = src.SOURCE_EPS[slot], target.target(slot)
    rng = seeds(slot)
    public, reference = src.shift_data.fresh_observations(
        slot, e2, target.REFERENCE_N, rng['reference'], 'evaluation')
    covariance = bias.all_cov(encode(public['x']))
    perm = base.historical_match(slot, 0)['perm']
    native_rows = bias.unit_rows(np.eye(4)[perm], covariance)
    fit = np.arange(len(a)) % 4 == 0
    historical = src.eq.fit_drift(train[fit], a[fit], u[fit])
    panels = []
    for i, (cs, es) in enumerate(rng['panels']):
        s, sr = src.shift_data.fresh_observations(slot, e1, len(a), cs, 'calibration')
        t, tr = src.shift_data.fresh_observations(slot, e2, len(a), cs, 'calibration')
        assert sr['physical_rng_stream_hashes'] == tr['physical_rng_stream_hashes']
        evaluation, er = src.shift_data.fresh_observations(slot, e2, src.previous.EVAL, es, 'evaluation')
        cal = {'s': s, 't': t}
        enc = {k: encode(obs['x']) for k, obs in cal.items()}
        drift = {k: src.eq.fit_drift(enc[k][fit], o['actions'][fit], o['regimes'][fit])
                 for k, o in cal.items()}
        drift['h'] = historical
        rows = {}
        mm = base.score_matrix(encode(evaluation['x']))
        scores, missing = {'native': bias.project(mm, native_rows)}, {}
        for k in METHODS[1:]:
            obs = cal[k[0]]
            rows[k] = src.eq.rows(enc[k[0]], obs['actions'], obs['regimes'],
                                  drift[k[1]], covariance, ~fit)
            src.projected(scores, mm, k, rows[k], missing)
        points = {m: base.point_sets(v) for m, v in scores.items()}
        native = points['native']
        safe = np.where((native >= 0).all(1)[:, None], native, [0, 1])
        flags = {k: src.previous.alarm_mask(base.margin(v, safe)) for k, v in scores.items()}
        diag = {k: covariance_diagnostic(enc[k], cal[k], ~fit, drift['s'], drift['t'],
                                         rows[k + 's'], rows[k + 't']) for k in cal}
        panels.append(dict(index=i, calibration=[sr, tr], evaluation=er, point=points,
                           alerts=flags, missing=missing, diagnostics=diag))
    return dict(reference=reference, panels=panels)


def score_slot(slot, data):
    reference = src.shift_data.replay_physical(data['reference']).astype(np.float64)
    scale = reference.std((0, 1), ddof=0)
    assert (scale > 0).all()
    measured = [target.measure(panel, scale) for panel in data['panels']]
    rng = np.random.default_rng(2026127901 + SLOTS.index(slot))
    draws = rng.integers(src.previous.PANELS, size=(rep.RESAMPLES, src.previous.PANELS))
    weights = np.stack([np.bincount(row, minlength=src.previous.PANELS) for row in draws])
    point = {m: {k: rep.bootstrap([p['point'][m][k] for p in measured], weights)
                 for k in ('all', 'near')} for m in METHODS}
    alarms = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in measured], weights)
                  for k in ('hit', 'fpr')} for m in METHODS}
    risk = {m: rep.upper(v['fpr'][1], .95) for m, v in alarms.items()}
    comparisons = {}
    for a, b in PAIRS:
        near = rep.paired_interval(point[a]['near'], point[b]['near'])
        alarm = rep.paired_interval(alarms[a]['hit'], alarms[b]['hit'])
        near['lower9875'] = rep.upper(point[a]['near'][1] - point[b]['near'][1], .0125)
        alarm['lower9875'] = rep.upper(alarms[a]['hit'][1] - alarms[b]['hit'][1], .0125)
        near['passed'] = near['gain'] is not None and near['gain'] >= .03 and near['lower9875'] > 0
        alarm['passed'] = (alarm['gain'] is not None and alarm['gain'] >= .10 and alarm['lower9875'] > 0
                           and risk[a] is not None and risk[b] is not None and max(risk[a], risk[b]) < .05)
        comparisons[f'{a}_vs_{b}'] = dict(near=near, alarm=alarm,
                                          all=rep.paired_interval(point[a]['all'], point[b]['all']))
    diagnostics = {}
    for label in ('s', 't'):
        d = [p['diagnostics'][label] for p in data['panels'] if p['diagnostics'][label] is not None]
        diagnostics[label] = {k: np.median([p[k] for p in d], axis=0) if d else None for k in
                              ('relative_frobenius', 'target_to_source_trace', 'axis_angle_deg')}
    return dict(comparisons=comparisons, diagnostics=diagnostics,
                point={m: {k: v[0] for k, v in d.items()} for m, d in point.items()},
                alarms={m: dict(detection=d['hit'][0], fpr=d['fpr'][0], upper95=risk[m])
                        for m, d in alarms.items()},
                unavailable={m: sum(p['missing'][m] for p in data['panels']) for m in METHODS[1:]},
                panels=measured)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock, prediction = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS},
                                    arm_labels=METHODS, comparisons=PAIRS))
        return
    sha = base.sha(lock)
    assert base.read(lock)['sources'] == sources()
    if phase == 'forecast':
        slots = {s: forecast_slot(s) for s in SLOTS}
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=sha, slots=slots))
        return
    data = base.read(prediction)
    assert data['lock_sha'] == sha
    summary = {s: score_slot(s, data['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=sha, forecast_sha=base.sha(prediction),
                                   slots=summary))


if __name__ == '__main__':
    main()
