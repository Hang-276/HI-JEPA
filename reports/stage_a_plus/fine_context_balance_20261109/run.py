"""Cross-fitted common context adjustment for passive regime contrasts."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MOMENT = HERE.parent / 'fine_drift_moment_20261109'
sys.path.insert(0, str(MOMENT))
spec = importlib.util.spec_from_file_location('paired_reference', MOMENT / 'run.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
src, base, bias, rep, target = prior.src, prior.base, prior.bias, prior.rep, prior.target
rep.HERE = HERE

import numpy as np

SLOTS, PANELS = prior.SLOTS, 32
ARMS = ('native', 'source', 'balanced', 'placebo', 'target')
RIDGE = .001


def seeds(slot):
    i = SLOTS.index(slot)
    root = 2026147000 + 1000 * i
    return dict(reference=2026146000 + i,
                panels=[(root + 2 * k, root + 2 * k + 1) for k in range(PANELS)])


def sources():
    paths = (Path(__file__), HERE / 'PROTOCOL.md', MOMENT / 'LOCK.json', MOMENT / 'run.py',
             MOMENT / 'SUMMARY.json', MOMENT / 'PASSIVE_TERM_AUDIT.json')
    return prior.source_paths() | {str(p): base.sha(p) for p in paths}


def outer(x):
    return x[..., :, None] * x[..., None, :]


def nuisance(y, a, u, source_drift, fit):
    design = src.eq._design(y[:, :-1], a)
    predicted = design @ source_drift
    residual = np.diff(y, axis=1) - predicted
    ti = np.triu_indices(4)
    phi = np.concatenate((design[..., 1:], outer(predicted)[..., ti[0], ti[1]]), axis=-1)
    valid = fit[:, None] & np.isin(u, base.old.CODES)
    train = phi[valid]
    mean, scale = train.mean(0), train.std(0)
    keep = scale > 1e-12
    x = (train[:, keep] - mean[keep]) / scale[keep]
    yy = outer(residual[valid]).reshape(-1, 16)
    center = yy.mean(0)
    cross = x.T @ (yy - center) / len(x)
    gram = x.T @ x / len(x)
    coeff = np.linalg.solve(gram + RIDGE * np.eye(len(gram)), cross)
    all_x = (phi[..., keep] - mean[keep]) / scale[keep]
    fitted = (all_x @ coeff + center).reshape(*residual.shape[:2], 4, 4)
    return residual, (fitted + fitted.swapaxes(-1, -2)) / 2


def contrast(values, u):
    stats, count = base.old.episode_stats(values, u)
    total = count.sum(0)
    assert (total > 0).all()
    means = stats.sum(0) / total[:, None, None]
    return means[0], means[1:] - means[0]


def forecast_slot(slot):
    _, views, a, _ = src.reference.historical(slot)
    _, encode = views['fine']
    old, new = src.SOURCE_EPS[slot], target.target(slot)
    rng = seeds(slot)
    p, ref = src.shift_data.fresh_observations(
        slot, new, target.REFERENCE_N, rng['reference'], 'evaluation')
    cov = bias.all_cov(encode(p['x']))
    native = bias.unit_rows(np.eye(4)[base.historical_match(slot, 0)['perm']], cov)
    n = len(a)
    fit = np.arange(n) % 4 == 0
    panels = []
    for i, (cs, es) in enumerate(rng['panels']):
        s, sr = src.shift_data.fresh_observations(slot, old, n, cs, 'calibration')
        t, tr = src.shift_data.fresh_observations(slot, new, n, cs, 'calibration')
        assert sr['physical_rng_stream_hashes'] == tr['physical_rng_stream_hashes']
        evaluation, er = src.shift_data.fresh_observations(slot, new, src.previous.EVAL, es, 'evaluation')
        ys, yt = encode(s['x']), encode(t['x'])
        cs0, ct0 = (src.eq.fit_drift(y[fit], p['actions'][fit], p['regimes'][fit])
                    for y, p in ((ys, s), (yt, t)))
        assert cs0 is not None and ct0 is not None
        r, h = nuisance(yt, t['actions'], t['regimes'], cs0, fit)
        u = t['regimes'][~fit]
        b, d = contrast(outer(r[~fit]), u)
        _, db = contrast(h[~fit], u)
        _, dp = contrast(np.roll(h[~fit], 1, axis=0), u)
        mm = base.score_matrix(encode(evaluation['x']))
        scores = {'native': bias.project(mm, native)}
        missing = {}
        for key, row in (('source', prior.readout(d, b, cov)),
                         ('balanced', prior.readout(d - db, b, cov)),
                         ('placebo', prior.readout(d - dp, b, cov)),
                         ('target', src.eq.rows(yt, t['actions'], t['regimes'], ct0, cov, ~fit))):
            src.projected(scores, mm, key, row, missing)
        point = {k: base.point_sets(v) for k, v in scores.items()}
        native_set = point['native']
        safe = np.where((native_set >= 0).all(1)[:, None], native_set, [0, 1])
        alarms = {k: src.previous.alarm_mask(base.margin(v, safe)) for k, v in scores.items()}
        panels.append(dict(index=i, calibration=[sr, tr], evaluation=er, point=point, alerts=alarms,
                           scores=scores, missing=missing))
    return dict(reference=ref, panels=panels)


def score_slot(slot, results):
    states = src.shift_data.replay_physical(results['reference']).astype(np.float64)
    scale = states.std((0, 1))
    assert (scale > 0).all()
    measured = [prior.measure(p, scale) for p in results['panels']]
    rng = np.random.default_rng(2026147901 + SLOTS.index(slot))
    draws = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(row, minlength=PANELS) for row in draws])
    points = {k: {m: rep.bootstrap([p['point'][k][m] for p in measured], weights)
                  for m in ('near', 'all')} for k in ARMS}
    alarm = {k: {m: rep.bootstrap([p['alerts'][k][m] for p in measured], weights)
                 for m in ('hit', 'fpr')} for k in ARMS}
    comparisons = {}
    for k in ('balanced', 'placebo', 'target'):
        comparisons[k] = dict(near=rep.paired_interval(points[k]['near'], points['source']['near']),
                              hit=rep.paired_interval(alarm[k]['hit'], alarm['source']['hit']))
    fraction = (comparisons['balanced']['near']['gain'] / comparisons['target']['near']['gain']
                if comparisons['target']['near']['gain'] > 0 else None)
    positive = (fraction is not None and fraction >= .30 and comparisons['balanced']['near']['gain'] >= .03
                and comparisons['balanced']['near']['low_975'] > 0
                and comparisons['balanced']['near']['gain'] > comparisons['placebo']['near']['gain'])
    return dict(passed=positive, recovery_fraction=fraction, comparisons=comparisons,
                point={k: {m: v[0] for m, v in d.items()} for k, d in points.items()},
                alarm={k: {m: v[0] for m, v in d.items()} for k, d in alarm.items()},
                mae={k: float(np.mean([p['gap_mae'][k] for p in measured])) for k in ARMS},
                unavailable={k: sum(p['missing'][k] for p in results['panels']) for k in ARMS[1:]},
                panels=measured)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = p.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS},
                                    panels=PANELS, ridge=RIDGE))
        return
    sha = base.sha(lock)
    assert base.read(lock)['sources'] == sources()
    if phase == 'forecast':
        slots = {s: forecast_slot(s) for s in SLOTS}
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=sha, slots=slots))
        return
    data = base.read(forecast)
    assert data['lock_sha'] == sha
    summary = {s: score_slot(s, data['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=sha, forecast_sha=base.sha(forecast),
                                   slots=summary))


if __name__ == '__main__':
    main()
