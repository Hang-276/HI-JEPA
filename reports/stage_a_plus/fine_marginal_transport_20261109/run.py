"""Compare affine, Gaussian-rank, and unpaired physical marginal transport."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / 'fine_target_reference_20261109'
sys.path.insert(0, str(TARGET))
spec = importlib.util.spec_from_file_location('target_reference', TARGET / 'run.py')
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)
src, base, bias, rep = tr.src, tr.base, tr.bias, tr.rep
rep.HERE = HERE

import numpy as np
import torch

SLOTS, PANELS = tr.SLOTS, 32
METHODS = ('native', 'affine', 'normal', 'marginal')


def seeds(slot):
    i = SLOTS.index(slot)
    root = 2026155000 + 1000 * i
    return dict(reference=2026154000 + i,
                panels=[(root + 2 * k, root + 2 * k + 1) for k in range(PANELS)])


def sources():
    return tr.sources() | {str(p): base.sha(p) for p in (
        Path(__file__), HERE / 'PROTOCOL.md', HERE / 'THEORY.md', TARGET / 'run.py', TARGET / 'SUMMARY.json')}


def orient(y, a, u, coef, rows, keep):
    predicted = src.eq._design(y[keep, :-1], a[keep]) @ coef
    residual = np.diff(y[keep], axis=1) - predicted
    labels = u[keep]
    reference = residual[labels == 0].mean(0)
    changes = np.stack([residual[labels == j].mean(0) - reference for j in range(1, 5)])
    mean_response = (changes * rows).sum(1)
    return rows * np.where(mean_response >= 0., 1., -1.)[:, None], mean_response


def sorted_columns(axes):
    return np.sort(axes.reshape(-1, 4), axis=0)


def transport(y, source, target):
    return np.stack([np.interp(y[..., j], source[:, j], target[:, j]) for j in range(4)], axis=-1)


def scores_on(y):
    return np.diagonal(base.score_matrix(y), axis1=1, axis2=2)


def flags(scores, native):
    safe = np.where((native >= 0).all(1)[:, None], native, [0, 1])
    return dict(point=base.point_sets(scores), alarm=src.previous.alarm_mask(base.margin(scores, safe)))


def public(slot, receipt, role):
    obs, rebuilt = src.shift_data.fresh_observations(
        slot, receipt['epsilon'], receipt['count'], receipt['seed'], role)
    assert rebuilt == receipt
    return obs


def forecast_slot(slot):
    _, views, a, _ = src.reference.historical(slot)
    _, encode = views['fine']
    eps, rng = tr.target(slot), seeds(slot)
    ref, rr = src.shift_data.fresh_observations(slot, eps, tr.REFERENCE_N, rng['reference'], 'evaluation')
    yref = encode(ref['x'])
    cov = bias.all_cov(yref)
    native = bias.unit_rows(np.eye(4)[base.historical_match(slot, 0)['perm']], cov)
    grid = (np.arange(yref.shape[0] * yref.shape[1]) + .5) / (yref.shape[0] * yref.shape[1])
    normal = torch.special.ndtri(torch.from_numpy(grid)).numpy()
    normal = np.broadcast_to((normal / normal.std())[:, None], (len(normal), 4))
    fit = np.arange(len(a)) % 4 == 0
    panels = []
    for i, (cs, es) in enumerate(rng['panels']):
        c, cr = src.shift_data.fresh_observations(slot, eps, len(a), cs, 'calibration')
        e, er = src.shift_data.fresh_observations(slot, eps, src.previous.EVAL, es, 'evaluation')
        yc, ye = encode(c['x']), encode(e['x'])
        drift = src.eq.fit_drift(yc[fit], c['actions'][fit], c['regimes'][fit])
        w = src.eq.rows(yc, c['actions'], c['regimes'], drift, cov, ~fit)
        native_scores = scores_on(ye @ native.T)
        own = base.point_sets(native_scores)
        if w is None:
            affine, gaussian, mean_shift = native_scores, native_scores, None
        else:
            w, mean_shift = orient(yc, c['actions'], c['regimes'], drift, w, ~fit)
            projection = ye @ w.T
            affine = scores_on(projection)
            gaussian = scores_on(transport(projection, sorted_columns(yref @ w.T), normal))
        pred = {k: flags(v, own) for k, v in (('native', native_scores), ('affine', affine), ('normal', gaussian))}
        panels.append(dict(index=i, calibration=cr, evaluation=er, rows=w, mean_shift=mean_shift, predictions=pred))
    return dict(reference=rr, panels=panels)


def physical_marginal_slot(slot, frozen):
    # Only the marginal reference is revealed here, never an evaluation trajectory.
    ref = src.shift_data.replay_physical(frozen['reference']).astype(np.float64)
    physical = sorted_columns(ref)
    del ref
    physical /= physical.std(0)
    _, views, _, _ = src.reference.historical(slot)
    _, encode = views['fine']
    ref_y = encode(public(slot, frozen['reference'], 'evaluation')['x'])
    panels = []
    for row in frozen['panels']:
        own = np.asarray(row['predictions']['native']['point'])
        if row['rows'] is None:
            panels.append(row['predictions']['native'])
            continue
        w = np.asarray(row['rows'])
        y = encode(public(slot, row['evaluation'], 'evaluation')['x'])
        rescaled = transport(y @ w.T, sorted_columns(ref_y @ w.T), physical)
        panels.append(flags(scores_on(rescaled), own))
    return panels


def score_slot(slot, frozen, oracle):
    reference = src.shift_data.replay_physical(frozen['reference']).astype(np.float64)
    scale = reference.std((0, 1))
    _, views, _, _ = src.reference.historical(slot)
    _, encode = views['fine']
    measured, directions = [], []
    for row, marginal in zip(frozen['panels'], oracle):
        pred = row['predictions'] | {'marginal': marginal}
        z = src.shift_data.replay_physical(row['evaluation']).astype(np.float64) / scale
        zscore = scores_on(z)
        truth = base.point_sets(zscore)
        strict = (truth >= 0).all(1)
        sorted_score = np.sort(zscore, axis=1)
        near = strict & ((sorted_score[:, 2] - sorted_score[:, 1]) <= .05)
        ok = {m: (np.asarray(v['point']) == truth).all(1) & strict for m, v in pred.items()}
        eligible = strict & (np.asarray(pred['native']['point']) >= 0).all(1)
        good, bad = eligible & ok['native'], eligible & ~ok['native']
        alert = {m: np.asarray(v['alarm']) for m, v in pred.items()}
        measured.append(dict(point={m: dict(all=rep.count(v, strict), near=rep.count(v, near)) for m, v in ok.items()},
                             alarm={m: dict(hit=rep.count(v, bad), fpr=rep.count(v, good)) for m, v in alert.items()}))
        if row['rows'] is not None:
            q = encode(public(slot, row['evaluation'], 'evaluation')['x']) @ np.asarray(row['rows']).T
            directions.append(((q-q.mean(1, keepdims=True)) * (z-z.mean(1, keepdims=True))).mean(1) > 0)
    rng = np.random.default_rng(2026155901 + SLOTS.index(slot))
    draws = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(row, minlength=PANELS) for row in draws])
    point = {m: {k: rep.bootstrap([r['point'][m][k] for r in measured], weights) for k in ('near', 'all')}
             for m in METHODS}
    alarm = {m: {k: rep.bootstrap([r['alarm'][m][k] for r in measured], weights) for k in ('hit', 'fpr')}
             for m in METHODS}
    contrasts = {m: dict(near=rep.paired_interval(point[m]['near'], point['affine']['near']),
                         all=rep.paired_interval(point[m]['all'], point['affine']['all']),
                         hit=rep.paired_interval(alarm[m]['hit'], alarm['affine']['hit']))
                 for m in ('normal', 'marginal')}
    main = contrasts['marginal']
    passed = main['near']['gain'] >= .03 and main['near']['low_975'] > 0 and main['all']['gain'] > -.01
    return dict(passed=passed, contrasts=contrasts,
                signs_positive=np.concatenate(directions).mean(0) if directions else None,
                point={m: {k: v[0] for k, v in d.items()} for m, d in point.items()},
                alarm={m: {k: v[0] for k, v in d.items()} for m, d in alarm.items()},
                unavailable=sum(row['rows'] is None for row in frozen['panels']), panels=measured)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('lock', 'forecast', 'oracle', 'score'))
    phase = p.parse_args().phase
    lock, forecast, oracle = HERE / 'LOCK.json', HERE / 'FORECAST.json', HERE / 'ORACLE.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS}, panels=PANELS))
        return
    sha = base.sha(lock)
    assert base.read(lock)['sources'] == sources()
    if phase == 'forecast':
        slots = {s: forecast_slot(s) for s in SLOTS}
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=sha, slots=slots))
        return
    predictions = base.read(forecast)
    assert predictions['lock_sha'] == sha
    forecast_sha = base.sha(forecast)
    if phase == 'oracle':
        slots = {s: physical_marginal_slot(s, predictions['slots'][s]) for s in SLOTS}
        rep.write('ORACLE.json', dict(utc=base.now(), lock_sha=sha, forecast_sha=forecast_sha, slots=slots))
        return
    reference = base.read(oracle)
    assert reference['forecast_sha'] == forecast_sha and reference['lock_sha'] == sha
    slots = {s: score_slot(s, predictions['slots'][s], reference['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=sha, forecast_sha=forecast_sha,
                                   oracle_sha=base.sha(oracle), slots=slots))


if __name__ == '__main__':
    main()
