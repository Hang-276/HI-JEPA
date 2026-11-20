"""Fit observational randomized-regime moments on fresh FHN calibration."""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'fine_active_shift_20261109'
NOISE = HERE.parent / 'fine_active_noise_20261109'
sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location('active_shift', OLD / 'run.py')
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)
tr, src, base, bias, rep, mt, pr = ac.tr, ac.src, ac.base, ac.bias, ac.rep, ac.mt, ac.pr
rep.HERE = HERE

SLOTS, PANELS = tr.SLOTS, 32
METHODS = ('native', 'passive') + tuple(
    f'{view}_{kind}_{degree}' for view in ('fine', 'raw')
    for kind in ('direct', 'residual') for degree in ('linear', 'cubic'))
CONTRASTS = (('fine_residual_cubic', 'passive'), ('fine_residual_cubic', 'fine_residual_linear'),
             ('fine_residual_cubic', 'fine_direct_cubic'), ('fine_residual_cubic', 'raw_residual_cubic'))
DOSE = .035 * np.eye(4)


def sources():
    return ac.sources() | {str(p): base.sha(p) for p in (
        Path(__file__), HERE / 'PROTOCOL.md', HERE / 'THEORY.md', NOISE / 'LOCK.json', NOISE / 'SUMMARY.json')}


def seeds(slot):
    i = SLOTS.index(slot)
    start = 2026187000 + 1000 * i
    return dict(reference=2026186000 + i,
                panels=[(start + 2 * k, start + 2 * k + 1) for k in range(PANELS)])


def contexts(raw, actions):
    return np.concatenate((raw[:, :-1], actions), axis=-1)


def regime_moment(inst, residual, regimes):
    n = len(regimes)
    assert n > 0 and np.all((0 <= regimes) & (regimes <= 4))
    base_moment = inst[regimes == 0].T @ residual[regimes == 0]
    a = np.concatenate([(inst[regimes == j].T @ residual[regimes == j] - base_moment) * (5. / n)
                        for j in range(1, 5)], axis=0)
    b = np.concatenate([inst.mean(0)[:, None] * d[None] for d in DOSE], axis=0)
    return a, b


def fit_drift_residual(outcomes, design, folds):
    residual = np.empty_like(outcomes)
    design_scale = np.sqrt(np.mean(design[folds != 3] ** 2, axis=0))
    assert np.all(design_scale > 0)
    design = design / design_scale
    for fold in range(4):
        keep = folds == fold
        train = (folds != 3) & ~keep
        beta, _, rank, _ = np.linalg.lstsq(design[train], outcomes[train], rcond=None)
        assert rank == design.shape[1]
        residual[keep] = outcomes[keep] - design[keep] @ beta
    return residual


def fit_readout(train, check, reference, evaluation):
    a, b = train
    columns = np.sqrt(np.mean(a ** 2, axis=0))
    if np.any(columns <= 0):
        return None, dict(condition=None, check=None)
    beta, _, rank, singular = np.linalg.lstsq(a / columns, b, rcond=None)
    cond = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    if rank < a.shape[1] or cond is None or cond >= 1e6:
        return None, dict(condition=cond, check=None)
    beta /= columns[:, None]
    a_check, b_check = check
    err = np.linalg.norm(a_check @ beta - b_check) / np.linalg.norm(b_check)
    sd = (reference @ beta).std((0, 1))
    assert np.all(sd > 0)
    return mt.scores_on((evaluation @ beta) / sd), dict(condition=cond, check=err)


def forecast_slot(slot):
    _, views, actions, _ = src.reference.historical(slot)
    encode = {name: views[name][1] for name in ('fine', 'raw')}
    eps, roots = tr.target(slot), seeds(slot)
    _, env, _, cfg, _ = src.shift_data._recipe(slot, eps, 0)
    means, sigma = env.regimes(cfg)
    np.testing.assert_array_equal(means[1:5] - means[0], DOSE)
    np.testing.assert_array_equal(sigma[1:5], np.broadcast_to(sigma[0], (4, 4)))
    public, reference = src.shift_data.fresh_observations(slot, eps, tr.REFERENCE_N, roots['reference'], 'evaluation')
    ref = {name: read(public['x']) for name, read in encode.items()}
    center = {name: y.mean((0, 1)) for name, y in ref.items()}
    ref_features = {name: pr.polynomial(y - center[name], 3)[..., 1:] for name, y in ref.items()}
    ref_context = contexts(ref['raw'], public['actions'])
    context_center, context_scale = ref_context.mean((0, 1)), ref_context.std((0, 1))
    ref_instruments = pr.polynomial((ref_context - context_center) / context_scale, 2)
    instrument_scale = np.sqrt(np.mean(ref_instruments ** 2, axis=(0, 1)))
    cov = bias.all_cov(ref['fine'])
    native = bias.unit_rows(np.eye(4)[base.historical_match(slot, 0)['perm']], cov)
    folds = np.arange(len(actions)) % 4
    panels = []
    for k, (cs, es) in enumerate(roots['panels']):
        cal, cr = src.shift_data.fresh_observations(slot, eps, len(actions), cs, 'calibration')
        eva, er = src.shift_data.fresh_observations(slot, eps, src.previous.EVAL, es, 'evaluation')
        yc = {name: read(cal['x']) for name, read in encode.items()}
        ye = {name: read(eva['x']) for name, read in encode.items()}
        drift = src.eq.fit_drift(yc['fine'][folds == 0], cal['actions'][folds == 0], cal['regimes'][folds == 0])
        w = src.eq.rows(yc['fine'], cal['actions'], cal['regimes'], drift, cov, folds != 0)
        ns = mt.scores_on(ye['fine'] @ native.T)
        own = base.point_sets(ns)
        scores = dict(native=ns, passive=ns if w is None else mt.scores_on(ye['fine'] @ w.T))
        eligible = cal['regimes'] <= 4
        u = cal['regimes'][eligible]
        f = np.broadcast_to(folds[:, None], eligible.shape)[eligible]
        train = f != 3
        context = ((contexts(yc['raw'], cal['actions']) - context_center) / context_scale)[eligible]
        instruments = pr.polynomial(context, 2) / instrument_scale
        nuisance = pr.polynomial(context, 3)
        audit = {}
        for name in ('fine', 'raw'):
            obs_features = pr.polynomial(yc[name] - center[name], 3)[..., 1:]
            output = np.diff(obs_features, axis=1)[eligible]
            residual = fit_drift_residual(output, nuisance, f)
            ev_features = pr.polynomial(ye[name] - center[name], 3)[..., 1:]
            for kind, variation in (('direct', output), ('residual', residual)):
                a, b = regime_moment(instruments[train], variation[train], u[train])
                c, d = regime_moment(instruments[~train], variation[~train], u[~train])
                for degree, columns in (('linear', slice(0, 4)), ('cubic', slice(None))):
                    method = f'{name}_{kind}_{degree}'
                    result, audit[method] = fit_readout((a[:, columns], b), (c[:, columns], d),
                                                         ref_features[name][..., columns], ev_features[..., columns])
                    scores[method] = ns if result is None else result
        panels.append(dict(index=k, calibration=cr, evaluation=er, audit=audit, missing_passive=w is None,
                           predictions={name: mt.flags(values, own) for name, values in scores.items()}))
    return dict(reference=reference, panels=panels)


def score_slot(slot, frozen):
    scale = src.shift_data.replay_physical(frozen['reference']).astype(np.float64).std((0, 1))
    panels = []
    for row in frozen['panels']:
        values = row['predictions']
        panels.append(tr.measure(dict(evaluation=row['evaluation'], point={m: v['point'] for m, v in values.items()},
                                      alerts={m: v['alarm'] for m, v in values.items()}), scale))
    rng = np.random.default_rng(2026187901 + SLOTS.index(slot))
    draws = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(d, minlength=PANELS) for d in draws])
    point = {m: {k: rep.bootstrap([p['point'][m][k] for p in panels], weights) for k in ('near', 'all')}
             for m in METHODS}
    alarm = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in panels], weights) for k in ('hit', 'fpr')}
             for m in METHODS}
    contrasts = {}
    for m, r in CONTRASTS:
        near = rep.paired_interval(point[m]['near'], point[r]['near'])
        near['lower_family'] = rep.upper(point[m]['near'][1] - point[r]['near'][1], .0125)
        all_windows = rep.paired_interval(point[m]['all'], point[r]['all'])
        contrasts[f'{m}_vs_{r}'] = dict(near=near, all=all_windows, hit=rep.paired_interval(alarm[m]['hit'], alarm[r]['hit']),
                                      passed=near['gain'] >= .03 and near['lower_family'] > 0 and all_windows['gain'] > -.01)
    diagnostic = {}
    for m in METHODS[2:]:
        audits = [p['audit'][m] for p in frozen['panels']]
        valid = [x for x in audits if x['check'] is not None]
        diagnostic[m] = dict(unavailable=len(audits) - len(valid), summary={
            k: [float(fn([v[k] for v in valid])) for fn in (np.min, np.median, np.max)] if valid else None
            for k in ('condition', 'check')})
    return dict(point={m: {k: v[0] for k, v in d.items()} for m, d in point.items()},
                alarm={m: {k: v[0] for k, v in d.items()} for m, d in alarm.items()}, contrasts=contrasts,
                diagnostics=diagnostic, missing_passive=sum(p['missing_passive'] for p in frozen['panels']),
                panels=panels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS}))
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
