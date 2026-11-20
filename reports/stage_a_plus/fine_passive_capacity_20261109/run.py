"""Cross passive variance readout capacity with raw versus frozen Fine."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / 'fine_target_reference_20261109'
CAPACITY = HERE.parent / 'fine_active_capacity_20261109'
sys.path.insert(0, str(TARGET))
spec = importlib.util.spec_from_file_location('target_reference', TARGET / 'run.py')
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)
src, base, bias, rep = target.src, target.base, target.bias, target.rep
rep.HERE = HERE

import numpy as np
import platform

SLOTS, PANELS = target.SLOTS, 32
DEGREES = {'linear': 4, 'cubic': 34}
VARIANTS = ('affine', *DEGREES)
METHODS = ('native', *(f'{view}_{variant}' for view in ('fine', 'raw') for variant in VARIANTS))
PRIMARY = (('raw_cubic', 'raw_linear'), ('fine_cubic', 'raw_cubic'))
AUXILIARY = (('fine_cubic', 'fine_linear'), ('fine_linear', 'raw_linear'),
             ('raw_cubic', 'raw_affine'), ('fine_cubic', 'fine_affine'),
             ('fine_affine', 'raw_affine'), ('fine_cubic', 'native'))
RIDGE, EPS = 1e-3, 1e-12


def seeds(slot):
    i = SLOTS.index(slot)
    root = 2026237000 + 1000 * i
    return dict(reference=2026236000 + i,
                panels=[(root + 2 * k, root + 2 * k + 1) for k in range(PANELS)])


def sources():
    parents = (Path(__file__), HERE / 'PROTOCOL.md', TARGET / 'run.py', TARGET / 'LOCK.json',
               CAPACITY / 'LOCK.json', CAPACITY / 'SUMMARY.json')
    return target.sources() | {str(p): base.sha(p) for p in parents}


def stream_audit():
    def roots(item):
        if isinstance(item, dict):
            if type(item.get('trajectory_seed')) is int:
                yield item['trajectory_seed']
            for child in item.values():
                yield from roots(child)
        elif isinstance(item, list):
            for child in item:
                yield from roots(child)

    paths = sorted({p for name in ('FORECAST.json', 'PASSIVE.json', 'CONTROLLED.json')
                    for p in HERE.parent.rglob(name) if HERE not in p.parents})
    archived = {str(p.relative_to(HERE.parent)): list(roots(base.read(p))) for p in paths}
    assert str(CAPACITY.name + '/FORECAST.json') in archived
    old = {r for found in archived.values() for r in found}
    _, rng = src.shift_data.base._sources()
    old.update(rng.seed_for("DataOwner", slot, 0, "data", role, namespace=namespace)
               for slot in rng.SLOTS for role in ('trajectory', 'mix')
               for namespace in (rng.NAMESPACE, rng.ENGINEERING_NAMESPACE))
    new = []
    for slot in SLOTS:
        plan = seeds(slot)
        new.append(src.shift_data._stream_seed(slot, plan['reference'], 'evaluation'))
        for c, e in plan['panels']:
            new.extend((src.shift_data._stream_seed(slot, c, 'calibration'),
                        src.shift_data._stream_seed(slot, e, 'evaluation')))
    assert len(new) == len(set(new)) and not (set(new) & old)
    return dict(archive_unique_roots=len(old), new_unique_roots=len(new),
                intersections=[], root_occurrences={p: len(r) for p, r in archived.items()})


def features(y):
    """Nonconstant state monomials, ordered by total degree."""
    return np.concatenate((y, *(np.prod(y[..., ix], axis=-1) for ix in src.eq._MONOMIALS)), axis=-1)


def ref_transform(reference):
    mu, sd = base.scale_fit(reference)
    values = features((reference - mu) / sd).reshape(-1, 34)
    centered = values - values.mean(0)
    return mu, sd, centered.T @ centered / len(centered)


def fit_feature_moments(cal, actions, regimes, fit):
    eligible = np.isin(regimes, base.old.CODES)
    training = fit[:, None] & eligible
    design = src.eq._design(cal[:, :-1], actions)
    differences = np.diff(features(cal), axis=1)
    coef, _, rank, _ = np.linalg.lstsq(design[training], differences[training], rcond=EPS)
    if rank != coef.shape[0]:
        return None, None, int(rank)
    residual = differences[~fit] - design[~fit] @ coef
    labels = regimes[~fit]
    moments = []
    for code in base.old.CODES:
        selected = residual[labels == code]
        if not len(selected):
            return coef, None, int(rank)
        moments.append(selected.T @ selected / len(selected))
    return coef, np.stack(moments), int(rank)


def generalized_rows(moments, covariance):
    rows, gaps = [], []
    for arm in range(1, 5):
        background = np.delete(moments, arm, axis=0).mean(0) + RIDGE * covariance
        eigen, vectors = np.linalg.eigh(background)
        if not np.all(eigen > 0):
            return None, None
        whiten = (vectors / np.sqrt(eigen)) @ vectors.T
        contrast = whiten @ (moments[arm] - moments[0]) @ whiten
        values, vectors = np.linalg.eigh((contrast + contrast.T) / 2)
        gap = (values[-1] - values[-2]) / max(np.abs(values).max(), np.finfo(float).tiny)
        if values[-1] <= 0 or gap <= EPS:
            return None, None
        rows.append(vectors[:, -1] @ whiten)
        gaps.append(gap)
    rows = np.stack(rows)
    output = rows @ covariance @ rows.T
    variances = np.diag(output)
    if not np.all(variances > 0):
        return None, None
    scale = np.sqrt(variances)
    condition = float(np.linalg.cond(output / np.outer(scale, scale)))
    info = dict(condition=condition if np.isfinite(condition) else None, eigengap=min(gaps))
    if condition > 100:
        return None, info
    return rows / scale[:, None], info


def readouts(cal, evaluation, actions, regimes, fit, reference):
    mu, sd, covariance = reference
    cal = (cal - mu) / sd
    evaluation = (evaluation - mu) / sd
    coeff, moments, rank = fit_feature_moments(cal, actions, regimes, fit)
    scores, audit = {}, {'drift_rank': rank}
    affine = src.eq.rows(cal, actions, regimes, None if coeff is None else coeff[:, :4],
                         covariance[:4, :4], ~fit)
    scores['affine'] = None if affine is None else bias.project(base.score_matrix(evaluation), affine)
    lifted = features(evaluation)
    for variant, d in DEGREES.items():
        rows, audit[variant] = (None, None) if moments is None else generalized_rows(
            moments[:, :d, :d], covariance[:d, :d])
        scores[variant] = None if rows is None else np.diagonal(
            base.score_matrix(lifted[..., :d] @ rows.T), axis1=1, axis2=2)
    return scores, audit


def smoke():
    checks = {}
    for slot in SLOTS:
        _, views, actions, regimes = src.reference.historical(slot)
        fit = np.arange(len(actions)) % 4 == 0
        checks[slot] = {}
        for view, (train, _) in views.items():
            ref = ref_transform(train)
            values, audit = readouts(train, train[:1], actions, regimes, fit, ref)
            assert audit['drift_rank'] == 37
            for scores in values.values():
                if scores is not None:
                    assert scores.shape == (1, 4) and np.isfinite(scores).all()
            assert values['affine'] is not None and values['linear'] is not None
            checks[slot][view] = dict(available={k: v is not None for k, v in values.items()},
                                      diagnostics=audit)
    return checks


def forecast_slot(slot):
    _, views, actions, _ = src.reference.historical(slot)
    encoders = {view: encode for view, (_, encode) in views.items()}
    fit = np.arange(len(actions)) % 4 == 0
    plan, epsilon = seeds(slot), src.SOURCE_EPS[slot]
    public, reference = src.shift_data.fresh_observations(
        slot, epsilon, target.REFERENCE_N, plan['reference'], 'evaluation')
    ref = {view: ref_transform(encode(public['x'])) for view, encode in encoders.items()}
    perm = base.historical_match(slot, 0)['perm']
    panels = []
    for i, (cs, es) in enumerate(plan['panels']):
        cal, cr = src.shift_data.fresh_observations(slot, epsilon, len(actions), cs, 'calibration')
        evaluation, er = src.shift_data.fresh_observations(slot, epsilon, src.previous.EVAL, es, 'evaluation')
        scores, missing, audit = {}, {}, {}
        for view, encode in encoders.items():
            ye = encode(evaluation['x'])
            if view == 'fine':
                ns = np.diagonal(base.score_matrix((ye - ref[view][0]) / ref[view][1]),
                                 axis1=1, axis2=2)[:, perm]
                scores['native'] = ns
            estimates, audit[view] = readouts(encode(cal['x']), ye, cal['actions'], cal['regimes'], fit, ref[view])
            scores.update({f'{view}_{variant}': value for variant, value in estimates.items()})
            missing.update({f'{view}_{variant}': value is None for variant, value in estimates.items()})
        native = base.point_sets(scores['native'])
        safe = np.where((native >= 0).all(1)[:, None], native, [0, 1])
        points, alerts = {}, {}
        for name, value in scores.items():
            points[name] = (np.full_like(native, -1) if value is None else base.point_sets(value))
            if value is None:
                value = scores[f'{name.split("_")[0]}_affine']
            if value is None:
                value = ns
            alerts[name] = src.previous.alarm_mask(base.margin(value, safe))
        panels.append(dict(index=i, calibration=cr, evaluation=er, point=points, alerts=alerts,
                           missing=missing, audit=audit))
    return dict(reference=reference, panels=panels)


def score_slot(slot, frozen):
    scale = src.shift_data.replay_physical(frozen['reference']).astype(np.float64).std((0, 1))
    measured = [target.measure(p, scale) for p in frozen['panels']]
    rng = np.random.default_rng(2026237901 + SLOTS.index(slot))
    indexes = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(row, minlength=PANELS) for row in indexes])
    points = {m: {k: rep.bootstrap([p['point'][m][k] for p in measured], weights)
                  for k in ('near', 'all')} for m in METHODS}
    alarms = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in measured], weights)
                  for k in ('hit', 'fpr')} for m in METHODS}
    misses = {m: sum(p['missing'][m] for p in frozen['panels']) for m in METHODS[1:]}
    contrasts = {}
    for m, r in PRIMARY + AUXILIARY:
        near = rep.paired_interval(points[m]['near'], points[r]['near'])
        near['lower_family'] = rep.upper(points[m]['near'][1] - points[r]['near'][1], .0125)
        whole = rep.paired_interval(points[m]['all'], points[r]['all'])
        passed = None
        if (m, r) in PRIMARY:
            passed = (max(misses[m], misses[r]) <= 2 and near['gain'] is not None
                      and near['gain'] >= .03 and near['lower_family'] is not None
                      and near['lower_family'] > 0 and whole['gain'] >= -.01)
        contrasts[f'{m}_vs_{r}'] = dict(near=near, all=whole, passed=passed,
                                        hit=rep.paired_interval(alarms[m]['hit'], alarms[r]['hit']))
    return dict(point={m: {k: v[0] for k, v in p.items()} for m, p in points.items()},
                alarm={m: {k: v[0] for k, v in p.items()} for m, p in alarms.items()},
                unavailable=misses, contrasts=contrasts, panels=measured)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'smoke', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock, engineering, forecast = (HERE / f'{name}.json' for name in ('LOCK', 'ENGINEERING', 'FORECAST'))
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), streams=stream_audit(),
                                    seeds={s: seeds(s) for s in SLOTS}))
        return
    sha = base.sha(lock)
    assert base.read(lock)['sources'] == sources()
    if phase == 'smoke':
        assert platform.machine() == 'arm64'
        rep.write('ENGINEERING.json', dict(utc=base.now(), lock_sha=sha, architecture=platform.machine(),
                                           checks=smoke()))
        return
    assert base.read(engineering)['lock_sha'] == sha
    if phase == 'forecast':
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=sha, engineering_sha=base.sha(engineering),
                                        slots={slot: forecast_slot(slot) for slot in SLOTS}))
        return
    frozen = base.read(forecast)
    assert frozen['lock_sha'] == sha and frozen['engineering_sha'] == base.sha(engineering)
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=sha, forecast_sha=base.sha(forecast),
                                   slots={slot: score_slot(slot, frozen['slots'][slot]) for slot in SLOTS}))


if __name__ == '__main__':
    main()
