"""Replicate independent passive calibration on independent whole panels."""
import argparse
import importlib.util
from pathlib import Path
import sys
import os

HERE = Path(__file__).resolve().parent
BIAS = HERE.parent / 'fine_anchor_bias_20261109'
PREVIOUS = HERE.parent / 'fine_independent_validation_20261109'
sys.path.insert(0, str(PREVIOUS))
spec = importlib.util.spec_from_file_location('anchor_bias', BIAS / 'run.py')
bias = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bias)
base = bias.base

import json
import numpy as np
from fine_forward import load_fine
from fresh_data import fresh_observations, replay_physical

PANEL_COUNT, EVAL_N, RESAMPLES = 64, 128, 4096
POINT_RULES = ('native', 'historical', 'fine', 'raw')
WARNING_RULES = ('native_margin', 'fine_point', 'fine_robust', 'raw_point', 'raw_robust')


def write(name, result):
    target = HERE / name
    temporary = target.with_suffix('.part')
    if target.exists():
        raise FileExistsError(target)
    with temporary.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False,
                  separators=(',', ':'), default=lambda x: x.tolist())
        stream.write('\n')
    try:
        os.link(temporary, target)
    finally:
        temporary.unlink()


def sources():
    return base.current_sources() | {str(p): base.sha(p) for p in (
        Path(__file__), HERE / 'PROTOCOL.md', BIAS / 'run.py', BIAS / 'LOCK.json',
        BIAS / 'SUMMARY.json')}


def schedule(slot):
    s = bias.SLOTS.index(slot)
    return [(2026111000 + 1000 * s + 2 * k, 2026111001 + 1000 * s + 2 * k)
            for k in range(PANEL_COUNT)]


def direction_weights(groups, seed):
    rng = np.random.default_rng(seed)
    weights = []
    for group in groups:
        n = len(group)
        draws = rng.integers(n, size=(base.old.NBOOT, n))
        weights.append(np.stack([np.bincount(row, minlength=n) for row in draws]))
    return np.concatenate(weights, axis=1)


def fits(slot):
    x, a, u, owner = base.inputs(slot)
    n, t, _ = base.old.SLOTS[slot]
    mx, sx = base.scale_fit(x)
    train = {'raw': ((x - mx) / sx, lambda values: (values - mx) / sx, None)}
    for rep in range(3):
        y, _ = base.old.fine_train(slot, rep, n, t, owner)
        encoder = load_fine(slot, rep)

        def encode(values, model=encoder):
            return (model.encode(values) - model.mean) / model.std

        order = base.historical_match(slot, rep)['perm']
        train[f'fine{rep}'] = y, encode, np.eye(4)[order]

    result = {}
    for name, (history, encode, native) in train.items():
        coef = bias.fit_drift(history, a, u)
        rr = bias.innovation(history, a, coef)
        hist = bias.pooled(bias.stats(rr, u), np.arange(n) % 4 != 0)
        cov = bias.all_cov(history)
        result[name] = dict(coef=coef, hist=hist, cov=cov, encode=encode, native=native)
    return result


def predict(config, cal, evaluation, groups, weights):
    y = config['encode'](cal['x'])
    e = config['encode'](evaluation['x'])
    coef, cov = config['coef'], config['cov']
    mm = base.score_matrix(e)
    stats = bias.stats(bias.innovation(y, cal['actions'], coef), cal['regimes'])
    sums, counts = stats
    indexes = np.concatenate(groups)
    pooled = base.old.axes(sums[indexes].sum(0), counts[indexes].sum(0))
    group_axes = [base.old.axes(sums[g].sum(0), counts[g].sum(0)) for g in groups]
    boot = base.old.axes(np.einsum('be,ecij->bcij', weights, sums[indexes]), weights @ counts[indexes])
    valid = int(boot['good'].sum())
    unavailable = np.full((len(e), 2), -1)
    values = dict(points=unavailable, accepted=np.zeros(len(e), dtype=bool),
                  draws=valid, condition=float(pooled['condition']) if np.isfinite(pooled['condition']) else None,
                  pooled_usable=bool(pooled['good']))
    if pooled['good']:
        scores = base.old.slowness(pooled['columns'][None], mm, cov)[0]
        top = base.point_sets(scores)
        usable = (top >= 0).all(1)
        sets = np.where(usable[:, None], top, [0, 1])
        accepted = np.zeros(len(e), dtype=bool)
        if valid / base.old.NBOOT >= .95 and all(group['good'] for group in group_axes):
            folds = base.old.slowness(np.stack([g['columns'] for g in group_axes]), mm, cov)
            draws = base.old.slowness(boot['columns'][boot['good']], mm, cov)
            accepted = (usable & (base.margin(folds, sets) > 0).all(0)
                        & (np.quantile(base.margin(draws, sets), .025, axis=0) > 0))
        values.update(points=top, accepted=accepted)
    if config['native'] is not None:
        native_scores = bias.project(mm, config['native'])
        native = base.point_sets(native_scores)
        safe = np.where((native >= 0).all(1)[:, None], native, [0, 1])
        native_margin = base.margin(native_scores, safe)
        native_low = np.zeros(len(native), dtype=bool)
        native_low[np.argsort(native_margin, kind='stable')[:6]] = True
        historical = bias.from_columns(config['hist'], cov, mm)['scores']
        values.update(native=native, native_margin_warn=native_low,
                      historical=base.point_sets(historical) if historical is not None else unavailable)
    return values


def forecast_slot(slot):
    config = fits(slot)
    n, _, _ = base.old.SLOTS[slot]
    groups = [np.arange(i, n, 4) for i in (1, 2, 3)]
    panels = []
    for i, (cs, es) in enumerate(schedule(slot)):
        cal, cr = fresh_observations(slot, n, cs)
        evaluation, er = fresh_observations(slot, EVAL_N, es)
        weights = direction_weights(groups, es + 900000)
        arms = {name: predict(view, cal, evaluation, groups, weights) for name, view in config.items()}
        panels.append(dict(index=i, calibration=cr, evaluation=er, arms=arms))
    return panels


def count(mask, subset):
    return np.array([int((mask & subset).sum()), int(subset.sum())])


def evaluation_panel(slot, panel):
    sd = base.historical_match(slot, 0)['train_sd_z']
    z = replay_physical(panel['evaluation']).astype(np.float64) / sd
    scores = np.diagonal(base.score_matrix(z), axis1=1, axis2=2)
    truth = base.point_sets(scores)
    strict = (truth >= 0).all(1)
    edge = np.sort(scores, axis=1)
    a, b = base.old.SLOTS[slot][-1]
    straddles = (truth == a).any(1) ^ (truth == b).any(1)
    masks = {'all': strict, 'specified_0.05': strict & straddles & (abs(scores[:, b] - scores[:, a]) <= .05)}
    for width in (.05, .02, .005):
        masks[f'near_{width}'] = strict & ((edge[:, 2] - edge[:, 1]) <= width)
    result = {}
    for rep in range(3):
        fine, raw = panel['arms'][f'fine{rep}'], panel['arms']['raw']
        native = np.asarray(fine['native'])
        methods = {'native': native, 'historical': np.asarray(fine['historical']),
                   'fine': np.asarray(fine['points']), 'raw': np.asarray(raw['points'])}
        hits = {k: (v == truth).all(1) & strict for k, v in methods.items()}
        eligible = strict & (native >= 0).all(1)
        wrong = eligible & ~hits['native']
        correct = eligible & hits['native']
        warns = {'native_margin': np.asarray(fine['native_margin_warn'])}
        selective = {}
        for label, rule in (('fine', fine), ('raw', raw)):
            accepted = np.asarray(rule['accepted'])
            point = (methods[label] != native).any(1) & (methods[label] >= 0).all(1)
            warns[f'{label}_point'] = point
            warns[f'{label}_robust'] = point & accepted
            selective[label] = dict(coverage=count(accepted, strict),
                                    wrong_and_decide=count(~hits[label] & accepted, strict),
                                    wrong_given_decide=count(~hits[label], strict & accepted))
        result[str(rep)] = dict(
            points={m: {k: count(v, mask) for k, mask in masks.items()} for m, v in hits.items()},
            warning={k: dict(hit=count(v, wrong), fpr=count(v, correct)) for k, v in warns.items()},
            selective=selective)
    return dict(models=result, support={k: int(m.sum()) for k, m in masks.items()})


def bootstrap(values, weights):
    counts = np.asarray(values)
    whole = counts.sum(0)
    draws = weights @ counts
    samples = np.divide(draws[:, 0], draws[:, 1], out=np.full(len(weights), np.nan), where=draws[:, 1] > 0)
    return dict(count=int(whole[0]), n=int(whole[1]),
                rate=float(whole[0] / whole[1]) if whole[1] else None), samples


def upper(samples, q):
    valid = samples[np.isfinite(samples)]
    return float(np.quantile(valid, q)) if len(valid) else None


def paired_interval(first, second):
    (a, sample_a), (b, sample_b) = first, second
    delta = sample_a - sample_b
    return dict(gain=a['rate'] - b['rate'] if a['rate'] is not None and b['rate'] is not None else None,
                low_975=upper(delta, .025), high_975=upper(delta, .975))


def score_slot(slot, panels):
    assert [p['index'] for p in panels] == list(range(PANEL_COUNT))
    measured = [evaluation_panel(slot, p) for p in panels]
    rng = np.random.default_rng(2026111901 + bias.SLOTS.index(slot))
    indexes = rng.integers(PANEL_COUNT, size=(RESAMPLES, PANEL_COUNT))
    weights = np.stack([np.bincount(row, minlength=PANEL_COUNT) for row in indexes])
    keys = measured[0]['support']
    models = []
    for rep in range(3):
        rows = [p['models'][str(rep)] for p in measured]
        point = {m: {k: bootstrap([r['points'][m][k] for r in rows], weights)
                     for k in keys} for m in POINT_RULES}
        comparisons = {m: {k: paired_interval(point['fine'][k], point[m][k]) for k in keys}
                       for m in ('native', 'historical', 'raw')}
        warning = {m: {k: bootstrap([r['warning'][m][k] for r in rows], weights)
                       for k in ('hit', 'fpr')} for m in WARNING_RULES}
        warning_comparisons = {m: paired_interval(warning[m]['hit'], warning['native_margin']['hit'])
                               for m in WARNING_RULES if m != 'native_margin'}
        point_out = {m: {k: v[0] for k, v in stats.items()} for m, stats in point.items()}
        warn_out = {m: dict(detection=r['hit'][0], false_alarm=r['fpr'][0],
                            false_alarm_upper95=upper(r['fpr'][1], .95)) for m, r in warning.items()}
        selective = {label: {k: bootstrap([r['selective'][label][k] for r in rows], weights)[0]
                             for k in ('coverage', 'wrong_and_decide', 'wrong_given_decide')}
                     for label in ('fine', 'raw')}
        sample_gains = [int(r['points']['fine']['near_0.05'][0] - r['points']['native']['near_0.05'][0])
                        for r in rows]
        models.append(dict(rep=rep, point=point_out, comparisons=comparisons,
                           warning=warn_out, warning_comparisons=warning_comparisons,
                           selective=selective, panel_fine_minus_native_near=sample_gains))
    main = models[0]
    key = 'near_0.05'
    primary_gate = (
        all(main['comparisons'][m][key]['gain'] >= .03 and main['comparisons'][m][key]['low_975'] > 0
            for m in ('native', 'raw'))
        and main['comparisons']['native']['all']['gain'] >= -.01
        and any(all(model['comparisons'][m][key]['gain'] > 0 for m in ('native', 'raw'))
                for model in models[1:]))
    warning_gates = {}
    for rule in ('fine_point', 'fine_robust'):
        rate, inc = main['warning'][rule], main['warning_comparisons'][rule]
        warning_gates[rule] = (rate['detection']['n'] >= 20 and rate['false_alarm_upper95'] < .05
                               and inc['gain'] >= .10 and inc['low_975'] > 0)
    diagnostics = {arm: dict(
        usable_panels=sum(p['arms'][arm]['pooled_usable'] for p in panels),
        minimum_valid_draws=min(p['arms'][arm]['draws'] for p in panels))
        for arm in ('raw', 'fine0', 'fine1', 'fine2')}
    return dict(models=models, support={k: sum(p['support'][k] for p in measured) for k in keys},
                primary_gate=primary_gate, warning_gates=warning_gates, direction_availability=diagnostics,
                panels=measured)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = cli.parse_args().phase
    lock = HERE / 'LOCK.json'
    forecast = HERE / 'FORECAST.json'
    if phase == 'lock':
        write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: schedule(s) for s in bias.SLOTS},
                                evaluation_trajectories_per_panel=EVAL_N, resamples=RESAMPLES))
        return
    assert base.read(lock)['sources'] == sources()
    lock_sha = base.sha(lock)
    if phase == 'forecast':
        panels = {s: forecast_slot(s) for s in bias.SLOTS}
        write('FORECAST.json', dict(utc=base.now(), lock_sha=lock_sha, slots=panels))
        return
    frozen = base.read(forecast)
    assert frozen['lock_sha'] == lock_sha
    summary = {s: score_slot(s, frozen['slots'][s]) for s in bias.SLOTS}
    write('SUMMARY.json', dict(utc=base.now(), lock_sha=lock_sha,
                               forecast_sha=base.sha(forecast), slots=summary))


if __name__ == '__main__':
    main()
