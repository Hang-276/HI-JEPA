"""Equal-budget alarm ranking and unlabeled covariance-based model selection."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPLICATION = HERE.parent / 'fine_calibration_replication_20261109'
EXTENSION = HERE.parent / 'fine_replication_extension_20261109'
spec = importlib.util.spec_from_file_location('alarm_replication', REPLICATION / 'run.py')
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)
base, bias = rep.base, rep.bias
rep.HERE = HERE

import numpy as np

sys.path.insert(0, str(EXTENSION))
import extra_fine_forward as extra_fine
import extra_fresh_data as extra_data
import fine_forward
import fresh_data

SLOTS = {'ar_rich_fresh1': (128, 129, (0, 2)),
         'ar_rich_fresh2': (128, 129, (1, 2)),
         'fhn_fresh_eps008': (96, 193, (0, 3)),
         'fhn_fresh_eps05': (96, 193, (1, 2))}
base.old.SLOTS = SLOTS
RULES = ('native', 'historical', 'fresh_fine', 'fresh_raw')
PANELS, EVAL, ALARMS = 64, 128, 6
CHOICES = RULES[:3]


def schedules(slot):
    start = 2026115000 + 1000 * tuple(SLOTS).index(slot)
    return [(start + 2 * p, start + 2 * p + 1) for p in range(PANELS)]


def implementations(slot):
    if slot in extra_data.SLOTS:
        return extra_data, extra_fine
    return fresh_data, fine_forward


def sources():
    paths = (HERE / 'PROTOCOL.md', Path(__file__), REPLICATION / 'run.py',
             REPLICATION / 'LOCK.json', REPLICATION / 'SUMMARY.json',
             EXTENSION / 'extra_fine_forward.py', EXTENSION / 'extra_fresh_data.py',
             EXTENSION / 'LOCK.json', EXTENSION / 'SUMMARY.json', BIAS_RUN)
    out = base.current_sources() | extra_data.source_hashes() | {str(p): base.sha(p) for p in paths}
    for row in base.read(EXTENSION / 'ENGINEERING.json')['fine']:
        for ref in (row['completion'], row['job'], row['checkpoint'], row['g0'], *row['sources']):
            out[ref['path']] = base.sha(Path(ref['path']))
            assert out[ref['path']] == ref['sha256']
    return out


BIAS_RUN = HERE.parent / 'fine_anchor_bias_20261109/run.py'


def conditional_gap(residual, u, rows):
    scores = []
    for code in base.old.CODES:
        e = residual[u == code] @ rows.T
        e = e - e.mean(0)
        covariance = e.T @ e / len(e)
        sign, ld = np.linalg.slogdet(covariance)
        if sign <= 0 or (np.diag(covariance) <= 0).any():
            return None
        score = .5 * (np.log(np.diag(covariance)).sum() - ld)
        scores.append(max(0., float(score)))
    return float(np.mean(scores))


def direction_rows(stats, subset, cov):
    basis = bias.pooled(stats, subset)
    if not basis['good']:
        return None
    return bias.unit_rows(np.linalg.inv(basis['columns']), cov)


def alarm_mask(margin):
    selected = np.argsort(margin, kind='stable')[:ALARMS]
    out = np.zeros(len(margin), dtype=bool)
    out[selected] = True
    return out


def forecast_slot(slot):
    data, fine = implementations(slot)
    n, t, _ = SLOTS[slot]
    train_x, a, u, owner = base.inputs(slot)
    train_fine, _ = base.old.fine_train(slot, 0, n, t, owner)
    frozen = fine.load_fine(slot, 0)
    mu_x, sd_x = base.scale_fit(train_x)
    train_raw = (train_x - mu_x) / sd_x
    fine_cov, raw_cov = bias.all_cov(train_fine), bias.all_cov(train_raw)
    fine_drift = bias.fit_drift(train_fine, a, u)
    raw_drift = bias.fit_drift(train_raw, a, u)
    estimate_group = np.arange(n) % 4 != 0
    hold = ~estimate_group
    historic_stats = bias.stats(bias.innovation(train_fine, a, fine_drift), u)
    historical = direction_rows(historic_stats, estimate_group, fine_cov)
    perm = base.historical_match(slot, 0)['perm']
    native_rows = np.eye(4)[perm]
    forecasts = []
    for i, (cs, es) in enumerate(schedules(slot)):
        cal, cr = data.fresh_observations(slot, n, cs)
        evaluation, er = data.fresh_observations(slot, EVAL, es)
        yc = (frozen.encode(cal['x']) - frozen.mean) / frozen.std
        ye = (frozen.encode(evaluation['x']) - frozen.mean) / frozen.std
        rc = (cal['x'] - mu_x) / sd_x
        re = (evaluation['x'] - mu_x) / sd_x
        fine_r = bias.innovation(yc, cal['actions'], fine_drift)
        raw_r = bias.innovation(rc, cal['actions'], raw_drift)
        f = direction_rows(bias.stats(fine_r, cal['regimes']), estimate_group, fine_cov)
        r = direction_rows(bias.stats(raw_r, cal['regimes']), estimate_group, raw_cov)
        fy, ry = base.score_matrix(ye), base.score_matrix(re)
        scores = {'native': bias.project(fy, native_rows)}
        for method, w, mm in (('historical', historical, fy), ('fresh_fine', f, fy), ('fresh_raw', r, ry)):
            scores[method] = bias.project(mm, w) if w is not None else scores['native']
        native = base.point_sets(scores['native'])
        safe_native = np.where((native >= 0).all(1)[:, None], native, [0, 1])
        flags = {key: alarm_mask(base.margin(value, safe_native)) for key, value in scores.items()}
        gaps = {key: conditional_gap(fine_r[hold], cal['regimes'][hold], w) if w is not None else None
                for key, w in zip(CHOICES, (native_rows, historical, f))}
        choice = min((k for k in CHOICES if gaps[k] is not None), key=gaps.get, default='native')
        outputs = {key: base.point_sets(value) for key, value in scores.items()}
        outputs['selector'] = outputs[choice]
        forecasts.append(dict(index=i, calibration=cr, evaluation=er, alerts=flags,
                              point=outputs, gaps=gaps, chosen=choice,
                              available=[historical is not None, f is not None, r is not None]))
    return forecasts


def panel_score(slot, panel):
    data, _ = implementations(slot)
    truth = data.replay_physical(panel['evaluation']).astype(np.float64)
    truth /= base.historical_match(slot, 0)['train_sd_z']
    scores = np.diagonal(base.score_matrix(truth), axis1=1, axis2=2)
    target = base.point_sets(scores)
    strict = (target >= 0).all(1)
    sorted_scores = np.sort(scores, axis=1)
    near = strict & ((sorted_scores[:, 2] - sorted_scores[:, 1]) <= .05)
    point = {m: (np.asarray(p) == target).all(1) & strict for m, p in panel['point'].items()}
    eligible = strict & (np.asarray(panel['point']['native']) >= 0).all(1)
    error = eligible & ~point['native']
    correct = eligible & point['native']
    flags = {m: np.asarray(p) for m, p in panel['alerts'].items()}
    assert all(int(a.sum()) == ALARMS for a in flags.values())
    return dict(point={k: {'all': rep.count(v, strict), 'near': rep.count(v, near)}
                       for k, v in point.items()},
                alerts={k: dict(hit=rep.count(v, error), fpr=rep.count(v, correct)) for k, v in flags.items()})


def summarize(slot, panels):
    measures = [panel_score(slot, p) for p in panels]
    rng = np.random.default_rng(2026115901 + tuple(SLOTS).index(slot))
    index = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(row, minlength=PANELS) for row in index])
    alarm_stats = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in measures], weights)
                       for k in ('hit', 'fpr')} for m in RULES}
    point_stats = {m: {k: rep.bootstrap([p['point'][m][k] for p in measures], weights)
                       for k in ('all', 'near')} for m in (*RULES, 'selector')}
    alarm = {m: dict(detection=v['hit'][0], false_alarm=v['fpr'][0],
                     fpr_upper95=rep.upper(v['fpr'][1], .95)) for m, v in alarm_stats.items()}
    differences = {}
    ours = alarm_stats['fresh_fine']['hit']
    for m in ('native', 'historical', 'fresh_raw'):
        other = alarm_stats[m]['hit']
        comparison = rep.paired_interval(ours, other)
        comparison['low_9875'] = rep.upper(ours[1] - other[1], .0125)
        comparison['passed'] = (comparison['gain'] >= .10 and comparison['low_9875'] > 0
                                and max(alarm[m]['fpr_upper95'], alarm['fresh_fine']['fpr_upper95']) < .05)
        differences[m] = comparison
    selection = {m: {k: rep.paired_interval(point_stats['selector'][k], point_stats[m][k])
                     for k in ('all', 'near')} for m in CHOICES}
    vs_native = selection['native']
    selector_gate = (vs_native['near']['gain'] >= .03 and vs_native['near']['low_975'] > 0
                     and vs_native['all']['gain'] >= -.01)
    return dict(alarms=alarm, alarms_vs_fresh_fine=differences,
                point={m: {k: v[0] for k, v in d.items()} for m, d in point_stats.items()},
                selection_comparisons=selection, selector_gate=selector_gate,
                selections={k: sum(p['chosen'] == k for p in panels) for k in CHOICES},
                usable=[sum(p['available'][i] for p in panels) for i in range(3)],
                panels=measures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: schedules(s) for s in SLOTS},
                                    panels=PANELS, evaluation=EVAL, alarms=ALARMS))
        return
    assert base.read(lock)['sources'] == sources()
    lock_sha = base.sha(lock)
    if phase == 'forecast':
        slots = {slot: forecast_slot(slot) for slot in SLOTS}
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=lock_sha, slots=slots))
        return
    predictions = base.read(forecast)
    assert predictions['lock_sha'] == lock_sha
    result = {s: summarize(s, predictions['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=lock_sha, forecast_sha=base.sha(forecast), slots=result))


if __name__ == '__main__':
    main()
