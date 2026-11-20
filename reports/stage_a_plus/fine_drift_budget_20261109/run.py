"""Nested budget for target drift while leaving direction data fixed."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / 'fine_target_reference_20261109'
BALANCE = HERE.parent / 'fine_context_balance_20261109'
sys.path.insert(0, str(TARGET))
spec = importlib.util.spec_from_file_location('target_reference', TARGET / 'run.py')
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)
src, base, bias, rep = tr.src, tr.base, tr.bias, tr.rep
rep.HERE = HERE

import numpy as np

SLOTS, PANELS = tr.SLOTS, 32
BUDGETS = (6, 12, 24)
METHODS = ('native', 'source', *(f'target{n}' for n in BUDGETS))


def seeds(slot):
    i = SLOTS.index(slot)
    root = 2026150000 + 1000 * i
    return dict(reference=2026149000 + i,
                panels=[(root + 2 * k, root + 2 * k + 1) for k in range(PANELS)])


def sources():
    paths = (Path(__file__), HERE / 'PROTOCOL.md', TARGET / 'run.py', TARGET / 'LOCK.json',
             BALANCE / 'LOCK.json', BALANCE / 'SUMMARY.json')
    return tr.sources() | {str(p): base.sha(p) for p in paths}


def forecast_slot(slot):
    _, views, a, _ = src.reference.historical(slot)
    _, encode = views['fine']
    n = len(a)
    target_eps = tr.target(slot)
    rng = seeds(slot)
    pub, reference = src.shift_data.fresh_observations(
        slot, target_eps, tr.REFERENCE_N, rng['reference'], 'evaluation')
    cov = bias.all_cov(encode(pub['x']))
    native = bias.unit_rows(np.eye(4)[base.historical_match(slot, 0)['perm']], cov)
    fit = np.arange(n) % 4 == 0
    subset = np.flatnonzero(fit)
    assert len(subset) == max(BUDGETS)
    panels = []
    for i, (cs, es) in enumerate(rng['panels']):
        s, sr = src.shift_data.fresh_observations(slot, src.SOURCE_EPS[slot], n, cs, 'calibration')
        t, trr = src.shift_data.fresh_observations(slot, target_eps, n, cs, 'calibration')
        assert sr['physical_rng_stream_hashes'] == trr['physical_rng_stream_hashes']
        ev, er = src.shift_data.fresh_observations(slot, target_eps, src.previous.EVAL, es, 'evaluation')
        ys, yt = encode(s['x']), encode(t['x'])
        source = src.eq.fit_drift(ys[fit], s['actions'][fit], s['regimes'][fit])
        coefs = {'source': source}
        for k in BUDGETS:
            ix = subset[:k]
            coefs[f'target{k}'] = src.eq.fit_drift(yt[ix], t['actions'][ix], t['regimes'][ix])
        mm = base.score_matrix(encode(ev['x']))
        scores, missing = {'native': bias.project(mm, native)}, {}
        for k, coeff in coefs.items():
            w = src.eq.rows(yt, t['actions'], t['regimes'], coeff, cov, ~fit)
            src.projected(scores, mm, k, w, missing)
        points = {k: base.point_sets(v) for k, v in scores.items()}
        chosen = points['native']
        safe = np.where((chosen >= 0).all(1)[:, None], chosen, [0, 1])
        flags = {k: src.previous.alarm_mask(base.margin(v, safe)) for k, v in scores.items()}
        panels.append(dict(index=i, calibration=[sr, trr], evaluation=er, point=points, alerts=flags,
                           missing=missing, rank_failure={k: v is None for k, v in coefs.items()}))
    return dict(reference=reference, panels=panels)


def score_slot(slot, forecast):
    z = src.shift_data.replay_physical(forecast['reference']).astype(np.float64)
    scale = z.std((0, 1), ddof=0)
    measured = [tr.measure(p, scale) for p in forecast['panels']]
    rng = np.random.default_rng(2026150901 + SLOTS.index(slot))
    draws = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(row, minlength=PANELS) for row in draws])
    points = {m: {k: rep.bootstrap([x['point'][m][k] for x in measured], weights)
                  for k in ('all', 'near')} for m in METHODS}
    alarms = {m: {k: rep.bootstrap([x['alerts'][m][k] for x in measured], weights)
                  for k in ('hit', 'fpr')} for m in METHODS}
    risk = {m: rep.upper(a['fpr'][1], .95) for m, a in alarms.items()}
    noninferiority, improvement = {}, {}
    for k in BUDGETS:
        arm = f'target{k}'
        reference = 'target24'
        if k != 24:
            near = rep.paired_interval(points[arm]['near'], points[reference]['near'])
            hit = rep.paired_interval(alarms[arm]['hit'], alarms[reference]['hit'])
            near['low99375'] = rep.upper(points[arm]['near'][1] - points[reference]['near'][1], .00625)
            hit['low99375'] = rep.upper(alarms[arm]['hit'][1] - alarms[reference]['hit'][1], .00625)
            noninferiority[arm] = dict(near=near, hit=hit,
                passed=near['low99375'] is not None and near['low99375'] > -.02
                and hit['low99375'] is not None and hit['low99375'] > -.05)
        near = rep.paired_interval(points[arm]['near'], points['source']['near'])
        hit = rep.paired_interval(alarms[arm]['hit'], alarms['source']['hit'])
        improvement[arm] = dict(near=near, hit=hit,
            passed=near['gain'] is not None and near['gain'] >= .03 and near['low_975'] > 0
                and hit['gain'] is not None and hit['gain'] >= .10 and hit['low_975'] > 0
                and risk[arm] is not None and risk[arm] < .05)
    return dict(noninferiority=noninferiority, improvement=improvement,
                point={m: {k: v[0] for k, v in p.items()} for m, p in points.items()},
                alarm={m: dict(hit=a['hit'][0], fpr=a['fpr'][0], fpr_upper95=risk[m]) for m, a in alarms.items()},
                unavailable={m: sum(p['missing'][m] for p in forecast['panels']) for m in METHODS[1:]},
                rank_failure={m: sum(p['rank_failure'][m] for p in forecast['panels']) for m in METHODS[1:]},
                panels=measured)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = p.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS},
                                    panels=PANELS, budgets=BUDGETS))
        return
    sha = base.sha(lock)
    assert base.read(lock)['sources'] == sources()
    if phase == 'forecast':
        slots = {s: forecast_slot(s) for s in SLOTS}
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=sha, slots=slots))
        return
    predictions = base.read(forecast)
    assert predictions['lock_sha'] == sha
    summary = {s: score_slot(s, predictions['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=sha, forecast_sha=base.sha(forecast), slots=summary))


if __name__ == '__main__':
    main()
