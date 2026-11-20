"""Test matching calibration under a fixed, independent target reference scale."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SHIFT = HERE.parent / 'fine_calibration_shift_20261109'
sys.path.insert(0, str(SHIFT))
spec = importlib.util.spec_from_file_location('source_scale_shift', SHIFT / 'run.py')
src = importlib.util.module_from_spec(spec)
spec.loader.exec_module(src)
base, bias, rep = src.base, src.bias, src.rep
rep.HERE = HERE

import numpy as np

METHODS = src.METHODS
SLOTS = tuple(src.SOURCE_EPS)
REFERENCE_N = 256


def target(slot):
    return next(e for e in src.EPS if e != src.SOURCE_EPS[slot])


def seeds(slot):
    i = SLOTS.index(slot)
    start = 2026125000 + 1000 * i
    return dict(reference=2026124000 + i,
                panels=[(start + 2 * k, start + 2 * k + 1) for k in range(src.previous.PANELS)])


def sources():
    paths = (Path(__file__), HERE / 'PROTOCOL.md', SHIFT / 'LOCK.json', SHIFT / 'SUMMARY.json',
             SHIFT / 'REFERENCE_DIAGNOSTIC.json', SHIFT / 'run.py')
    return src.sources() | {str(p): base.sha(p) for p in paths}


def forecast_slot(slot):
    _, reps, a, u = src.reference.historical(slot)
    _, fine = reps['fine']
    _, raw = reps['raw']
    target_eps, source_eps = target(slot), src.SOURCE_EPS[slot]
    rng = seeds(slot)
    pub, reference = src.shift_data.fresh_observations(
        slot, target_eps, REFERENCE_N, rng['reference'], 'evaluation')
    fine_cov = bias.all_cov(fine(pub['x']))
    raw_cov = bias.all_cov(raw(pub['x']))
    perm = base.historical_match(slot, 0)['perm']
    native_rows = bias.unit_rows(np.eye(4)[perm], fine_cov)
    group0 = np.arange(len(a)) % 4 == 0
    drift = {name: src.eq.fit_drift(train[group0], a[group0], u[group0])
             for name, (train, _) in reps.items()}
    result = []
    for i, (cs, es) in enumerate(rng['panels']):
        source_cal, sr = src.shift_data.fresh_observations(slot, source_eps, len(a), cs, 'calibration')
        target_cal, tr = src.shift_data.fresh_observations(slot, target_eps, len(a), cs, 'calibration')
        assert sr['physical_rng_stream_hashes'] == tr['physical_rng_stream_hashes']
        obs, er = src.shift_data.fresh_observations(slot, target_eps, src.previous.EVAL, es, 'evaluation')
        mf, mr = base.score_matrix(fine(obs['x'])), base.score_matrix(raw(obs['x']))
        estimates = {'matched': src.candidate_rows(target_cal, fine, drift['fine'], fine_cov, ~group0),
                     'mismatched': src.candidate_rows(source_cal, fine, drift['fine'], fine_cov, ~group0),
                     'matched_raw': src.candidate_rows(target_cal, raw, drift['raw'], raw_cov, ~group0)}
        y = fine(target_cal['x'])
        updated_drift = src.eq.fit_drift(y[group0], target_cal['actions'][group0], target_cal['regimes'][group0])
        estimates['matched_refit'] = src.eq.rows(y, target_cal['actions'], target_cal['regimes'],
                                                 updated_drift, fine_cov, ~group0)
        scores, missing = {'native': bias.project(mf, native_rows)}, {}
        for k, row in estimates.items():
            src.projected(scores, mr if k == 'matched_raw' else mf, k, row, missing)
        points = {m: base.point_sets(v) for m, v in scores.items()}
        native = points['native']
        safe = np.where((native >= 0).all(1)[:, None], native, [0, 1])
        flags = {k: src.previous.alarm_mask(base.margin(v, safe)) for k, v in scores.items()}
        result.append(dict(index=i, calibration=[sr, tr], evaluation=er,
                           point=points, alerts=flags, missing=missing))
    return dict(reference=reference, target_eps=target_eps, panels=result)


def measure(data, scale):
    z = src.shift_data.replay_physical(data['evaluation']).astype(np.float64) / scale
    scores = np.diagonal(base.score_matrix(z), axis1=1, axis2=2)
    truth = base.point_sets(scores)
    strict = (truth >= 0).all(1)
    order = np.sort(scores, axis=1)
    near = strict & ((order[:, 2] - order[:, 1]) <= .05)
    ok = {m: (np.asarray(p) == truth).all(1) & strict for m, p in data['point'].items()}
    eligible = (np.asarray(data['point']['native']) >= 0).all(1) & strict
    good, bad = eligible & ok['native'], eligible & ~ok['native']
    flags = {m: np.asarray(v) for m, v in data['alerts'].items()}
    assert all(int(v.sum()) == src.previous.ALARMS for v in flags.values())
    return dict(point={m: dict(all=rep.count(v, strict), near=rep.count(v, near)) for m, v in ok.items()},
                alerts={m: dict(hit=rep.count(v, bad), fpr=rep.count(v, good)) for m, v in flags.items()})


def score_slot(slot, data):
    reference = src.shift_data.replay_physical(data['reference']).astype(np.float64)
    scale = reference.std((0, 1), ddof=0)
    assert (scale > 0).all()
    rows = [measure(p, scale) for p in data['panels']]
    rng = np.random.default_rng(2026120901 + SLOTS.index(slot))
    draws = rng.integers(src.previous.PANELS, size=(rep.RESAMPLES, src.previous.PANELS))
    weights = np.stack([np.bincount(row, minlength=src.previous.PANELS) for row in draws])
    point = {m: {k: rep.bootstrap([p['point'][m][k] for p in rows], weights)
                 for k in ('all', 'near')} for m in METHODS}
    alarms = {m: {k: rep.bootstrap([p['alerts'][m][k] for p in rows], weights)
                  for k in ('hit', 'fpr')} for m in METHODS}
    risk = {m: rep.upper(d['fpr'][1], .95) for m, d in alarms.items()}
    comparisons = {}
    for m, r in (('matched', 'mismatched'), ('matched', 'native'),
                 ('matched_refit', 'matched'), ('matched', 'matched_raw')):
        near = rep.paired_interval(point[m]['near'], point[r]['near'])
        hit = rep.paired_interval(alarms[m]['hit'], alarms[r]['hit'])
        near['lower9875'] = rep.upper(point[m]['near'][1] - point[r]['near'][1], .0125)
        hit['lower9875'] = rep.upper(alarms[m]['hit'][1] - alarms[r]['hit'][1], .0125)
        near['passed'] = near['gain'] is not None and near['gain'] >= .03 and near['lower9875'] > 0
        hit['passed'] = (hit['gain'] is not None and hit['gain'] >= .10 and hit['lower9875'] > 0
                         and risk[m] is not None and risk[r] is not None and max(risk[m], risk[r]) < .05)
        comparisons[f'{m}_vs_{r}'] = dict(near=near, hit=hit,
                                           all=rep.paired_interval(point[m]['all'], point[r]['all']))
    return dict(reference_scale=scale, comparisons=comparisons,
                point={m: {k: v[0] for k, v in d.items()} for m, d in point.items()},
                alarms={m: dict(detection=d['hit'][0], fpr=d['fpr'][0], upper95=risk[m])
                        for m, d in alarms.items()},
                unavailable={m: sum(p['missing'][m] for p in data['panels']) for m in METHODS[1:]},
                panels=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds={s: seeds(s) for s in SLOTS},
                                    target={s: target(s) for s in SLOTS}, reference_n=REFERENCE_N))
        return
    sha = base.sha(lock)
    assert base.read(lock)['sources'] == sources()
    if phase == 'forecast':
        slots = {slot: forecast_slot(slot) for slot in SLOTS}
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=sha, slots=slots))
        return
    predictions = base.read(forecast)
    assert predictions['lock_sha'] == sha
    summary = {s: score_slot(s, predictions['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=sha, forecast_sha=base.sha(forecast),
                                   slots=summary))


if __name__ == '__main__':
    main()
