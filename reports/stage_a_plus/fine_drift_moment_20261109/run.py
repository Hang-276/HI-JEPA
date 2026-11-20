"""Paired-moment and whitening hybrid experiment on shifted FHN windows."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / 'fine_target_reference_20261109'
FACTOR = HERE.parent / 'fine_drift_factorial_20261109'
sys.path.insert(0, str(TARGET))
spec = importlib.util.spec_from_file_location('target_reference', TARGET / 'run.py')
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)
src, base, bias, rep = target.src, target.base, target.bias, target.rep
rep.HERE = HERE

import numpy as np
from paired_shift import paired_observations, source_hashes

SLOTS = target.SLOTS
PANELS = 32
TIMES = (8, 24, 40, 56, 72, 88, 104, 120)
METHODS = ('native', 'passive_s', 'passive_t', 'sparse_s', 'sparse_t',
           'd_ss', 'd_st', 'd_ts', 'd_tt', 'p_s', 'p_t')
COMPARISONS = (('passive_t', 'passive_s'), ('sparse_t', 'sparse_s'),
               ('d_tt', 'd_ss'), ('d_st', 'd_ss'), ('d_tt', 'd_ts'),
               ('d_ts', 'd_ss'), ('d_tt', 'd_st'), ('p_t', 'p_s'))


def source_paths():
    paths = (Path(__file__), HERE / 'PROTOCOL.md', HERE / 'THEORY.md', HERE / 'paired_shift.py',
             TARGET / 'LOCK.json', TARGET / 'SUMMARY.json', FACTOR / 'LOCK.json', FACTOR / 'SUMMARY.json')
    return target.sources() | source_hashes() | {str(p): base.sha(p) for p in paths}


def seeds(slot):
    i = SLOTS.index(slot)
    start = 2026129000 + 1000 * i
    return dict(reference=2026128000 + i,
                panels=[(start + 2 * k, start + 2 * k + 1) for k in range(PANELS)])


def readout(contrast, baseline, cov):
    values, vectors = np.linalg.eigh(baseline)
    if values[0] <= 0:
        return None
    whitener = (vectors / np.sqrt(values)) @ vectors.T
    whitened = whitener @ contrast @ whitener
    whitened = (whitened + whitened.swapaxes(-1, -2)) * .5
    values, vectors = np.linalg.eigh(whitened)
    gap = values[:, -1] - values[:, -2]
    if np.any(values[:, -1] <= 0) or np.any(gap <= 1e-12 * abs(values).max(1)):
        return None
    axes = vectors[:, :, -1].T
    if np.linalg.cond(axes) > 10:
        return None
    return bias.unit_rows(np.linalg.solve(axes, whitener), cov)


def moment(array):
    flat = array.reshape(-1, 4)
    return flat.T @ flat / len(flat)


def sparse_readout(y, a, u, drift, keep, cov):
    y0, y1 = y[keep][:, TIMES], y[keep][:, np.asarray(TIMES) + 1]
    action, regime = a[keep][:, TIMES], u[keep][:, TIMES]
    r = y1 - y0 - src.eq._design(y0, action) @ drift
    if any(not (regime == k).any() for k in base.old.CODES):
        return None
    baseline, *spikes = [moment(r[regime == k]) for k in base.old.CODES]
    return readout(np.stack(spikes) - baseline, baseline, cov)


def paired_moments(encode, siblings, keep, drift):
    current = encode(siblings['current'][keep])
    a = siblings['actions'][keep]
    y0 = encode(siblings['baseline'][:, keep])
    y = encode(siblings['spikes'][:, :, keep])
    phi = src.eq._design(current, a)
    ms, mt = (phi @ drift[k] for k in ('s', 't'))
    r0 = y0 - current[None]
    r = y - current[None, None]
    bs, bt = moment(r0 - ms), moment(r0 - mt)
    ds = np.stack([moment(item - ms) - bs for item in r])
    dt = np.stack([moment(item - mt) - bt for item in r])
    diff = y - y0[None]
    p = np.stack([moment(item) for item in diff])
    drift_delta = np.broadcast_to((ms - mt)[None], y0.shape).reshape(-1, 4)
    reconstructed = []
    for item in diff:
        flat = item.reshape(-1, 4)
        term = drift_delta.T @ flat / len(flat)
        reconstructed.append(term + term.T)
    equality = float(np.max(abs(dt - ds - reconstructed)))
    assert equality < 1e-10
    diagnostic = dict(
        identity_maxabs=equality,
        residual_trace_ratio=float(np.trace(bt) / np.trace(bs)),
        relative_B=float(np.linalg.norm(bt - bs) / np.linalg.norm(bs)),
        relative_D=np.linalg.norm(dt - ds, axis=(1, 2)) / np.linalg.norm(ds, axis=(1, 2)))
    return dict(s=ds, t=dt, p=p), dict(s=bs, t=bt), diagnostic


def forecast_slot(slot):
    _, reps, a, _ = src.reference.historical(slot)
    _, encode = reps['fine']
    old_eps, new_eps = src.SOURCE_EPS[slot], target.target(slot)
    rng = seeds(slot)
    public, ref = src.shift_data.fresh_observations(
        slot, new_eps, target.REFERENCE_N, rng['reference'], 'evaluation')
    cov = bias.all_cov(encode(public['x']))
    perm = base.historical_match(slot, 0)['perm']
    native = bias.unit_rows(np.eye(4)[perm], cov)
    fit = np.arange(len(a)) % 4 == 0
    keep = ~fit
    panels = []
    for i, (cs, es) in enumerate(rng['panels']):
        s, sr = src.shift_data.fresh_observations(slot, old_eps, len(a), cs, 'calibration')
        t, tr = src.shift_data.fresh_observations(slot, new_eps, len(a), cs, 'calibration')
        assert sr['physical_rng_stream_hashes'] == tr['physical_rng_stream_hashes']
        ev, er = src.shift_data.fresh_observations(slot, new_eps, src.previous.EVAL, es, 'evaluation')
        y = {'s': encode(s['x']), 't': encode(t['x'])}
        drift = {k: src.eq.fit_drift(y[k][fit], q['actions'][fit], q['regimes'][fit])
                 for k, q in (('s', s), ('t', t))}
        assert all(coef is not None for coef in drift.values())
        siblings, receipt = paired_observations(tr, cs + 500000, TIMES, antithetic=True)
        ds, bs, diagnostic = paired_moments(encode, siblings, keep, drift)
        rows = {'native': native}
        for k in ('s', 't'):
            rows[f'passive_{k}'] = src.eq.rows(y['t'], t['actions'], t['regimes'], drift[k], cov, keep)
            rows[f'sparse_{k}'] = sparse_readout(y['t'], t['actions'], t['regimes'], drift[k], keep, cov)
            rows[f'p_{k}'] = readout(ds['p'], bs[k], cov)
            for j in ('s', 't'):
                rows[f'd_{j}{k}'] = readout(ds[j], bs[k], cov)
        mm = base.score_matrix(encode(ev['x']))
        scores = {'native': bias.project(mm, native)}
        missing = {}
        for key, row in rows.items():
            src.projected(scores, mm, key, row, missing)
        points = {k: base.point_sets(v) for k, v in scores.items()}
        sets = points['native']
        safe = np.where((sets >= 0).all(1)[:, None], sets, [0, 1])
        flags = {m: src.previous.alarm_mask(base.margin(s, safe)) for m, s in scores.items()}
        panels.append(dict(index=i, calibration=[sr, tr], evaluation=er, point=points, alerts=flags,
                           scores=scores, missing=missing, paired_receipt=receipt, diagnostic=diagnostic))
    return dict(reference=ref, panels=panels)


def measure(panel, scale):
    z = src.shift_data.replay_physical(panel['evaluation']).astype(np.float64) / scale
    target_scores = np.diagonal(base.score_matrix(z), axis1=1, axis2=2)
    truth = base.point_sets(target_scores)
    strict = (truth >= 0).all(1)
    order = np.sort(target_scores, axis=1)
    near = strict & ((order[:, 2] - order[:, 1]) <= .05)
    ok = {m: (np.asarray(p) == truth).all(1) & strict for m, p in panel['point'].items()}
    eligible = strict & (np.asarray(panel['point']['native']) >= 0).all(1)
    good, bad = eligible & ok['native'], eligible & ~ok['native']
    flags = {m: np.asarray(v) for m, v in panel['alerts'].items()}
    assert all(int(v.sum()) == src.previous.ALARMS for v in flags.values())
    return dict(point={m: dict(all=rep.count(v, strict), near=rep.count(v, near)) for m, v in ok.items()},
                alerts={m: dict(hit=rep.count(v, bad), fpr=rep.count(v, good)) for m, v in flags.items()},
                gap_mae={m: base.interval_mae(v, target_scores, strict) for m, v in panel['scores'].items()})


def score_slot(slot, data):
    reference = src.shift_data.replay_physical(data['reference']).astype(np.float64)
    scale = reference.std((0, 1), ddof=0)
    assert (scale > 0).all()
    measured = [measure(p, scale) for p in data['panels']]
    rng = np.random.default_rng(2026137901 + SLOTS.index(slot))
    draw = rng.integers(PANELS, size=(rep.RESAMPLES, PANELS))
    weights = np.stack([np.bincount(row, minlength=PANELS) for row in draw])
    point = {m: {k: rep.bootstrap([d['point'][m][k] for d in measured], weights)
                 for k in ('all', 'near')} for m in METHODS}
    alarms = {m: {k: rep.bootstrap([d['alerts'][m][k] for d in measured], weights)
                  for k in ('hit', 'fpr')} for m in METHODS}
    mae = {m: np.asarray([d['gap_mae'][m] for d in measured]) for m in METHODS}
    comparisons = {}
    for a, b in COMPARISONS:
        delta = (weights @ (mae[a] - mae[b])) / PANELS
        comparisons[f'{a}_vs_{b}'] = dict(
            near=rep.paired_interval(point[a]['near'], point[b]['near']),
            alarm=rep.paired_interval(alarms[a]['hit'], alarms[b]['hit']),
            mae_delta=float((mae[a] - mae[b]).mean()), mae_delta_ci=np.quantile(delta, [.025, .975]))
    diag = [p['diagnostic'] for p in data['panels']]
    return dict(comparisons=comparisons, gap_mae={m: float(v.mean()) for m, v in mae.items()},
                point={m: {k: v[0] for k, v in o.items()} for m, o in point.items()},
                alarm={m: {k: v[0] for k, v in o.items()} for m, o in alarms.items()},
                unavailable={m: sum(p['missing'][m] for p in data['panels']) for m in METHODS},
                identity_maxabs=max(o['identity_maxabs'] for o in diag),
                diagnostic={k: np.median([o[k] for o in diag], axis=0)
                            for k in ('residual_trace_ratio', 'relative_B', 'relative_D')},
                panels=measured)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock, forecast = HERE / 'LOCK.json', HERE / 'FORECAST.json'
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=base.now(), sources=source_paths(), times=TIMES,
                                    panels=PANELS, seeds={s: seeds(s) for s in SLOTS}))
        return
    sha = base.sha(lock)
    assert base.read(lock)['sources'] == source_paths()
    if phase == 'forecast':
        slots = {slot: forecast_slot(slot) for slot in SLOTS}
        rep.write('FORECAST.json', dict(utc=base.now(), lock_sha=sha, slots=slots))
        return
    data = base.read(forecast)
    assert data['lock_sha'] == sha
    summary = {s: score_slot(s, data['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=base.now(), lock_sha=sha, forecast_sha=base.sha(forecast), slots=summary))


if __name__ == '__main__':
    main()
