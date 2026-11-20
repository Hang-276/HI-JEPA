"""Split passive moment drift dependence into observable squared and cross terms."""
from pathlib import Path
import numpy as np

from run import HERE, SLOTS, base, bias, rep, src, target, readout, measure

METHODS = ('native', 'source', 'target', 'remove_energy', 'remove_cross', 'replace_D', 'replace_B')
PAIRS = (('target', 'source'), ('remove_energy', 'source'), ('remove_cross', 'source'),
         ('replace_D', 'source'), ('replace_B', 'source'))


def reload(slot, receipt, role):
    public, checked = src.shift_data.fresh_observations(
        slot, receipt['epsilon'], receipt['count'], receipt['seed'], role)
    assert checked == receipt
    return public


def moments(a, b, u):
    s, n = base.old.episode_stats(a[..., None] * b[..., None, :], u)
    averages = s.sum(0) / n.sum(0)[:, None, None]
    return averages[0], averages[1:] - averages[0]


def panel_reconstruction(slot, panel, encode, cov, native_rows):
    s, t = (reload(slot, p, 'calibration') for p in panel['calibration'])
    fit = np.arange(len(t['x'])) % 4 == 0
    y_s, y_t = encode(s['x']), encode(t['x'])
    coefs = [src.eq.fit_drift(y[fit], obs['actions'][fit], obs['regimes'][fit])
             for y, obs in ((y_s, s), (y_t, t))]
    assert all(w is not None for w in coefs)
    y, a, u = y_t[~fit], t['actions'][~fit], t['regimes'][~fit]
    design = src.eq._design(y[:, :-1], a)
    mu_s, mu_t = (design @ w for w in coefs)
    r_t = np.diff(y, axis=1) - mu_t
    e = mu_t - mu_s
    r_s = r_t + e
    bs, ds = moments(r_s, r_s, u)
    bt, dt = moments(r_t, r_t, u)
    be, de = moments(e, e, u)
    bc, dc = moments(e, r_t, u)
    bc, dc = bc + bc.T, dc + dc.swapaxes(1, 2)
    discrepancy = max(float(abs(ds - dt - de - dc).max()), float(abs(bs - bt - be - bc).max()))
    assert discrepancy < 1e-10
    directions = dict(source=readout(ds, bs, cov), target=readout(dt, bt, cov),
                      remove_energy=readout(ds-de, bs, cov), remove_cross=readout(ds-dc, bs, cov),
                      replace_D=readout(dt, bs, cov), replace_B=readout(ds, bt, cov))
    evaluation = reload(slot, panel['evaluation'], 'evaluation')
    mm = base.score_matrix(encode(evaluation['x']))
    scores = {'native': bias.project(mm, native_rows)}
    missing = {}
    for name, rows in directions.items():
        src.projected(scores, mm, name, rows, missing)
    np.testing.assert_allclose(scores['source'], panel['scores']['passive_s'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(scores['target'], panel['scores']['passive_t'], atol=1e-10, rtol=0)
    points = {k: base.point_sets(v) for k, v in scores.items()}
    own = points['native']
    safe = np.where((own >= 0).all(1)[:, None], own, [0, 1])
    flags = {k: src.previous.alarm_mask(base.margin(s, safe)) for k, s in scores.items()}
    return dict(evaluation=panel['evaluation'], point=points, alerts=flags, scores=scores,
                missing=missing, identity=discrepancy,
                relative_energy=np.linalg.norm(de)/np.linalg.norm(ds-dt),
                relative_cross=np.linalg.norm(dc)/np.linalg.norm(ds-dt))


def audit_slot(slot, data):
    _, reps, _, _ = src.reference.historical(slot)
    _, encode = reps['fine']
    public = reload(slot, data['reference'], 'evaluation')
    cov = bias.all_cov(encode(public['x']))
    native = bias.unit_rows(np.eye(4)[base.historical_match(slot, 0)['perm']], cov)
    reconstructed = [panel_reconstruction(slot, p, encode, cov, native) for p in data['panels']]
    z = src.shift_data.replay_physical(data['reference']).astype(np.float64)
    scale = z.std((0, 1))
    measured = [measure(p, scale) for p in reconstructed]
    n = len(measured)
    rng = np.random.default_rng(2026143901 + SLOTS.index(slot))
    draw = rng.integers(n, size=(rep.RESAMPLES, n))
    weights = np.stack([np.bincount(row, minlength=n) for row in draw])
    points = {m: {k: rep.bootstrap([d['point'][m][k] for d in measured], weights)
                  for k in ('all', 'near')} for m in METHODS}
    alarms = {m: {k: rep.bootstrap([d['alerts'][m][k] for d in measured], weights)
                  for k in ('hit', 'fpr')} for m in METHODS}
    return dict(
        comparisons={f'{a}_vs_{b}': dict(near=rep.paired_interval(points[a]['near'], points[b]['near']),
                        hit=rep.paired_interval(alarms[a]['hit'], alarms[b]['hit'])) for a, b in PAIRS},
        point={m: {k: v[0] for k, v in d.items()} for m, d in points.items()},
        alarm={m: {k: v[0] for k, v in d.items()} for m, d in alarms.items()},
        gap_mae={m: float(np.mean([d['gap_mae'][m] for d in measured])) for m in METHODS},
        identity_max=max(p['identity'] for p in reconstructed),
        energy_to_change=float(np.median([p['relative_energy'] for p in reconstructed])),
        cross_to_change=float(np.median([p['relative_cross'] for p in reconstructed])),
        unavailable={m: sum(p['missing'][m] for p in reconstructed) for m in METHODS[1:]})


def main():
    forecast, summary = HERE / 'FORECAST.json', HERE / 'SUMMARY.json'
    assert base.read(summary)['forecast_sha'] == base.sha(forecast)
    data = base.read(forecast)['slots']
    result = {s: audit_slot(s, data[s]) for s in SLOTS}
    rep.write('PASSIVE_TERM_AUDIT.json', dict(utc=base.now(), forecast_sha=base.sha(forecast),
              summary_sha=base.sha(summary), script_sha=base.sha(Path(__file__)), slots=result,
              scope='post-result diagnostic on exactly the same public and scored trajectories'))


if __name__ == '__main__':
    main()
