"""Post-reveal joint axis and slow-selection audit of sealed factorial predictions."""
from pathlib import Path
import numpy as np

from run import HERE, SLOTS, base, bias, rep, seeds, src, target

METHODS = ('native', 'ss', 'ts', 'tt')


def replay(slot, receipt, n, role):
    public, checked = src.shift_data.fresh_observations(
        slot, receipt['epsilon'], n, receipt['seed'], role)
    assert receipt == checked
    return public


def rank_metrics(q, z, second, third):
    yc = q - q.mean(1, keepdims=True)
    zc = z - z.mean(1, keepdims=True)
    norm = (yc ** 2).mean(1) * (zc ** 2).mean(1)
    assert (norm > 0).all()
    inner = (yc * zc).mean(1)
    r2 = inner ** 2 / norm
    q = q * np.where(inner >= 0, 1., -1.)[:, None, :]
    energy, cross = [], []
    for h in base.old.LAGS:
        dq = q[:, h:] - q[:, :-h]
        error = z[:, h:] - z[:, :-h] - dq
        energy.append((error ** 2).mean(1))
        cross.append((2. * dq * error).mean(1))
    energy, cross = np.mean(energy, axis=0), np.mean(cross, axis=0)
    target_score = np.diagonal(base.score_matrix(z), axis1=1, axis2=2)
    q_score = np.diagonal(base.score_matrix(q), axis1=1, axis2=2)
    reconstruction = abs(target_score - q_score - cross - energy).max()
    assert reconstruction < 1e-12
    i = np.arange(len(z))
    return dict(r2=r2, cross_gap=cross[i, third] - cross[i, second],
                energy_gap=energy[i, third] - energy[i, second],
                reconstruction=np.repeat(reconstruction, len(z)))


def recover_panel(slot, p, encoder, cov, native, scale):
    c = {name: replay(slot, receipt, 96, 'calibration')
         for name, receipt in zip(('s', 't'), p['calibration'])}
    assert p['calibration'][0]['physical_rng_stream_hashes'] == p['calibration'][1]['physical_rng_stream_hashes']
    y = {k: encoder(obs['x']) for k, obs in c.items()}
    fit = np.arange(96) % 4 == 0
    drift = {k: src.eq.fit_drift(y[k][fit], obs['actions'][fit], obs['regimes'][fit])
             for k, obs in c.items()}
    rows = {'native': native}
    for k in METHODS[1:]:
        data = c[k[0]]
        rows[k] = src.eq.rows(y[k[0]], data['actions'], data['regimes'], drift[k[1]], cov, ~fit)
        assert rows[k] is not None
    evaluation = replay(slot, p['evaluation'], src.previous.EVAL, 'evaluation')
    q = encoder(evaluation['x'])
    mm = base.score_matrix(q)
    for m in METHODS:
        np.testing.assert_array_equal(base.point_sets(bias.project(mm, rows[m])), p['point'][m])
    # Physical labels are consulted only after reproducing every sealed point prediction.
    z = src.shift_data.replay_physical(p['evaluation']).astype(np.float64) / scale
    score = np.diagonal(base.score_matrix(z), axis1=1, axis2=2)
    target = base.point_sets(score)
    order = np.argsort(score, axis=1)
    i = np.arange(len(z))
    near = (score[i, order[:, 2]] - score[i, order[:, 1]]) <= .05
    methods = {}
    for m, w in rows.items():
        methods[m] = rank_metrics(q @ w.T, z, order[:, 1], order[:, 2])
        methods[m]['point_ok'] = (np.asarray(p['point'][m]) == target).all(1)
    return near, methods


def summarize_method(results, near):
    r2, point = results['r2'], results['point_ok']
    strict = (r2 >= .95).all(1)
    historical = (r2.mean(1) >= .95) & (r2.min(1) >= .90)
    out = {'axis_worst_quantile': np.quantile(r2.min(1), [.05, .5, .95]),
           'axis_worst_global': float(r2.min()),
           'reconstruction_max': float(results['reconstruction'].max())}
    for name, group in (('all', np.ones(len(point), dtype=bool)), ('near', near)):
        wrong = ~point & strict & group
        out[name] = dict(
            point=rep.count(point, group), strict_axis=rep.count(strict, group),
            historical_axis=rep.count(historical, group),
            strict_joint=rep.count(strict & point, group),
            strict_axis_wrong=rep.count(~point, strict & group),
            high_r2_wrong_boundary_cross_mae=(float(abs(results['cross_gap'][wrong]).mean())
                                               if wrong.any() else None),
            high_r2_wrong_boundary_energy_mae=(float(abs(results['energy_gap'][wrong]).mean())
                                                if wrong.any() else None))
    return out


def summarize_pair(results, near, a, b):
    first, second = results[a], results[b]
    close = abs(first['r2'] - second['r2']).max(1) <= .01
    both_precise = (first['r2'] >= .95).all(1) & (second['r2'] >= .95).all(1)
    return {name: dict(n=int(mask.sum()),
                       win=int((mask & first['point_ok'] & ~second['point_ok']).sum()),
                       loss=int((mask & ~first['point_ok'] & second['point_ok']).sum()))
            for name, mask in (('all', np.ones(len(near), dtype=bool)), ('near', near),
                               ('near_axis_close', near & close),
                               ('near_axis_close_both_precise', near & close & both_precise))}


def audit_slot(slot, data):
    _, views, _, _ = src.reference.historical(slot)
    _, encode = views['fine']
    public = replay(slot, data['reference'], target.REFERENCE_N, 'evaluation')
    cov = bias.all_cov(encode(public['x']))
    scale = src.shift_data.replay_physical(data['reference']).astype(np.float64).std((0, 1))
    native = bias.unit_rows(np.eye(4)[base.historical_match(slot, 0)['perm']], cov)
    panels = [recover_panel(slot, p, encode, cov, native, scale) for p in data['panels']]
    near = np.concatenate([x[0] for x in panels])
    matrices = {m: {key: np.concatenate([value[m][key] for _, value in panels])
                    for key in panels[0][1][m]} for m in METHODS}
    return dict(n=len(near), near=int(near.sum()),
                methods={m: summarize_method(v, near) for m, v in matrices.items()},
                paired={f'{a}_vs_{b}': summarize_pair(matrices, near, a, b)
                        for a, b in (('tt', 'ts'), ('tt', 'ss'), ('tt', 'native'))})


def main():
    forecast, outcome = HERE / 'FORECAST.json', HERE / 'SUMMARY.json'
    assert base.read(outcome)['forecast_sha'] == base.sha(forecast)
    obj = base.read(forecast)
    results = {slot: audit_slot(slot, obj['slots'][slot]) for slot in SLOTS}
    rep.write('AXIS_AUDIT.json', dict(utc=base.now(), forecast_sha=base.sha(forecast),
              score_sha=base.sha(outcome), script_sha=base.sha(Path(__file__)), slots=results,
              status='post-reveal paired forensic audit; no new data or changed forecasts'))


if __name__ == '__main__':
    main()
