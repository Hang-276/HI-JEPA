"""Post-result same-budget factorial for context and Gaussian-noise sharing."""
from pathlib import Path
import numpy as np

from run import HERE, SLOTS, TIMES, base, rep, src, moment
from paired_shift import shift, _exact_contexts

REPEATS = 14


def verify_public(slot, receipt):
    data, check = shift.fresh_observations(slot, receipt['epsilon'], receipt['count'],
                                           receipt['seed'], 'calibration')
    assert check == receipt
    return data


def intervention_obs(encode, transform, mix, context, noise, sigma):
    return encode(np.ascontiguousarray(transform(context + sigma * noise, mix), dtype=np.float32))


def arm(encode, context, action, determin, noise, sigma, transform, mix, drift):
    mu = src.eq._design(context, action)
    residuals = []
    for sign in (1, -1):
        obs = intervention_obs(encode, transform, mix, determin, sign * noise, sigma)
        residuals.append(obs - context)
    delta = np.stack(residuals)
    return {k: delta - mu @ w for k, w in drift.items()}


def relative_change(baseline, spike, antithetic):
    support = slice(None) if antithetic else slice(0, 1)
    source = moment(spike['s'][support]) - moment(baseline['s'][support])
    target = moment(spike['t'][support]) - moment(baseline['t'][support])
    return float(np.linalg.norm(source - target) / np.linalg.norm(source))


def score_panel(slot, panel, encode):
    data = {k: verify_public(slot, receipt) for k, receipt in zip(('s', 't'), panel['calibration'])}
    fit = np.arange(96) % 4 == 0
    drift = {}
    for k, p in data.items():
        y = encode(p['x'])
        drift[k] = src.eq.fit_drift(y[fit], p['actions'][fit], p['regimes'][fit])
        assert drift[k] is not None
    receipt = panel['calibration'][1]
    raw, actions, deterministic, sigma0, sigmas, transform, mix, err = _exact_contexts(
        slot, receipt['epsilon'], receipt['count'], receipt['trajectory_seed'],
        shift.FRESH_TAG, np.asarray(TIMES), receipt['public_arrays'], receipt['physical_rng_stream_hashes'])
    rows = np.flatnonzero(~fit)[:5 * REPEATS].reshape(5, REPEATS)
    assert len(np.unique(rows)) == 5 * REPEATS
    states = encode(raw)
    rng = np.random.default_rng(2026139001 + 1000 * SLOTS.index(slot) + panel['index'])
    eps = rng.standard_normal((5, REPEATS, len(TIMES), 4))
    baseline = arm(encode, states[rows[0]], actions[rows[0]], deterministic[rows[0]],
                   eps[0], sigma0, transform, mix, drift)
    values = {}
    for context_same in (False, True):
        for noise_same in (False, True):
            spikes = []
            for j, sigma in enumerate(sigmas, 1):
                i = 0 if context_same else j
                noise = eps[0 if noise_same else j]
                spikes.append(arm(encode, states[rows[i]], actions[rows[i]], deterministic[rows[i]],
                                  noise, sigma, transform, mix, drift))
            for anti in (False, True):
                code = f'context_{int(context_same)}_noise_{int(noise_same)}_anti_{int(anti)}'
                values[code] = [relative_change(baseline, r, anti) for r in spikes]
    return dict(replay_error=err, values=values)


def main():
    forecast = HERE / 'FORECAST.json'
    summary = HERE / 'SUMMARY.json'
    assert base.read(summary)['forecast_sha'] == base.sha(forecast)
    slots = base.read(forecast)['slots']
    results = {}
    for slot in SLOTS:
        _, views, _, _ = src.reference.historical(slot)
        _, encode = views['fine']
        panels = [score_panel(slot, p, encode) for p in slots[slot]['panels']]
        stats = {k: np.quantile([p['values'][k] for p in panels], [.1, .5, .9], axis=0)
                 for k in panels[0]['values']}
        overall = {k: np.quantile(np.asarray([p['values'][k] for p in panels]).max(1), [.1, .5, .9])
                   for k in panels[0]['values']}
        results[slot] = dict(axis_quantiles=stats, worst_axis_quantiles=overall,
                             panels=panels, max_replay_error=max(p['replay_error'] for p in panels))
    rep.write('CONTEXT_NOISE_AUDIT.json', dict(utc=base.now(), forecast_sha=base.sha(forecast),
        summary_sha=base.sha(summary), script_sha=base.sha(Path(__file__)), results=results,
        scope='post-result generator-side moment diagnostic; no evaluation scores',
        episodes_per_arm=REPEATS, contexts_per_episode=len(TIMES)))


if __name__ == '__main__':
    main()
