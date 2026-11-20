"""Measure selective top-2 information at two fixed orientation endpoints."""
import argparse
import math

import run as model
import numpy as np

PILOT, DESIGN, EVAL = 2026259001, 2026259002, 2026259003
N, BUDGET, ERROR = 8192, .035, .05
TAIL = .05 / 96
METHODS = ('task', 'side')
SOURCES = ('endpoint.py', 'ENDPOINT_PROTOCOL.md', 'ENDPOINT_THEORY.md',
           'run.py', 'THEORY.md', 'LOCK.json')


def sides(root, dynamic, cell, n):
    return 2 * model.streams(root, dynamic, cell, 0).integers(0, 2, n) - 1


def calibration_ratio(root, dynamic, cell, side):
    rng = model.streams(root, dynamic, cell, 3)
    tau, d = model.STRENGTH[cell], (model.HI - model.LO) / 2
    km = 2 * tau ** 2 * np.sin(d) ** 2
    kv = tau ** 2 * np.sin(2 * d) ** 2 / (2 * (1 + tau))
    nm, nv, _ = rng.multinomial(model.CAL, [8 / 13, 4 / 13, 1 / 13], size=len(side)).T
    result = nm * km + np.sqrt(2 * nm * km) * rng.standard_normal(len(side))
    take = nv > 0
    difference = math.sqrt(kv ** 2 + 2 * kv)
    result[take] += .5 * ((kv + difference) * rng.chisquare(nv[take])
                          + (kv - difference) * rng.chisquare(nv[take]))
    return side * result


def window_ratio(innovations, u, tau):
    mean, sd = np.zeros((13, 4)), np.ones((13, 4))
    mean[1:9:2], mean[2:9:2] = tau * np.eye(4), -tau * np.eye(4)
    sd[9 + np.arange(4), np.arange(4)] = math.sqrt(1 + tau)
    if tau == 0:
        return np.zeros(innovations.shape[1])
    standardized = []
    for theta in (model.HI, model.LO):
        physical = innovations @ model.rotate(theta, model.PLANES).T
        standardized.append(((physical - mean[u]) / sd[u]) ** 2)
    return -.5 * (standardized[0] - standardized[1]).sum(axis=(0, 2))


def sample(root, dynamic, cell, n):
    true_side = sides(root, dynamic, cell, n)
    theta = np.where(true_side == 1, model.HI, model.LO)
    x, u, _, _ = model.observable_data(root, dynamic, cell, n, theta)
    tau = model.STRENGTH[cell]
    m = model.moment(x, tau)
    plus = model.selection(model.direct_score(m, np.full(n, model.HI)))
    minus = model.selection(model.direct_score(m, np.full(n, model.LO)))
    assert np.all(plus > 0) and np.all(minus > 0)
    window = window_ratio((x[1:] - x[:-1] @ model.drift(dynamic).T) / model.SIGMA, u, tau)
    cal = calibration_ratio(root, dynamic, cell, true_side)
    rand = model.streams(root, dynamic, cell, 4).uniform(size=(2, n))
    return plus, minus, (window, window + cal), rand


def bayes_error(ratio):
    exp = np.exp(-np.abs(ratio))
    return exp / (1 + exp)


def threshold(eta):
    ordered = np.sort(eta)
    spend = BUDGET * len(eta)
    count = np.searchsorted(np.cumsum(ordered), spend, side='right')
    if count == len(eta):
        return dict(value=1., equality_probability=1.)
    cutoff = ordered[count]
    assert cutoff > 0
    below = eta < cutoff
    tied = eta == cutoff
    probability = (spend - eta[below].sum()) / (cutoff * tied.sum())
    return dict(value=float(cutoff), equality_probability=float(np.clip(probability, 0, 1)))


def make_design():
    cutoffs = {}
    for d in range(2):
        for j in range(3):
            plus, minus, ratios, _ = sample(DESIGN, d, j, N)
            for b, ratio in enumerate(ratios):
                error = bayes_error(ratio)
                key = f'{d}_{j}_{b}'
                cutoffs[key] = {'task': threshold(error * (plus != minus)), 'side': threshold(error)}
    return cutoffs


def make_forecast(cutoffs):
    arrays = {}
    for d in range(2):
        for j in range(3):
            plus, minus, ratios, rand = sample(EVAL, d, j, N)
            for b, ratio in enumerate(ratios):
                key = f'{d}_{j}_{b}'
                mistake = bayes_error(ratio)
                side = np.where((ratio > 0) | ((ratio == 0) & (rand[0] < .5)), 1, -1)
                arrays[key] = np.stack([plus, minus, ratio, side], axis=1)
                for rule in METHODS:
                    eta = mistake * (plus != minus) if rule == 'task' else mistake
                    t, p = (cutoffs[key][rule][k] for k in ('value', 'equality_probability'))
                    arrays[f'{key}_{rule}'] = (eta < t) | ((eta == t) & (rand[1] < p))
    with (model.HERE / 'ENDPOINT_FORECAST.npz').open('xb') as file:
        np.savez_compressed(file, **arrays)


def wrong_upper(count):
    empirical = count / N
    if empirical == 1:
        return 1.
    low, high = empirical, 1.
    target = math.log(1 / TAIL) / N
    for _ in range(48):
        mid = (low + high) / 2
        kl = (1 - empirical) * math.log((1 - empirical) / (1 - mid))
        if empirical:
            kl += empirical * math.log(empirical / mid)
        if kl <= target:
            low = mid
        else:
            high = mid
    return high


def measure(cutoffs):
    radius = math.sqrt(math.log(1 / TAIL) / (2 * N))
    rows = []
    with np.load(model.HERE / 'ENDPOINT_FORECAST.npz', allow_pickle=False) as f:
        for d, psi in enumerate(model.PSI):
            for j, tau in enumerate(model.STRENGTH):
                true_side = sides(EVAL, d, j, N)
                cells = []
                for b in range(2):
                    key = f'{d}_{j}_{b}'
                    plus, minus, ratio, guess = f[key].T
                    truth = np.where(true_side == 1, plus, minus)
                    answer = np.where(guess == 1, plus, minus)
                    disagreement = plus != minus
                    eta = bayes_error(ratio) * disagreement
                    t = cutoffs[key]['task']['value']
                    dual = ERROR / t + np.maximum(0., 1. - eta / t).mean() + radius
                    data = dict(calibration=b * model.CAL, disagreement=float(disagreement.mean()),
                                task_optimal_upper=float(min(1, dual)), task_bayes_wrong=float(eta.mean()))
                    if tau == 0:
                        assert np.all(ratio == 0)
                        data['exact_null_coverage_plugin'] = float(min(1, 1 - disagreement.mean() + 2 * ERROR))
                        data['task_optimal_upper'] = float(min(1, data['exact_null_coverage_plugin'] + radius))
                    for m in METHODS:
                        accepted = f[f'{key}_{m}']
                        count = int((accepted & (answer != truth)).sum())
                        data[m] = dict(coverage=float(accepted.mean()), wrong=count,
                                       wrong_upper=wrong_upper(count),
                                       side_wrong=int((accepted & (guess != true_side)).sum()))
                    cells.append(data)
                gain = cells[1]['task']['coverage'] - cells[0]['task']['coverage']
                rows.append(dict(psi=psi, tau=tau, n=N, cells=cells, gain=gain, gain_lower=gain - 2 * radius,
                                 weak_gate=(gain >= .10 and gain > 2 * radius) if tau == .12 else None))
    return dict(rows=rows, risk_gate=all(c['task']['wrong_upper'] < ERROR for r in rows for c in r['cells']))


def pilot():
    d = (model.HI - model.LO) / 2
    max_error = 0.
    mean_error = 0.
    for tau in model.STRENGTH:
        a, b = model.rotate(model.HI, ((0, 1),))[:2, :2], model.rotate(model.LO, ((0, 1),))[:2, :2]
        root = np.diag([math.sqrt(1 + tau), 1])
        plus = a.T @ root
        minus = b.T @ root
        matrix = plus.T @ np.linalg.solve(minus @ minus.T, plus) - np.eye(2)
        kv = tau ** 2 * np.sin(2 * d) ** 2 / (2 * (1 + tau))
        expected = kv + np.array([-1, 1]) * math.sqrt(kv ** 2 + 2 * kv)
        max_error = max(max_error, float(np.max(np.abs(np.linalg.eigvalsh(matrix) - expected))))
        assert max_error < 1e-12
        signed_axes = np.repeat(np.eye(4), 2, axis=0) * np.tile([tau, -tau], 4)[:, None]
        ep = signed_axes @ model.rotate(model.HI, model.PLANES)
        em = signed_axes @ model.rotate(model.LO, model.PLANES)
        km = 2 * tau ** 2 * np.sin(d) ** 2
        ratio = window_ratio(ep[:, None, :], np.arange(1, 9)[:, None], tau)[0]
        mean_error = max(mean_error, abs(float(ratio) / 8 - km),
                         float(np.max(np.abs(((ep - em) ** 2).sum(1) - 2 * km))))
        assert mean_error < 1e-12
        for dyn in range(2):
            cell = model.STRENGTH.index(tau)
            plus, minus, ratios, _ = sample(PILOT, dyn, cell, 16)
            if tau == 0:
                assert all(np.all(r == 0) for r in ratios)
            assert np.all(plus > 0) and np.all(minus > 0)
    return dict(quadratic_spectrum_max_error=max_error, mean_identity_max_error=mean_error,
                null_likelihood_identical=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('lock', 'pilot', 'design', 'forecast', 'score'))
    phase = p.parse_args().phase
    sources = {name: model.digest(name) for name in SOURCES}
    if phase == 'lock':
        model.write('ENDPOINT_LOCK.json', dict(utc=model.utc(), sources=sources, pilot=PILOT, design=DESIGN, eval=EVAL))
        return
    lock = model.digest('ENDPOINT_LOCK.json')
    assert model.read('ENDPOINT_LOCK.json')['sources'] == sources
    if phase == 'pilot':
        model.write('ENDPOINT_PILOT.json', dict(utc=model.utc(), lock=lock, **pilot()))
        return
    assert model.read('ENDPOINT_PILOT.json')['lock'] == lock
    if phase == 'design':
        model.write('ENDPOINT_DESIGN.json', dict(utc=model.utc(), lock=lock, cutoffs=make_design()))
        return
    design = model.read('ENDPOINT_DESIGN.json')
    assert design['lock'] == lock
    design_sha = model.digest('ENDPOINT_DESIGN.json')
    if phase == 'forecast':
        make_forecast(design['cutoffs'])
        model.write('ENDPOINT_RECEIPT.json', dict(utc=model.utc(), lock=lock, design=design_sha,
                                                 forecast=model.digest('ENDPOINT_FORECAST.npz')))
        return
    receipt = model.read('ENDPOINT_RECEIPT.json')
    assert receipt['lock'] == lock and receipt['design'] == design_sha
    assert receipt['forecast'] == model.digest('ENDPOINT_FORECAST.npz')
    model.write('ENDPOINT_SUMMARY.json', dict(utc=model.utc(), receipt=model.digest('ENDPOINT_RECEIPT.json'),
                                             **measure(design['cutoffs'])))


if __name__ == '__main__':
    main()
