"""Single-threaded, sealed CPU validation of the Gaussian selection boundary."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from score import summarize


HERE = Path(__file__).resolve().parent
SOURCE = ('CONFIG.json', 'PROTOCOL.md', 'run.py', 'score.py',
          '../phase2_boundary_20261107/BOUNDARY.md')
I = np.eye(4)


def read(name):
    return json.loads((HERE / name).read_text())


def digest(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def write(name, data):
    with (HERE / name).open('x') as out:
        json.dump(data, out, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
        out.write('\n')


def rotation(a, b, angle):
    c, s = math.cos(angle), math.sin(angle)
    mat = I.copy()
    mat[np.ix_([a, b], [a, b])] = [[c, s], [-s, c]]
    return mat


def dynamics(cfg, setting):
    phi = math.radians(cfg['phi_deg'])
    w = rotation(1, 2, phi)
    a = cfg['rho'] * (w @ rotation(0, 1, math.radians(setting['psi_deg'])) @ w.T)
    return a, math.pi / 4 - phi, math.sqrt(1 - cfg['rho']**2)


def gaussian_contexts(sigma, tau):
    means = np.zeros((13, 4))
    means[1:9:2] = sigma * tau * I
    means[2:9:2] = -sigma * tau * I
    scales = np.full((13, 4), sigma)
    scales[9 + np.arange(4), np.arange(4)] *= math.sqrt(1 + tau)
    return means, scales


def simulate(rng, n, reps, a, physical, tau, sigma):
    u = rng.integers(0, 13, (n, reps), dtype=np.int8)
    means, scales = gaussian_contexts(sigma, tau)
    noise = (rng.standard_normal((n, reps, 4)) * scales[u] + means[u]) @ physical
    lam = 1 + (2 * tau**2 + tau) / 13
    x = np.empty((n + 1, reps, 4))
    x[0] = math.sqrt(lam) * rng.standard_normal((reps, 4))
    for t in range(n):
        x[t + 1] = x[t] @ a.T + noise[t]
    return x, u


def rng_for(seed, ix, jx, side, start, stream):
    return np.random.default_rng(np.random.SeedSequence(
        [seed, ix, jx, (side + 1)//2, start, stream]))


def fit_drift(x):
    previous = x[:-1].swapaxes(0, 1)
    after = x[1:].swapaxes(0, 1)
    transposed = previous.swapaxes(1, 2)
    return np.linalg.solve(transposed @ previous, transposed @ after).swapaxes(1, 2)


def projections(x, u, a, beta, sigma, alpha):
    if a.ndim == 2:
        residual = x[1:] - x[:-1] @ a.T
    else:
        residual = x[1:] - np.einsum('tbi,bji->tbj', x[:-1], a)
    weight = np.zeros((13, 4))
    weight[3, 1:3] = (-math.sin(beta), math.cos(beta))
    weight[5, 1:3] = (-math.cos(beta), -math.sin(beta))
    weight[[4, 6]] = -weight[[3, 5]]
    active = ((3 <= u) & (u <= 6)).sum(0)
    signed_sum = np.einsum('tbi,tbi->b', residual, weight[u]) / sigma
    stat = signed_sum / np.sqrt(np.maximum(active, 1))
    cut = math.sqrt(2 * math.log(2 / alpha))
    signs = (stat > cut).astype(int) - (stat < -cut).astype(int)
    means = [float(s/c) if c else None for s, c in zip(signed_sum, active)]
    assert np.isfinite(stat).all()
    return active.tolist(), means, signs.tolist()


def scores(x, lags, tau):
    lam = 1 + (2*tau**2 + tau)/13
    return np.mean([np.mean((x[h:] - x[:-h])**2, axis=0) / lam for h in lags], axis=0)


def selection(x, lags, tau):
    slow = scores(x, lags, tau)
    order = np.argsort(slow, axis=-1)
    rank = np.take_along_axis(slow, order, axis=-1)
    mask = np.sum(1 << order[:, :2], axis=-1)
    return np.where(rank[:, 1] < rank[:, 2], mask, 0).tolist()


def state_hash(x, u):
    check = hashlib.sha256(memoryview(x).cast('B'))
    check.update(memoryview(u).cast('B'))
    return check.hexdigest()


def verify_lock():
    receipt = read('LOCK.json')
    assert receipt['source_sha256'] == {n: digest(n) for n in SOURCE}
    return read('CONFIG.json')


def pilot(cfg):
    a, beta, sigma = dynamics(cfg, cfg['dynamics'][0])
    n, reps = 64, 16
    seed = cfg['pilot_seed']
    cases = []
    for side in (-1, 1):
        d = math.radians(cfg['cells'][0]['d_deg'])
        theta = beta + side*d
        transform = rotation(1, 2, theta)
        tau = cfg['cells'][0]['tau']
        x, u = simulate(rng_for(seed, 0, 0, side, 0, 0), n, reps, a, transform, tau, sigma)
        replay, replay_u = simulate(rng_for(seed, 0, 0, side, 0, 0),
                                    n, reps, a, transform, tau, sigma)
        assert x.shape == (n+1, reps, 4) and u.shape == (n, reps)
        assert state_hash(x, u) == state_hash(replay, replay_u)
        np.testing.assert_allclose(x @ transform.T @ transform, x, atol=1e-13, rtol=0)
        z = x @ transform.T
        az = transform @ a @ transform.T
        np.testing.assert_allclose(z[1:] - z[:-1] @ az.T,
                                   (x[1:] - x[:-1] @ a.T) @ transform.T, atol=1e-13, rtol=0)
        means, scales = gaussian_contexts(sigma, tau)
        shifted = means @ transform
        dirs = (np.array([-math.sin(beta), math.cos(beta)]),
                np.array([-math.cos(beta), -math.sin(beta)]))
        np.testing.assert_allclose([shifted[3, 1:3] @ dirs[0]/sigma,
                                    shifted[5, 1:3] @ dirs[1]/sigma],
                                   side*tau*math.sin(d), atol=1e-14, rtol=0)
        assert np.all(scales[0:9] == sigma) and np.count_nonzero(scales[9:] > sigma) == 4
        counts, vals, signs = projections(x, u, a, beta, sigma, cfg['alpha'])
        residual = x[1:, 0] - x[:-1, 0] @ a.T
        explicit = [((-1 if ui % 2 == 0 else 1) * (v[1:3] @ dirs[(ui-3)//2]) / sigma)
                    for ui, v in zip(u[:, 0], residual) if 3 <= ui <= 6]
        assert len(explicit) == counts[0] > 0
        np.testing.assert_allclose(vals[0], np.mean(explicit), atol=1e-14)
        direction = math.sqrt(2*math.log(2/cfg['alpha'])/len(explicit))
        assert signs[0] == (int(vals[0] > direction) - int(vals[0] < -direction))
        ahat = fit_drift(x)
        np.testing.assert_allclose(ahat[0], np.linalg.lstsq(x[:-1, 0], x[1:, 0], rcond=None)[0].T,
                                   atol=1e-12, rtol=0)
        for h in cfg['lags']:
            expected = sum((z[t+h, 0]-z[t, 0])**2 for t in range(n+1-h)) / (n+1-h)
            actual = scores(z, [h], tau)[0] * (1+(2*tau**2+tau)/13)
            np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
        cases.append(dict(side=side, active_min=min(counts), active_max=max(counts),
                          dataset_sha256=state_hash(x, u)))
    write('PILOT.json', dict(passed=True, lock_sha256=digest('LOCK.json'), n=n,
                             reps_per_side=reps, checks=cases))


def predict(cfg, ix):
    gate = read('PILOT.json')
    assert gate['passed'] and gate['lock_sha256'] == digest('LOCK.json')
    if ix:
        prev = read(f'SUMMARY_{ix-1}.json')
        assert prev['passed_by_dynamics'][cfg['dynamics'][ix-1]['name']]
    start_time = time.monotonic()
    setting = cfg['dynamics'][ix]
    a, beta, sigma = dynamics(cfg, setting)
    cells = []
    for jx, cell in enumerate(cfg['cells']):
        n, tau, d = (cell[k] for k in ('n', 'tau', 'd_deg'))
        panels = []
        for side in (-1, 1):
            transform = rotation(1, 2, beta + side*math.radians(d))
            panel = dict(side=side, active=[], oracle=[], ols=[], oracle_mean=[], ols_mean=[],
                         drift_rmse=[], batch_hashes=[])
            for start in range(0, cfg['reps_per_side'], cfg['batch']):
                reps = min(cfg['batch'], cfg['reps_per_side']-start)
                x, u = simulate(rng_for(cfg['seed'], ix, jx, side, start, 0),
                                n, reps, a, transform, tau, sigma)
                train, _ = simulate(rng_for(cfg['seed'], ix, jx, side, start, 1),
                                    cfg['ols_fit_transitions_per_rep'], reps, a, transform, tau, sigma)
                ahat = fit_drift(train)
                count, oracle_mean, oracle = projections(x, u, a, beta, sigma, cfg['alpha'])
                ols_count, ols_mean, ols = projections(x, u, ahat, beta, sigma, cfg['alpha'])
                assert count == ols_count
                panel['batch_hashes'].append(state_hash(x, u))
                panel['active'].extend(count)
                panel['oracle'].extend(oracle)
                panel['ols'].extend(ols)
                panel['oracle_mean'].extend(oracle_mean)
                panel['ols_mean'].extend(ols_mean)
                panel['drift_rmse'].extend(np.sqrt(np.mean((ahat-a)**2, axis=(1, 2))).tolist())
            panels.append(panel)
        cells.append(dict(**cell, dynamics=setting['name'], panels=panels))
    write(f'PREDICT_{ix}.json', dict(lock_sha256=digest('LOCK.json'), alpha=cfg['alpha'],
                                    familywise_alpha=cfg['familywise_alpha'], elapsed_seconds=time.monotonic()-start_time,
                                    planned_cells=len(cfg['cells'])*len(cfg['dynamics']), cells=cells))


def reveal(cfg, ix):
    started = time.monotonic()
    pred = read(f'PREDICT_{ix}.json')
    assert pred['lock_sha256'] == digest('LOCK.json')
    setting = cfg['dynamics'][ix]
    a, beta, sigma = dynamics(cfg, setting)
    dscale = 2*np.mean([cfg['rho']**h*(1-math.cos(h*math.radians(setting['psi_deg'])))
                       for h in cfg['lags']])
    labels = []
    for jx, cell in enumerate(pred['cells']):
        n, tau, d = (cell[k] for k in ('n', 'tau', 'd_deg'))
        widths = [*cfg['reference_windows'], n] if cell['name'] == 'ref' else [n]
        cell_labels = dict(name=cell['name'], population_gap=dscale*math.sin(2*math.radians(d)), panels=[])
        for panel in cell['panels']:
            side = panel['side']
            transform = rotation(1, 2, beta + side*math.radians(d))
            windows = {str(w): dict(physical=[], observed=[]) for w in widths}
            for i, start in enumerate(range(0, cfg['reps_per_side'], cfg['batch'])):
                reps = min(cfg['batch'], cfg['reps_per_side']-start)
                x, u = simulate(rng_for(cfg['seed'], ix, jx, side, start, 0),
                                n, reps, a, transform, tau, sigma)
                assert state_hash(x, u) == panel['batch_hashes'][i]
                z = x @ transform.T
                for w in widths:
                    windows[str(w)]['physical'].extend(selection(z[:w+1], cfg['lags'], tau))
                    windows[str(w)]['observed'].extend(selection(x[:w+1], cfg['lags'], tau))
            panel['windows'] = windows
            cell_labels['panels'].append(dict(side=side, windows=windows))
        labels.append(cell_labels)
    result = summarize(pred)
    write(f'WINDOW_{ix}.json', dict(prediction_sha256=digest(f'PREDICT_{ix}.json'), labels=labels))
    write(f'SUMMARY_{ix}.json', dict(**result, prediction_sha256=digest(f'PREDICT_{ix}.json'),
                                    window_sha256=digest(f'WINDOW_{ix}.json'), seconds=time.monotonic()-started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('lock', 'pilot', 'predict', 'reveal'))
    parser.add_argument('dynamics_index', type=int, default=0, nargs='?')
    args = parser.parse_args()
    if args.command == 'lock':
        write('LOCK.json', dict(source_sha256={n: digest(n) for n in SOURCE},
                                numpy=np.__version__, cpu_threads=1))
        return
    cfg = verify_lock()
    assert 0 <= args.dynamics_index < len(cfg['dynamics'])
    if args.command == 'pilot':
        pilot(cfg)
    else:
        action = predict if args.command == 'predict' else reveal
        action(cfg, args.dynamics_index)


if __name__ == '__main__':
    main()
