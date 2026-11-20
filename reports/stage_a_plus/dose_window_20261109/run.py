"""Frozen CPU verification of the finite-window commensurate-dose bounds."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
LENGTHS = {65: (16, 1024), 193: (32, 2048), 769: (16, 1024)}
ELLS = np.array([.16, .08, .04, .02])
RHO = np.array([.95, .9495, .995, .20])
LAGS, AMPLITUDE = (4, 16, 32), .8
PILOT, FORMAL, BOOT = 2026284001, 2026284002, 2026284003
RULES = ('phase', 'global', 'population')
EPS = 1e-12
SOURCES = ('run.py', 'PROTOCOL.md', 'THEORY.md', '../DOSE_IDENTIFIABILITY_20261109.md')


def digest(path):
    return hashlib.sha256((HERE / path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(name, record):
    with (HERE / name).open('x') as f:
        json.dump(record, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def read(name):
    return json.loads((HERE / name).read_text())


def trajectory(root, length, group, size):
    rng = np.random.default_rng(np.random.SeedSequence([root, length, group]))
    x = rng.standard_normal((size, length, 4))
    sigma = np.sqrt(1 - RHO ** 2)
    for t in range(1, length):
        x[:, t] = RHO * x[:, t - 1] + sigma * x[:, t]
    return x


def scores(x):
    return sum(((x[:, h:] - x[:, :-h]) ** 2).mean(axis=1) for h in LAGS) / len(LAGS)


def cross_weights(x):
    w = np.zeros_like(x)
    for lag in LAGS:
        dx = (x[:, lag:] - x[:, :-lag]) / (len(LAGS) * (x.shape[1] - lag))
        w[:, lag:] += dx
        w[:, :-lag] -= dx
    return w


def cross_extrema(x, w, ell):
    phase = np.remainder(x / ell, 1.)
    order = np.argsort(phase, axis=1, kind='stable')
    p = np.take_along_axis(phase, order, axis=1)
    coefficients = np.take_along_axis(w, order, axis=1)
    tails = np.cumsum(coefficients[:, ::-1], axis=1)[:, ::-1]
    # A threshold cannot split observations with identical phase.
    valid = np.concatenate([np.ones((len(p), 1), dtype=bool), p[:, 1:] > p[:, :-1]], axis=1)
    constant = (w * phase).sum(axis=1)
    lo = np.minimum(0., np.where(valid, tails, np.inf).min(axis=1)) - constant
    hi = np.maximum(0., np.where(valid, tails, -np.inf).max(axis=1)) - constant
    return np.minimum(lo, 0.), np.maximum(hi, 0.)


def envelope(d, lo, hi, ell):
    q = max(np.finfo(float).tiny, math.exp(-2 * np.pi ** 2 / ell ** 2))
    fourier = 4 * q / (1 - q)
    vmin = max((1 - ell / 2) ** 2, 1 - fourier)
    low = d + (2 * ell * lo - d * (fourier + ell ** 2 / 4)) / vmin
    high = d + (2 * ell * hi + d * fourier + ell ** 2) / vmin
    return np.maximum(low - EPS, 0.), high + EPS


def population_bound(ell):
    return ell ** 2 * (RHO[0] ** np.asarray(LAGS)).mean() / (2 * (1 - ell / 2) ** 2)


def masks(x):
    score = scores(x)
    d = score[:, 0]
    cutoff = np.sort(score[:, 1:], axis=1)[:, 1]
    w = cross_weights(x[:, :, 0])
    predictions = []
    for ell in ELLS:
        lo, hi = cross_extrema(x[:, :, 0], w, ell)
        phase = envelope(d, lo, hi, ell)
        global_box = envelope(d, -np.sqrt(d), np.sqrt(d), ell)
        error = population_bound(ell)
        boxes = (phase, global_box, (d - error, d + error))
        flags = np.stack([(high < cutoff) | (low > cutoff) for low, high in boxes], axis=1)
        assert np.all(~flags[:, 1] | flags[:, 0])
        predictions.append((flags, np.stack(phase, axis=1)))
    return score, cutoff, predictions


def warp_variance(ell):
    k = 2 * np.pi / ell
    a = AMPLITUDE / k
    return 1 + 2 * a * k * math.exp(-k ** 2 / 2) + a ** 2 * (-math.expm1(-2 * k ** 2)) / 2


def warp(x, ell):
    return x + AMPLITUDE * ell / (2 * np.pi) * np.sin(2 * np.pi * x / ell)


def population_change(ell):
    k = 2 * np.pi / ell
    r = RHO[0] ** np.asarray(LAGS)
    a = AMPLITUDE / k
    power = -np.expm1(-2 * k ** 2) / 2
    covariance = (np.exp(-(1 - r) * k ** 2) - np.exp(-(1 + r) * k ** 2)) / 2
    return float((2 * a ** 2 * (r * power - covariance) / warp_variance(ell)).mean())


def leading_constant(length):
    laplacian = np.zeros((length, length))
    for lag in LAGS:
        i = np.arange(length - lag)
        j = i + lag
        w = 1 / (len(LAGS) * (length - lag))
        laplacian[i, i] += w
        laplacian[j, j] += w
        laplacian[i, j] -= w
        laplacian[j, i] -= w
    idx = np.arange(length)
    cov = RHO[0] ** abs(idx[:, None] - idx[None, :])
    return float(AMPLITUDE / (np.sqrt(2) * np.pi) * np.sqrt(np.sum((laplacian @ cov) * laplacian)))


def pilot():
    x = trajectory(PILOT, 193, 0, 64)
    d, _, predictions = masks(x)
    w = cross_weights(x[:, :, 0])
    np.testing.assert_allclose((w * x[:, :, 0]).sum(1), d[:, 0], atol=1e-12)
    maximum = 0.
    for ell, (_, band) in zip(ELLS, predictions, strict=True):
        values = x[:, :, 0]
        phase = np.remainder(values / ell, 1.)
        lo, hi = cross_extrema(values, w, ell)
        exhaustive = ((phase[0, :, None] >= np.r_[0, np.sort(phase[0]), 1.]) * w[0, :, None]).sum(0)
        exhaustive -= (w[0] * phase[0]).sum()
        np.testing.assert_allclose([lo[0], hi[0]], [exhaustive.min(), exhaustive.max()], atol=1e-12)
        transformed = warp(values, ell) / np.sqrt(warp_variance(ell))
        actual = scores(transformed[..., None])[:, 0]
        assert np.all((band[:, 0] <= actual) & (actual <= band[:, 1]))
        for shift in (ell, 2 * ell):
            maximum = max(maximum, float(np.max(abs(warp(values + shift, ell) - warp(values, ell) - shift))))
    assert maximum < 1e-12
    return dict(n=64, bound_contains_sine=True, tail_extrema_parity=True, period_max_error=maximum)


def forecast():
    public, flags = {}, {}
    for length, (groups, size) in LENGTHS.items():
        for group in range(groups):
            x = trajectory(FORMAL, length, group, size)
            key = f'{length}_{group}'
            public[key] = hashlib.sha256(x.tobytes()).hexdigest()
            d, cutoff, predictions = masks(x)
            flags[key + '_d'] = np.column_stack([d[:, 0], cutoff])
            flags[key + '_f'] = np.stack([flag for flag, _ in predictions])
            flags[key + '_b'] = np.stack([band for _, band in predictions])
    with (HERE / 'FORECAST.npz').open('xb') as f:
        np.savez_compressed(f, **flags)
    return public


def slope(rms):
    x = np.log(ELLS)
    return np.log(rms) @ (x - x.mean()) / np.sum((x - x.mean()) ** 2)


def score(record):
    runs = {}
    with np.load(HERE / 'FORECAST.npz', allow_pickle=False) as data:
        for length, (groups, size) in LENGTHS.items():
            summaries = []
            for group in range(groups):
                key = f'{length}_{group}'
                x = trajectory(FORMAL, length, group, size)
                assert hashlib.sha256(x.tobytes()).hexdigest() == record['public'][key]
                d, cutoff = data[key + '_d'].T
                flags, bands = data[key + '_f'], data[key + '_b']
                row = []
                for i, ell in enumerate(ELLS):
                    physical = warp(x[:, :, 0], ell) / np.sqrt(warp_variance(ell))
                    value = scores(physical[..., None])[:, 0]
                    delta = value - d
                    different = (value < cutoff) != (d < cutoff)
                    assert np.all((d != cutoff) & (value != cutoff))
                    accepted = flags[i].sum(0)
                    errors = (flags[i] & different[:, None]).sum(0)
                    escaped = ((value < bands[i, :, 0]) | (value > bands[i, :, 1])).sum()
                    row.append([delta @ delta, int(different.sum()), int((delta < 0).sum()),
                                int(escaped), *accepted, *errors])
                summaries.append(row)
            arr = np.asarray(summaries)
            assert arr.shape == (groups, len(ELLS), 10)
            total = arr.sum(0)
            n = groups * size
            rms = np.sqrt(total[:, 0] / n)
            boot = np.random.default_rng(np.random.SeedSequence([BOOT, length]))
            resamples = boot.integers(groups, size=(4096, groups))
            draws = arr[resamples].sum(1)
            slopes = slope(np.sqrt(draws[:, :, 0] / n))
            coverage_gap = (draws[:, 0, 4] - draws[:, 0, 5]) / n
            limit = leading_constant(length)
            cells = []
            for i, ell in enumerate(ELLS):
                cells.append(dict(ell=float(ell), n=n, rms=float(rms[i]), rms_per_ell=float(rms[i] / ell),
                                  predicted_small_period_rms=float(limit * ell), population=population_change(ell),
                                  different=int(total[i, 1]), negative_changes=int(total[i, 2]),
                                  outside_phase_band=int(total[i, 3]),
                                  accepted=dict(zip(RULES, total[i, 4:7].astype(int).tolist(), strict=True)),
                                  wrong=dict(zip(RULES, total[i, 7:10].astype(int).tolist(), strict=True))))
            runs[str(length)] = dict(cells=cells, empirical_slope=float(slope(rms)),
                                     population_slope=float(slope(np.array([c['population'] for c in cells]))),
                                     slope_ci=np.quantile(slopes, [.025, .975]).tolist(),
                                     phase_gain_at_016=float((total[0, 4] - total[0, 5]) / n),
                                     phase_gain_ci=np.quantile(coverage_gap, [.025, .975]).tolist())
    main = runs['193']
    bound_gate = all(not (c['outside_phase_band'] or c['wrong']['phase'] or c['wrong']['global'])
                     for r in runs.values() for c in r['cells'])
    return dict(runs=runs, gates=dict(scale=.80 <= main['empirical_slope'] <= 1.25
                                     and 1.80 <= main['population_slope'] <= 2.15,
                                     coverage=main['phase_gain_at_016'] >= .05,
                                     sound_bounds=bound_gate,
                                     population_misuse=main['cells'][2]['wrong']['population'] >= 8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'pilot', 'forecast', 'score'))
    phase = parser.parse_args().phase
    sources = {name: digest(name) for name in SOURCES}
    if phase == 'lock':
        save('LOCK.json', dict(utc=utc(), sources=sources, pilot=PILOT, formal=FORMAL, bootstrap=BOOT))
        return
    lock = digest('LOCK.json')
    assert read('LOCK.json')['sources'] == sources
    if phase == 'pilot':
        save('PILOT.json', dict(utc=utc(), lock=lock, **pilot()))
        return
    assert read('PILOT.json')['lock'] == lock
    if phase == 'forecast':
        public = forecast()
        save('FORECAST.json', dict(utc=utc(), lock=lock, pilot=digest('PILOT.json'),
                                   predictions=digest('FORECAST.npz'), public=public))
        return
    receipt = read('FORECAST.json')
    assert receipt['lock'] == lock and receipt['pilot'] == digest('PILOT.json')
    assert receipt['predictions'] == digest('FORECAST.npz')
    save('SUMMARY.json', dict(utc=utc(), forecast=digest('FORECAST.json'), **score(receipt)))


if __name__ == '__main__':
    main()
