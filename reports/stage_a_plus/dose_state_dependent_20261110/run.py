"""Frozen CPU comparison of state-dependent dose-response bounds."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from statistics import NormalDist

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = '../dose_window_20261109/run.py'
SPEC = importlib.util.spec_from_file_location('dose_window', HERE / PARENT)
window = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(window)

PILOT, FORMAL, BOOT = 2026286101, 2026286102, 2026286103
GROUPS, SIZE, LENGTH = 24, 128, 193
GRID = np.linspace(-4, 4, 101)
STEP = GRID[1] - GRID[0]
MLO, MHI, SIGMA, ALPHA, EPS = .70, 1.30, .02, .01, 1e-11
DOSES = np.array([.16, .32])
REPEATS = (20, 80, 0)
WORLDS = ('identity', 'tanh', 'sin')
RULES = ('prior', 'numerator', 'variance', 'joint')
SOURCES = ('run.py', 'PROTOCOL.md', 'METHOD.md', PARENT)
Z = NormalDist().inv_cdf(1 - ALPHA / (2 * GRID.size * DOSES.size))


def digest(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def array_hash(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


def save(name, record):
    with (HERE / name).open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read(name):
    return json.loads((HERE / name).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def pdf(x):
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)


def cdf(x):
    x = np.asarray(x)
    values = (.5 * math.erfc(-value / math.sqrt(2)) for value in x.flat)
    return np.fromiter(values, dtype=float, count=x.size).reshape(x.shape)


def physical(x, world):
    if world == 'tanh':
        return x + .25 * np.tanh(x)
    if world == 'sin':
        return x + .25 * np.sin(x)
    return x


def means(world):
    target = physical(GRID, world) + DOSES[:, None]
    low = GRID + DOSES[:, None] / MHI
    high = GRID + DOSES[:, None] / MLO
    for _ in range(52):
        mid = (low + high) / 2
        less = physical(mid, world) < target
        low = np.where(less, mid, low)
        high = np.where(less, high, mid)
    return (low + high) / 2


def calibration_noise(root, group):
    rng = np.random.default_rng(np.random.SeedSequence([root, 0, group]))
    return SIGMA * rng.standard_normal((DOSES.size, GRID.size, max(REPEATS)))


def confidence(reads, exact, repeat):
    if repeat == 0:
        return exact, exact
    mid = reads[:, :, :repeat].mean(axis=2)
    radius = Z * SIGMA / math.sqrt(repeat)
    return mid - radius, mid + radius


def graph(a, c):
    dx = GRID[None, :] - GRID[:, None]
    upper = np.maximum(MLO * dx, MHI * dx)
    for dose, start, end in zip(DOSES, a, c, strict=True):
        from_end = GRID[None, :] - end[:, None]
        from_start = GRID[None, :] - start[:, None]
        lower = dose + np.minimum(MLO * from_end, MHI * from_end)
        high = dose + np.maximum(MLO * from_start, MHI * from_start)
        upper = np.minimum(upper, np.minimum(high, -lower.T))
    upper += EPS
    np.fill_diagonal(upper, np.minimum(upper.diagonal(), 0.))
    for node in range(len(GRID)):
        upper = np.minimum(upper, upper[:, node, None] + upper[None, node, :])
    if np.min(upper.diagonal()) < -EPS:
        return None
    return -upper.T - EPS, upper + EPS


def ref_weights():
    cumulative = cdf(GRID)
    mass = np.diff(cumulative)
    moment = pdf(GRID[:-1]) - pdf(GRID[1:])
    right = (moment - GRID[:-1] * mass) / np.diff(GRID)
    probability = np.r_[mass - right, 1 - cumulative[-1]] + np.r_[cumulative[0], right]
    np.testing.assert_allclose(probability.sum(), 1.)
    radius = GRID[-1]
    tail2 = 2 * ((1 + radius**2) * float(cdf(-radius)) - radius * float(pdf(radius)))
    chord = (MHI - MLO) * STEP / 4
    error = math.sqrt(chord**2 * mass.sum() + MHI**2 * tail2)
    return probability, mass, error


REF_WEIGHT, REF_MASS, REF_ERROR = ref_weights()


def reference(low, high):
    # The ordered lower/upper absolute differences, used on both triangles.
    positive = np.triu(np.maximum(low, 0.), 1)
    negative = np.triu(np.maximum(high, 0.), 1)
    vlo = REF_WEIGHT @ (positive**2) @ REF_WEIGHT
    vhi = REF_WEIGHT @ (negative**2) @ REF_WEIGHT
    vlo -= REF_MASS @ (np.diag(negative, 1)**2) / 4
    sdlo = max(MLO, math.sqrt(max(vlo, 0.)) - REF_ERROR)
    sdhi = min(MHI, math.sqrt(vhi) + REF_ERROR)
    return sdlo**2 - EPS, sdhi**2 + EPS


def interpolation(x):
    clamped = np.clip(x, GRID[0], GRID[-1])
    left = np.clip(np.searchsorted(GRID, clamped, side='right') - 1, 0, GRID.size - 2)
    local_step = GRID[left + 1] - GRID[left]
    theta = (clamped - GRID[left]) / local_step
    error = (MHI - MLO) * theta * (1 - theta) * local_step + MHI * np.abs(x - clamped)
    return left, theta, error


def numerator(low, high, endpoints):
    result_low, result_high = 0., 0.
    for distance, left_x, tx, ex, left_y, ty, ey in endpoints:
        error = ex + ey
        b0 = (1 - tx) * (1 - ty)
        b1 = (1 - tx) * ty
        b2 = tx * (1 - ty)
        b3 = tx * ty
        lo = (b0 * low[left_x, left_y] + b1 * low[left_x, left_y + 1]
              + b2 * low[left_x + 1, left_y] + b3 * low[left_x + 1, left_y + 1])
        hi = (b0 * high[left_x, left_y] + b1 * high[left_x, left_y + 1]
              + b2 * high[left_x + 1, left_y] + b3 * high[left_x + 1, left_y + 1])
        lo = np.maximum(MLO * distance, lo - error - EPS)
        hi = np.minimum(MHI * distance, hi + error + EPS)
        result_low += (lo**2).mean(axis=1)
        result_high += (hi**2).mean(axis=1)
    return result_low / len(endpoints) - EPS, result_high / len(endpoints) + EPS


def slope_prior(x, d, cutoff):
    weights = window.cross_weights(x)
    order = np.argsort(x, axis=1)
    ordered = np.take_along_axis(x, order, axis=1)
    b = np.take_along_axis(weights, order, axis=1)
    tail = np.cumsum(b[:, :0:-1], axis=1)[:, ::-1]
    a, c = ordered[:, :-1], ordered[:, 1:]
    ratio = tail / (cutoff[:, None] * pdf(0.))
    root = np.sqrt(-2 * np.log(np.clip(ratio, np.finfo(float).tiny, 1.)))
    left_root = np.clip(-root, a, c)
    right_root = np.clip(root, a, c)
    bcut = cutoff[:, None]
    # An antiderivative on each constant-tail interval, split at its two roots.
    fa = tail * a - bcut * cdf(a)
    fl = tail * left_root - bcut * cdf(left_root)
    fr = tail * right_root - bcut * cdf(right_root)
    fc = tail * c - bcut * cdf(c)
    integral = (np.abs(fl - fa) + np.abs(fr - fl) + np.abs(fc - fr)).sum(axis=1)
    integral += cutoff * (cdf(ordered[:, 0]) + cdf(-ordered[:, -1]))
    slope_error = max(abs(1 - MLO), abs(MHI - 1))
    linear = 2 * slope_error * integral
    sign = d - cutoff
    signed = sign - linear - cutoff * slope_error**2, sign + linear + d * slope_error**2
    signed_answer = classify(signed, 0.)
    crude = classify((d * (MLO / MHI)**2, d * (MHI / MLO)**2), cutoff)
    return combine(signed_answer, crude), np.stack(signed)


def classify(bounds, cutoff):
    return np.where(bounds[1] < cutoff - EPS, 1, np.where(bounds[0] > cutoff + EPS, -1, 0))


def combine(prior, new):
    assert not np.any((prior != 0) & (new != 0) & (prior != new))
    return np.where(prior != 0, prior, new)


def geometry(x):
    score = window.scores(x)
    d = score[:, 0]
    cutoff = np.sort(score[:, 1:], axis=1)[:, 1]
    values = x[:, :, 0]
    prior, signed = slope_prior(values, d, cutoff)
    endpoints = []
    for lag in window.LAGS:
        left = np.minimum(values[:, lag:], values[:, :-lag])
        right = np.maximum(values[:, lag:], values[:, :-lag])
        endpoints.append((right - left, *interpolation(left), *interpolation(right)))
    return values, d, cutoff, prior, signed, endpoints


def predict(d, cutoff, prior, endpoints, lower, upper):
    node_bounds = graph(lower, upper)
    if node_bounds is None:
        score_bound = np.stack([d * (MLO / MHI)**2, d * (MHI / MLO)**2])
        return np.repeat(prior[None, :], len(RULES), axis=0), score_bound, (MLO**2, MHI**2), False
    vlo, vhi = reference(*node_bounds)
    nlo, nhi = numerator(*node_bounds, endpoints)
    assert vlo <= vhi and np.all(nlo <= nhi)
    bands = (nlo / MHI**2, nhi / MLO**2), (d * MLO**2 / vhi, d * MHI**2 / vlo), (nlo / vhi, nhi / vlo)
    decisions = [prior, *(combine(prior, classify(band, cutoff)) for band in bands)]
    decisions[-1] = combine(combine(decisions[1], decisions[2]), decisions[-1])
    return np.stack(decisions), np.stack(bands[-1]), (vlo, vhi), True


def truth(values, world, order=96):
    z, weights = np.polynomial.hermite.hermgauss(order)
    g = physical(math.sqrt(2) * z, world)
    mean = weights @ g / math.sqrt(np.pi)
    variance = weights @ (g - mean)**2 / math.sqrt(np.pi)
    return window.scores(physical(values, world)[..., None])[:, 0] / variance, variance


def pilot():
    x = window.trajectory(PILOT, LENGTH, 0, 16)
    values, d, cutoff, prior, signed, endpoints = geometry(x)
    noise = calibration_noise(PILOT, 0)
    for world in WORLDS:
        q = means(world)
        np.testing.assert_allclose(physical(q, world) - physical(GRID, world),
                                   np.broadcast_to(DOSES[:, None], q.shape), atol=1e-13)
        value, variance = truth(values, world)
        np.testing.assert_allclose(variance, truth(values, world, order=160)[1],
                                   rtol=0, atol=2e-9)
        g = (value - cutoff) * variance
        assert np.all((signed[0] - EPS <= g) & (g <= signed[1] + EPS))
        for repeat in REPEATS:
            lower, upper = confidence(q[:, :, None] + noise, q, repeat)
            bounds = graph(lower, upper)
            assert bounds is not None
            hgrid = physical(GRID, world)
            difference = hgrid[None, :] - hgrid[:, None]
            assert np.all((bounds[0] - EPS <= difference) & (difference <= bounds[1] + EPS))
            answer, score, ref, feasible = predict(d, cutoff, prior, endpoints, lower, upper)
            assert feasible and ref[0] <= variance <= ref[1]
            assert np.all((score[0] <= value) & (value <= score[1]))
            assert np.all((answer == 0) | (answer == np.where(value < cutoff, 1, -1)))
    return dict(windows=16, exact_response=True, node_parity=True, physical_bounds=True,
                reference_quadrature_parity=True)


def forecast():
    public, output = {}, {}
    exact_means = {world: means(world) for world in WORLDS}
    for group in range(GROUPS):
        x = window.trajectory(FORMAL, LENGTH, group, SIZE)
        _, d, cutoff, prior, signed, endpoints = geometry(x)
        noise = calibration_noise(FORMAL, group)
        output[f'g{group}_signed'] = signed
        for world, q in exact_means.items():
            key = f'g{group}_{world}'
            reads = q[:, :, None] + noise
            public[key] = dict(x=array_hash(x), calibration=array_hash(reads))
            results = []
            for repeat in REPEATS:
                a, c = confidence(reads, q, repeat)
                answer, score, ref, feasible = predict(d, cutoff, prior, endpoints, a, c)
                results.append((answer, score, ref, feasible, np.stack([a, c])))
            for name, values in zip(('answer', 'score', 'ref', 'feasible', 'ci'), zip(*results), strict=True):
                output[key + '_' + name] = np.stack(values)
    with (HERE / 'FORECAST.npz').open('xb') as stream:
        np.savez_compressed(stream, **output)
    return public


def score(public):
    exact = {world: means(world) for world in WORLDS}
    shape = len(WORLDS), GROUPS, len(REPEATS)
    counts = np.zeros((*shape, len(RULES), 2), dtype=int)
    audit = np.zeros((*shape, 4), dtype=int)
    with np.load(HERE / 'FORECAST.npz', allow_pickle=False) as packet:
        for group in range(GROUPS):
            x = window.trajectory(FORMAL, LENGTH, group, SIZE)
            cutoff = np.sort(window.scores(x)[:, 1:], axis=1)[:, 1]
            noise = calibration_noise(FORMAL, group)
            signed = packet[f'g{group}_signed']
            for j, (world, q) in enumerate(exact.items()):
                key = f'g{group}_{world}'
                assert public[key] == dict(x=array_hash(x), calibration=array_hash(q[:, :, None] + noise))
                value, variance = truth(x[:, :, 0], world)
                assert np.all(value != cutoff)
                answer = np.where(value < cutoff, 1, -1)
                g = (value - cutoff) * variance
                assert np.all((signed[0] - EPS <= g) & (g <= signed[1] + EPS))
                a = packet[key + '_answer']
                score = packet[key + '_score']
                ref = packet[key + '_ref']
                ci = packet[key + '_ci']
                counts[j, group, ..., 0] = (a != 0).sum(axis=-1)
                counts[j, group, ..., 1] = ((a != 0) & (a != answer)).sum(axis=-1)
                audit[j, group, :, 0] = ((ci[:, 0] > q) | (ci[:, 1] < q)).any(axis=(1, 2))
                audit[j, group, :, 1] = ~packet[key + '_feasible']
                audit[j, group, :, 2] = ((score[:, 0] > value) | (score[:, 1] < value)).sum(axis=-1)
                audit[j, group, :, 3] = (ref[:, 0] > variance) | (ref[:, 1] < variance)
    sound = not (np.any(counts[..., 1] * (audit[..., 0] == 0)[..., None])
                 or np.any(audit[..., 1:] * (audit[..., 0] == 0)[..., None]))
    sample = np.random.default_rng(BOOT).integers(GROUPS, size=(4096, GROUPS))
    nwin = GROUPS * SIZE
    runs = {}
    for j, world in enumerate(WORLDS):
        scenarios = {}
        for k, repeat in enumerate(REPEATS):
            data = counts[j, :, k]
            total = data.sum(axis=0)
            draws = data[sample].sum(axis=1)
            gains = (draws[:, -1, 0] - draws[:, 0, 0]) / nwin
            exact_loss = (counts[j, :, -1][sample].sum(axis=1)[:, -1, 0] - draws[:, -1, 0]) / nwin
            stats = audit[j, :, k].sum(axis=0).tolist()
            scenarios['exact_grid' if repeat == 0 else str(repeat * DOSES.size * GRID.size)] = dict(
                diagnostics=dict(zip(('ci_miss_panels', 'infeasible_panels', 'outside_scores',
                                      'outside_reference_panels'), stats, strict=True)),
                gain=float((total[-1, 0] - total[0, 0]) / nwin),
                gain_ci=np.quantile(gains, [.025, .975]).tolist(),
                exact_minus_cell_ci=np.quantile(exact_loss, [.025, .975]).tolist(),
                rules={name: dict(accepted=int(row[0]), wrong=int(row[1]),
                                  unconditional_wrong=float(row[1] / nwin),
                                  conditional_wrong=float(row[1] / row[0]) if row[0] else None)
                       for name, row in zip(RULES, total, strict=True)})
        residual = exact[world] - GRID
        runs[world] = dict(response_span=np.ptp(residual, axis=1).tolist(), scenarios=scenarios)
    gate = all(runs[w]['scenarios']['4040']['gain'] >= .05 for w in ('tanh', 'sin'))
    return dict(windows=nwin, groups=GROUPS, runs=runs,
                gates=dict(nonconstant_gain=gate, confidence_event_sound=bool(sound)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'pilot', 'forecast', 'score'))
    phase = parser.parse_args().phase
    sources = {name: digest(name) for name in SOURCES}
    if phase == 'lock':
        save('LOCK.json', dict(utc=utc(), sources=sources, pilot=PILOT, formal=FORMAL, bootstrap=BOOT))
        return
    assert read('LOCK.json')['sources'] == sources
    lock = digest('LOCK.json')
    if phase == 'pilot':
        save('PILOT.json', dict(utc=utc(), lock=lock, **pilot()))
        return
    assert read('PILOT.json')['lock'] == lock
    if phase == 'forecast':
        public = forecast()
        save('FORECAST.json', dict(utc=utc(), lock=lock, pilot=digest('PILOT.json'),
                                  public=public, predictions=digest('FORECAST.npz')))
        return
    receipt = read('FORECAST.json')
    assert receipt['lock'] == lock and receipt['pilot'] == digest('PILOT.json')
    assert receipt['predictions'] == digest('FORECAST.npz')
    save('SUMMARY.json', dict(utc=utc(), forecast=digest('FORECAST.json'), **score(receipt['public'])))


if __name__ == '__main__':
    main()
