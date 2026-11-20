"""CPU verification of commensurate-dose scale separation after a cubic warp."""

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
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
ROOTS = (2026289401, 2026289402, 2026289403)
ELLS = np.array([.32, .16, .08, .04])
RHO = np.array([.95, .9495, .995, .20])
LAGS = (4, 16, 32)
CUBIC, AMPLITUDE = .20, .8
SCALE = math.sqrt(1 + 6 * CUBIC + 15 * CUBIC**2)
LENGTH, GROUPS, SIZE = 193, 32, 1024
COARSE, FINE = (6., 8), (6.5, 16)
SOURCES = ('PROTOCOL.md', 'THEORY.md', 'run.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(name, data):
    with (HERE / name).open('x') as stream:
        json.dump(dict(utc=utc(), **data), stream, indent=2, allow_nan=False)
        stream.write('\n')


def read(name):
    return json.loads((HERE / name).read_text())


def transform(z):
    return (z + CUBIC * z**3) / SCALE


def inverse(x):
    factor = math.sqrt(3 * CUBIC)
    return 2 / factor * np.sinh(np.arcsinh(1.5 * factor * SCALE * x) / 3)


def jacobian(z):
    return SCALE / (1 + 3 * CUBIC * z**2)


def phi(z):
    return np.exp(-z**2 / 2) / math.sqrt(2 * math.pi)


def correlation(r):
    return ((1 + 3 * CUBIC)**2 * r + 6 * CUBIC**2 * r**3) / SCALE**2


def conditional(z, r):
    return (r * (1 + 3 * CUBIC * (1 - r**2)) * z + CUBIC * r**3 * z**3) / SCALE


def marginal_moments(ell):
    """Direct one-dimensional Fourier-weighted integrations in observed x."""
    k = 2 * math.pi / ell
    radius = FINE[0]
    stop = float(transform(radius))

    def density(x):
        z = inverse(x)
        return phi(z) * jacobian(z)

    def integ(fun, kind, frequency):
        value, error = quad(fun, 0., stop, weight=kind, wvar=frequency,
                            epsabs=5e-13, epsrel=0., limit=500)
        return 2 * value, 2 * error

    x_sin, first_error = integ(lambda x: x * density(x), 'sin', k)
    y_sin, errors = [], [first_error]
    for lag in LAGS:
        r = RHO[0]**lag
        value, error = integ(lambda x: conditional(inverse(x), r) * density(x), 'sin', k)
        y_sin.append(value)
        errors.append(error)
    cos_2, error = integ(density, 'cos', 2 * k)
    errors.append(error)
    # For |z| beyond the cutoff, the conditional polynomial is smaller than |g(z)|.
    cross_tail = float(2 * phi(radius) * (1 + CUBIC * (radius**2 + 2)) / SCALE)
    return x_sin, np.array(y_sin), (1 - cos_2) / 2, max(errors), cross_tail


def pair_moments(ell, setting):
    """Fold the bivariate integral into x,y >= 0 and integrate in blocks."""
    radius, per_period = setting
    dx = ell / per_period
    length = math.ceil(float(transform(radius)) / dx)
    x = np.arange(length + 1) * dx
    z = inverse(x)
    wt = dx * jacobian(z) * np.sin(2 * math.pi * x / ell)
    wt[[0, -1]] *= .5
    output = []
    for lag in LAGS:
        r = RHO[0]**lag
        variance = 1 - r**2
        values = []
        for start in range(0, len(z), 128):
            block = slice(start, start + 128)
            zi = z[block, None]
            positive = np.exp(-((z[None, :] - r * zi)**2 / variance + zi**2) / 2)
            negative = np.exp(-((z[None, :] + r * zi)**2 / variance + zi**2) / 2)
            values.append(float(wt[block] @ ((positive - negative) @ wt)))
        output.append(math.fsum(values) / (math.pi * math.sqrt(variance)))
    return np.asarray(output)


def population():
    r = correlation(RHO[0]**np.asarray(LAGS))
    rows = []
    for ell in ELLS:
        x_sin, y_sin, sin2, integration_error, cross_tail = marginal_moments(ell)
        coarse = pair_moments(ell, COARSE)
        fine = pair_moments(ell, FINE)
        amplitude = AMPLITUDE * ell / (2 * math.pi)
        variance = 1 + 2 * amplitude * x_sin + amplitude**2 * sin2
        linear = float(4 * amplitude * np.mean(r * x_sin - y_sin) / variance)
        quadratic = float(2 * amplitude**2 * np.mean(r * sin2 - fine) / variance)
        rows.append(dict(ell=float(ell), covariance_x_sin=x_sin, cross_y_sin=y_sin.tolist(),
                         sine_second_moment=sin2, sine_pair_moments=fine.tolist(),
                         pair_refinement_difference=float(np.max(np.abs(coarse - fine))),
                         quadrature_error_estimate=integration_error,
                         cross_moment_truncation_bound=cross_tail, warped_variance=variance,
                         population_linear_contribution=linear,
                         population_quadratic_contribution=quadratic,
                         population_change=linear + quadratic))
    return rows


def trajectory(root, group, size):
    rng = np.random.default_rng(np.random.SeedSequence([root, LENGTH, group]))
    x = rng.standard_normal((size, LENGTH, 4))
    innovation = np.sqrt(1 - RHO**2)
    for t in range(1, LENGTH):
        x[:, t] = RHO * x[:, t - 1] + innovation * x[:, t]
    x[..., 0] = transform(x[..., 0])
    return x


def score(x):
    return sum(((x[:, lag:] - x[:, :-lag])**2).mean(axis=1) for lag in LAGS) / len(LAGS)


def laplacian():
    matrix = np.zeros((LENGTH, LENGTH))
    for lag in LAGS:
        i = np.arange(LENGTH - lag)
        j = i + lag
        weight = 1 / (len(LAGS) * (LENGTH - lag))
        matrix[i, i] += weight
        matrix[j, j] += weight
        matrix[i, j] -= weight
        matrix[j, i] -= weight
    return matrix


def pilot():
    x = trajectory(ROOTS[0], 0, 64)[..., 0]
    np.testing.assert_allclose(transform(inverse(x)), x, atol=1e-12)
    matrix = laplacian()
    weights = x @ matrix
    base = score(x)
    np.testing.assert_allclose((x * weights).sum(axis=1), base, atol=1e-12)
    maximum = 0.
    for ell in ELLS:
        amplitude = AMPLITUDE * ell / (2 * math.pi)
        p = amplitude * np.sin(2 * math.pi * x / ell)
        for dose in (ell, 2 * ell):
            shifted = x + dose + amplitude * np.sin(2 * math.pi * (x + dose) / ell)
            maximum = max(maximum, float(np.max(np.abs(shifted - x - p - dose))))
        direct = score(x + p)
        expansion = base + 2 * (weights * p).sum(axis=1) + score(p)
        np.testing.assert_allclose(direct, expansion, atol=1e-12)
    assert maximum < 1e-12
    return dict(windows=64, inverse=True, weights=True, expansion=True,
                dose_pair_maximum_absolute_residual=maximum)


def slope(values):
    log_ell = np.log(ELLS)
    centered = log_ell - log_ell.mean()
    return np.log(values) @ centered / (centered @ centered)


def limits():
    lags = np.arange(LENGTH)
    covariance = correlation(RHO[0]**np.abs(lags[:, None] - lags[None, :]))
    matrix = laplacian()
    average_r = float(correlation(RHO[0]**np.asarray(LAGS)).mean())
    population_constant = AMPLITUDE**2 * average_r / (4 * math.pi**2)
    window_constant = AMPLITUDE * math.sqrt(float(np.sum((matrix @ covariance) * matrix))) / (
        math.sqrt(2) * math.pi)
    gaussian_fourth = lambda k: math.prod(range(1, k, 2))
    fourth = math.fsum(math.comb(4, j) * CUBIC**j * gaussian_fourth(4 + 2*j)
                       for j in range(5)) / SCALE**4
    return population_constant, window_constant, fourth


def formal():
    rows = population()
    mean_square = np.zeros((GROUPS, len(ELLS)))
    rank_change = np.zeros(len(ELLS), dtype=int)
    score_decrease = np.zeros(len(ELLS), dtype=int)
    hashes = []
    for group in range(GROUPS):
        x = trajectory(ROOTS[1], group, SIZE)
        hashes.append(hashlib.sha256(x.tobytes()).hexdigest())
        d = score(x)
        cutoff = np.sort(d[:, 1:], axis=1)[:, 1]
        axis = x[..., 0]
        for j, (ell, row) in enumerate(zip(ELLS, rows, strict=True)):
            warped = axis + AMPLITUDE * ell / (2 * math.pi) * np.sin(2 * math.pi * axis / ell)
            difference = score(warped) / row['warped_variance'] - d[:, 0]
            mean_square[group, j] = np.mean(difference**2)
            rank_change[j] += np.count_nonzero((d[:, 0] < cutoff) != (d[:, 0] + difference < cutoff))
            score_decrease[j] += np.count_nonzero(difference < 0)
    assert len(set(hashes)) == GROUPS
    rms = np.sqrt(mean_square.mean(axis=0))
    draws = np.random.default_rng(ROOTS[2]).integers(GROUPS, size=(4096, GROUPS))
    window_slope = float(slope(rms))
    window_slope_ci = np.quantile(slope(np.sqrt(mean_square[draws].mean(axis=1))), [.025, .975])
    pop = np.array([row['population_change'] for row in rows])
    pop_slope = float(slope(pop))
    pop_limit, rms_limit, kurtosis = limits()
    pop_relative = abs(pop / ELLS**2 / pop_limit - 1)
    rms_relative = abs(rms / ELLS / rms_limit - 1)
    for j, row in enumerate(rows):
        row.update(window_rms=float(rms[j]), slow_top2_disagreement=int(rank_change[j]),
                   window_score_decreases=int(score_decrease[j]),
                   population_limit_relative_error=float(pop_relative[j]),
                   window_limit_relative_error=float(rms_relative[j]))
    gates = dict(
        paired_moment_precision=bool(max(row['pair_refinement_difference'] for row in rows) < 5e-9),
        one_dimensional_precision=bool(max(row['quadrature_error_estimate'] for row in rows) < 1e-10),
        population_order=bool(1.8 <= pop_slope <= 2.15),
        window_order=bool(.8 <= window_slope <= 1.25),
        population_constant=bool(np.all(pop_relative[-2:] <= .05)),
        window_constant=bool(np.all(rms_relative[-2:] <= .05)),
    )
    return dict(windows=GROUPS*SIZE, length=LENGTH, groups=GROUPS, trajectory_hashes=hashes,
                pair_truncation_bounds=[2 * math.erfc(radius / math.sqrt(2))
                                        for radius, _ in (COARSE, FINE)],
                observation_kurtosis=kurtosis, expected_population_constant=pop_limit,
                expected_window_rms_constant=rms_limit, population_slope=pop_slope,
                window_slope=window_slope, window_slope_ci=window_slope_ci.tolist(),
                results=rows, gates=gates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'pilot', 'formal'))
    phase = parser.parse_args().phase
    sources = {name: digest(HERE / name) for name in SOURCES}
    if phase == 'lock':
        save('LOCK.json', dict(sources=sources, roots=ROOTS))
        return
    assert read('LOCK.json')['sources'] == sources
    lock = digest(HERE / 'LOCK.json')
    if phase == 'pilot':
        save('PILOT.json', dict(lock=lock, **pilot()))
        return
    assert read('PILOT.json')['lock'] == lock
    save('SUMMARY.json', dict(lock=lock, pilot=digest(HERE / 'PILOT.json'), **formal()))


if __name__ == '__main__':
    main()
