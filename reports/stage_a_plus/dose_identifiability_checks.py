"""Deterministic quadrature checks for the known-dose slow-set bounds."""
import json
import os

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

LAGS = np.array([4, 16, 32])
R = .95 ** LAGS
K, AC, LIMIT, SIGMA = 3., .16, .28, .04


def increase(r, a, k):
    exponent = k ** 2
    second = .5 * (-np.expm1(-2 * exponent))
    correlated = .5 * (np.exp(-(1 - r) * exponent) - np.exp(-(1 + r) * exponent))
    variance = 1 + 2 * a * k * np.exp(-exponent / 2) + a ** 2 * second
    return 2 * a ** 2 * (r * second - correlated) / variance


def gap(a):
    return (increase(R, a, K) - increase(R, AC, K)).mean()


def solve_for_gap(target):
    lo, hi = 0., LIMIT
    assert gap(lo) < target < gap(hi)
    for _ in range(60):
        a = (lo + hi) / 2
        if gap(a) < target:
            lo = a
        else:
            hi = a
    return (lo + hi) / 2


def branch(x, d, a, k):
    target = x + a * np.sin(k * x) + d
    lo, hi = target - a, target + a
    for _ in range(60):
        m = (lo + hi) / 2
        below = m + a * np.sin(k * m) < target
        lo, hi = np.where(below, m, lo), np.where(below, hi, m)
    return (lo + hi) / 2


def run():
    points, weights = np.polynomial.hermite.hermgauss(96)
    x, p = points * np.sqrt(2), weights / np.sqrt(np.pi)
    hx = x + AC * np.sin(K * x)
    variance = p @ (hx ** 2)
    errors = []
    for r in R:
        y = r * x[:, None] + np.sqrt(1 - r ** 2) * x[None, :]
        hy = y + AC * np.sin(K * y)
        cov = p @ ((hx[:, None] * hy) @ p)
        errors.append(abs(2 * (r - cov / variance) - increase(r, AC, K)))
    assert max(errors) < 1e-12

    period = 2 * np.pi / K
    resonance = max(float(np.max(abs(branch(x, d, a, K) - x - d)))
                    for a in (AC, LIMIT) for d in (period, 2 * period))
    assert resonance < 1e-12

    sharp = []
    for period_gcd in (1., .5, .25, .125):
        k = 2 * np.pi / period_gcd
        value = increase(R, .6 / k, k).mean()
        limit = .6 ** 2 * R.mean() / (4 * np.pi ** 2)
        bound = period_gcd ** 2 * R.mean() / (2 * (1 - period_gcd / 2) ** 2)
        assert 0 < value < bound
        sharp.append(dict(gcd=period_gcd, increase=float(value), theoretical_upper=float(bound),
                          normalized_by_gcd_squared=float(value / period_gcd ** 2), asymptote=float(limit)))
    assert abs(sharp[-1]['normalized_by_gcd_squared'] / sharp[-1]['asymptote'] - 1) < .001

    rows = []
    for gamma in (.003, .006, .009):
        a1, a2 = (solve_for_gap(sign * gamma) for sign in (-1, 1))
        assert max(abs(gap(a1) + gamma), abs(gap(a2) - gamma)) < 1e-12
        for epsilon in (.008, .016, .032):
            difference = branch(x, epsilon, a1, K) - branch(x, epsilon, a2, K)
            kl = .5 * (p @ (difference ** 2)) / (2 * SIGMA ** 2)
            normalized = kl * SIGMA ** 2 / (epsilon * gamma) ** 2
            rows.append(dict(gamma=gamma, epsilon=epsilon, a_minus=a1, a_plus=a2,
                             per_query_kl=float(kl), normalized_kl=float(normalized)))
    ratios = [r['normalized_kl'] for r in rows]
    assert max(ratios) / min(ratios) < 1.2
    return dict(quadrature_max_error=float(max(errors)), resonant_translation_max_error=resonance,
                commensurate_order=sharp, detuned_kl=rows)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
