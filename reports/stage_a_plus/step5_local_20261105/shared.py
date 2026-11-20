"""Frozen transformations, quadratic scores and Student quantiles for step 5."""
import math
from functools import lru_cache
import numpy as np

PHIS = (0, 60, 80)
LAGS = (4, 16, 32)
AMPLITUDES = (0, 2, 4, 5, 6, 9, 12, 15, 18, 30, 45)


def rotation(degrees, axes=(0, 1)):
    a, b = axes
    angle = math.radians(degrees)
    co, si = math.cos(angle), math.sin(angle)
    transform = np.eye(4)
    transform[np.ix_([a, b], [a, b])] = [[co, si], [-si, co]]
    return transform


def candidates(phi):
    angles = sorted({sign*m for m in AMPLITUDES for sign in (-1, 1)})
    rows = [(f'r{angle:+d}', rotation(angle)) for angle in angles]
    if phi == 80:
        rows.append(('block+9', rotation(9, (0, 2))))
        for slope in (-.22, .22):
            q = np.eye(4)
            q[0, :2] = np.array([1., slope]) / math.sqrt(1 + slope*slope)
            rows.append((f'shear{slope:+.2f}', q))
    return rows


def scores(Q, M):
    return np.einsum('ai,...ij,aj->...a', Q, M, Q)


@lru_cache(None)
def tcrit(p, df):
    # Integrate the t density from zero. Only central quantiles at df>=31 are used.
    nodes, weights = np.polynomial.legendre.leggauss(48)
    log_constant = math.lgamma((df+1)/2) - math.lgamma(df/2) - .5*math.log(df*math.pi)
    lo, hi = 0., 10.
    for _ in range(57):
        half = (hi + lo)/4
        x = half*(nodes+1)
        cdf = .5 + half*np.dot(weights, np.exp(log_constant - ((df+1)/2)*np.log1p(x*x/df)))
        if cdf < p:
            lo = 2*half
        else:
            hi = 2*half
    return (lo+hi)/2
