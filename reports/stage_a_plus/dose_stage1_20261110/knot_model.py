"""Exact knot implications of bracketing a monotone dose response."""

from dataclasses import dataclass, replace
from fractions import Fraction
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import csr_matrix, eye, hstack, vstack
from scipy.sparse.csgraph import floyd_warshall, NegativeCycleError
from scipy.special import ndtr


class InfeasibleKnots(ValueError):
    """The relaxed knot graph has a negative cycle."""


@dataclass(frozen=True)
class KnotModel:
    nodes: np.ndarray
    left: np.ndarray
    right: np.ndarray
    cap: np.ndarray
    anchor: int
    grid_index: np.ndarray
    distance: np.ndarray
    center: np.ndarray | None
    m: float
    M: float
    slack: float

    def validate(self, values):
        return validate_witness(self, values)

    def variance(self, values):
        return gaussian_variance(self.nodes, values)

    def score(self, values, trajectory, lags=(4, 16, 32), *, verify=True):
        return score_witness(self, values, trajectory, lags, verify=verify)


def make_knots(grid, doses, lower, upper, m=.7, M=1.3):
    """Bracket arrays are dose-by-grid. Infinite sides impose no constraint.

    The edge ``i -> j`` constrains ``h(nodes[j]) - h(nodes[i])`` from above.
    ``distance`` is an outward-padded all-pairs implication, while ``cap``
    records the original edge caps. ``center`` is a verified witness if found.
    """
    grid = np.asarray(grid, dtype=float)
    doses = np.atleast_1d(np.asarray(doses, dtype=float))
    lower, upper = (np.asarray(side, dtype=float) for side in (lower, upper))
    m, M = float(m), float(M)
    if (grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid))
            or doses.ndim != 1 or doses.size == 0 or not np.all(np.isfinite(doses))
            or not 0 < m <= M < np.inf):
        raise ValueError('Invalid grid, doses, or slope bounds')
    shape = (len(doses), len(grid))
    if shape[0] == 1 and lower.shape == upper.shape == (shape[1],):
        lower, upper = lower[None, :], upper[None, :]
    if lower.shape != shape and lower.shape == upper.shape == shape[::-1]:
        lower, upper = lower.T, upper.T
    if (lower.shape != shape or upper.shape != shape or np.any(np.isnan(lower))
            or np.any(np.isnan(upper)) or np.any(np.isposinf(lower))
            or np.any(np.isneginf(upper)) or np.any(lower > upper)):
        raise ValueError('Invalid response brackets')

    use_lower = np.isfinite(lower)
    use_upper = np.isfinite(upper)
    nodes = np.unique(np.r_[grid, lower[use_lower], upper[use_upper], 0.])
    anchor = int(np.searchsorted(nodes, 0.))
    grid_index = np.searchsorted(nodes, grid)
    sources = np.broadcast_to(grid_index, shape)
    doses_by_grid = np.broadcast_to(doses[:, None], shape)
    lo_index = np.searchsorted(nodes, lower[use_lower])
    hi_index = np.searchsorted(nodes, upper[use_upper])
    start = np.arange(len(nodes) - 1)
    step = np.diff(nodes)
    left = np.r_[start, start + 1, sources[use_lower], hi_index]
    right = np.r_[start + 1, start, lo_index, sources[use_upper]]
    cap = np.r_[M * step, -m * step, doses_by_grid[use_lower], -doses_by_grid[use_upper]]

    # Initialize with valid all-pairs secants, so finite intermediates in the
    # min-plus closure are bounded above by the range of these secants. Give
    # every off-diagonal inequality enough margin to absorb edge and addition
    # roundoff. An update using two off-diagonal edges adds two such margins.
    difference = nodes[None, :] - nodes[:, None]
    distance = np.where(difference >= 0, M * difference, m * difference)
    np.minimum.at(distance, (left, right), cap)
    scale = max(1., float(np.abs(distance).max()), float(np.abs(cap).max(initial=0.)),
                float(np.abs(nodes).max()))
    slack = 128 * np.finfo(float).eps * scale
    if np.any(distance.diagonal() < -slack):
        raise InfeasibleKnots('A bracket requires a negative self-edge')
    distance += slack
    np.fill_diagonal(distance, 0.)
    # scipy treats zeros in a dense input as absent edges.
    zeros = distance == 0.
    np.fill_diagonal(zeros, False)
    distance[zeros] = np.nextafter(0., np.inf)
    try:
        distance = floyd_warshall(distance, directed=True, overwrite=True)
    except NegativeCycleError as exc:
        raise InfeasibleKnots('Response brackets conflict with the slope bounds') from exc

    model = KnotModel(nodes, left, right, cap, anchor, grid_index, distance,
                      None, m, M, slack)
    for proposal in (nodes, (distance[anchor, :] - distance[:, anchor]) / 2):
        if validate_witness(model, proposal):
            return replace(model, center=proposal.copy())
    return replace(model, center=strict_center(model))


def strict_center(model):
    """Find a common-slack witness in adjacent slopes, then verify it exactly."""
    step = np.diff(model.nodes)
    count = len(step)
    if count == 0:
        return None
    left, right = model.left[2 * count:], model.right[2 * count:]
    lo, hi = np.minimum(left, right), np.maximum(left, right)
    increments = np.arange(count)
    crossed = (lo[:, None] <= increments) & (increments < hi[:, None])
    responses = crossed * step * np.sign(right - left)[:, None]
    identity = eye(count, format='csr')
    inequalities = vstack([identity, -identity, csr_matrix(responses)], format='csr')
    inequalities = hstack([inequalities, np.ones((inequalities.shape[0], 1))], format='csr')
    caps = np.r_[np.full(count, model.M), np.full(count, -model.m), model.cap[2 * count:]]
    # SciPy passes these native HiGHS options through but warns about them.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected:',
                                category=OptimizeWarning)
        fit = linprog(np.r_[np.zeros(count), -1.], A_ub=inequalities, b_ub=caps,
                      bounds=[(None, None)] * count + [(0., None)], method='highs-ds',
                      options={'threads': 1, 'parallel': False})
    if not fit.success:
        return None
    increment = fit.x[:-1] * step
    i = model.anchor
    proposal = np.r_[-np.cumsum(increment[:i][::-1])[::-1], 0., np.cumsum(increment[i:])]
    return proposal if validate_witness(model, proposal) else None


def validate_witness(model, values):
    """Check the original inequalities, with exact arithmetic at roundoff ties.

    Values extend with unit slope on both tails. Interpolation between knots
    and these tail slopes define the actual candidate coordinate.
    """
    values = np.asarray(values, dtype=float)
    if (values.shape != model.nodes.shape or not np.all(np.isfinite(values))
            or values[model.anchor] != 0. or not model.m <= 1. <= model.M):
        return False
    gap = values[model.right] - values[model.left]
    error = (8 * np.finfo(float).eps
             * np.maximum(1., np.maximum(np.abs(gap), np.abs(model.cap))))
    if np.any(gap > model.cap + error):
        return False
    close = np.flatnonzero(gap >= model.cap - error)
    if len(close) == 0:
        return True
    exact = lambda x: Fraction(float(x))
    steps = len(model.nodes) - 1
    for i in close:
        left, right = model.left[i], model.right[i]
        exact_gap = exact(values[right]) - exact(values[left])
        if i < 2 * steps:
            slope = model.M if i < steps else model.m
            cap = exact(slope) * (exact(model.nodes[right]) - exact(model.nodes[left]))
        else:
            cap = exact(model.cap[i])
        if exact_gap > cap:
            return False
    return True


def interpolate(nodes, values, x):
    """Evaluate the piecewise-linear witness with unit-slope tails."""
    x = np.asarray(x, dtype=float)
    return np.interp(x, nodes, values) + (x - np.clip(x, nodes[0], nodes[-1]))


def gaussian_variance(nodes, values):
    """Analytic Gaussian variance via truncated moments on each linear piece."""
    nodes, values = np.asarray(nodes, dtype=float), np.asarray(values, dtype=float)
    slope = np.r_[1., np.diff(values) / np.diff(nodes), 1.]
    intercept = np.r_[values[0] - nodes[0],
                      values[:-1] - slope[1:-1] * nodes[:-1],
                      values[-1] - nodes[-1]]
    endpoints = np.r_[-np.inf, nodes, np.inf]
    lo, hi = endpoints[:-1], endpoints[1:]
    density = np.exp(-endpoints**2 / 2) / np.sqrt(2 * np.pi)
    product = np.r_[0., nodes * density[1:-1], 0.]
    mass = np.where(lo >= 0., ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    first = density[:-1] - density[1:]
    second = mass + product[:-1] - product[1:]
    mean = np.sum(slope * first + intercept * mass)
    centered = intercept - mean
    return float(np.sum(slope**2 * second + 2 * slope * centered * first
                        + centered**2 * mass))


def score_witness(model, values, trajectory, lags=(4, 16, 32), *, verify=True):
    """Score piecewise-linear values on one or more windows along the last axis."""
    if verify and not validate_witness(model, values):
        raise ValueError('The witness violates an original knot constraint')
    h = interpolate(model.nodes, values, trajectory)
    numerator = sum(np.mean((h[..., lag:] - h[..., :-lag])**2, axis=-1)
                    for lag in lags) / len(lags)
    return numerator / gaussian_variance(model.nodes, values)
