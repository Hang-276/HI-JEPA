"""Project calibration constraints from arbitrary origins to a fixed grid."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np


KNOT_PATH = Path(__file__).resolve().parent.parent / 'dose_stage1_20261110' / 'knot_model.py'
if 'knot_model' not in sys.modules:
    spec = importlib.util.spec_from_file_location('knot_model', KNOT_PATH)
    knot_model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = knot_model
    spec.loader.exec_module(knot_model)
else:
    knot_model = sys.modules['knot_model']

InfeasibleKnots = knot_model.InfeasibleKnots


def _align(grid, origins, doses, low, high, m, M):
    grid, origins = (np.asarray(x, dtype=float) for x in (grid, origins))
    doses = np.atleast_1d(np.asarray(doses, dtype=float))
    low, high = (np.asarray(x, dtype=float) for x in (low, high))
    if (grid.ndim != 1 or not grid.size or np.any(np.diff(grid) <= 0)
            or origins.ndim != 1 or not origins.size or doses.ndim != 1 or not doses.size
            or not all(np.isfinite(x).all() for x in (grid, origins, doses))
            or not 0 < m <= M < np.inf):
        raise ValueError('invalid grid, origins, doses, or slopes')
    shape = (doses.size, origins.size)
    if doses.size == 1 and low.shape == high.shape == (origins.size,):
        low, high = low[None, :], high[None, :]
    if (low.shape != shape or high.shape != shape or np.isnan(low).any() or np.isnan(high).any()
            or np.isposinf(low).any() or np.isneginf(high).any() or np.any(low > high)):
        raise ValueError('invalid dose-by-origin response brackets')

    nodes = np.unique(np.r_[grid, origins])
    grid_index = np.searchsorted(nodes, grid)
    origin_index = np.searchsorted(nodes, origins)
    row = np.arange(doses.size)[:, None]
    aligned_low = np.full((doses.size, nodes.size), -np.inf)
    aligned_high = np.full_like(aligned_low, np.inf)
    np.maximum.at(aligned_low, (row, origin_index), low)
    np.minimum.at(aligned_high, (row, origin_index), high)
    if np.any(aligned_low > aligned_high):
        raise InfeasibleKnots('repeated origins have disjoint brackets')
    return nodes, grid_index, doses, aligned_low, aligned_high


def exact_model(grid, origins, doses, low, high, m=.7, M=1.3):
    """Use all query endpoints, but keep the original evaluation grid indices."""
    nodes, grid_index, doses, low, high = _align(grid, origins, doses, low, high, m, M)
    model = knot_model.make_knots(nodes, doses, low, high, m=m, M=M)
    return replace(model, grid_index=model.grid_index[grid_index])


def propagated_ranges(grid, origins, doses, low, high, m=.7, M=1.3):
    """Apply the old response propagation to all origins and grid nodes."""
    try:
        nodes, grid_index, doses, low, high = _align(grid, origins, doses, low, high, m, M)
    except InfeasibleKnots:
        return None
    dx = nodes[None, :] - nodes[:, None]
    upper = np.maximum(m * dx, M * dx)
    for dose, start, end in zip(doses, low, high, strict=True):
        from_end = nodes[None, :] - end[:, None]
        from_start = nodes[None, :] - start[:, None]
        lower = dose + np.minimum(m * from_end, M * from_end)
        high = dose + np.maximum(m * from_start, M * from_start)
        upper = np.minimum(upper, np.minimum(high, -lower.T))
    if not np.isfinite(upper).all():
        raise ValueError('nonfinite propagated graph')
    scale = max(1., float(np.abs(upper).max()), float(np.abs(nodes).max()),
                float(np.abs(doses).max()))
    pad = max(1e-11, 128 * np.finfo(float).eps * scale)
    upper += pad
    np.fill_diagonal(upper, np.minimum(upper.diagonal(), 0.))
    for node in range(nodes.size):
        upper = np.minimum(upper, upper[:, node, None] + upper[None, node, :])
    if np.min(upper.diagonal()) < -pad:
        return None
    upper = upper[np.ix_(grid_index, grid_index)]
    return -upper.T - pad, upper + pad
