"""Frozen origin layouts and shared-noise dose responses for Stage 2."""

import numpy as np


ORIGIN_LAYOUTS = ('grid', 'random', 'spread')
POOL_SIZE = 512


def layout_origins(root, group, grid):
    """Return the original Gaussian pool and three sorted origin layouts."""
    grid = np.asarray(grid, dtype=float)
    if (grid.ndim != 1 or not 2 <= len(grid) <= POOL_SIZE
            or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0)):
        raise ValueError('Grid must be finite, sorted, and strictly increasing')
    rng = np.random.default_rng(np.random.SeedSequence([root, 1, group]))
    pool = rng.standard_normal(POOL_SIZE)

    low, high = np.argmin(pool), np.argmax(pool)
    selected = [low, high]
    distance = np.minimum(pool - pool[low], pool[high] - pool)
    distance[selected] = -np.inf
    while len(selected) < len(grid):
        next_index = np.argmax(distance)
        selected.append(next_index)
        distance = np.minimum(distance, np.abs(pool - pool[next_index]))
        distance[next_index] = -np.inf

    return pool, {'grid': grid.copy(), 'random': np.sort(pool[:len(grid)]),
                  'spread': np.sort(pool[selected])}


def response_means(origins, doses, physical, m, M):
    """Invert h(y)=h(x)+dose by the same bisection used in the prior round."""
    origins = np.asarray(origins, dtype=float)
    doses = np.asarray(doses, dtype=float)[:, None]
    target = physical(origins) + doses
    first = origins + doses / M
    second = origins + doses / m
    low, high = np.minimum(first, second), np.maximum(first, second)
    for _ in range(52):
        mid = (low + high) / 2
        less = physical(mid) < target
        low = np.where(less, mid, low)
        high = np.where(less, high, mid)
    return (low + high) / 2


def calibration_reads(root, group, q, sigma, repeat=20):
    """Add one paired array of independent Gaussian errors to each mean grid."""
    q = np.asarray(q, dtype=float)
    rng = np.random.default_rng(np.random.SeedSequence([root, 0, group]))
    return q[..., None] + sigma * rng.standard_normal(q.shape + (repeat,))
