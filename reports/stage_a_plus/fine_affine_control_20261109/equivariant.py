"""Affine-equivariant direction readout for four state axes and regime labels."""

from itertools import combinations_with_replacement

import numpy as np


STATE_DIM = 4
REGIMES = (0, 5, 6, 7, 8)
MAX_CONDITION = 10
RANK_TOLERANCE = 1e-12
_MONOMIALS = [
    np.array(tuple(combinations_with_replacement(range(STATE_DIM), degree)))
    for degree in (2, 3)
]


def _design(previous, actions):
    """Polynomial basis of total state degree at most three, plus linear actions."""
    ones = np.ones(previous.shape[:-1] + (1,))
    products = [np.prod(previous[..., indices], axis=-1) for indices in _MONOMIALS]
    return np.concatenate([ones, previous, *products, actions], axis=-1)


def _history(y, a, u):
    y = np.asarray(y, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    u = np.asarray(u)
    if (y.ndim != 3 or y.shape[-1] != STATE_DIM or y.shape[1] < 2
            or a.ndim != 3 or a.shape[:2] != y[:, :-1].shape[:2] or u.shape != a.shape[:2]):
        raise ValueError("Expected aligned trajectories and one action/regime per transition")
    if not np.isfinite(y).all() or not np.isfinite(a).all():
        raise ValueError("State and action inputs must be finite")
    return y, a, u


def fit_drift(y, a, u):
    """Fit shared increment drift from the supplied group-0 trajectory slice."""
    y, a, u = _history(y, a, u)
    eligible = np.isin(u, REGIMES)
    x = _design(y[:, :-1][eligible], a[eligible])
    if x.shape[0] < x.shape[1]:
        return None
    delta = np.diff(y, axis=1)[eligible]
    coefficient, _, rank, _ = np.linalg.lstsq(x, delta, rcond=RANK_TOLERANCE)
    return coefficient if rank == x.shape[1] else None


def rows(ycal, a, u, drift, covariance, episode_mask):
    """Return variance-normalized semantic unmixing rows, or None when unusable."""
    if drift is None:
        return None
    ycal, a, u = _history(ycal, a, u)
    ycal, a, u = ycal[episode_mask], a[episode_mask], u[episode_mask]
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (STATE_DIM, STATE_DIM) or not np.isfinite(covariance).all():
        raise ValueError("History covariance must be a finite four-by-four matrix")
    moments = []
    for code in REGIMES:
        cell = u == code
        if not cell.any():
            return None
        previous = ycal[:, :-1][cell]
        residual = ycal[:, 1:][cell] - previous - _design(previous, a[cell]) @ drift
        moments.append(residual.T @ residual / len(residual))
    baseline, *spikes = moments
    if not np.isfinite(moments).all():
        return None

    eigvals, basis = np.linalg.eigh(baseline)
    if eigvals[0] <= 0:
        return None
    whiten = (basis / np.sqrt(eigvals)) @ basis.T
    contrasts = whiten @ (np.stack(spikes) - baseline) @ whiten
    contrasts = (contrasts + contrasts.swapaxes(-1, -2)) * 0.5
    eigvals, basis = np.linalg.eigh(contrasts)
    eigengap = eigvals[:, -1] - eigvals[:, -2]
    tolerance = RANK_TOLERANCE * np.max(np.abs(eigvals), axis=1)
    if np.any(eigvals[:, -1] <= 0) or np.any(eigengap <= tolerance):
        return None

    whitened_columns = basis[:, :, -1].T
    if np.linalg.cond(whitened_columns) > MAX_CONDITION:
        return None
    # Inverse of the columns mapped back by the baseline square root.
    unmixing = np.linalg.solve(whitened_columns, whiten)
    variances = np.einsum("ij,jk,ik->i", unmixing, covariance, unmixing)
    if not np.isfinite(variances).all() or np.any(variances <= 0):
        return None
    return unmixing / np.sqrt(variances[:, None])
