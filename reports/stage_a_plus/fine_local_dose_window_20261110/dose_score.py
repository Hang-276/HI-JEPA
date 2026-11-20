"""Observable local-dose correction of a fixed readout's slow scores."""

import numpy as np


LAGS = (4, 16, 32)
MAX_CONDITION = 20.
MIN_SINGULAR = 1e-3
UNDEFINED_CONDITION = np.finfo(float).max


def _lags(lags, length):
    lags = tuple(lags)
    if not lags or any(not isinstance(lag, (int, np.integer)) or not 0 < lag < length
                       for lag in lags):
        raise ValueError('Each scoring lag must be positive and shorter than the window')
    return lags


def lag_scores(values, lags=LAGS):
    """Average squared increments equally across lags, on standardized paths."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 4 or not values.size:
        raise ValueError('Expected paths with shape (windows, states, 4)')
    lags = _lags(lags, values.shape[1])
    return np.mean([((values[:, lag:] - values[:, :-lag])**2).mean(axis=1)
                    for lag in lags], axis=0)


def _order(scores, k):
    order = np.argsort(scores, axis=1, kind='stable')
    boundary = np.take_along_axis(scores, order[:, k-1:k+1], axis=1)
    points = np.sort(order[:, :k], axis=1)
    points[boundary[:, 0] == boundary[:, 1]] = -1
    return order, points


def _gap(scores, native_mask):
    return (np.min(np.where(native_mask, np.inf, scores), axis=1)
            - np.max(np.where(native_mask, scores, -np.inf), axis=1))


def _response_quality(jacobian):
    singular = np.linalg.svd(jacobian, compute_uv=False)
    smallest = singular[..., -1]
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        condition = np.divide(singular[..., 0], smallest,
                              out=np.full(smallest.shape, UNDEFINED_CONDITION),
                              where=smallest > 0.)
    condition = np.minimum(condition, UNDEFINED_CONDITION)
    min_singular = smallest.min(axis=1)
    max_condition = condition.max(axis=1)
    safe = (min_singular >= MIN_SINGULAR) & (max_condition <= MAX_CONDITION)
    return safe, min_singular, max_condition


def evaluate(native, branch_plus, branch_minus, fine_ref_sd, affine, affine_ref_sd,
             dose, k=2, lags=LAGS):
    """Score local-dose residuals using visible data only.

    ``native`` is (N,T,4), already permuted to the named physical slots.
    Branches are (N,T,4,4), with the last two axes named intervention and
    identically permuted Fine feature. All three arrays use the input units
    of ``affine``. Fine scoring uses ``fine_ref_sd``; all other rules share
    ``affine_ref_sd``. Gaps always test the original Fine-selected set.

    Numerical safety requires every local response matrix in a window to
    pass the conditioning and absolute singular-value checks. If a window
    fails, both corrected scores use the fixed affine fallback. This flag
    does not certify the accuracy of the trapezoidal approximation.
    """
    native, plus, minus, fine_sd, affine, affine_sd = (
        np.asarray(arr, dtype=np.float64)
        for arr in (native, branch_plus, branch_minus, fine_ref_sd, affine, affine_ref_sd))
    dose = float(dose)
    if (native.ndim != 3 or native.shape[-1] != 4 or not native.size
            or plus.shape != native.shape + (4,) or minus.shape != plus.shape
            or affine.shape != (4, 4) or fine_sd.shape != (4,) or affine_sd.shape != (4,)
            or not all(np.isfinite(a).all() for a in (native, plus, minus, fine_sd, affine, affine_sd))
            or not np.all(fine_sd > 0.) or not np.all(affine_sd > 0.)
            or not np.isfinite(dose) or dose <= 0.
            or not isinstance(k, (int, np.integer)) or not 0 < k < 4):
        raise ValueError('Invalid path, dose branch, reference scale, or fixed readout')
    lags = _lags(lags, native.shape[1])

    z = native @ affine
    affine_scores = lag_scores(z / affine_sd, lags)
    native_scores = lag_scores(native / fine_sd, lags)
    jacobian = ((plus - minus) @ affine) / (2 * dose)
    safe, min_singular, max_condition = _response_quality(jacobian)
    residual = np.zeros_like(z)
    if np.any(safe):
        # Row-vector form: d(physical) @ J = d(z), so d(z) @ J^-1
        # is the estimated physical step. Sum consecutive corrected steps.
        inverse = np.linalg.inv(jacobian[safe])
        midpoint = (inverse[:, :-1] + inverse[:, 1:]) / 2
        step = np.diff(z[safe], axis=1)
        integrated_step = (step[..., None, :] @ midpoint)[..., 0, :]
        residual[safe, 1:] = np.cumsum(integrated_step - step, axis=1)

    standardized = z / affine_sd
    residual /= affine_sd
    energy, cross, signed = [], [], []
    for lag in lags:
        dx = standardized[:, lag:] - standardized[:, :-lag]
        dr = residual[:, lag:] - residual[:, :-lag]
        energy.append((dr**2).mean(axis=1))
        cross.append(2 * (dx*dr).mean(axis=1))
        signed.append(((dx + dr)**2).mean(axis=1))
    energy, cross, signed_scores = (np.mean(terms, axis=0) for terms in (energy, cross, signed))
    magnitude_scores = affine_scores + energy
    # Both direct and expanded signed scores are available for independent parity checks.
    algebra_error = np.max(np.abs(signed_scores - (magnitude_scores + cross)), axis=1)

    native_order, native_set = _order(native_scores, k)
    native_mask = np.zeros(native_scores.shape, dtype=bool)
    np.put_along_axis(native_mask, native_order[:, :k], True, axis=1)
    out = dict(safe=safe, min_singular=min_singular, max_condition=max_condition,
               residual_energy=energy, cross_term=cross, algebra_error=algebra_error)
    for rule, scores in (('native', native_scores), ('affine', affine_scores),
                         ('magnitude', magnitude_scores), ('signed', signed_scores)):
        out[rule + '_scores'] = scores
        out[rule + '_gap'] = _gap(scores, native_mask)
        out[rule + '_set'] = native_set if rule == 'native' else _order(scores, k)[1]
    return out
