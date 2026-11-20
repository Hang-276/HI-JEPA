"""Labelled calibration and frozen corrections for dynamic slow-score errors."""

import numpy as np


DIMENSION = 4
UPPER = np.triu_indices(DIMENSION)
ARMS = ("base", "cross_mean", "cross_diag", "cross_full", "cross_full_reverse", "total_full")


def normalize(z_cal, u_cal):
    """Fit the physical forward affine map and keep its named-axis scale."""
    z = np.asarray(z_cal, dtype=np.float64)
    u = np.asarray(u_cal, dtype=np.float64)
    if z.shape != u.shape or z.ndim != 2 or z.shape[1] != DIMENSION or len(z) < DIMENSION + 1:
        raise ValueError("expected matched CAL states and readouts of shape [n,4]")
    if not np.isfinite(z).all() or not np.isfinite(u).all():
        raise ValueError("nonfinite calibration inputs")
    affine = np.linalg.lstsq(np.column_stack([np.ones(len(z)), z]), u, rcond=None)[0]
    scale = affine[1:].diagonal()
    if np.any(np.abs(scale) < 1e-8):
        raise ValueError("own-axis slope too small for physical normalization")
    return {"offset": affine[0].tolist(), "scale": scale.tolist()}


def _paths(values, lags):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] != DIMENSION or not len(x):
        raise ValueError("expected trajectories of shape [n,t,4]")
    if not lags or any(not 0 < lag < x.shape[1] for lag in lags):
        raise ValueError("lags must be positive and shorter than the trajectories")
    if not np.isfinite(x).all():
        raise ValueError("nonfinite trajectories")
    return x


def grams(v, lags):
    """Window moments, averaged over each lag's admissible endpoints then lags."""
    values = _paths(v, lags)
    moment = np.zeros((len(values), DIMENSION, DIMENSION))
    for lag in lags:
        increment = values[:, lag:] - values[:, :-lag]
        moment += np.einsum("nti,ntj->nij", increment, increment) / increment.shape[1]
    return moment / len(lags)


def decompose(z, v, lags):
    """Return physical and visible numerators, signed cross term K, and energy E."""
    z, v = _paths(z, lags), _paths(v, lags)
    if z.shape != v.shape:
        raise ValueError("physical and visible trajectories must match")
    parts = {name: np.zeros((len(v), DIMENSION)) for name in ("truth", "visible", "cross", "energy")}
    for lag in lags:
        physical = z[:, lag:] - z[:, :-lag]
        visible = v[:, lag:] - v[:, :-lag]
        error = visible - physical
        parts["truth"] += np.mean(physical**2, axis=1)
        parts["visible"] += np.mean(visible**2, axis=1)
        parts["cross"] += 2 * np.mean(physical * error, axis=1)
        parts["energy"] += np.mean(error**2, axis=1)
    parts = {name: value / len(lags) for name, value in parts.items()}
    np.testing.assert_allclose(
        parts["visible"] - parts["truth"], parts["cross"] + parts["energy"],
        rtol=0, atol=5e-14,
    )
    return parts


def _variance(values):
    v = np.asarray(values, dtype=np.float64)
    if v.shape != (DIMENSION,) or not np.isfinite(v).all() or np.any(v <= 0):
        raise ValueError("expected four positive finite reference variances")
    return v


def _features(gram):
    values = np.asarray(gram, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (DIMENSION, DIMENSION) or not len(values):
        raise ValueError("expected window moments of shape [n,4,4]")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite window moments")
    return values.diagonal(axis1=1, axis2=2), values[:, UPPER[0], UPPER[1]]


def _fit_ridge(x, y, xv, yv, alphas):
    center, scale = x.mean(axis=0), x.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("constant CAL feature")
    if y.shape != (len(x), DIMENSION) or yv.shape != (len(xv), DIMENSION):
        raise ValueError("CAL and VAL targets must have four physical outputs")
    target_center = y.mean(axis=0)
    target = y - target_center
    train = (x - center) / scale
    validation = (xv - center) / scale
    covariance = train.T @ train / len(train)
    cross = train.T @ target / len(train)
    candidates = [
        np.linalg.lstsq(train, target, rcond=None)[0] if alpha == 0 else
        np.linalg.solve(covariance + alpha * np.eye(x.shape[1]), cross)
        for alpha in alphas
    ]
    errors = [float(np.mean((validation @ coefficients + target_center - yv) ** 2)) for coefficients in candidates]
    selected = int(np.argmin(errors))
    return {
        "feature_mean": center.tolist(), "feature_std": scale.tolist(), "target_mean": target_center.tolist(),
        "coefficients": candidates[selected].tolist(), "alpha": alphas[selected],
        "validation_mse_by_alpha": errors, "validation_mse": errors[selected],
    }


def fit_predictors(fit_gram, fit_parts, val_gram, val_parts, variance, config):
    """Fit frozen CAL-mean and ridge corrections, selecting penalties on VAL."""
    variance = _variance(variance)
    diag, full = _features(fit_gram)
    val_diag, val_full = _features(val_gram)
    for diagonal, parts in ((diag, fit_parts), (val_diag, val_parts)):
        np.testing.assert_allclose(diagonal, parts["visible"], rtol=0, atol=5e-14)
    cross, val_cross = fit_parts["cross"], val_parts["cross"]
    total = fit_parts["visible"] / variance - fit_parts["truth"]
    val_total = val_parts["visible"] / variance - val_parts["truth"]
    alphas = config["ridge_alphas"]
    mean_cross = cross.mean(axis=0)
    return {
        "ridge_alphas": alphas,
        "mean_cross": mean_cross.tolist(),
        "mean_cross_validation_mse": float(np.mean((val_cross - mean_cross) ** 2)),
        "cross_diag": _fit_ridge(diag, cross, val_diag, val_cross, alphas),
        "cross_full": _fit_ridge(full, cross, val_full, val_cross, alphas),
        "total_full": _fit_ridge(full, total, val_full, val_total, alphas),
    }


def _ridge_predict(features, fit):
    normalized = (features - np.asarray(fit["feature_mean"])) / np.asarray(fit["feature_std"])
    return normalized @ np.asarray(fit["coefficients"]) + np.asarray(fit["target_mean"])


def predict_corrections(gram, fit):
    """Return K estimates in numerator units, and total error in score units."""
    diag, full = _features(gram)
    return {
        "cross_mean": np.broadcast_to(np.asarray(fit["mean_cross"]), diag.shape),
        "cross_diag": _ridge_predict(diag, fit["cross_diag"]),
        "cross_full": _ridge_predict(full, fit["cross_full"]),
        "total_full": _ridge_predict(full, fit["total_full"]),
    }


def predict(gram, fit, variance, config):
    """Predict six four-axis score arrays using only frozen fit and visible moments."""
    if tuple(config["arms"]) != ARMS:
        raise ValueError("unexpected arm order")
    variance = _variance(variance)
    numerator, _ = _features(gram)
    correction = predict_corrections(gram, fit)
    base = numerator / variance
    return {
        "base": base,
        "cross_mean": (numerator - correction["cross_mean"]) / variance,
        "cross_diag": (numerator - correction["cross_diag"]) / variance,
        "cross_full": (numerator - correction["cross_full"]) / variance,
        "cross_full_reverse": (numerator + correction["cross_full"]) / variance,
        "total_full": base - correction["total_full"],
    }
