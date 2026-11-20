"""Frozen calibration and observable Hermite corrections in named MLP coordinates."""

import itertools

import numpy as np


DIMENSION = 4
RECIPIENTS = (1, 2)
DEGREES = (2, 3)
MULTI_INDICES = np.array([
    (0, 0, 0, 0),
    *np.eye(DIMENSION, dtype=int).tolist(),
    *(
        indices
        for degree in DEGREES
        for indices in itertools.product(range(degree + 1), repeat=DIMENSION)
        if sum(indices) == degree
    ),
])
NONLINEAR = MULTI_INDICES.sum(axis=1) >= 2
MIXED = NONLINEAR & (np.count_nonzero(MULTI_INDICES, axis=1) >= 2)
GROUPS = ("mixed", "self", "all")
ARMS = (
    "base", "mixed_axis2", "mixed_axis3", "mixed_both", "reverse_both",
    "self_both", "raw_mixed_both", "all_nonlinear_both",
)


def basis(z):
    """Unit-variance Hermite products up to total degree three, including constant."""
    z = np.asarray(z, dtype=np.float64)
    if not z.ndim or z.shape[-1] != DIMENSION:
        raise ValueError("expected physical coordinates with trailing dimension four")
    square = z**2
    univariate = np.stack([
        np.ones_like(z),
        z,
        (square - 1) / np.sqrt(2),
        z * (square - 3) / np.sqrt(6),
    ], axis=-1)
    return np.prod(univariate[..., np.arange(DIMENSION)[None, :], MULTI_INDICES], axis=-1)


def group_indices(group, recipient):
    if group == "mixed":
        mask = MIXED
    elif group == "self":
        mask = NONLINEAR & (MULTI_INDICES[:, recipient] == MULTI_INDICES.sum(axis=1))
    elif group == "all":
        mask = NONLINEAR
    else:
        raise ValueError(f"unknown correction group: {group}")
    return np.flatnonzero(mask)


def _cosine(first, second):
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 0:
        return None
    return float(np.clip(first @ second / denominator, -1.0, 1.0))


def _group_fit(recipient, group, visible, physical_basis, visible_basis, residual_coefficients,
               split_coefficients, forward_linear, linear_condition, scale, config):
    indices = group_indices(group, recipient)
    coefficients = residual_coefficients[indices, recipient]
    norm = float(np.linalg.norm(coefficients))
    half_a, half_b = split_coefficients[:, indices, recipient]
    split_cosine = _cosine(half_a, half_b)
    raw = visible_basis[:, indices] @ coefficients
    raw_center = float(raw.mean())
    projection = np.linalg.lstsq(physical_basis, raw, rcond=None)[0]
    beta = np.zeros(DIMENSION)
    proxy_coefficients = np.zeros(len(indices))
    offset = raw_center
    proxy_cosine = None
    gamma = None
    reasons = []
    if np.any(np.abs(scale) < config["minimum_diagonal_slope"]):
        reasons.append("diagonal_slope")
    if not linear_condition < config["maximum_linear_condition"]:
        reasons.append("forward_condition")
    if norm < config["minimum_group_norm"]:
        reasons.append("group_norm")
    if split_cosine is None or split_cosine < config["minimum_split_cosine"]:
        reasons.append("split_cosine")
    if "forward_condition" not in reasons:
        beta = np.linalg.solve(forward_linear, projection[1:5])
        q_without_center = raw - visible @ beta
        offset = float(q_without_center.mean())
        centered = q_without_center - offset
        centered_projection = np.linalg.lstsq(physical_basis, centered, rcond=None)[0]
        proxy_coefficients = centered_projection[indices]
        proxy_cosine = _cosine(coefficients, proxy_coefficients)
        alignment = float(coefficients @ proxy_coefficients)
        if alignment > 0:
            gamma = float(norm**2 / alignment)
    if proxy_cosine is None or proxy_cosine < config["minimum_proxy_cosine"]:
        reasons.append("proxy_cosine")
    if gamma is None or not config["minimum_proxy_gain"] <= gamma <= config["maximum_proxy_gain"]:
        reasons.append("gamma")
    active = not reasons
    reduction = 1 - np.linalg.norm(coefficients - gamma * proxy_coefficients) / norm if active else 0.0
    return {
        "recipient": recipient,
        "indices": indices.tolist(),
        "coefficients": coefficients.tolist(),
        "beta": beta.tolist(),
        "offset": offset,
        "gamma": gamma if gamma is not None else 0.0,
        "active": active,
        "raw_center": raw_center,
        "split_cosine": split_cosine,
        "proxy_cosine": proxy_cosine,
        "norm": norm,
        "reason": ",".join(reasons) if reasons else "active",
        "cal_reduction_ratio": float(reduction),
        "proxy_coefficients": proxy_coefficients.tolist(),
    }


def fit_calibration(z, u, config):
    """Fit only on labelled calibration states. Returns a JSON-compatible fit."""
    z = np.asarray(z, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if z.shape != u.shape or z.ndim != 2 or z.shape[1] != DIMENSION or len(z) < 2 * len(MULTI_INDICES):
        raise ValueError("expected aligned calibration states and readouts of shape [n,4]")
    if not np.isfinite(z).all() or not np.isfinite(u).all():
        raise ValueError("nonfinite calibration inputs")
    design = np.column_stack((np.ones(len(z)), z))
    affine = np.linalg.lstsq(design, u, rcond=None)[0]
    offset = affine[0]
    scale = affine[1:].diagonal().copy()
    if np.any(np.abs(scale) < 1e-8):
        raise ValueError("own-axis slope too small for physical normalization")
    visible = (u - offset) / scale
    physical_basis = basis(z)
    visible_basis = basis(visible)
    residual = visible - z
    forward_coefficients = np.linalg.lstsq(physical_basis, visible, rcond=None)[0]
    residual_coefficients = np.linalg.lstsq(physical_basis, residual, rcond=None)[0]
    middle = len(z) // 2
    split_coefficients = np.array([
        np.linalg.lstsq(physical_basis[sl], residual[sl], rcond=None)[0]
        for sl in (slice(0, middle), slice(middle, None))
    ])
    forward_linear = forward_coefficients[1:5]
    linear_condition = float(np.linalg.cond(forward_linear))
    if not np.isfinite(linear_condition):
        linear_condition = float(np.finfo(np.float64).max)
    groups = {
        group: [
            _group_fit(
                recipient, group, visible, physical_basis, visible_basis, residual_coefficients,
                split_coefficients, forward_linear, linear_condition, scale, config
            )
            for recipient in RECIPIENTS
        ]
        for group in GROUPS
    }
    return {
        "multi_indices": MULTI_INDICES.tolist(),
        "offset": offset.tolist(),
        "scale": scale.tolist(),
        "normalizer": scale.tolist(),
        "forward_linear_condition": linear_condition,
        "residual_coefficients": residual_coefficients.tolist(),
        "forward_coefficients": forward_coefficients.tolist(),
        "split_residual_coefficients": split_coefficients.tolist(),
        "groups": groups,
    }


def _correction(visible, features, item, raw):
    indices = item["indices"]
    response = features[..., indices] @ np.asarray(item["coefficients"])
    if raw:
        return response - item["raw_center"]
    return item["gamma"] * (response - visible @ np.asarray(item["beta"]) - item["offset"])


def outputs(u, fit, config):
    """Apply frozen calibration and the prespecified correction arms to U."""
    arm_names = tuple(config["arms"])
    if arm_names != ARMS:
        raise ValueError("unexpected arm order")
    visible = (np.asarray(u, dtype=np.float64) - np.asarray(fit["offset"])) / np.asarray(fit["normalizer"])
    features = basis(visible)
    predictions = {arm: visible.copy() for arm in arm_names}
    for group, items in fit["groups"].items():
        for item in items:
            if not item["active"]:
                continue
            j = item["recipient"]
            correction = _correction(visible, features, item, raw=False)
            if group == "mixed":
                predictions[f"mixed_axis{j + 1}"][..., j] -= correction
                predictions["mixed_both"][..., j] -= correction
                predictions["reverse_both"][..., j] += correction
                predictions["raw_mixed_both"][..., j] -= _correction(visible, features, item, raw=True)
            elif group == "self":
                predictions["self_both"][..., j] -= correction
            else:
                predictions["all_nonlinear_both"][..., j] -= correction
    return predictions
