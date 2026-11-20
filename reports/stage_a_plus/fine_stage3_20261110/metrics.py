"""Dose-response diagnostics with whole-panel paired resampling."""

from itertools import permutations
import math

import numpy as np


DOSES = np.asarray((.035, .035 * math.sqrt(2), -.0175))
BOOTSTRAPS = 4096
AXES = np.arange(4)
PERMUTATIONS = np.asarray(tuple(permutations(AXES)))


def _scaled(changes):
    changes = np.asarray(changes, dtype=float)
    if (changes.ndim != 6 or changes.shape[1:3] != (len(DOSES), 4)
            or changes.shape[-1] != 4 or not all(changes.shape)
            or not np.isfinite(changes).all()):
        raise ValueError('expected finite changes [panel, dose, target, trajectory, context, output]')
    return changes / DOSES[None, :, None, None, None, None]


def _weights(groups, seed):
    draws = np.random.default_rng(seed).integers(groups, size=(BOOTSTRAPS, groups))
    counts = np.zeros((BOOTSTRAPS, groups), dtype=int)
    np.add.at(counts, (np.arange(BOOTSTRAPS)[:, None], draws), 1)
    return counts / groups


def _resample(weights, group_values):
    return (weights @ group_values.reshape(len(group_values), -1)).reshape(
        (len(weights), *group_values.shape[1:]))


def _interval(estimate, draws):
    quantiles = np.quantile(draws, [.025, .975], axis=0)
    return dict(estimate=np.asarray(estimate).tolist(), ci95=np.moveaxis(quantiles, 0, -1).tolist())


def _fraction(numerator, denominator):
    # An identically zero response has no off-target energy to diagnose.
    return np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def _off_target(energy):
    total = energy.sum(axis=(-2, -1))
    diagonal = energy[..., AXES, AXES].sum(axis=-1)
    by_permutation = energy[..., AXES, PERMUTATIONS].sum(axis=-1)
    best = by_permutation.max(axis=-1)
    return (_fraction(np.maximum(total - diagonal, 0.), total),
            _fraction(np.maximum(total - best, 0.), total))


def _rmse_squared(scaled):
    target = np.eye(4)[None, None, :, None, None, :]
    return ((scaled - target)**2).sum(axis=-1).mean(axis=(2, 3, 4))


def summarize(changes, bootstrap_seed, mapped=True):
    """Return matrices in physical-target × output order, separately per dose.

    Signed means divide by the signed physical dose. RMS divides by its
    magnitude. Off-target fractions pool squared changes before taking ratios.
    The best permutation is chosen separately per dose and bootstrap replicate.
    Only mapped outputs in physical units admit a relative delta RMSE.
    """
    scaled = _scaled(changes)
    weights = _weights(len(scaled), bootstrap_seed)
    energy_group = (scaled**2).mean(axis=(3, 4))
    energy = energy_group.mean(axis=0)
    draws = _resample(weights, energy_group)
    fixed, best = _off_target(energy)
    fixed_draws, best_draws = _off_target(draws)
    ordered = np.sort(energy, axis=-2)
    second = np.sqrt(ordered[:, -2, :])
    second_draws = np.sqrt(np.sort(draws, axis=-2)[..., -2, :])
    answer = dict(
        groups=len(scaled), doses=DOSES.tolist(),
        dose_scaled_mean=scaled.mean(axis=(0, 3, 4)).tolist(),
        dose_scaled_rms=np.sqrt(energy).tolist(),
        fixed_off_target=_interval(fixed, fixed_draws),
        best_permutation_off_target=_interval(best, best_draws),
        zero_energy_doses=(energy.sum(axis=(-2, -1)) == 0).tolist(),
        second_rms=second.tolist(),
        second_rms_original_units=(second * abs(DOSES[:, None])).tolist(),
        second_to_first=_fraction(second, np.sqrt(ordered[:, -1, :])).tolist(),
        second_energy_share=_fraction(ordered[:, -2, :], energy.sum(axis=-2)).tolist(),
        second_rms_min_by_dose=_interval(second.min(axis=-1), second_draws.min(axis=-1)),
        second_rms_min=_interval(second.min(), second_draws.min(axis=(-2, -1))),
    )
    if mapped:
        squared = _rmse_squared(scaled)
        answer['relative_rmse'] = _interval(np.sqrt(squared.mean(axis=0)),
                                            np.sqrt(_resample(weights, squared)))
    return answer


def comparison(changes_a, changes_b, seed):
    """Report paired relative physical-unit delta RMSE differences, a minus b."""
    a, b = _scaled(changes_a), _scaled(changes_b)
    if a.shape != b.shape:
        raise ValueError('paired changes must have the same shape and panel order')
    a, b = _rmse_squared(a), _rmse_squared(b)
    weights = _weights(len(a), seed)
    estimate = np.sqrt(a.mean(axis=0)) - np.sqrt(b.mean(axis=0))
    draws = np.sqrt(_resample(weights, a)) - np.sqrt(_resample(weights, b))
    return _interval(estimate, draws)
