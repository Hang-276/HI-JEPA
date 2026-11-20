"""Reference and window bounds for the entire monotone periodic lift class."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.special import ndtr


PRIOR_PATH = Path(__file__).resolve().parent.parent / 'dose_window_20261109' / 'run.py'
_spec = importlib.util.spec_from_file_location('finite_reference_phase_bounds', PRIOR_PATH)
_prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prior)
LAGS = (4, 16, 32)
ROUNDING = 1e-10


def reference_bands(reference, ell=.35, radius=3., alpha=.01):
    """Return uniform physical variance bounds from an independent reference.

    Rows are independent, with four named coordinates. Only coordinate zero
    permits a monotone periodic lift. The probability event uses four DKW
    bands. The known normal tail envelope supplies the clipping error.
    """
    reference = np.asarray(reference, dtype=np.float64)
    if (reference.ndim != 2 or reference.shape[1] != 4 or len(reference) < 2
            or not np.isfinite(reference).all() or not 0 < ell < 2
            or not radius > 0 or not 0 < alpha < 1):
        raise ValueError('Expected finite independent four-coordinate reference rows and valid constants')
    clipped = np.clip(reference, -radius, radius)
    centered = clipped - clipped.mean(axis=0)
    empirical = np.mean(centered**2, axis=0)
    weights = centered[:, 0][None, :] / len(clipped)
    cov_low, cov_high = _prior.cross_extrema(clipped[:, 0][None, :], weights, ell)
    empirical_low, empirical_high = empirical.copy(), empirical.copy()
    empirical_low[0] += 2 * ell * cov_low[0]
    empirical_high[0] += 2 * ell * cov_high[0] + ell**2 / 4
    displacement = np.array([ell, 0., 0., 0.])
    cdf_radius = np.sqrt(np.log(8 / alpha) / (2 * len(reference)))
    span = 2 * radius + displacement
    generic_error = span**2 * cdf_radius
    center_offset = np.abs(clipped.mean(axis=0)) + displacement
    if cdf_radius < 1:
        refined_error = cdf_radius / (1 - cdf_radius) * (
            2 * empirical_high + 2 * center_offset**2
            + 2 * span * cdf_radius * center_offset
            + span**2 * cdf_radius**2 + span**2 / 2)
        variance_error = np.minimum(generic_error, refined_error)
    else:
        variance_error = generic_error
    density = np.exp(-radius**2 / 2) / np.sqrt(2 * np.pi)
    tail_squared = 2 * ((2 * displacement - radius) * density
                        + (1 + (radius - displacement)**2) * ndtr(-radius))
    tail_l2 = np.sqrt(np.maximum(tail_squared, 0.))
    sd_low = np.maximum(np.sqrt(np.maximum(empirical_low - variance_error - ROUNDING, 0.))
                        - tail_l2 - ROUNDING, 0.)
    sd_high = np.sqrt(np.maximum(empirical_high + variance_error + ROUNDING, 0.)) + tail_l2 + ROUNDING
    audit = dict(variance_low=np.maximum(sd_low**2 - ROUNDING, 0.),
                variance_high=sd_high**2 + ROUNDING,
                sd_low=sd_low, sd_high=sd_high, cdf_radius=float(cdf_radius),
                variance_error=variance_error, tail_l2=tail_l2,
                empirical_clipped_variance=empirical,
                empirical_variance_low=empirical_low,
                empirical_variance_high=empirical_high,
                covariance_extrema=np.array([cov_low[0], cov_high[0]]))
    lower, upper = audit.pop('variance_low'), audit.pop('variance_high')
    return lower, upper, {key: value.tolist() if isinstance(value, np.ndarray) else value
                          for key, value in audit.items()}


def _maximum_threshold_energy(phase):
    """Maximize threshold-lift increment energy, including all tied events."""
    count, length = phase.shape
    events = np.zeros_like(phase)
    baseline = np.zeros(count)
    for lag in LAGS:
        difference = phase[:, lag:] - phase[:, :-lag]
        weight = 1 / (len(LAGS) * (length - lag))
        baseline += weight * np.sum(difference**2, axis=1)
        signed = weight * np.sign(difference) * (1 - 2 * np.abs(difference))
        events[:, :-lag] += signed
        events[:, lag:] -= signed
    order = np.argsort(phase, axis=1, kind='stable')
    ordered_phase = np.take_along_axis(phase, order, axis=1)
    running = np.cumsum(np.take_along_axis(events, order, axis=1), axis=1)
    group_end = np.c_[ordered_phase[:, :-1] < ordered_phase[:, 1:], np.ones(count, dtype=bool)]
    maximum = np.max(np.where(group_end, running, -np.inf), axis=1)
    return baseline + np.maximum(maximum, 0.)


def numerator_bounds(window, ell=.35):
    """Return B-by-4 lower and upper unnormalized increment energies."""
    window = np.asarray(window, dtype=np.float64)
    if (window.ndim != 3 or window.shape[-1] != 4 or window.shape[1] <= max(LAGS)
            or not np.isfinite(window).all() or not 0 < ell < 2):
        raise ValueError('Expected finite four-coordinate windows longer than all scoring lags')
    score = _prior.scores(window)
    first = window[..., 0]
    low_cross, high_cross = _prior.cross_extrema(first, _prior.cross_weights(first), ell)
    energy_max = _maximum_threshold_energy(np.remainder(first / ell, 1.))
    low, high = score.copy(), score.copy()
    low[:, 0] += 2 * ell * low_cross
    high[:, 0] += 2 * ell * high_cross + ell**2 * energy_max
    return np.maximum(low - ROUNDING, 0.), high + ROUNDING


def certify(low_numerator, high_numerator, low_variance, high_variance, selection):
    """Certify a supplied named top-2 using uniform positive score intervals."""
    low_numerator, high_numerator = (np.asarray(x, dtype=float)
                                     for x in (low_numerator, high_numerator))
    low_variance, high_variance = (np.asarray(x, dtype=float)
                                   for x in (low_variance, high_variance))
    selection = np.asarray(selection, dtype=int)
    if (low_numerator.ndim != 2 or low_numerator.shape[1] != 4
            or high_numerator.shape != low_numerator.shape
            or low_variance.shape != (4,) or high_variance.shape != (4,)
            or selection.shape != (len(low_numerator), 2)
            or np.any(low_numerator < 0) or np.any(low_variance < 0)
            or np.any(high_variance <= 0) or np.any(high_numerator < low_numerator)
            or np.any(high_variance < low_variance)):
        raise ValueError('Invalid score bounds or named top-2')
    lower = low_numerator / high_variance
    upper = np.divide(high_numerator, low_variance,
                      out=np.full_like(high_numerator, np.inf), where=low_variance > 0)
    mask = np.any(np.arange(4)[None, None, :] == selection[:, :, None], axis=1)
    valid_selection = ((selection >= 0) & (selection < 4)).all(axis=1) & (selection[:, 0] != selection[:, 1])
    gap = np.min(np.where(mask, np.inf, lower), axis=1) - np.max(np.where(mask, upper, -np.inf), axis=1)
    return valid_selection & (gap > ROUNDING)


def self_check():
    """Verify the event sweep and moment envelopes on fixed arithmetic paths."""
    time = np.arange(65)
    first = np.stack([np.sin(time * .31), np.cos(time * .17), (time % 7) / 20])
    phase = np.remainder(first / .35, 1.)
    fast = _maximum_threshold_energy(phase)
    exhaustive = []
    for row in phase:
        thresholds = np.unique(np.r_[0., row, 1.])
        residual = (row[None, :] >= thresholds[:, None]) - row
        exhaustive.append(_prior.scores(residual[..., None])[:, 0].max())
    sweep_error = float(np.max(np.abs(fast - exhaustive)))
    reference = np.stack([np.linspace(-3.7, 3.7, 101), np.sin(np.arange(101)),
                          np.cos(np.arange(101)), np.linspace(-2., 2., 101)], axis=1)
    _, _, bands = reference_bands(reference)
    clipped = np.clip(reference[:, 0], -3., 3.)
    coordinates = np.stack([first, np.cos(first), np.sin(first), first / 3], axis=-1)
    low, high = numerator_bounds(coordinates)
    moment_violation, numerator_violation = 0., 0.
    for kind in ('sine', 'ramp'):
        def lift(x):
            phase = np.remainder(x / .35, 1.)
            residual = (.8 * np.sin(2 * np.pi * phase) / (2 * np.pi) if kind == 'sine'
                        else .98 * (np.clip((phase - .49) / .02, 0., 1.) - phase))
            return x + .35 * residual
        variance = np.var(lift(clipped))
        moment_violation = max(moment_violation, bands['empirical_variance_low'][0] - variance,
                               variance - bands['empirical_variance_high'][0])
        actual = _prior.scores(lift(first)[..., None])[:, 0]
        numerator_violation = max(numerator_violation, float(np.max(low[:, 0] - actual)),
                                  float(np.max(actual - high[:, 0])))
    assert sweep_error < 1e-12 and moment_violation <= ROUNDING and numerator_violation <= ROUNDING
    return dict(threshold_sweep_error=sweep_error, moment_envelope_violation=moment_violation,
                numerator_envelope_violation=numerator_violation)
