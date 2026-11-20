"""evidence_tier: exploratory_numerical_sandbox

NumPy-only helpers for the complete 800-cell slow-factor phase grid.
No frozen experiment file is modified. Monte Carlo rows are independent full
stationary paths; time prefixes and readout-noise budgets are deliberately paired.
The main target uses the archived low-dimensional physical variance of one.
An additional target uses the same independent physical REF estimate, as allowed
by the generic metrics API. Predictions always use their own ddof=0 REF variance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np


EVIDENCE_TIER = "exploratory_numerical_sandbox"
ROOT = Path(__file__).resolve().parents[2]
LAGS = (4, 16, 32)
LENGTHS = (65, 97, 129, 193, 289, 385, 577, 769, 1153, 1537)
R2_GRID = (.80, .85, .90, .95, .97, .985, .995, 1.)
ALPHAS = (.25, .5, 1., 2., 4.)
N_CAL, N_TEST, N_REF = 2048, 4096, 32768


def top2_gap(scores):
    """Exactly metrics.py's stable top-2 and nonnegative third-minus-second gap."""
    order = np.argsort(scores, axis=1, kind="stable")
    ordered = np.take_along_axis(scores, order, axis=1)
    return np.sort(order[:, :2], axis=1), ordered[:, 2] - ordered[:, 1]


def wilson_interval(errors, count, z=1.959963984540054):
    """Pointwise approximate 95% interval; never a simultaneous grid certificate."""
    if count == 0:
        return [0., 1.]
    p = errors / count
    den = 1 + z*z/count
    mid = (p + z*z/(2*count)) / den
    rad = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / den
    return [max(0., mid-rad), min(1., mid+rad)]


def describe(wrong, accept=None, threshold=None):
    if accept is None:
        accept = np.ones(len(wrong), dtype=bool)
    accepted = int(np.count_nonzero(accept))
    errors = int(np.count_nonzero(wrong & accept))
    result = {"n": len(wrong), "accepted": accepted, "errors": errors,
              "coverage": accepted/len(wrong),
              "risk": errors/accepted if accepted else None,
              "ci95_wilson": wilson_interval(errors, accepted),
              "interval_scope": "pointwise_approximate_not_simultaneous"}
    if threshold is not None:
        result["threshold"] = float(threshold)
    return result


def path_sufficient_stats(rng, rho, count, lengths=LENGTHS, lags=LAGS):
    """Stream full vectorized path rows through a 33-frame ring buffer.

    Return for every requested prefix T the three exact weighted numerators
    S(z), 2<z,e>_Delta, S(e), with e independent iid standard Gaussian. No loop
    over Monte Carlo rows occurs. Only time and the three scoring lags are looped.
    """
    rho = np.asarray(rho, dtype=np.float64)
    dimension = len(rho)
    ring_size = max(lags) + 1
    z = np.empty((ring_size, count, dimension), dtype=np.float64)
    e = np.empty_like(z)
    innovation = np.empty((count, dimension), dtype=np.float64)
    dz, de, scratch = np.empty_like(innovation), np.empty_like(innovation), np.empty_like(innovation)
    sums_z = np.zeros((len(lags), count, dimension), dtype=np.float64)
    sums_cross, sums_e = np.zeros_like(sums_z), np.zeros_like(sums_z)
    scale = np.sqrt(1-rho*rho)
    checkpoints = set(lengths)
    output = {}
    for t in range(max(lengths)):
        slot = t % ring_size
        current_z, current_e = z[slot], e[slot]
        if t == 0:
            rng.standard_normal(out=current_z)
        else:
            rng.standard_normal(out=innovation)
            np.multiply(z[(t-1) % ring_size], rho, out=current_z)
            np.multiply(innovation, scale, out=scratch)
            current_z += scratch
        rng.standard_normal(out=current_e)
        for k, h in enumerate(lags):
            if t < h:
                continue
            np.subtract(current_z, z[(t-h) % ring_size], out=dz)
            np.subtract(current_e, e[(t-h) % ring_size], out=de)
            np.multiply(dz, dz, out=scratch)
            sums_z[k] += scratch
            np.multiply(dz, de, out=scratch)
            scratch *= 2
            sums_cross[k] += scratch
            np.multiply(de, de, out=scratch)
            sums_e[k] += scratch
        length = t + 1
        if length in checkpoints:
            weights = 1. / (len(lags) * (length-np.asarray(lags)))
            output[length] = tuple(np.einsum("h,hnd->nd", weights, values)
                                   for values in (sums_z, sums_cross, sums_e))
    return output


def gaussian_variance_coefficients(rho, length, lags=LAGS):
    """Exact Var(S(z+sqrt(v)e)) = a + b*v + c*v**2.

    This finite lag-pair covariance sum uses O(T |L|^2) work and O(T) memory.
    Each offset has its exact endpoint multiplicity. There is no dense T-by-T
    matrix and no assumption that different lag scores are independent.
    """
    coefficients = np.zeros(3, dtype=np.float64)
    for h in lags:
        nh = length-h
        for ell in lags:
            nl = length-ell
            u = np.arange(-(nl-1), nh)
            multiplicity = np.minimum(nh-1, u+nl-1) - np.maximum(0, u) + 1
            signal = (rho**np.abs(u+h-ell) - rho**np.abs(u+h)
                      - rho**np.abs(u-ell) + rho**np.abs(u))
            noise = ((u+h-ell == 0).astype(float) - (u+h == 0)
                     - (u-ell == 0) + (u == 0))
            weight = 2*multiplicity/(len(lags)**2*nh*nl)
            coefficients[0] += np.sum(weight*signal*signal)
            coefficients[1] += np.sum(weight*2*signal*noise)
            coefficients[2] += np.sum(weight*noise*noise)
    return coefficients


def analytic_cell(rho, length, residual_variance, predicted_ref, coefficients=None):
    """Population-normalized predicted Gaussian moments plus fixed-REF moments.

    This is a population boundary-pair diagnostic, not a same-window risk law.
    At residual variance v, each output has variance 1+v, and its normalized
    mean is (mu_i+2v)/(1+v). The physical gap and normalized predicted gap are
    both reported to prevent a hidden normalization mismatch.
    """
    rho = np.asarray(rho, dtype=float)
    if coefficients is None:
        coefficients = np.array([gaussian_variance_coefficients(r, length) for r in rho])
    v = residual_variance
    physical_means = np.mean([2*(1-rho**h) for h in LAGS], axis=0)
    numerator_means = physical_means + 2*v
    numerator_variances = coefficients @ np.array([1., v, v*v])
    means = numerator_means/(1+v)
    variances = numerator_variances/(1+v)**2
    gap = float(means[2]-means[1])
    sigma = math.sqrt(float(variances[1]+variances[2]))
    conditional_means = numerator_means/predicted_ref
    conditional_vars = numerator_variances/predicted_ref**2
    cond_gap = float(conditional_means[2]-conditional_means[1])
    cond_sigma = math.sqrt(float(conditional_vars[1]+conditional_vars[2]))
    physical_gap = float(physical_means[2]-physical_means[1])
    return {"delta_mu": gap, "physical_delta_mu": physical_gap,
            "sigma_diff": sigma, "snr_eff": gap/sigma,
            "snr_requested_physical_gap_over_predicted_sigma": physical_gap/sigma,
            "snr_definition": "normalized predicted population mean gap / normalized predicted score-difference SD",
            "predicted_population_means": means.tolist(),
            "predicted_population_variances": variances.tolist(),
            "physical_score_variances": coefficients[:, 0].tolist(),
            "normal_population_pair_proxy": .5*math.erfc(gap/sigma/math.sqrt(2)),
            "conditioned_on_ref": {"delta_mu": cond_gap, "sigma_diff": cond_sigma,
                "snr_eff": cond_gap/cond_sigma,
                "normal_population_pair_proxy": .5*math.erfc(cond_gap/cond_sigma/math.sqrt(2))}}


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.nonzero(sorted_values[1:] != sorted_values[:-1])[0]+1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=float)
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = (a+b-1)/2
    return ranks


def spearman(x, y):
    a, b = average_ranks(x), average_ranks(y)
    a -= a.mean()
    b -= b.mean()
    denominator = math.sqrt(float(a@a)*float(b@b))
    return float(a@b/denominator) if denominator else None


def minimum_window_table(cells):
    """First observed crossing on this grid, with censoring and nonmonotonicity."""
    keys = sorted({(c["process"], c["alpha"], c["r2_population"]) for c in cells})
    output = []
    for process, alpha, r2 in keys:
        selected = sorted([c for c in cells if (c["process"], c["alpha"], c["r2_population"]) ==
                           (process, alpha, r2)], key=lambda c: c["T"])
        row = {"process": process, "alpha": alpha, "r2_population": r2}
        for rule in ("all", "own_cal"):
            below = [c[rule]["risk"] is not None and c[rule]["risk"] < .01 for c in selected]
            observed = next((c["T"] for c, yes in zip(selected, below) if yes), None)
            pointwise = next((c["T"] for c in selected if c[rule]["ci95_wilson"][1] < .01), None)
            row[rule] = {"first_observed_below_1pct": observed,
                         "first_pointwise_wilson_upper_below_1pct": pointwise,
                         "right_censored_at": max(LENGTHS) if observed is None else None,
                         "below_then_above_on_grid": any(below[a] and not below[b]
                                                        for a in range(len(below)) for b in range(a+1, len(below))),
                         "interpretation": "observed_grid_crossing_not_a_monotone_or_certified_T_min_law"}
        output.append(row)
    return output


def diagnostics(cells):
    snr = [c["snr_eff"] for c in cells]
    requested_snr = [c["snr_requested_physical_gap_over_predicted_sigma"] for c in cells]
    all_risk = [c["all"]["risk"] for c in cells]
    own_risk = [c["own_cal"]["risk"] for c in cells]
    population_proxy = np.array([c["normal_population_pair_proxy"] for c in cells])
    population_observed = np.array([c["population_pair_inversion_observed"] for c in cells])
    conditional_proxy = np.array([c["conditioned_on_ref"]["normal_population_pair_proxy"] for c in cells])
    return {"spearman_snr_vs_same_window_all_risk": spearman(snr, all_risk),
            "spearman_snr_vs_same_window_own_cal_risk": spearman(snr, own_risk),
            "spearman_snr_vs_fixed_coverage": spearman(snr, [c["fixed"]["coverage"] for c in cells]),
            "spearman_requested_physical_gap_snr_vs_same_window_all_risk": spearman(requested_snr, all_risk),
            "spearman_requested_physical_gap_snr_vs_same_window_own_cal_risk": spearman(requested_snr, own_risk),
            "spearman_requested_physical_gap_snr_vs_fixed_coverage": spearman(requested_snr, [c["fixed"]["coverage"] for c in cells]),
            "normal_proxy_population_pair_mae": float(np.mean(np.abs(population_proxy-population_observed))),
            "normal_proxy_conditioned_ref_population_pair_mae": float(np.mean(np.abs(conditional_proxy-population_observed))),
            "normal_proxy_vs_same_window_risk_target_mismatch_mae": float(np.mean(np.abs(population_proxy-all_risk))),
            "normal_approximation_scope": "proxy is for fixed population pair inversion, not same-window top2 mismatch",
            "paired_grid_scope": "800 cells share path prefixes, residual noise and REF draws; correlations are descriptive",
            "by_r2": [{"r2_population": r2,
                       "spearman_snr_vs_all_risk": spearman([c["snr_eff"] for c in cells if c["r2_population"] == r2],
                                                          [c["all"]["risk"] for c in cells if c["r2_population"] == r2])}
                      for r2 in R2_GRID]}


def run_phase(seed=20261116):
    start = time.perf_counter()
    config_path = ROOT/"reports/lejepa_b_transfer_20261115/config.json"
    cfg = json.loads(config_path.read_text())
    fixed_threshold = cfg["fixed_threshold"]
    count = N_CAL + N_TEST
    cells = []
    condition_times = []
    for process_number, (process, rates) in enumerate(cfg["processes"].items()):
        # REF is independent of every CAL/TEST path; paired across R2,T,alpha.
        ref_rng = np.random.default_rng(np.random.SeedSequence([seed, 2101, process_number]))
        ref_z, ref_e = ref_rng.standard_normal((2, N_REF, 4))
        ref_noise_mean_square = np.mean(ref_e*ref_e, axis=0)
        ref_z -= ref_z.mean(axis=0)
        ref_e -= ref_e.mean(axis=0)
        var_z, var_e = np.mean(ref_z*ref_z, axis=0), np.mean(ref_e*ref_e, axis=0)
        cov_ze = np.mean(ref_z*ref_e, axis=0)
        empirical_r2 = {r2: (1-(1-r2)*ref_noise_mean_square/var_z).tolist() for r2 in R2_GRID}
        for alpha in ALPHAS:
            condition_start = time.perf_counter()
            rho = np.asarray(rates)**alpha
            # Reset same innovations for paired alpha interventions, separate processes.
            rng = np.random.default_rng(np.random.SeedSequence([seed, 2102, process_number]))
            sufficient = path_sufficient_stats(rng, rho, count)
            for length in LENGTHS:
                signal, cross, noise = sufficient[length]
                physical_top, _ = top2_gap(signal[N_CAL:])
                physical_ref_top, _ = top2_gap(signal[N_CAL:]/var_z)
                coeff = np.array([gaussian_variance_coefficients(r, length) for r in rho])
                for r2 in R2_GRID:
                    v = 1-r2
                    predicted_ref = var_z + 2*math.sqrt(v)*cov_ze + v*var_e
                    predicted = (signal + math.sqrt(v)*cross + v*noise)/predicted_ref
                    _, cal_gap = top2_gap(predicted[:N_CAL])
                    pred_top, gap = top2_gap(predicted[N_CAL:])
                    threshold = float(np.quantile(cal_gap, .5, method="higher"))
                    wrong = np.any(pred_top != physical_top, axis=1)
                    ref_wrong = np.any(pred_top != physical_ref_top, axis=1)
                    own_accept, fixed_accept = gap >= threshold, gap >= fixed_threshold
                    analytic = analytic_cell(rho, length, v, predicted_ref, coeff)
                    row = {"process": process, "alpha": alpha, "T": length,
                           "r2_population": r2, "r2_ref_empirical_by_axis": empirical_r2[r2],
                           "rho": rho.tolist(), "residual_variance": v,
                           "physical_reference_variance": [1.]*4,
                           "predicted_reference_variance": predicted_ref.tolist(),
                           "empirical_physical_reference_variance": var_z.tolist(),
                           **analytic,
                           "all": describe(wrong),
                           "own_cal": describe(wrong, own_accept, threshold),
                           "fixed": describe(wrong, fixed_accept, fixed_threshold),
                           "predicted_gap_median": float(np.median(gap)),
                           "population_pair_inversion_observed": float(np.mean(predicted[N_CAL:, 2] <= predicted[N_CAL:, 1])),
                           "empirical_physical_ref_sensitivity": {
                               "all": describe(ref_wrong),
                               "own_cal": describe(ref_wrong, own_accept, threshold),
                               "fixed": describe(ref_wrong, fixed_accept, fixed_threshold)}}
                    if r2 == 1.:
                        assert row["empirical_physical_ref_sensitivity"]["all"]["errors"] == 0
                    cells.append(row)
            condition_times.append({"process": process, "alpha": alpha,
                                    "seconds": time.perf_counter()-condition_start})
    assert len(cells) == 800
    return {"evidence_tier": EVIDENCE_TIER, "cells": cells,
            "grid": {"processes": cfg["processes"], "alphas": list(ALPHAS),
                     "T": list(LENGTHS), "r2_population": list(R2_GRID), "cells": 800,
                     "n_cal": N_CAL, "n_test": N_TEST, "n_ref": N_REF, "lags": list(LAGS)},
            "design": {"seed": seed, "residual": "independent iid Gaussian, zhat=z+sqrt(1-R2)*e",
                       "r2_semantics": "population 1-E[(zhat-z)^2]/Var(z), not signal-fraction parameter",
                       "physical_truth": "known unit variance, exactly archived low-dimensional target",
                       "secondary_truth": "physical REF ddof0 variance, shared reference states",
                       "cal_threshold": "per-cell CAL gap median, np.quantile(method='higher')",
                       "fixed_threshold": fixed_threshold,
                       "paired_reuse": "independent path rows; nested T prefixes; shared innovations across alpha and R2; one independent REF per process",
                       "analytic_snr": "population-normalized predicted boundary pair; all Gaussian AR + iid noise covariance terms exact; finite REF conditional version also reported",
                       "uncertainty": "pointwise approximate Wilson95, no simultaneous post-grid certificate"},
            "t_min": minimum_window_table(cells), "diagnostics": diagnostics(cells),
            "condition_timings": condition_times, "run_elapsed_seconds": time.perf_counter()-start}
