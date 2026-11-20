#!/usr/bin/env python3
"""evidence_tier: exploratory_numerical_sandbox

Vectorized NumPy mechanism sandbox, not a neural-network experiment.
Harmonic fitting uses independent lag-one pairs; phase fitting uses independent
stationary trajectories. No archived experiment module is imported or executed.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TIER = "exploratory_numerical_sandbox"
SEED = 2026111601
HADAMARD = np.array([[1, 1, 1, 1], [1, -1, 1, -1],
                     [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float) / 2
PERMS = np.array(list(itertools.permutations(range(4))))
LAGS = (4, 16, 32)


def observe(z):
    q = z.copy()
    q[..., 0] += .55*np.sin(1.5*z[..., 1])
    q[..., 1] += .4*(z[..., 2]**2-1)
    q[..., 2] += .55*np.sin(1.5*z[..., 3])
    return q @ HADAMARD


def poly2(x, hermite=False):
    terms = [x]
    terms += [(x[:, i]*x[:, j] - (1. if i == j and hermite else 0.))[:, None]
              / (math.sqrt(2) if i == j and hermite else 1.)
              for i in range(4) for j in range(i, 4)]
    return np.concatenate(terms, axis=1)


def invroot(c):
    val, vec = np.linalg.eigh((c+c.T)/2)
    cutoff = max(float(val[-1])*1e-10, 1e-12)
    if val[0] <= cutoff:
        raise ValueError(f"Covariance too ill-conditioned for unregularized fit: {val[0]}")
    return (vec/np.sqrt(val))@vec.T


def fit_pair(left, right):
    n = len(left)
    center = (left.mean(0)+right.mean(0))/2
    c0 = (left.T@left+right.T@right)/(2*n)-np.outer(center, center)
    lag = (left.T@right+right.T@left)/(2*n)-np.outer(center, center)
    white = invroot(c0)
    val, vec = np.linalg.eigh(white@lag@white)
    return center, white@vec[:, ::-1], val[::-1]


def function_metrics(u, z):
    u = u-u.mean(0)
    slow = z[:, :2]-z[:, :2].mean(0)
    cz = slow.T@slow/len(z)
    cu = u.T@u/len(z)
    cross = slow.T@u/len(z)
    singular = np.linalg.svd(invroot(cz)@cross@invroot(cu), compute_uv=False)
    harmonic = (z[:, 0]**2-1)/math.sqrt(2)
    harmonic -= harmonic.mean()
    hc = u.T@harmonic/len(z)
    pollution = float(hc@np.linalg.solve(cu, hc)/(harmonic@harmonic/len(z)))
    r2 = float(np.trace(cross@np.linalg.solve(cu, cross.T))/np.trace(cz))
    return {"function_subspace_sin_theta_max": math.sqrt(max(0., 1-float(singular[-1])**2)),
            "slow_multivariate_linear_regression_R2": r2,
            "harmonic_H2_projection_R2": pollution,
            "canonical_correlations": singular.tolist()}


def harmonic_experiment(rng):
    cases = [(f"rho1_{r:g}", [.95, r, .86, .82])
             for r in (.87, .885, .895, .9025, .91, .925, .94)]
    cases += [("separated_wide", [.93, .918, .892, .88]),
              ("separated_narrow", [.93, .906, .904, .88])]
    ns = [1024, 4096, 65536]
    qa, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    audit_z = rng.normal(size=(16384, 4))
    audit_e = rng.normal(size=audit_z.shape)
    rows = []
    for case, rates in cases:
        rho = np.array(rates)
        left = rng.normal(size=(max(ns), 4))
        right = rho*left+np.sqrt(1-rho**2)*rng.normal(size=left.shape)
        eleft = rng.normal(size=left.shape)
        eright = .5*eleft+math.sqrt(.75)*rng.normal(size=left.shape)
        features = [("raw_linear_sfa", observe(left), observe(right), observe(audit_z)),
                    ("poly2_observed_sfa", poly2(observe(left)), poly2(observe(right)), poly2(observe(audit_z))),
                    ("oracle_latent_hermite_dictionary", poly2(left, True), poly2(right, True), poly2(audit_z, True))]
        for power in (1., .99, .95):
            features.append((f"identifiable_power_{power:g}",
                             (math.sqrt(power)*left+math.sqrt(1-power)*eleft)@qa.T,
                             (math.sqrt(power)*right+math.sqrt(1-power)*eright)@qa.T,
                             (math.sqrt(power)*audit_z+math.sqrt(1-power)*audit_e)@qa.T))
        for n in ns:
            for label, fl, fr, fa in features:
                center, basis, values = fit_pair(fl[:n], fr[:n])
                m = function_metrics((fa-center)@basis[:, :2], audit_z)
                row = {"case": case, "rho": rates, "N_independent_lag1_pairs": n,
                       "method": label, "feature_dimension": int(fl.shape[1]),
                       "calibration_eigenvalues_top3": values[:3].tolist(), **m}
                if label.startswith("identifiable_power_"):
                    power = float(label.rsplit("_", 1)[1])
                    direct = fa@qa
                    row["signal_power_parameter"] = power
                    row["oracle_inverse_rotation_direct_state_R2_by_axis"] = (
                        1-((direct-audit_z)**2).mean(0)/audit_z.var(0)).tolist()
                    row["population_direct_state_R2"] = 2*math.sqrt(power)-1
                    row["population_optimal_linear_readout_R2"] = power
                rows.append(row)
    return {"design": {"rho_cases": dict(cases), "independent_pair_budgets": ns,
                       "calibration_lag": 1, "audit_independent_states": len(audit_z),
                       "residual_AR1_rho": .5, "calibration_replicates": 1,
                       "N_is_trajectory_count": False,
                       "pair_interpretation": "Each pair is an independent stationary draw of (Z_t,Z_t+1); this is not one long trajectory.",
                       "ground_truth_use": "Only oracle dictionary and evaluation use Z; raw/poly2/identifiable SFA fitting is unlabeled.",
                       "poly2_boundary_warning": "Poly2 of nonlinear observations does not exactly contain Z0 or H2(Z0); rho0^2 boundary is exact only for the oracle latent Hermite dictionary.",
                       "error_parameter_warning": "signal_power is not the code MSE-based R2; both actual audit and analytic direct/optimal R2 are reported."},
            "rows": rows,
            "oracle_population": [{"case": name, "second_linear_eigenvalue": r[1],
                                    "leading_quadratic_eigenvalue": r[0]**2,
                                    "population_second_mode": "tie" if abs(r[1]-r[0]**2)<1e-12 else
                                    ("H2_Z0" if r[1]<r[0]**2 else "Z1")}
                                   for name, r in cases]}


def stationary_paths(rng, rates, n, length):
    """Vectorized exact stationary AR recursion via cumulative geometric sum."""
    rates = np.asarray(rates)
    draws = rng.normal(size=(n, length, len(rates)))
    draws[:, 1:] *= np.sqrt(1-rates**2)
    powers = rates[None, :]**np.arange(length)[:, None]
    return np.cumsum(draws/powers[None], axis=1)*powers[None]


def centered_moments(values, lags):
    mean = values.mean(axis=(0, 1))
    flat = values.reshape(-1, values.shape[-1])
    zero = flat.T@flat/len(flat)-np.outer(mean, mean)
    lagged = []
    for lag in lags:
        left = values[:, :-lag].reshape(-1, values.shape[-1])
        right = values[:, lag:].reshape(-1, values.shape[-1])
        raw = left.T@right/len(left)
        endpoints = (left.mean(0)+right.mean(0))/2
        lagged.append((raw+raw.T)/2-np.outer(endpoints, mean)-np.outer(mean, endpoints)+np.outer(mean, mean))
    return mean, zero, lagged


def window_grams(values):
    result = np.zeros((len(values), values.shape[-1], values.shape[-1]))
    for lag in LAGS:
        delta = values[:, lag:]-values[:, :-lag]
        result += (delta.swapaxes(1, 2)@delta)/(values.shape[1]-lag)/len(LAGS)
    return result


def top2(scores):
    return np.sort(np.argsort(scores, axis=1, kind="stable")[:, :2], axis=1)


def rank(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    _, first, count = np.unique(values[order], return_index=True, return_counts=True)
    for a, n in zip(first, count):
        ranks[order[a:a+n]] = a+(n-1)/2
    return ranks


def spearman(a, b):
    aa, bb = rank(a), rank(b)
    if aa.std() == 0 or bb.std() == 0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def phase_experiment(rng):
    budgets, length, testn, reps = (128, 1024), 193, 2048, 3
    powers = [1-e for e in (.001, .005, .01, .02, .05, .10, .20)]
    gaps, clags = (.002, .006, .014, .026, .050), (1, 4, 16)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    ref = rng.normal(size=(32768, 8))
    ref_cov = np.cov(ref, rowvar=False, ddof=0)
    audit = rng.normal(size=(16384, 8))
    audit_raw = audit.T@audit/len(audit)
    audit_mean = audit.mean(0)
    truth_var = audit[:, :4].var(0)
    rows = []
    for gap in gaps:
        rho = np.array([.95, .905+gap/2, .905-gap/2, .85])
        ztest = stationary_paths(rng, rho, testn, length)
        for residual in ("isotropic_ar05", "slow_noise_on_fast_axes"):
            kappa = np.array([.5]*4 if residual == "isotropic_ar05" else [.5, .5, .94, .94])
            etest = stationary_paths(rng, kappa, testn, length)
            test_gram = window_grams(np.concatenate((ztest, etest), axis=2))
            true_scores = np.diagonal(test_gram[:, :4, :4], axis1=1, axis2=2)
            true_top = top2(true_scores)
            for rep in range(reps):
                zcal = stationary_paths(rng, rho, max(budgets), length)
                ecal = stationary_paths(rng, kappa, max(budgets), length)
                joint = np.concatenate((zcal, ecal), axis=2)
                for ncal in budgets:
                    mean8, zero8, lag8 = centered_moments(joint[:ncal], clags)
                    for power in powers:
                        mixing = np.vstack((math.sqrt(power)*np.eye(4), math.sqrt(1-power)*np.eye(4)))
                        mean = mean8@mixing
                        zero = mixing.T@zero8@mixing
                        zero_rot = rotation@zero@rotation.T
                        white_rot = invroot(zero_rot)
                        direct = math.sqrt(power)*audit[:, :4]+math.sqrt(1-power)*audit[:, 4:]
                        input_r2 = 1-((direct-audit[:, :4])**2).mean(0)/truth_var
                        for lag, mat8 in zip(clags, lag8):
                            lag_phys = mixing.T@mat8@mixing
                            chat_rot = white_rot@rotation@lag_phys@rotation.T@white_rot
                            val, vec = np.linalg.eigh(chat_rot)
                            basis = rotation.T@white_rot@vec[:, ::-1]
                            qsub, _ = np.linalg.qr(basis[:, :2])
                            sv = np.linalg.svd(qsub[:2], compute_uv=False)
                            angle = math.sqrt(max(0., 1-float(sv[-1])**2))
                            function_angle = math.sqrt(max(0., 1-power*float(sv[-1])**2))
                            normalized = np.abs(basis)/np.linalg.norm(basis, axis=0)
                            utility = normalized[np.arange(4)[None], PERMS].sum(1)
                            perm = PERMS[int(np.argmax(utility))]
                            named = basis[:, perm].copy()
                            named *= np.sign(np.diag(named))[None]
                            named /= np.linalg.norm(named, axis=0)[None]
                            map8 = mixing@named
                            ref_var = np.einsum("ai,ab,bi->i", map8, ref_cov, map8)
                            scores = np.einsum("ai,nab,bi->ni", map8, test_gram, map8, optimize=True)/ref_var
                            error = float(np.any(top2(scores) != true_top, axis=1).mean())
                            bias = -mean@named
                            diffmap = map8-np.vstack((np.eye(4), np.zeros((4, 4))))
                            mse = np.einsum("ai,ab,bi->i", diffmap, audit_raw, diffmap)+2*bias*(audit_mean@diffmap)+bias*bias
                            audit_r2 = 1-mse/truth_var
                            population = power*rho**lag+(1-power)*kappa**lag
                            pop_rot = rotation@np.diag(population)@rotation.T
                            noise = float(np.linalg.norm(chat_rot-pop_rot, ord=2))
                            clean_gap = float(power*(rho[1]**lag-rho[2]**lag))
                            contrast = float((1-power)*(kappa[2]**lag-kappa[1]**lag))
                            rows.append({"gap_rho": gap, "rho": rho.tolist(), "signal_power_parameter": power,
                                         "residual": residual, "residual_rho": kappa.tolist(), "N_cal_trajectories": ncal,
                                         "T_cal": length, "cal_lag": lag, "replicate": rep,
                                         "signal_loading_subspace_sin_theta_max": angle,
                                         "function_subspace_sin_theta_max": function_angle,
                                         "oracle_aligned_four_axis_same_window_top2_error": error,
                                         "oracle_alignment_permutation": perm.tolist(),
                                         "oracle_aligned_audit_state_R2_by_axis": audit_r2.tolist(),
                                         "oracle_inverse_rotation_input_audit_R2_by_axis": input_r2.tolist(),
                                         "population_direct_input_R2": 2*math.sqrt(power)-1,
                                         "population_lag_eigenvalues_physical_order": population.tolist(),
                                         "population_target_boundary_signed_gap": float(population[1]-population[2]),
                                         "population_slow_subspace_is_target": bool(set(np.argsort(-population)[:2]) == {0, 1}),
                                         "sample_operator_error": noise,
                                         "sampling_to_clean_gap_ratio": noise/clean_gap,
                                         "residual_contrast_to_clean_gap_ratio": abs(contrast)/clean_gap,
                                         "total_upper_perturbation_to_clean_gap_ratio": (noise+abs(contrast))/clean_gap,
                                         "noise_to_actual_signed_gap_magnitude_ratio": noise/max(abs(float(population[1]-population[2])), 1e-15)})
    groups = {}
    for row in rows:
        key = (row["signal_power_parameter"], row["gap_rho"], row["N_cal_trajectories"], row["cal_lag"], row["residual"])
        groups.setdefault(key, []).append(row)
    cells = []
    for key, group in groups.items():
        cell = dict(zip(("signal_power_parameter", "gap_rho", "N_cal_trajectories", "cal_lag", "residual"), key))
        for field in ("signal_loading_subspace_sin_theta_max", "function_subspace_sin_theta_max", "oracle_aligned_four_axis_same_window_top2_error",
                      "sampling_to_clean_gap_ratio", "residual_contrast_to_clean_gap_ratio", "total_upper_perturbation_to_clean_gap_ratio"):
            vals = [r[field] for r in group]
            cell[field+"_mean"] = float(np.mean(vals))
            cell[field+"_sd_calibration_replicates"] = float(np.std(vals, ddof=1))
        cell["population_slow_subspace_is_target"] = group[0]["population_slow_subspace_is_target"]
        cell["population_target_boundary_signed_gap"] = group[0]["population_target_boundary_signed_gap"]
        cells.append(cell)
    diagnostics = {}
    for residual in ("isotropic_ar05", "slow_noise_on_fast_axes"):
        subset = [r for r in rows if r["residual"] == residual]
        angles = [r["signal_loading_subspace_sin_theta_max"] for r in subset]
        diagnostics[residual] = {
            "replicate_cells": len(subset),
            "collapse_definition": "signal-loading sin(theta_max)>sqrt(1/2), descriptive threshold",
            "collapse_fraction": float(np.mean(np.array(angles)>math.sqrt(.5))),
            "spearman_angle_vs_sampling_ratio": spearman(angles, [r["sampling_to_clean_gap_ratio"] for r in subset]),
            "spearman_angle_vs_total_ratio": spearman(angles, [r["total_upper_perturbation_to_clean_gap_ratio"] for r in subset]),
            "spearman_task_error_vs_total_ratio": spearman([r["oracle_aligned_four_axis_same_window_top2_error"] for r in subset],
                                                         [r["total_upper_perturbation_to_clean_gap_ratio"] for r in subset]),
            "population_target_failure_fraction": float(np.mean([not r["population_slow_subspace_is_target"] for r in subset]))}
    boundaries = []
    for gap in gaps:
        for lag in clags:
            physical_gap = (.905+gap/2)**lag-(.905-gap/2)**lag
            residual_advantage = .94**lag-.5**lag
            epsilon_critical = physical_gap/(physical_gap+residual_advantage)
            boundaries.append({"gap_rho": gap, "cal_lag": lag,
                               "physical_boundary_eigengap": physical_gap,
                               "fast_residual_spectral_advantage": residual_advantage,
                               "critical_error_power_1_minus_R": epsilon_critical,
                               "critical_signal_power_R": 1-epsilon_critical,
                               "population_direct_MSE_R2_at_critical": 2*math.sqrt(1-epsilon_critical)-1})
    return {"design": {"signal_power_parameters": powers, "boundary_rho_gaps": gaps,
                       "N_cal_independent_trajectories": list(budgets), "T_cal": length,
                       "calibration_lags": list(clags), "CAL_replicates": reps,
                       "TEST_independent_trajectories_per_rho_residual": testn, "T_test": length,
                       "reference_independent_states": len(ref), "audit_independent_states": len(audit),
                       "paired_test_across_calibration_replicates": True,
                       "same_window_truth_denominator": "known physical variance 1, matching archived low-dimensional A4/B experiment",
                       "predicted_denominator": "independent REF variance ddof=0",
                       "score_lags": list(LAGS), "score_lag_weights": [1/3]*3,
                       "ground_truth_use": "No Z is used to fit whitening or lag eigenvectors. Principal-angle evaluation and signed permutation alignment use oracle Z-coordinate information. Top2 diagnosis uses all FOUR aligned eigenvectors, not only the extracted 2D representation.",
                       "subspace_metrics": "signal-loading angle removes additive-noise ceiling; function-space angle includes the ceiling sqrt(1-signal_power).",
                       "residual_design": "Independent unit-variance Gaussian AR residual. Isotropic rates all .5; slow-leak rates [.5,.5,.94,.94] in physical fast directions before a common orthogonal rotation.",
                       "population_bias": "Signed boundary gap=R*(rho1^lag-rho2^lag)+(1-R)*(kappa1^lag-kappa2^lag). Isotropic residual has zero contrast bias. Negative boundary gap is a population ordering failure regardless of CAL size."},
            "population_boundary_conditions": boundaries,
            "replicate_rows": rows, "cell_means": cells, "dimensionless_ratio_diagnostics": diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/exploratory_20261116")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    allowed = (ROOT/"results/exploratory_20261116").resolve()
    if output != allowed and not output.is_relative_to(allowed):
        parser.error("Output must be under results/exploratory_20261116")
    output.mkdir(parents=True, exist_ok=True)
    target = output/"probe1_summary.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing numerical receipt: {target}")
    start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    harmonic = harmonic_experiment(rng)
    phase = phase_experiment(rng)
    elapsed = time.perf_counter()-start
    result = {"evidence_tier": TIER, "seed": SEED, "numpy_version": np.__version__,
              "runtime_seconds": elapsed, "runtime_under_60_seconds": elapsed < 60,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "harmonic": harmonic, "subspace_phase": phase,
              "limitations": ["Synthetic Gaussian representation and oracle evaluation; no LeJEPA network is trained or evaluated.",
                              "R parameter is signal power; actual code MSE R2 is separately reported.",
                              "Poly2 observed dictionary is not the oracle latent Hermite span; an exact harmonic boundary cannot be transferred between them.",
                              "Harmonic study has one calibration draw per case/budget; nested budgets and common audit states are paired, not independent replications.",
                              "Phase has three CAL replicates; shared TEST across CAL fits cannot be counted as extra independent physical windows.",
                              "Full-axis top2 alignment uses physical identity oracle and is not a deployable unlabeled naming method.",
                              "No uniqueness/necessity theorem for identifiability or general neural-network phase transition is established."]}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"evidence_tier": TIER, "runtime_seconds": elapsed,
                      "harmonic_cells": len(harmonic["rows"]), "phase_replicate_cells": len(phase["replicate_rows"]),
                      "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
