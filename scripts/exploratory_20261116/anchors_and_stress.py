"""Exploratory read-only sealed anchors and synthetic fast-axis stress.

Only Python standard library and NumPy. Public helpers return JSON-ready dicts;
they never write files, import archived runners, load models, or read predictions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

EVIDENCE_TIER = "exploratory_numerical_sandbox"
ROOT = Path(__file__).resolve().parents[2]
LAGS = (4, 16, 32)
STRESS_RHO = np.array([.96, .95, .70, .68], dtype=float)
FAST_SIGNAL_FRACTIONS = (.90, .85, .70, .50, .20, .05, .01)
LEAK_AMPLITUDES = (0., .02, .05, .10)
NOISE_BRANCHES = (("fixed_small_noise", .02), ("fixed_noise_floor", .10))
EXTRA_SIGNAL_FRACTIONS = (.0001, 0.)


def _read(relative: str, provenance: dict) -> dict:
    path = ROOT / relative
    raw = path.read_bytes()
    provenance[relative] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return json.loads(raw)


def _audit(values) -> dict:
    x = np.asarray(values, dtype=float)
    if x.shape != (4,) or not np.isfinite(x).all():
        raise ValueError("Expected four finite sealed AUDIT R2 values")
    return {"by_axis": x.tolist(), "mean": float(x.mean()), "min": float(x.min()),
            "max": float(x.max()), "range": float(np.ptp(x))}


def _risk_record(item: dict, *, threshold=None) -> dict:
    n = int(item.get("windows", item.get("n", 0)))
    accepted = int(item.get("accepted", n))
    errors = int(item.get("errors", item.get("wrong_accepted", item.get("wrong_all", 0))))
    if not n or not 0 <= errors <= accepted <= n:
        raise ValueError(f"Malformed sealed counts: {item}")
    out = {"n": n, "accepted": accepted, "errors": errors, "coverage": accepted / n,
           "risk": errors / accepted if accepted else None}
    if threshold is not None:
        out["threshold"] = float(threshold)
    for src in ("simultaneous_error_upper", "upper"):
        if src in item:
            out["sealed_error_upper"] = float(item[src])
    for src in ("passes_both", "pass"):
        if src in item:
            out["sealed_pass"] = bool(item[src])
    return out


def _simple_phase(cell: dict) -> dict:
    return {"r2_population": float(cell["r2_population"]),
            "snr_eff": float(cell["snr_eff"]),
            "delta_mu": float(cell["delta_mu"]),
            "sigma_diff": float(cell["sigma_diff"]),
            "all_risk": float(cell["all"]["risk"]),
            "own_cal_risk": cell["own_cal"]["risk"],
            "own_cal_coverage": float(cell["own_cal"]["coverage"]),
            "own_cal_threshold": float(cell["own_cal"]["threshold"]),
            "fixed_risk": cell["fixed"]["risk"],
            "fixed_coverage": float(cell["fixed"]["coverage"]),
            "fixed_threshold": float(cell["fixed"]["threshold"])}


def _projection(anchor: dict, phase_cells: list[dict], coordinate: str) -> dict:
    candidates = sorted((c for c in phase_cells if c["process"] == anchor["process"]
                         and int(c["T"]) == 193
                         and float(c["alpha"]) == anchor["alpha"]),
                        key=lambda c: c["r2_population"])
    if not candidates:
        raise ValueError(f"Missing phase cell for {anchor['process']}, alpha={anchor['alpha']}")
    target = anchor["audit_r2"][coordinate]
    nearest = min(candidates, key=lambda c: (abs(c["r2_population"] - target), c["r2_population"]))
    nearest_record = _simple_phase(nearest)
    lower = max((c for c in candidates if c["r2_population"] <= target),
                key=lambda c: c["r2_population"], default=candidates[0])
    upper = min((c for c in candidates if c["r2_population"] >= target),
                key=lambda c: c["r2_population"], default=candidates[-1])
    rlo, rhi = float(lower["r2_population"]), float(upper["r2_population"])
    weight = 0. if rlo == rhi else min(1., max(0., (target - rlo) / (rhi - rlo)))
    lo, hi = _simple_phase(lower), _simple_phase(upper)
    interpolated = {}
    for key in lo:
        if lo[key] is None or hi[key] is None:
            interpolated[key] = None
        else:
            interpolated[key] = float((1 - weight) * lo[key] + weight * hi[key])
    differences = {"all_risk_actual_minus_nearest_pp": 100 * (anchor["all"]["risk"] - nearest_record["all_risk"]),
                   "all_risk_actual_minus_interpolated_pp": 100 * (anchor["all"]["risk"] - interpolated["all_risk"])}
    matched_rules = {}
    for key, record in anchor["rules"].items():
        if key == "fixed" or (anchor["experiment"] == "A3" and key == "g50"):
            sandbox_key = "fixed"
            comparable = math.isclose(record["threshold"], nearest_record["fixed_threshold"], abs_tol=1e-12)
            caveat = "same numeric threshold; independent synthetic CAL/REF/readout and AUDIT matching assumptions still differ"
        elif key == "recalibrated":
            sandbox_key, comparable = "own_cal", False
            caveat = "sealed threshold pools 16 model-process CAL cells per regime; sandbox uses its own condition CAL median"
        else:
            matched_rules[key] = {"sandbox_prediction": None,
                                  "reason": "phase grid has no matched 20%/80% or pair-adjusted batch-ranking rule"}
            continue
        prisk, pcov = interpolated[sandbox_key + "_risk"], interpolated[sandbox_key + "_coverage"]
        matched_rules[key] = {"phase_rule": sandbox_key, "numeric_threshold_matches": comparable,
                             "caveat": caveat, "interpolated_risk": prisk, "interpolated_coverage": pcov,
                             "risk_difference_pp": None if prisk is None or record["risk"] is None else 100*(record["risk"]-prisk),
                             "coverage_difference_pp": 100*(record["coverage"]-pcov)}
    ideal_variance = 2. - target
    ref = np.asarray(anchor["predicted_reference_variance"])
    return {"audit_coordinate": coordinate, "target_audit_r2": target,
            "nearest_grid_r2_distance": abs(nearest_record["r2_population"]-target),
            "nearest": nearest_record,
            "interpolation": {"definition": "linear descriptive interpolation between adjacent Monte Carlo phase rows; not a new simulation or a confidence interval",
                              "lower_r2": rlo, "upper_r2": rhi, "upper_weight": weight,
                              "outside_grid": target < candidates[0]["r2_population"] or target > candidates[-1]["r2_population"],
                              "prediction": interpolated},
            "coupling_exactly_matched": anchor["coupling"] == 0,
            "coupling_note": "exact diagonal dynamics" if anchor["coupling"] == 0 else "uncoupled proxy only: phase has no c=.12 covariance block",
            "ideal_population_predicted_variance": ideal_variance,
            "actual_ref_minus_ideal_variance_by_axis": (ref-ideal_variance).tolist(),
            "actual_ref_minus_ideal_variance_rms": float(np.sqrt(np.mean((ref-ideal_variance)**2))),
            "differences": differences, "selective_comparisons": matched_rules}


def _ranks(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="stable")
    out = np.empty(len(a), dtype=float)
    start = 0
    while start < len(a):
        end = start + 1
        while end < len(a) and a[order[end]] == a[order[start]]:
            end += 1
        out[order[start:end]] = (start + end - 1) / 2
        start = end
    return out


def _correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_anchors(phase_data: dict) -> dict:
    """Read sealed JSON, attach exact audit coordinates, and overlay phase rows."""
    started = time.perf_counter()
    provenance = {}
    a4 = _read("reports/lejepa_a4_20261115/SUMMARY.json", provenance)
    a4_forecast = _read("reports/lejepa_a4_20261115/NEW_FORECAST.json", provenance)
    a4_config = _read("reports/lejepa_a4_20261115/config.json", provenance)
    a3 = _read("reports/lejepa_a3_gap_20261115/SUMMARY.json", provenance)
    a3_config = _read("reports/lejepa_a3_gap_20261115/config.json", provenance)
    a2 = _read("reports/lejepa_a2_readout_20261115/SUMMARY.json", provenance)
    a2_audit = _read("reports/lejepa_a2_readout_20261115/weak/RECEIPT.json", provenance)
    a2_forecast = _read("reports/lejepa_a2_readout_20261115/FORECAST.json", provenance)
    b = _read("reports/lejepa_b_transfer_20261115/SUMMARY.json", provenance)
    b_config = _read("reports/lejepa_b_transfer_20261115/config.json", provenance)
    b_thresholds = _read("reports/lejepa_b_transfer_20261115/THRESHOLDS.json", provenance)
    refs = {(c["name"], c["process"]): c["reference_variance"] for c in a4_forecast["cells"]}
    a2_refs = {(c["name"], c["process"], c["arm"]): c["reference_variance"] for c in a2_forecast["cells"]}
    cells = []

    def base(experiment, name, process, r2, ref, all_answers, *, regime="baseline", arm="strong", alpha=1., coupling=0.):
        rates = a4_config["processes"][process]
        return {"experiment": experiment, "name": name, "process": process, "regime": regime, "arm": arm,
                "T": 193, "rho_base": rates, "alpha": float(alpha), "coupling": float(coupling),
                "rho_diagonal_effective": [float(r**alpha) for r in rates],
                "coupling_code_indices": [1, 2] if coupling else None,
                "audit_r2": _audit(r2), "predicted_reference_variance": ref,
                "physical_reference_variance": [1., 1., 1., 1.], "n_reference": a4_config["n_reference"],
                "all": _risk_record(all_answers), "rules": {}}

    for key, c in a4["by_cell"].items():
        name, process = key.split("__")
        row = base("A4", name, process, c["state_r2"], refs[name, process], c)
        row["audit_source"] = "A4 SUMMARY.by_cell.state_r2 (original new-model AUDIT)"
        row["rules"] = {rule: _risk_record({"windows": c["windows"], **record}) for rule, record in c["rules"].items()}
        row["selection_design"] = "409/2048 highest TEST scores per cell; adjusted scores use directed-pair allowances"
        cells.append(row)
    for key, c in a3["all_answers"]["by_cell"].items():
        name, process = key.split("__")
        row = base("A3", name, process, a4["by_cell"][key]["state_r2"], refs[name, process], c)
        row["audit_source"] = "A4 original AUDIT reused by frozen A3 readout; no new AUDIT inferred"
        row["rules"] = {rule: _risk_record(record["by_cell"][key], threshold=record["threshold"])
                        for rule, record in a3["thresholds"].items()}
        row["selection_design"] = "fixed thresholds from pooled original CAL, 48-claim sealed family"
        cells.append(row)
    for arm, data in a2["arms"].items():
        for key, c in data["by_cell"].items():
            name, process = key.split("__")
            r2 = a2_audit["members"][name]["audit"][arm+"_r2"]
            row = base("A2", name, process, r2, a2_refs[name, process, arm], c["all_answers"], arm=arm)
            row["audit_source"] = "A2 weak/RECEIPT.members[name].audit: original role721 AUDIT, for both arms"
            row["rules"] = {"fixed": _risk_record(c, threshold=a2["threshold"])}
            row["selection_design"] = "fixed inherited A3 g50; strong 8192/weak 64 fit labels, both 4096 validation labels"
            cells.append(row)
    for c in b["cells"]:
        regime = c["regime"]
        desc = b_config["regimes"][regime]
        row = base("B1_B3", c["name"], c["process"], b_thresholds["iid_audit_r2"][c["name"]],
                   c["reference_variance"], c["all_answers"], regime=regime,
                   alpha=desc["exponent"], coupling=desc["coupling"])
        row["audit_source"] = "B THRESHOLDS.iid_audit_r2: common fresh iid audit, stationary N(0,I) unchanged across regimes"
        row["rules"] = {rule: _risk_record(record, threshold=record["threshold"]) for rule, record in c["rules"].items()}
        row["selection_design"] = "fixed inherited g50 or regime-wide pooled CAL median across all 16 model-process cells"
        row["predicted_gap_median"] = c["predicted_gap_quantiles"][2]
        row["score_mse_by_axis"] = c["score_mse_by_axis"]
        cells.append(row)
    for c in cells:
        c["projections"] = {mode: _projection(c, phase_data["cells"], mode) for mode in ("mean", "min")}

    metrics = {}
    for mode in ("mean", "min"):
        eligible = [c for c in cells if c["coupling"] == 0]
        actual = np.array([c["all"]["risk"] for c in eligible])
        predicted = np.array([c["projections"][mode]["interpolation"]["prediction"]["all_risk"] for c in eligible])
        snr = [c["projections"][mode]["interpolation"]["prediction"]["snr_eff"] for c in eligible]
        denominator = float(np.square(actual-actual.mean()).sum())
        metrics[mode] = {"n_cells": len(eligible), "exclude_coupled": True,
                         "spearman_snr_vs_actual_error": _correlation(_ranks(snr), _ranks(actual)),
                         "spearman_interpolated_error_vs_actual_error": _correlation(_ranks(predicted), _ranks(actual)),
                         "uncalibrated_prediction_r2": None if denominator == 0 else 1-float(np.square(actual-predicted).sum())/denominator,
                         "mae_pp": 100*float(np.mean(np.abs(actual-predicted))),
                         "rmse_pp": 100*float(np.sqrt(np.mean((actual-predicted)**2))),
                         "bias_actual_minus_sandbox_pp": 100*float(np.mean(actual-predicted)),
                         "scope": "descriptive cells with shared models/trajectories, no iid inference; negative R2 allowed; no fitting to anchors"}
    aggregate = []
    groups = sorted({(c["experiment"], c["arm"], c["process"], c["regime"]) for c in cells})
    for group in groups:
        selected = [c for c in cells if (c["experiment"], c["arm"], c["process"], c["regime"]) == group]
        total = sum(c["all"]["n"] for c in selected)
        entry = {"experiment": group[0], "arm": group[1], "process": group[2], "regime": group[3],
                 "cells": len(selected), "model_evaluations": total,
                 "actual_all_risk": sum(c["all"]["errors"] for c in selected)/total,
                 "audit_mean_range": [min(c["audit_r2"]["mean"] for c in selected), max(c["audit_r2"]["mean"] for c in selected)],
                 "audit_min_range": [min(c["audit_r2"]["min"] for c in selected), max(c["audit_r2"]["min"] for c in selected)],
                 "coupling_proxy": group[3] == "coupled"}
        for mode in ("mean", "min"):
            predicted = sum(c["all"]["n"]*c["projections"][mode]["interpolation"]["prediction"]["all_risk"] for c in selected)/total
            entry[mode+"_interpolated_phase_risk"] = predicted
            entry[mode+"_difference_pp"] = 100*(entry["actual_all_risk"]-predicted)
        aggregate.append(entry)
    return {"evidence_tier": EVIDENCE_TIER, "source_kind": "read_only_sealed_neural_summaries_overlay_on_new_synthetic_phase",
            "sources": provenance, "n_cells": len(cells), "cells": cells, "aggregate": aggregate,
            "diagnostics": metrics, "timing_seconds": time.perf_counter()-started,
            "limitations": ["AUDIT mean/min collapse real anisotropic readout residuals to a scalar; phase noise is independent isotropic AR, real MLP errors need not be.",
                            "Interpolation is descriptive between existing synthetic grid cells, not evidence of an exact neural risk law or a causal residual-direction effect.",
                            "Predicted denominator equals each model's own empirical REF; physical denominator is known one, matching these sealed low-dimensional runners.",
                            "A3 g20/g80 and A4 batch-rank selections have no matched phase rule; their sealed numbers are retained without fabricating selective predictions.",
                            "B coupled uses A[1,2]=.12 and Q=I-AA^T; diagonal phase rows are explicitly only proxies.",
                            "SNR/normal population-ranking tails are not predictions of the same-window model-selection error; only synthetic same-window risks enter differences."]}


def _top_and_gap(scores: np.ndarray):
    order = np.argsort(scores, axis=1, kind="stable")
    ordered = np.take_along_axis(scores, order, axis=1)
    return np.sort(order[:, :2], axis=1), ordered[:, 2]-ordered[:, 1]


def _wilson(errors: int, n: int):
    if n == 0:
        return None
    z = 1.959963984540054
    p = errors/n
    denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0., center-half), min(1., center+half)]


def _count(wrong: np.ndarray, mask: np.ndarray | None = None):
    if mask is None:
        mask = np.ones(len(wrong), dtype=bool)
    n = int(mask.sum())
    errors = int(np.count_nonzero(wrong & mask))
    return {"n": len(wrong), "accepted": n, "errors": errors, "coverage": n/len(wrong),
            "risk": errors/n if n else None, "ci95_wilson_descriptive": _wilson(errors, n)}


def _state_samples(rng, n):
    return np.column_stack((rng.standard_normal((n, 4)), rng.standard_normal((n, 4))))


def _window_statistics(rng, n, length):
    """Linear combinations reuse exact per-window quadratic sufficient statistics."""
    z = rng.standard_normal((n, length, 4))
    scale = np.sqrt(1-STRESS_RHO**2)
    for t in range(1, length):
        z[:, t] = STRESS_RHO*z[:, t-1] + scale*z[:, t]
    noise = rng.standard_normal(z.shape)
    zz = np.zeros((n, 4)); xx = np.zeros_like(zz); zx = np.zeros_like(zz)
    zd = np.zeros((n, 2)); dx = np.zeros_like(zd)
    for lag in LAGS:
        dz = z[:, lag:]-z[:, :-lag]
        dn = noise[:, lag:]-noise[:, :-lag]
        scale = 1/(len(LAGS)*(length-lag))
        zz += np.einsum("ntd,ntd->nd", dz, dz)*scale
        xx += np.einsum("ntd,ntd->nd", dn, dn)*scale
        zx += np.einsum("ntd,ntd->nd", dz, dn)*scale
        zd += np.einsum("ntd,ntd->nd", dz[..., 2:], dz[..., :2])*scale
        dx += np.einsum("ntd,ntd->nd", dz[..., :2], dn[..., 2:])*scale
    # Small direct-array cross-check exercises this shortcut independently.
    direct_sample = (z[:7].copy(), noise[:7].copy())
    return {"zz": zz, "xx": xx, "zx": zx, "zd": zd, "dx": dx}, direct_sample


def _coefficients(signal_fraction, leakage, noise_sd):
    w = np.zeros((8, 4))
    w[0, 0] = w[1, 1] = 1.
    w[4, 0] = w[5, 1] = math.sqrt(.06)
    w[2, 2] = w[3, 3] = math.sqrt(signal_fraction)
    w[0, 2] = w[1, 3] = leakage
    w[6, 2] = w[7, 3] = noise_sd
    return w


def _numerators(stats, signal_fraction, leakage, noise_sd):
    zz, xx, zx = stats["zz"], stats["xx"], stats["zx"]
    ans = np.empty_like(zz)
    ans[:, :2] = zz[:, :2]+.06*xx[:, :2]+2*math.sqrt(.06)*zx[:, :2]
    a = math.sqrt(signal_fraction)
    ans[:, 2:] = (signal_fraction*zz[:, 2:]+leakage**2*zz[:, :2]+noise_sd**2*xx[:, 2:]
                  +2*a*leakage*stats["zd"]+2*a*noise_sd*zx[:, 2:]+2*leakage*noise_sd*stats["dx"])
    return ans


def run_stress(seed: int) -> dict:
    """Run 56 requested stress cells plus 16 separately labelled limit controls."""
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    nref, naudit, ncal, ntest, length = 32768, 32768, 2048, 4096, 193
    ref = _state_samples(rng, nref)
    audit = _state_samples(rng, naudit)
    centered = ref-ref.mean(axis=0)
    ref_cov = centered.T@centered/nref
    stats, sample = _window_statistics(rng, ncal+ntest, length)
    truth = stats["zz"]
    true_top, true_gap = _top_and_gap(truth[ncal:])
    physical_ref = ref[:, :4].var(axis=0, ddof=0)
    empirical_truth_top, _ = _top_and_gap(truth[ncal:]/physical_ref)
    mu = 2*np.mean(1-STRESS_RHO[None, :]**np.asarray(LAGS)[:, None], axis=0)
    slow_population = (mu[:2]+.12)/1.06
    slower_output_score = float(slow_population.max())
    direct_check_max = 0.
    cells = []
    for branch, noise_sd in NOISE_BRANCHES:
        for signal in FAST_SIGNAL_FRACTIONS+EXTRA_SIGNAL_FRACTIONS:
            for leak in LEAK_AMPLITUDES:
                w = _coefficients(signal, leak, noise_sd)
                variance = np.einsum("ki,kl,li->i", w, ref_cov, w)
                if np.any(variance <= 0):
                    raise ValueError("Undefined predicted REF variance: do not clamp")
                numerator = _numerators(stats, signal, leak, noise_sd)
                scores = numerator/variance
                _, cal_gap = _top_and_gap(scores[:ncal])
                pred_top, pred_gap = _top_and_gap(scores[ncal:])
                threshold = float(np.quantile(cal_gap, .5, method="higher"))
                accept = pred_gap >= threshold
                wrong = np.any(pred_top != true_top, axis=1)
                empirical_wrong = np.any(pred_top != empirical_truth_top, axis=1)
                predict_audit = audit@w
                actual_r2 = 1-np.mean((predict_audit-audit[:, :4])**2, axis=0)/audit[:, :4].var(axis=0, ddof=0)
                predicted_audit_variance = predict_audit.var(axis=0, ddof=0)
                population_r2 = np.array([.94, .94, 1-(math.sqrt(signal)-1)**2-leak**2-noise_sd**2,
                                          1-(math.sqrt(signal)-1)**2-leak**2-noise_sd**2])
                fast_variance = signal+leak**2+noise_sd**2
                population_scores = np.r_[slow_population, (signal*mu[2:]+leak**2*mu[:2]+2*noise_sd**2)/fast_variance]
                pop_top = np.argsort(population_scores, kind="stable")[:2]
                extra = signal in EXTRA_SIGNAL_FRACTIONS
                true_member = np.any(pred_top[:, :, None] == true_top[:, None, :], axis=2)
                false_fast = ((pred_top >= 2) & ~true_member).any(axis=1)
                selected_fast = (pred_top >= 2).any(axis=1)
                gate = bool(np.min(actual_r2) >= .85)
                # Compare with raw numerators only to expose normalization's role;
                # this is a deliberately different score, never the primary metric.
                raw_top, _ = _top_and_gap(numerator[ncal:])
                row = {"branch": branch, "extra_limit_control": extra, "fast_signal_fraction": signal,
                       "leakage_amplitude": leak, "noise_sd": noise_sd,
                       "audit_r2": _audit(actual_r2), "population_state_r2_by_axis": population_r2.tolist(),
                       "audit_gate_min_r2_ge_085": gate,
                       "gate_action": "admit" if gate else "reject all windows because physical-axis recovery premise fails",
                       "predicted_reference_variance": variance.tolist(),
                       "audit_predicted_variance": predicted_audit_variance.tolist(),
                       "population_predicted_variance": [1.06, 1.06, fast_variance, fast_variance],
                       "population_scores": population_scores.tolist(), "population_top2": sorted(pop_top.tolist()),
                       "population_fast_axis_enters": bool(np.any(pop_top >= 2)),
                       "fast_variance_fractions": {"physical_signal": signal/fast_variance,
                                                   "slow_leakage": leak**2/fast_variance, "white_noise": noise_sd**2/fast_variance},
                       "critical_signal_fraction_for_population_fast_entry_by_axis":
                           ((leak**2*(slower_output_score-mu[:2])-noise_sd**2*(2-slower_output_score))/(mu[2:]-slower_output_score)).tolist(),
                       "all": _count(wrong), "own_cal": {**_count(wrong, accept), "threshold": threshold},
                       "median_predicted_gap": float(np.median(pred_gap)),
                       "false_fast_inclusion": _count(false_fast),
                       "false_fast_inclusion_own_cal": _count(false_fast, accept),
                       "any_predicted_fast_axis_frequency": float(selected_fast.mean()),
                       "raw_unnormalized_score_wrong_rate_diagnostic_only": float(np.any(raw_top != true_top, axis=1).mean()),
                       "empirical_physical_ref_sensitivity": {"all": _count(empirical_wrong), "own_cal": _count(empirical_wrong, accept)}}
                cells.append(row)
                if (signal, leak) in ((.90, 0.), (.01, .1), (0., .1)):
                    small_z, small_noise = sample
                    y = np.concatenate((small_z, small_noise), axis=2)@w
                    direct = np.mean([np.mean((y[:, h:]-y[:, :-h])**2, axis=1) for h in LAGS], axis=0)
                    direct_check_max = max(direct_check_max, float(np.max(np.abs(direct-numerator[:len(y)]))))
    requested = [c for c in cells if not c["extra_limit_control"]]
    additional = [c for c in cells if c["extra_limit_control"]]
    summary = {}
    for branch, _ in NOISE_BRANCHES:
        selected = [c for c in requested if c["branch"] == branch]
        unadmitted = [c for c in selected if not c["audit_gate_min_r2_ge_085"]]
        summary[branch] = {"requested_cells": len(selected),
                           "population_fast_entry_cells": sum(c["population_fast_axis_enters"] for c in selected),
                           "all_risk_range": [min(c["all"]["risk"] for c in selected), max(c["all"]["risk"] for c in selected)],
                           "own_cal_risk_range": [min(c["own_cal"]["risk"] for c in selected), max(c["own_cal"]["risk"] for c in selected)],
                           "audit_rejected_cells": len(unadmitted),
                           "rejected_but_observed_all_risk_below_one_percent": sum(c["all"]["risk"] < .01 for c in unadmitted),
                           "rejected_but_observed_own_cal_risk_below_one_percent": sum(c["own_cal"]["risk"] < .01 for c in unadmitted),
                           "max_actual_vs_population_r2_error": max(float(np.max(np.abs(np.asarray(c["audit_r2"]["by_axis"])-c["population_state_r2_by_axis"]))) for c in selected)}
    return {"evidence_tier": EVIDENCE_TIER, "seed": int(seed), "timing_seconds": time.perf_counter()-started,
            "design": {"rho": STRESS_RHO.tolist(), "T": length, "lags": list(LAGS), "N_ref": nref, "N_audit": naudit,
                       "N_cal": ncal, "N_test": ntest, "requested_fast_signal_fractions": list(FAST_SIGNAL_FRACTIONS),
                       "requested_leak_amplitudes": list(LEAK_AMPLITUDES), "noise_branches": dict(NOISE_BRANCHES),
                       "extra_limit_signal_fractions": list(EXTRA_SIGNAL_FRACTIONS),
                       "slow_readout": "y_j=z_j+sqrt(.06)*xi_j, j=0,1; population state R2=.94",
                       "fast_readout": "y_2=sqrt(s)*z_2+leak*z_0+sigma*xi_2; y_3=sqrt(s)*z_3+leak*z_1+sigma*xi_3",
                       "noise": "independent standard Gaussian white noise for every time, axis and window, independent of physical AR trajectories",
                       "fast_population_state_r2": "1-(sqrt(s)-1)^2-leak^2-sigma^2; s is NOT code-style state R2",
                       "reference": "independent iid physical/noise states; each prediction uses own ddof=0 empirical variance; truth known unit variance",
                       "sharing": "all stress cells share REF, AUDIT, CAL and TEST states/noise for paired comparisons; CAL/TEST rows and reference/audit panels are independent",
                       "critical_signal_formula": "[leak^2*(b-mu_donor)-sigma^2*(2-b)]/(mu_fast-b), b=max population normalized slow-readout score; fast enters iff s is strictly below a positive critical value (ties use stable axis index)",
                       "intervals": "per-cell descriptive Wilson intervals, no multiple-testing certificate; not a trained H-JEPA experiment"},
            "validation": {"sufficient_statistic_direct_score_max_absolute_error": direct_check_max,
                           "passed": direct_check_max < 1e-12},
            "summary_requested_grid": summary, "requested_cells": requested, "additional_limit_controls": additional,
            "limitations": ["A state-R2 threshold .85 is an operational admission gate, neither a proven necessary nor sufficient condition for slow-set correctness.",
                            "If the requested s>=.01/leak<=.10 grid does not exhibit a population hijack, that is a negative result; additional s=.0001/0 controls cannot replace it.",
                            "Normalization is scale-invariant: slow leakage dominates only after relative variance composition changes; small absolute denominator alone does not imply hijacking.",
                            "A fixed white-noise floor may prevent slow leakage from dominating even when the physical fast signal vanishes.",
                            "Synthetic slow/fast labels and leak donors are known here; this does not establish the corresponding mechanism in Layer 2 or actual MLP readouts."]}
