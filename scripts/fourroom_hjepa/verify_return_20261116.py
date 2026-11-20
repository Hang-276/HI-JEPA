#!/usr/bin/env python3
"""Verify the saved FourRoom return without loading models or changing evidence.

Python standard library and NumPy only. Computes existing-array scores and
binomial bounds independently; never fits, samples trajectories, or alters gates.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/fourroom_hjepa/fourroom_01"
OUTPUT = ROOT / "results/fourroom_hjepa/fourroom_01_analysis_20261116/VERIFICATION.json"
TIER = "formal_result_integrity_verification"
FLOAT_ATOL = 2e-12


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def numeric_match(actual, expected, name, atol=FLOAT_ATOL):
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}")
    check(np.isfinite(a).all() and np.isfinite(b).all(), f"{name}: nonfinite values")
    if a.dtype.kind in "biu" and b.dtype.kind in "biu":
        check(np.array_equal(a, b), f"{name}: discrete mismatch")
        return 0.
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    check(error <= atol, f"{name}: max absolute difference {error} > {atol}")
    return error


def npz(path, shapes, records):
    with np.load(path, allow_pickle=False) as data:
        check(set(data.files) == set(shapes), f"{path}: unexpected array names")
        arrays = {name: data[name] for name in data.files}
    details = []
    for name, shape in shapes.items():
        value = arrays[name]
        check(value.shape == shape, f"{path}/{name}: expected {shape}, got {value.shape}")
        check(value.dtype.kind in "biuf" and np.isfinite(value).all(), f"{path}/{name}: not finite numeric")
        details.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype), "all_finite": True})
    records.append({"path": str(path.relative_to(SOURCE)), "sha256": digest(path), "arrays": details})
    return arrays


def log_binomial_cdf(k, n, p, log_choose):
    if p <= 0:
        return 0.
    if p >= 1:
        return 0. if k == n else -math.inf
    j = np.arange(k+1, dtype=float)
    values = log_choose+j*math.log(p)+(n-j)*math.log1p(-p)
    largest = float(values.max())
    return largest+math.log(float(np.exp(values-largest).sum()))


def clopper_pearson_upper(k, n, alpha):
    """Invert P_p(Binomial(n,p)<=k)=alpha, equivalent to beta.ppf(1-alpha,k+1,n-k)."""
    if k == n:
        return 1.
    if k == 0:
        return -math.expm1(math.log(alpha)/n)
    logs = np.array([math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
                     for j in range(k+1)])
    lo, hi = k/n, 1.
    for _ in range(80):
        mid = (lo+hi)/2
        if log_binomial_cdf(k, n, mid, logs) > math.log(alpha):
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def verify_splits(cfg, splits):
    roles = ("train", "val", "audit", "ref", "cal", "test")
    check(set(splits) == set(roles), "Split roles do not match formal protocol")
    total = sum(cfg["episodes"].values())
    rng = np.random.default_rng(cfg["split_seed"])
    order = rng.permutation(total)
    cursor = 0
    all_ids, all_frames, rows = [], [], []
    for role in roles:
        count = cfg["episodes"][role]
        ids = np.asarray(splits[role]["episode_ids"], dtype=np.int64)
        frames = np.asarray(splits[role]["frame_indices"], dtype=np.int64)
        expected_ids = np.sort(order[cursor:cursor+count]); cursor += count
        check(np.array_equal(ids, expected_ids), f"{role}: split seed/order mismatch")
        width = cfg["fit_frames_per_episode"] if role in ("train", "val") else cfg["window_length"]
        check(frames.shape == (count, width), f"{role}: frame shape mismatch")
        check(np.all(np.diff(ids) > 0), f"{role}: ids not unique/increasing")
        check(np.all(np.diff(frames, axis=1) > 0), f"{role}: frame order not increasing")
        local = frames-ids[:, None]*401
        check(np.all((local >= 0) & (local < 401)), f"{role}: frame crosses episode")
        if role in ("train", "val"):
            regenerated = np.stack([np.sort(rng.choice(401, width, replace=False)) for _ in ids])
        else:
            regenerated = rng.integers(0, 402-width, size=count)[:, None]+np.arange(width)
            check(np.all(np.diff(frames, axis=1) == 1), f"{role}: noncontiguous scoring window")
        check(np.array_equal(local, regenerated), f"{role}: deterministic frame selection mismatch")
        all_ids.append(ids); all_frames.append(frames.ravel())
        rows.append({"role": role, "episodes": count, "frames_per_episode": width,
                     "frame_rows": int(frames.shape[0]), "total_selected_frames": int(frames.size),
                     "episode_ids_sorted_unique": True, "frames_stay_within_401_frame_episode": True,
                     "seed_and_selection_reconstruction_exact": True})
    combined_ids = np.concatenate(all_ids)
    combined_frames = np.concatenate(all_frames)
    check(np.array_equal(np.sort(combined_ids), np.arange(total)), "Episodes missing or shared between roles")
    check(len(np.unique(combined_frames)) == len(combined_frames), "Frames shared between roles")
    return {"roles": rows, "total_episodes": total, "roles_disjoint": True,
            "all_configured_episode_ids_used_once": True, "frame_indices_disjoint": True,
            "reconstruction": "Saved manifest exactly reproduces pinned split seed and code, assuming packed 401-frame episodes.",
            "raw_data_limit": "Original HDF5 and its ep_offset/pixels are not in the return. This checks manifest consistency, not independent image-to-coordinate provenance."}


def verify_csv(summary, path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    check(len(rows) == 12, "CSV does not contain 12 rows")
    keyed = {(r["cell"], r["rule"]): r for r in rows}
    check(len(keyed) == 12, "CSV duplicate cell/rule")
    differences = []
    for row in summary["rows"]:
        other = keyed[row["cell"], row["rule"]]
        check(set(other) == set(row), "CSV columns differ from JSON fields")
        for key, value in row.items():
            text = other[key]
            if isinstance(value, bool):
                check(text == str(value), f"CSV Boolean mismatch {key}")
            elif isinstance(value, (float, int)):
                check(float(text) == value, f"CSV numeric mismatch {key}")
            elif value is None:
                check(text == "", f"CSV None mismatch {key}")
            else:
                check(text == value, f"CSV text mismatch {key}")
    return {"rows": len(rows), "json_values_exactly_match_csv": True, "mismatches": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    check(output.is_relative_to(OUTPUT.parent), "Output must remain in the separate analysis directory")
    check(not output.exists(), "Refusing to replace an existing verification receipt")
    start = time.perf_counter()
    all_source_files = sorted(p for p in SOURCE.rglob("*") if p.is_file())
    before_hashes = {str(p.relative_to(SOURCE)): digest(p) for p in all_source_files}
    cfg = load_json(SOURCE/"config.json")
    status = load_json(SOURCE/"status.json")
    summary = load_json(SOURCE/"summary.json")
    thresholds = load_json(SOURCE/"thresholds.json")
    forecast = load_json(SOURCE/"forecast_receipt.json")
    check(status == {"status": "COMPLETE"}, "Run is not COMPLETE")
    check(cfg["engineering_smoke"] is False and summary["engineering_smoke"] is False, "Smoke results are not formal results")
    check(cfg["model_seeds"] == [42, 43, 44] and cfg["window_length"] == 193 and cfg["lags"] == [4, 16, 32], "Formal model/window/lag config mismatch")
    check(cfg["audit_min_r2"] == .85 and cfg["minimum_coverage"] == .35 and cfg["maximum_selective_error"] == .01 and cfg["family_alpha"] == .05, "Formal gate configuration changed")
    check(cfg["episodes"] == {"train": 256, "val": 64, "audit": 128, "ref": 256, "cal": 1024, "test": 2048}, "Formal episode budgets changed")
    check(cfg["fit_frames_per_episode"] == 32 and cfg["cal_gap_quantile"] == .5, "Frame budget/quantile mismatch")
    axes = cfg["axes"]
    check(axes == summary["axes"] == ["agent_x", "agent_y", "distractor_x", "distractor_y"], "Axes mismatch")
    cells = [f"seed{seed}_level{level}" for seed in cfg["model_seeds"] for level in (1, 2)]
    expected_keys = {(cell, rule) for cell in cells for rule in ("own_cal", "level1_cal")}
    check(len(summary["rows"]) == 12 and {(r["cell"], r["rule"]) for r in summary["rows"]} == expected_keys, "12-result Cartesian product incomplete")
    check(thresholds["family_size"] == 12 and thresholds["per_claim_alpha"] == .05/12, "Confidence family mismatch")
    csv_check = verify_csv(summary, SOURCE/"summary.csv")
    check(forecast["thresholds_sha256"] == digest(SOURCE/"thresholds.json"), "Forecast threshold SHA mismatch")
    check(set(forecast["prediction_sha256"]) == set(cells), "Forecast predicted cells mismatch")
    receipt_checks = [{"path": "thresholds.json", "expected_sha256": forecast["thresholds_sha256"], "matches": True}]
    for cell in cells:
        expected = forecast["prediction_sha256"][cell]
        check(expected == digest(SOURCE/cell/"test_prediction.npz"), f"Forecast prediction SHA mismatch {cell}")
        receipt_checks.append({"path": cell+"/test_prediction.npz", "expected_sha256": expected, "matches": True})
    split_check = verify_splits(cfg, load_json(SOURCE/"splits.json"))
    # Execute only already-inspected pure NumPy definitions, no SciPy import or model code.
    tree = ast.parse((SOURCE/"scripts/metrics.py").read_text())
    names = {"_variance", "_top2_and_gap", "score_windows"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    check(len(nodes) == 3, "Expected pure scoring functions absent")
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE/"scripts/metrics.py"), "exec"), namespace)
    score = namespace["score_windows"]
    array_records = []
    ref = npz(SOURCE/"reference_truth.npz", {"states": (256, 193, 4), "variance": (4,)}, array_records)
    truth = npz(SOURCE/"test_truth.npz", {"states": (2048, 193, 4)}, array_records)["states"]
    ref_error = numeric_match(ref["states"].reshape(-1, 4).var(axis=0, ddof=0), ref["variance"], "physical REF variance")
    check(np.all(ref["variance"] > 0), "Physical REF is degenerate")
    first_audit_truth = None
    audit_rows, rule_rows, cell_checks = [], [], []
    max_cp_difference = 0.
    for cell in cells:
        folder = SOURCE/cell
        audit = npz(folder/"audit.npz", {"prediction": (24704, 4), "truth": (24704, 4)}, array_records)
        audit_json = load_json(folder/"audit.json")
        if first_audit_truth is None:
            first_audit_truth = audit["truth"].copy()
        check(np.array_equal(audit["truth"], first_audit_truth), "AUDIT truth row order differs between cells")
        audit_variance = audit["truth"].var(axis=0, ddof=0)
        check(np.all(audit_variance > 0), "AUDIT variance is degenerate")
        r2 = 1-((audit["prediction"]-audit["truth"])**2).mean(axis=0)/audit_variance
        r2_error = numeric_match(r2, [audit_json["r2"][a] for a in axes], cell+" audit R2")
        eligible = bool(np.min(r2) >= cfg["audit_min_r2"])
        check(eligible == audit_json["eligible"], cell+" AUDIT admission mismatch")
        audit_rows.append({"cell": cell, "r2": dict(zip(axes, r2.tolist())), "minimum_r2": float(r2.min()),
                           "eligible": eligible, "below_gate_axes": [a for a, value in zip(axes, r2) if value < cfg["audit_min_r2"]]})
        predicted_ref = npz(folder/"reference.npz", {"prediction": (49408, 4), "variance": (4,)}, array_records)
        ref_var_error = numeric_match(predicted_ref["prediction"].var(axis=0, ddof=0), predicted_ref["variance"], cell+" predicted REF variance")
        check(np.all(predicted_ref["variance"] > 0), cell+" predicted REF degenerate")
        cal = npz(folder/"calibration.npz", {"prediction": (1024, 193, 4), "scores": (1024, 4), "gap": (1024,)}, array_records)
        v = predicted_ref["variance"]
        cal_computed = score(cal["prediction"], cal["prediction"], v, v, cfg["lags"])
        cal_errors = {"scores": numeric_match(cal_computed["predicted_scores"], cal["scores"], cell+" CAL scores"),
                      "gap": numeric_match(cal_computed["gap"], cal["gap"], cell+" CAL gap")}
        cal_threshold = float(np.quantile(cal_computed["gap"], cfg["cal_gap_quantile"], method="linear"))
        threshold_error = numeric_match(cal_threshold, thresholds["own_cal"][cell], cell+" CAL median")
        prediction = npz(folder/"test_prediction.npz", {"prediction": (2048, 193, 4)}, array_records)["prediction"]
        shapes = {"true_scores": (2048, 4), "predicted_scores": (2048, 4), "true_top2": (2048, 2),
                  "predicted_top2": (2048, 2), "gap": (2048,), "wrong": (2048,), "true_gap": (2048,),
                  "cross_term": (2048, 4), "error_energy": (2048, 4)}
        saved = npz(folder/"test_scores.npz", shapes, array_records)
        computed = score(prediction, truth, ref["variance"], v, cfg["lags"])
        score_errors = {name: numeric_match(computed[name], saved[name], cell+" TEST "+name) for name in shapes}
        check((folder/"probe.pt").stat().st_size > 0 and (folder/"probe.json").is_file(), cell+" readout artifacts missing")
        seed = cell.split("_")[0]
        check(thresholds["level1_cal"][seed] == thresholds["own_cal"][seed+"_level1"], seed+" level1 threshold mismatch")
        rows_by_rule = {r["rule"]: r for r in summary["rows"] if r["cell"] == cell}
        for rule in ("own_cal", "level1_cal"):
            threshold = thresholds["own_cal"][cell] if rule == "own_cal" else thresholds["level1_cal"][seed]
            accept = computed["gap"] >= threshold
            n = int(accept.sum()); k = int(np.count_nonzero(computed["wrong"] & accept))
            upper = clopper_pearson_upper(k, n, thresholds["per_claim_alpha"])
            coverage = n/len(truth)
            coverage_pass = bool(coverage >= cfg["minimum_coverage"])
            risk_pass = bool(n > 0 and upper <= cfg["maximum_selective_error"])
            formal_status = "RECOVERY_NOT_VERIFIED" if not eligible else "PASS" if coverage_pass and risk_pass else "CRITERIA_NOT_MET"
            calculated = {"cell": cell, "rule": rule, "threshold": threshold, "minimum_r2": float(r2.min()),
                          "eligible": eligible, "windows": len(truth), "accepted": n, "errors": k,
                          "coverage": coverage, "selective_error": k/n if n else None, "error_upper": upper,
                          "passes_coverage": coverage_pass, "passes_error_bound": risk_pass,
                          "passes": coverage_pass and risk_pass, "status": formal_status}
            old = rows_by_rule[rule]
            for field, value in calculated.items():
                if isinstance(value, str) or value is None or isinstance(value, bool):
                    check(value == old[field], f"{cell}/{rule}/{field}: stored summary mismatch")
                else:
                    numeric_match(value, old[field], f"{cell}/{rule}/{field}")
            cp_difference = abs(upper-old["error_upper"])
            max_cp_difference = max(max_cp_difference, cp_difference)
            rule_rows.append({**calculated, "stored_error_upper": old["error_upper"],
                              "cp_absolute_difference": cp_difference, "all_summary_fields_match": True,
                              "failures_by_gate": {"recovery": not eligible, "coverage": not coverage_pass, "risk_upper": not risk_pass}})
        cell_checks.append({"cell": cell, "audit_r2_max_absolute_difference": r2_error,
                            "predicted_ref_variance_max_absolute_difference": ref_var_error,
                            "cal_max_absolute_differences": cal_errors, "threshold_absolute_difference": threshold_error,
                            "test_max_absolute_differences": score_errors,
                            "all_answer_errors": int(np.count_nonzero(computed["wrong"])),
                            "all_answer_error_rate": float(np.mean(computed["wrong"])),
                            "physical_ref_variance": ref["variance"].tolist(),
                            "predicted_ref_variance": v.tolist(),
                            "test_state_mse_by_axis": ((prediction-truth)**2).mean(axis=(0, 1)).tolist(),
                            "test_state_mse_normalized_by_physical_ref_by_axis": (((prediction-truth)**2).mean(axis=(0, 1))/ref["variance"]).tolist(),
                            "probe_artifacts_present_not_deserialized": True})
    after_hashes = {str(p.relative_to(SOURCE)): digest(p) for p in sorted(SOURCE.rglob("*")) if p.is_file()}
    check(before_hashes == after_hashes, "Input package changed during verification")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"evidence_tier": TIER, "status": "passed", "source_directory": str(SOURCE),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__},
              "absolute_float_tolerance": FLOAT_ATOL, "formal_complete": True, "engineering_smoke": False,
              "summary_rows": 12, "summary_csv_check": csv_check, "forecast_hash_checks": receipt_checks,
              "split_manifest": split_check, "array_files_checked": len(array_records), "array_checks": array_records,
              "physical_ref_variance_max_absolute_difference": ref_error,
              "audit_truth_identical_across_six_cells": True, "audit_rows": audit_rows,
              "cell_checks": cell_checks, "rule_rows": rule_rows,
              "confidence_bound_method": "Independent stdlib/NumPy inversion of binomial lower tail; 80 bisection steps; no SciPy import.",
              "cp_max_absolute_difference": max_cp_difference,
              "cal_quantile_method": "linear, matching the archived NumPy default; not low-dimensional higher",
              "original_files_checked_before_after": len(before_hashes), "original_files_unchanged": True,
              "source_sha256": before_hashes, "script_sha256": digest(Path(__file__)),
              "counts": {"recovery_eligible_cells": sum(r["eligible"] for r in audit_rows),
                         "formal_pass_rows": sum(r["status"] == "PASS" for r in rule_rows),
                         "coverage_pass_rows": sum(r["passes_coverage"] for r in rule_rows),
                         "risk_upper_pass_rows": sum(r["passes_error_bound"] for r in rule_rows)},
              "execution_scope": {"model_loading": False, "training": False, "new_trajectory_sampling": False,
                                  "threshold_changed": False, "existing_array_score_recalculation": True},
              "limits": ["Prediction/truth arrays are positionally aligned by the saved manifest and archived code; raw HDF5 is absent, so image-to-coordinate identity is not independently reconstructed.",
                         "Matching saved source, hashes and scores is integrity/numerical consistency evidence, not a causal mechanism proof.",
                         "The six cells share TEST episodes; twelve claims are not twelve independent data samples.",
                         "RECOVERY_NOT_VERIFIED does not prove information loss; a failed risk upper bound is not itself proof that true risk exceeds the limit."]}
    result["elapsed_seconds"] = time.perf_counter()-start
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"evidence_tier": TIER, "status": "passed", "output": str(output),
                      "counts": result["counts"], "cp_max_absolute_difference": max_cp_difference,
                      "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
