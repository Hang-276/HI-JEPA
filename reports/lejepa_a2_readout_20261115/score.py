"""Score the locked A2 readout comparison on paired physical TEST windows."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A3 = HERE.parent / "lejepa_a3_gap_20261115"
ARMS = ("strong", "weak")
BINS = ("lower", "middle", "upper")
MATCH_COUNTS = (
    "windows", "both_accepted", "matched", "weak_wrong_strong_right", "weak_right_strong_wrong",
    "both_wrong", "both_right",
)

spec = importlib.util.spec_from_file_location("a2_fixed_threshold_helpers", A3 / "score.py")
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)


def checked_path(relative, parent):
    path = Path(relative)
    resolved = (HERE / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(parent):
        raise ValueError(f"Unexpected artifact path: {relative}")
    return resolved


def verify():
    receipt_path = HERE / "FORECAST.json"
    receipt = BASE.read(receipt_path)
    config = BASE.read(HERE / "config.json")
    if receipt["config"] != config:
        raise ValueError("Forecast config differs from locked config")
    sources = {checked_path(relative, ROOT): expected for relative, expected in receipt["sources"].items()}
    required = {HERE / "score.py", HERE / "config.json", HERE / "PROTOCOL.md", A3 / "score.py", A3 / "THRESHOLDS.json"}
    if not required <= set(sources):
        raise ValueError("Forecast does not bind the A2 scoring, config and protocol and its A3 inputs")
    for path, expected in sources.items():
        if BASE.digest(path) != expected:
            raise ValueError(f"Forecast source digest mismatch: {path}")
    published = BASE.read(A3 / "THRESHOLDS.json")["thresholds"]["g50"]
    edges = config["gap_edges"]
    if (config["threshold"] != published or len(edges) != 3 or edges[0] != published
            or not all(math.isfinite(e) for e in edges) or not 0 < edges[0] < edges[1] < edges[2]
            or config["matched_gap_max_difference"] <= 0):
        raise ValueError("A2 threshold, bin edges or matching tolerance do not conform to the locked design")

    models = {f"{case}_{seed}": (seed, case) for seed in config["seeds"] for case in config["training_cases"]}
    expected_grid = {(name, process, arm) for name in models for process in config["processes"] for arm in ARMS}
    if len(expected_grid) != 32:
        raise ValueError("Expected two arms across the full 16-cell panel")
    cells, master_by_pair = {}, {}
    for cell in receipt["cells"]:
        name, process, seed, arm, master = (cell[k] for k in ("name", "process", "seed", "arm", "test_master"))
        key = (name, process, arm)
        if (key not in expected_grid or key in cells or (seed, cell["training_case"]) != models[name]
                or type(master) is not int or master < 0):
            raise ValueError(f"Invalid model metadata or repeated cell: {key}")
        if master_by_pair.setdefault((seed, process), master) != master:
            raise ValueError(f"Readout arms and training regimes must share the TEST stream: {key}")
        path = checked_path(cell["path"], HERE)
        if BASE.digest(path) != cell["sha256"]:
            raise ValueError(f"Forecast array digest mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"predicted_scores"}:
                raise ValueError(f"Unexpected fields in sealed prediction: {path}")
            prediction = archive["predicted_scores"]
        if prediction.shape != (config["n_test"], 4) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid prediction array: {path}")
        ordered = np.sort(prediction, axis=1)
        cells[key] = {"metadata": cell, "top2": BASE.top_two(prediction), "gap": ordered[:, 2] - ordered[:, 1]}
    if cells.keys() != expected_grid or len(set(master_by_pair.values())) != len(master_by_pair):
        raise ValueError("Incomplete panel or unexpected sharing of TEST streams")
    return config, cells, receipt_path


def by_group(records, reducer):
    all_rows = list(records.values())
    result = {"overall": reducer(all_rows)}
    for field in ("name", "seed", "process", "training_case"):
        levels = sorted({row[field] for row in all_rows})
        result[f"by_{field}"] = {str(level): reducer([row for row in all_rows if row[field] == level]) for level in levels}
    return result


def evaluate_cell(cell, physical_top, config, adjusted_alpha):
    wrong = (cell["top2"] != physical_top).any(axis=1)
    gap = cell["gap"]
    accepted = gap >= config["threshold"]
    record = BASE.counts(int(accepted.sum()), int((accepted & wrong).sum()), len(wrong))
    upper = BASE.cp_upper(record["errors"], record["accepted"], adjusted_alpha)
    passes_coverage = record["coverage"] >= config["min_coverage"]
    passes_risk = upper <= config["max_error"]
    ranges = zip(config["gap_edges"], [*config["gap_edges"][1:], np.inf])
    bins = {}
    for label, (lo, hi) in zip(BINS, ranges):
        in_bin = (gap >= lo) & (gap < hi)
        bins[label] = BASE.counts(int(in_bin.sum()), int((in_bin & wrong).sum()), len(wrong))
    if sum(c["accepted"] for c in bins.values()) != record["accepted"]:
        raise AssertionError("Visible bins must partition the accepted windows")
    meta = {k: cell["metadata"][k] for k in ("name", "seed", "training_case", "process", "test_master")}
    return wrong, {**meta, **record, "simultaneous_error_upper": upper, "passes_coverage": passes_coverage,
                   "passes_error_bound": passes_risk, "passes_both": passes_coverage and passes_risk,
                   "all_answers": BASE.counts(len(wrong), int(wrong.sum()), len(wrong)), "by_gap_bin": bins}


def matched_rows(strong, weak, config):
    strong_gap, weak_gap = strong["gap"], weak["gap"]
    strong_wrong, weak_wrong = strong["wrong"], weak["wrong"]
    both_accepted = (strong_gap >= config["threshold"]) & (weak_gap >= config["threshold"])
    matched = both_accepted & (np.abs(strong_gap - weak_gap) <= config["matched_gap_max_difference"])
    result = {"windows": len(matched), "both_accepted": int(both_accepted.sum()), "matched": int(matched.sum()),
              "weak_wrong_strong_right": int((matched & weak_wrong & ~strong_wrong).sum()),
              "weak_right_strong_wrong": int((matched & ~weak_wrong & strong_wrong).sum()),
              "both_wrong": int((matched & weak_wrong & strong_wrong).sum()),
              "both_right": int((matched & ~weak_wrong & ~strong_wrong).sum())}
    return result


def summarize_match(rows):
    result = {k: sum(row[k] for row in rows) for k in MATCH_COUNTS}
    n = result["matched"]
    if sum(result[k] for k in MATCH_COUNTS[3:]) != n:
        raise AssertionError("Discordance table does not partition the matched rows")
    wrong_strong = result["weak_right_strong_wrong"] + result["both_wrong"]
    wrong_weak = result["weak_wrong_strong_right"] + result["both_wrong"]
    return {**result, "strong_matched_error": wrong_strong / n if n else None,
            "weak_matched_error": wrong_weak / n if n else None,
            "weak_minus_strong_matched_error": (wrong_weak - wrong_strong) / n if n else None}


def summarize_arm(records):
    worst_coverage = min(records, key=lambda key: (records[key]["coverage"], key))
    worst_upper = max(records, key=lambda key: (records[key]["simultaneous_error_upper"], key))
    observed = {key: row["selective_error"] for key, row in records.items() if row["selective_error"] is not None}
    worst_observed = max(observed, key=lambda key: (observed[key], key), default=None)
    for row in records.values():
        row["all_answers"] = {**row["all_answers"], **{k: row[k] for k in ("name", "seed", "process", "training_case")}}
        for counts in row["by_gap_bin"].values():
            counts.update({k: row[k] for k in ("name", "seed", "process", "training_case")})
    return {**by_group(records, BASE.pooled), "by_cell": records,
            "all_answers": by_group({key: row["all_answers"] for key, row in records.items()}, BASE.pooled),
            "by_gap_bin": {label: by_group({key: row["by_gap_bin"][label] for key, row in records.items()}, BASE.pooled)
                           for label in BINS},
            "worst_coverage_cell": worst_coverage, "worst_coverage": records[worst_coverage]["coverage"],
            "worst_error_upper_cell": worst_upper, "worst_error_upper": records[worst_upper]["simultaneous_error_upper"],
            "worst_observed_error_cell": worst_observed, "worst_observed_error": observed.get(worst_observed),
            "n_cells_passing_coverage": sum(c["passes_coverage"] for c in records.values()),
            "n_cells_passing_error_bound": sum(c["passes_error_bound"] for c in records.values()),
            "n_cells_passing_both": sum(c["passes_both"] for c in records.values()),
            "passes_joint_panel": all(c["passes_both"] for c in records.values())}


def score(config, cells, receipt_path):
    summary_path = HERE / "SUMMARY.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    alpha = config["alpha"] / len(cells)
    truth = {}
    by_arm = {arm: {} for arm in ARMS}
    for (name, process, arm), cell in cells.items():
        master = cell["metadata"]["test_master"]
        if (master, process) not in truth:
            truth[master, process] = BASE.top_two(BASE.physical_scores(config, master, process))
        cell["wrong"], by_arm[arm][f"{name}__{process}"] = evaluate_cell(cell, truth[master, process], config, alpha)
    matched = {}
    for (name, process, arm), strong in cells.items():
        if arm != "strong":
            continue
        result = matched_rows(strong, cells[name, process, "weak"], config)
        metadata = {k: strong["metadata"][k] for k in ("name", "seed", "training_case", "process", "test_master")}
        matched[f"{name}__{process}"] = {**metadata, **result}

    results = {arm: summarize_arm(records) for arm, records in by_arm.items()}
    paired = by_group(matched, summarize_match)
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump({"forecast_sha256": BASE.digest(receipt_path), "threshold": config["threshold"],
                   "familywise_alpha": config["alpha"], "family_size": len(cells), "per_claim_alpha": alpha,
                   "minimum_observed_coverage": config["min_coverage"], "maximum_error_upper": config["max_error"],
                   "gap_edges": config["gap_edges"], "matched_gap_max_difference": config["matched_gap_max_difference"],
                   "arms": results, "matched": {**paired, "by_cell": {key: {**row, **summarize_match([row])}
                                                                        for key, row in matched.items()}}},
                  handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    for arm, row in results.items():
        print(arm, "passes:", row["passes_joint_panel"], "worst coverage:", row["worst_coverage"],
              "worst simultaneous upper:", row["worst_error_upper"], flush=True)
    print("Matched:", paired["overall"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "score"))
    args = parser.parse_args()
    config, cells, receipt_path = verify()
    print(f"Verified sealed forecast for {len(cells)} readout cells. SHA256 {BASE.digest(receipt_path)}", flush=True)
    if args.stage == "score":
        score(config, cells, receipt_path)


if __name__ == "__main__":
    main()
