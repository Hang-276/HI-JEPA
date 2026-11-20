"""Verify the locked A3 forecast and score simultaneous fixed-threshold risks."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THRESHOLDS = ("g20", "g50", "g80")


def read(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked_path(relative, parent):
    path = Path(relative)
    resolved = (HERE / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(parent):
        raise ValueError(f"Unexpected artifact path: {relative}")
    return resolved


def top_two(values):
    return np.sort(np.argsort(values, axis=1, kind="stable")[:, :2], axis=1)


def verify():
    receipt_path = HERE / "FORECAST.json"
    receipt = read(receipt_path)
    config = read(HERE / "config.json")
    if receipt["config"] != config:
        raise ValueError("Forecast config differs from locked config")
    sources = {checked_path(relative, ROOT): expected for relative, expected in receipt["sources"].items()}
    required = {HERE / "score.py", HERE / "config.json", HERE / "PROTOCOL.md"}
    if not required <= set(sources):
        raise ValueError("Forecast does not bind the scoring code, config and protocol")
    for path, expected in sources.items():
        if digest(path) != expected:
            raise ValueError(f"Forecast source digest mismatch: {path}")
    thresholds = receipt["thresholds"]
    if (set(thresholds) != set(THRESHOLDS) or not all(math.isfinite(thresholds[k]) for k in THRESHOLDS)
            or not thresholds["g20"] >= thresholds["g50"] >= thresholds["g80"] >= 0):
        raise ValueError("Unexpected locked thresholds")
    models = {f"{case}_{seed}": (seed, case) for seed in config["seeds"] for case in config["training_cases"]}
    expected = {(name, process) for name in models for process in config["processes"]}
    if len(expected) != 16:
        raise ValueError("Expected the fixed four-seed, two-regime, two-process panel")
    cells, master_by_pair = {}, {}
    for cell in receipt["cells"]:
        name, process, seed, master = (cell[k] for k in ("name", "process", "seed", "test_master"))
        key = (name, process)
        if (key not in expected or key in cells or (seed, cell["training_case"]) != models[name]
                or type(master) is not int or master < 0):
            raise ValueError(f"Invalid model metadata or repeated cell: {key}")
        if master_by_pair.setdefault((seed, process), master) != master:
            raise ValueError(f"Two training regimes must share the TEST stream: {key}")
        path = checked_path(cell["path"], HERE)
        if digest(path) != cell["sha256"]:
            raise ValueError(f"Forecast array digest mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"predicted_scores"}:
                raise ValueError(f"Unexpected fields in sealed prediction: {path}")
            prediction = archive["predicted_scores"]
        if prediction.shape != (config["n_test"], 4) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid prediction array: {path}")
        ordered = np.sort(prediction, axis=1)
        cells[key] = {"metadata": cell, "top2": top_two(prediction), "gap": ordered[:, 2] - ordered[:, 1]}
    if cells.keys() != expected or len(set(master_by_pair.values())) != len(master_by_pair):
        raise ValueError("Incomplete panel or unexpected sharing of TEST streams")
    return config, thresholds, cells, receipt_path


def physical_scores(config, master, process):
    rho = np.asarray(config["processes"][process])
    rng = np.random.default_rng(np.random.SeedSequence([master, config["test_role"]]))
    innovations = rng.normal(size=(config["n_test"], config["length"], 4))
    states = innovations.copy()
    scale = np.sqrt(1 - rho**2)
    for time in range(1, config["length"]):
        states[:, time] = rho * states[:, time - 1] + scale * innovations[:, time]
    return sum(((states[:, lag:] - states[:, :-lag])**2).mean(axis=1)
               for lag in config["lags"]) / len(config["lags"])


def cp_upper(errors, accepted, alpha):
    """Invert the lower binomial tail for the exact one-sided CP upper bound."""
    if errors == accepted:
        return 1.0
    log_alpha = math.log(alpha)
    if errors == 0:
        return -math.expm1(log_alpha / accepted)
    log_choose = math.lgamma(accepted + 1) - math.lgamma(errors + 1) - math.lgamma(accepted - errors + 1)
    lo, hi = errors / accepted, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        if p == lo or p == hi:
            break
        # The ratio to the mass at errors is at most one for p >= errors/accepted.
        odds = (1 - p) / p
        tail_ratio = term = 1.0
        for j in range(errors, 0, -1):
            term *= j * odds / (accepted - j + 1)
            tail_ratio += term
        log_tail = log_choose + errors * math.log(p) + (accepted - errors) * math.log1p(-p) + math.log(tail_ratio)
        if log_tail > log_alpha:
            lo = p
        else:
            hi = p
    return hi


def counts(accepted, errors, total):
    return {"windows": total, "accepted": accepted, "errors": errors, "coverage": accepted / total,
            "selective_error": errors / accepted if accepted else None}


def pooled(cells):
    return counts(sum(c["accepted"] for c in cells), sum(c["errors"] for c in cells), sum(c["windows"] for c in cells))


def split_groups(records):
    result = {"overall": pooled(list(records.values()))}
    for field in ("seed", "process", "training_case"):
        keys = sorted({record[field] for record in records.values()})
        result[f"by_{field}"] = {str(key): pooled([row for row in records.values() if row[field] == key]) for key in keys}
    return result


def score(config, thresholds, cells, receipt_path):
    summary_path = HERE / "SUMMARY.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    family_size = len(THRESHOLDS) * len(cells)
    alpha = config["alpha"] / family_size
    truth, baseline = {}, {}
    by_threshold = {name: {} for name in THRESHOLDS}
    for (name, process), cell in cells.items():
        cell_meta = cell["metadata"]
        master = cell_meta["test_master"]
        if (master, process) not in truth:
            truth[master, process] = top_two(physical_scores(config, master, process))
        wrong = (cell["top2"] != truth[master, process]).any(axis=1)
        identifier = f"{name}__{process}"
        meta = {k: cell_meta[k] for k in ("name", "seed", "training_case", "process", "test_master")}
        baseline[identifier] = {**meta, **counts(len(wrong), int(wrong.sum()), len(wrong))}
        for label in THRESHOLDS:
            accepted = cell["gap"] >= thresholds[label]
            record = counts(int(accepted.sum()), int((accepted & wrong).sum()), len(wrong))
            upper = cp_upper(record["errors"], record["accepted"], alpha)
            reaches_coverage = record["coverage"] >= config["minimum_coverage"]
            reaches_error_bound = upper <= config["max_error"]
            by_threshold[label][identifier] = {**meta, **record, "simultaneous_error_upper": upper,
                                               "passes_coverage": reaches_coverage, "passes_error_bound": reaches_error_bound,
                                               "passes_both": reaches_coverage and reaches_error_bound}
    result = {}
    for label, records in by_threshold.items():
        worst_coverage = min(records, key=lambda key: (records[key]["coverage"], key))
        worst_upper = max(records, key=lambda key: (records[key]["simultaneous_error_upper"], key))
        result[label] = {"threshold": thresholds[label], **split_groups(records), "by_cell": records,
                         "worst_coverage_cell": worst_coverage, "worst_coverage": records[worst_coverage]["coverage"],
                         "worst_error_upper_cell": worst_upper, "worst_error_upper": records[worst_upper]["simultaneous_error_upper"],
                         "n_cells_passing_coverage": sum(c["passes_coverage"] for c in records.values()),
                         "n_cells_passing_error_bound": sum(c["passes_error_bound"] for c in records.values()),
                         "n_cells_passing_both": sum(c["passes_both"] for c in records.values()),
                         "passes_joint_panel": all(c["passes_both"] for c in records.values())}
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump({"forecast_sha256": digest(receipt_path), "familywise_alpha": config["alpha"],
                   "family_size": family_size, "per_cell_threshold_alpha": alpha,
                   "minimum_coverage": config["minimum_coverage"], "maximum_error_upper": config["max_error"],
                   "accepted_thresholds": [label for label in THRESHOLDS if result[label]["passes_joint_panel"]],
                   "all_answers": {**split_groups(baseline), "by_cell": baseline}, "thresholds": result},
                  handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    for label, row in result.items():
        print(label, "passes:", row["passes_joint_panel"], "worst coverage:", row["worst_coverage"],
              "worst simultaneous upper:", row["worst_error_upper"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "score"))
    args = parser.parse_args()
    config, thresholds, cells, receipt_path = verify()
    print(f"Verified sealed forecast for {len(cells)} cells. SHA256 {digest(receipt_path)}", flush=True)
    if args.stage == "score":
        score(config, thresholds, cells, receipt_path)


if __name__ == "__main__":
    main()
