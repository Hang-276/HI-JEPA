"""Score previously sealed natural-direction predictions."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from run import HERE, ORIGINAL, PAIRS, digest, read, top_two, pair_gaps, write, smoke_config


RULES = ("nominal", "diag", "full")
FIXED_PAIR = (1, 2)


def ordered_signs(scores):
    return pair_gaps(scores) >= 0


def score_rule(predicted, physical, risk, safe, widths, correct_nominal, coverages):
    chosen = top_two(predicted)
    correct = np.all(chosen == top_two(physical), axis=1)
    pairs_correct = ordered_signs(predicted) == ordered_signs(physical)
    result = {
        "top2_accuracy": float(correct.mean()),
        "six_pair_accuracy": pairs_correct.mean(axis=0).tolist(),
        "average_pair_accuracy": float(pairs_correct.mean()),
        "fixed_pair_accuracy": float(pairs_correct[:, 3].mean()),
        "risk_diagnostics": {},
    }
    assert tuple(PAIRS[3]) == FIXED_PAIR
    safest_order = np.argsort(-risk, kind="stable")
    for coverage in coverages:
        n = int(np.floor(coverage * len(physical)))
        accepted = safest_order[:n]
        result["risk_diagnostics"][str(coverage)] = {
            "n": n,
            "nominal_top2_error": float((~correct_nominal[accepted]).mean()) if n else None,
        }
    if safe is not None:
        error = np.abs(pair_gaps(physical - predicted))
        simultaneous = np.all(error <= widths[None, :], axis=1)
        nominal_errors_safe = safe & ~correct_nominal
        result["conformal"] = {
            "all_six_marginal_coverage": float(simultaneous.mean()),
            "accepted_windows": int(safe.sum()),
            "acceptance_coverage": float(safe.mean()),
            "accepted_nominal_errors": int(nominal_errors_safe.sum()),
            "unconditional_accepted_error": float(nominal_errors_safe.mean()),
            "observed_conditional_accepted_error": float(nominal_errors_safe.sum() / safe.sum()) if safe.any() else None,
        }
    return result, correct, pairs_correct


def summarize(config, aggregate):
    master = np.random.default_rng(config["bootstrap_seed"])
    n = config["n_test"]
    repetitions = config["bootstrap_repetitions"]
    draws = master.multinomial(n, np.full(n, 1 / n), size=repetitions)
    result = {}
    for group, rows in aggregate.items():
        matrix = np.array(rows)
        by_row = matrix.mean(axis=0)
        bootstrap = draws @ by_row / n
        result[group] = {
            "fixed_tasks": len(rows),
            "top2_delta_full_minus_diag": float(by_row[:, 0].mean()),
            "top2_paired_ci95": np.quantile(bootstrap[:, 0], [0.025, 0.975]).tolist(),
            "fixed_pair_delta_full_minus_diag": float(by_row[:, 1].mean()),
            "fixed_pair_paired_ci95": np.quantile(bootstrap[:, 1], [0.025, 0.975]).tolist(),
            "fixed_pair_accuracies": {
                rule: float(by_row[:, 2 + index].mean()) for index, rule in enumerate(RULES)
            },
            "fixed_pair_accuracy_ci95": {
                rule: np.quantile(bootstrap[:, 2 + index], [0.025, 0.975]).tolist()
                for index, rule in enumerate(RULES)
            },
            "top2_accuracies": {rule: float(by_row[:, 5 + index].mean()) for index, rule in enumerate(RULES)},
        }
    return result


def score(config, round_name, base):
    path = base / round_name
    forecast_path = path / "FORECAST_RECEIPT.json"
    forecast = read(forecast_path)
    reveal = read(path / "REVEAL_RECEIPT.json")
    label_file = path / "labels.npz"
    if reveal["forecast_receipt_sha256"] != digest(forecast_path) or reveal["labels_sha256"] != digest(label_file):
        raise ValueError("unrelated label archive")
    if forecast["config"] != config:
        raise ValueError("configuration changed")
    from run import sources
    if forecast["source_hashes"] != sources():
        raise ValueError("sealed analysis sources changed")
    original = base / "primary"
    original_receipt = read(original / "FORECAST_RECEIPT.json")
    aggregate = defaultdict(list)
    records = {}
    with np.load(label_file, allow_pickle=False) as labels:
        for file, expected_hash in forecast["prediction_sha256"].items():
            name = Path(file).stem
            process, checkpoint = name.split("__")
            physical = labels[process]
            if digest(path / file) != expected_hash:
                raise ValueError(f"prediction changed: {file}")
            parameter_path = original / f"{name}.json"
            if digest(parameter_path) != original_receipt["parameter_sha256"][parameter_path.name]:
                raise ValueError(f"calibration changed: {parameter_path}")
            parameters = read(parameter_path)
            with np.load(path / file, allow_pickle=False) as arrays:
                correct_nominal = np.all(top_two(arrays["nominal_scores"]) == top_two(physical), axis=1)
                metrics, top2, pair = {}, {}, {}
                for rule in RULES:
                    diagnostic = parameters["methods"].get(rule)
                    metrics[rule], top2[rule], pair[rule] = score_rule(
                        arrays[f"{rule}_scores"], physical, arrays[f"{rule}_risk"],
                        arrays[f"{rule}_safe"] if diagnostic else None,
                        np.asarray(diagnostic["gap_widths"]) if diagnostic else None,
                        correct_nominal, config["diagnostic_coverages"],
                    )
                records[name] = {"process": process, "checkpoint": checkpoint, "metrics": metrics}
            row = np.column_stack([
                top2["full"].astype(float) - top2["diag"],
                pair["full"][:, 3].astype(float) - pair["diag"][:, 3],
                *[pair[rule][:, 3] for rule in RULES],
                *[top2[rule] for rule in RULES],
            ])
            for group in ("all", "separated_train" if checkpoint.startswith("separated_") else "crossed_train", process):
                aggregate[group].append(row)
    groups = summarize(config, aggregate)
    primary = groups["all"]
    passed = primary["top2_delta_full_minus_diag"] >= config["primary_minimum_gain"] and primary["top2_paired_ci95"][0] > 0
    write(path / "RESULTS.json", {
        "forecast_receipt_sha256": digest(forecast_path), "labels_sha256": digest(label_file),
        "primary_gate_passes": passed, "fixed_pair_zero_indexed": list(FIXED_PAIR),
        "pair_order_zero_indexed": PAIRS.tolist(), "groups": groups, "tasks": records,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", choices=("primary", "replication"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = read(HERE / "protocol.json")
    base = HERE / "smoke" if args.smoke else HERE
    if args.smoke:
        config = smoke_config(config)
    score(config, args.round, base)


if __name__ == "__main__":
    main()
