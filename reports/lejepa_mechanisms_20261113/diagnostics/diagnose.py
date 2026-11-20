"""Post-reveal score diagnostics for fixed LeJEPA probe outputs and physical controls."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
READOUT = HERE.parent / "readout"
REPLICATION = READOUT / "replication"
RULES = ("linear", "cubic", "mlp")
AXES = np.arange(4)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(states, lags):
    return np.mean([np.mean((states[:, lag:] - states[:, :-lag])**2, axis=1) for lag in lags], axis=0)


def paths(config, rates):
    def normal(role, size):
        return np.random.default_rng(np.random.SeedSequence([config["master"], role])).normal(size=size)

    ref = normal(3, (config["n_reference"], 4))
    innovation = normal(4, (config["n_test"], config["trajectory_length"], 4))
    result = {}
    for case, rate in rates.items():
        rate = np.asarray(rate)
        sample = innovation.copy()
        noise = np.sqrt(1 - rate**2)
        for t in range(1, config["trajectory_length"]):
            sample[:, t] = rate * sample[:, t - 1] + noise * innovation[:, t]
        result[case] = scores(sample, config["lags"])
    return ref.var(axis=0), result


def members(score):
    subset = np.argsort(score, axis=1, kind="stable")[:, :2]
    result = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(result, subset, True, axis=1)
    return result


def metric(predicted, truth):
    expected = members(truth)
    proposed = members(predicted)
    correct = np.all(proposed == expected, axis=1)
    fastest = expected[:, 3]
    included = proposed[:, 3]
    delta = predicted - truth
    details = {}
    for label, mask in (("all", np.ones(len(truth), dtype=bool)), ("fast_in", fastest), ("fast_out", ~fastest)):
        details[label] = {
            "windows": int(mask.sum()),
            "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].mean()),
            "signed_bias_by_axis": delta[mask].mean(axis=0).tolist(),
            "mae_by_axis": np.abs(delta[mask]).mean(axis=0).tolist(),
        }
    details["axis_false_inclusion"] = np.sum(proposed & ~expected, axis=0).tolist()
    details["axis_false_exclusion"] = np.sum(~proposed & expected, axis=0).tolist()
    details["fastest_false_inclusion"] = int(np.count_nonzero(~fastest & included))
    details["fastest_false_exclusion"] = int(np.count_nonzero(fastest & ~included))
    return details, correct


def competition(scores, truth, windows):
    """Count omitted→admitted inversions. Multi-axis substitutions can yield >1 pair."""
    omitted = members(truth) & ~members(scores)
    admitted = members(scores) & ~members(truth)
    pairs = omitted[:, :, None] & admitted[:, None, :] & windows[:, None, None]
    true_gap = truth[:, None, :] - truth[:, :, None]
    estimated_gap = scores[:, None, :] - scores[:, :, None]
    values = []
    for missing in AXES:
        for intruder in AXES:
            selected = pairs[:, missing, intruder]
            if selected.any():
                values.append(
                    {
                        "omitted_axis": int(missing + 1),
                        "admitted_axis": int(intruder + 1),
                        "windows": int(selected.sum()),
                        "true_gap_mean": float(true_gap[selected, missing, intruder].mean()),
                        "estimated_gap_mean": float(estimated_gap[selected, missing, intruder].mean()),
                    }
                )
    return values


def paired(cubic, mlp, truth, cubic_correct, mlp_correct):
    repair = ~cubic_correct & mlp_correct
    harm = cubic_correct & ~mlp_correct
    fastest = members(truth)[:, 3]
    cubic_fast = members(cubic)[:, 3]
    mlp_fast = members(mlp)[:, 3]
    counts = {}
    for label, mask in (("all", np.ones(len(truth), dtype=bool)), ("fast_in", fastest), ("fast_out", ~fastest)):
        counts[label] = {
            "repairs": int(np.count_nonzero(repair & mask)),
            "new_errors": int(np.count_nonzero(harm & mask)),
            "net_correct": int(np.count_nonzero(repair & mask) - np.count_nonzero(harm & mask)),
            "repairs_fixed_fastest_membership": int(np.count_nonzero(repair & mask & (cubic_fast != fastest))),
            "new_errors_wrong_fastest_membership": int(np.count_nonzero(harm & mask & (mlp_fast != fastest))),
        }
    return {
        "counts": counts,
        "new_errors_omitted_to_admitted": competition(mlp, truth, harm),
        "repairs_preexisting_omitted_to_admitted": competition(cubic, truth, repair),
    }


def case_metrics(name, evaluation, forecast, truth, summary, config, is_replication):
    candidates = {}
    outcomes = {}
    correctness = {}
    corrected = {}
    corrected_correctness = {}
    for rule in RULES:
        if is_replication:
            state_prediction = forecast[f"{evaluation}_{rule}_states"].astype(np.float64)
            ref = forecast[f"{rule}_reference_variance"]
            values = scores(state_prediction, config["lags"]) / ref
            expected_accuracy = summary["evaluation"][evaluation][rule]["top2_accuracy"]
            fastest_r2 = summary["evaluation"][evaluation][rule]["fastest_r2"]
        else:
            values = forecast[f"{evaluation}_{rule}_scores"]
            metrics = summary["readouts"][rule]["evaluation"][evaluation]
            expected_accuracy = metrics["task"]["top2_accuracy"]
            fastest_r2 = metrics["heldout_physical_r2"][3]
        candidates[rule] = values
        outcomes[rule], correctness[rule] = metric(values, truth)
        outcomes[rule]["fastest_state_r2"] = fastest_r2
        if outcomes[rule]["all"]["accuracy"] != expected_accuracy:
            raise ValueError(f"Recomputed scores disagree with the frozen result: {name} {evaluation} {rule}")
        if not is_replication:
            np.testing.assert_array_equal(correctness[rule], forecast[f"{evaluation}_{rule}_top2_correct"])
        corrected[rule] = values.copy()
        corrected[rule][:, 3] = truth[:, 3]
        outcomes[rule]["true_fastest_oracle"], corrected_correctness[rule] = metric(corrected[rule], truth)
    return {
        "readouts": outcomes,
        "cubic_to_mlp": paired(candidates["cubic"], candidates["mlp"], truth, correctness["cubic"], correctness["mlp"]),
        "cubic_to_mlp_with_true_fastest_oracle": paired(
            corrected["cubic"], corrected["mlp"], truth, corrected_correctness["cubic"], corrected_correctness["mlp"]
        ),
    }


def run(is_replication):
    original = load(READOUT / "runs" / "results.json")
    rates = {r["case"]: r["rates"] for r in original["runs"].values()}
    if is_replication:
        result_path = REPLICATION / "RESULTS.json"
        formal = load(result_path)
        inputs = formal["results"]
        config = load(REPLICATION / "protocol.json")
        receipt = load(REPLICATION / "FORECAST_RECEIPT.json")
        directory = REPLICATION / "forecast"
    else:
        result_path = READOUT / "runs" / "results.json"
        inputs = original["runs"]
        config = original["config"]
        directory = READOUT / "runs"
    ref, target = paths(config, rates)
    results = {}
    source_hashes = {}
    for name, record in inputs.items():
        if record["mode"] != "frozen" and not name.endswith("_cubic_shear"):
            continue
        path = directory / f"{name}.npz"
        source_hashes[path.name] = digest(path)
        if is_replication and source_hashes[path.name] != receipt["forecast_files_sha256"][path.name]:
            raise ValueError(f"Replication forecasts no longer match their receipt: {name}")
        eval_cases = record["evaluation"] if is_replication else record["readouts"]["linear"]["evaluation"]
        with np.load(path, allow_pickle=False) as forecast:
            results[name] = {}
            for case in eval_cases:
                if not is_replication:
                    np.testing.assert_array_equal(target[case], forecast[f"{case}_physical_scores"])
                results[name][case] = case_metrics(name, case, forecast, target[case], record, config, is_replication)
    aggregate = Counter()
    for name, evaluated in results.items():
        if name.endswith("_cubic_shear"):
            continue
        same = name.rsplit("_", 1)[0]
        nominal = evaluated[same]["cubic_to_mlp"]["counts"]["all"]
        oracle = evaluated[same]["cubic_to_mlp_with_true_fastest_oracle"]["counts"]["all"]
        for label, row in (("nominal", nominal), ("true_fastest_oracle", oracle)):
            aggregate[f"{label}_repairs"] += row["repairs"]
            aggregate[f"{label}_new_errors"] += row["new_errors"]
            aggregate[f"{label}_checkpoints_top2_improved"] += row["net_correct"] > 0
    return {
        "scope": "post-reveal descriptive diagnosis, not a confirmatory causal contrast or deployable rule",
        "source_summary_sha256": digest(result_path),
        "source_prediction_sha256": source_hashes,
        "master_rng": config["master"],
        "same_condition_16_checkpoint_overview": dict(aggregate),
        "exact_coordinates_with_finite_reference": {
            "population_reference_variance": [1] * 4,
            "sample_reference_variance": ref.tolist(),
            "by_case": {case: metric(y / ref, y)[0] for case, y in target.items()},
        },
        "diagnostics": results,
    }


def reference_test_crossing(original, replication):
    old_summary = load(READOUT / "runs" / "results.json")
    new_config = load(REPLICATION / "protocol.json")
    rates = {record["case"]: record["rates"] for record in old_summary["runs"].values()}
    _, new_targets = paths(new_config, rates)
    cells = {}
    aggregate = {}
    for name, record in old_summary["runs"].items():
        if record["mode"] != "frozen":
            continue
        case = record["case"]
        with np.load(READOUT / "runs" / f"{name}.npz", allow_pickle=False) as old, np.load(
            REPLICATION / "forecast" / f"{name}.npz", allow_pickle=False
        ) as new:
            target = {"old": old[f"{case}_physical_scores"], "new": new_targets[case]}
            denominators = {}
            numerators = {}
            for rule in ("cubic", "mlp"):
                old_var = np.asarray(record["readouts"][rule]["evaluation"][case]["task"]["reference_variance"])
                new_var = new[f"{rule}_reference_variance"]
                denominators[rule] = {"old": old_var, "new": new_var}
                states = new[f"{case}_{rule}_states"].astype(np.float64)
                numerators[rule] = {
                    "old": old[f"{case}_{rule}_scores"] * old_var,
                    "new": scores(states, new_config["lags"]),
                }
            result = {}
            for test in ("old", "new"):
                for ref in ("old", "new"):
                    predicted = {rule: numerators[rule][test] / denominators[rule][ref] for rule in ("cubic", "mlp")}
                    correct = {rule: np.all(members(value) == members(target[test]), axis=1) for rule, value in predicted.items()}
                    label = f"{test}_test_{ref}_ref"
                    repairs = int((correct["mlp"] & ~correct["cubic"]).sum())
                    harms = int((~correct["mlp"] & correct["cubic"]).sum())
                    result[label] = {
                        "cubic_correct": int(correct["cubic"].sum()),
                        "mlp_correct": int(correct["mlp"].sum()),
                        "repairs": repairs,
                        "new_errors": harms,
                        "net_correct_mlp_minus_cubic": repairs - harms,
                        "windows": len(target[test]),
                    }
                    counts = aggregate.setdefault(label, Counter())
                    counts["checkpoint_count_improved"] += repairs > harms
                    for key, value in result[label].items():
                        counts[key] += value
                    if test == ref:
                        expected = original if test == "old" else replication
                        for rule in ("cubic", "mlp"):
                            assert result[label][f"{rule}_correct"] == expected["diagnostics"][name][case]["readouts"][rule]["all"]["correct"]
            cells[name] = result
    return {
        "scope": "post-reveal re-normalization of the same trained probes; explanatory, with old-new test and reference samples only",
        "same_condition_16_checkpoint_cells": {label: dict(counter) for label, counter in aggregate.items()},
        "by_checkpoint": cells,
    }


def main():
    original = run(False)
    replication = run(True)
    crossing = reference_test_crossing(original, replication)
    for result, filename in (
        (original, "ORIGINAL.json"), (replication, "REPLICATION.json"), (crossing, "TEST_REFERENCE_CROSSING.json")
    ):
        with (HERE / filename).open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
