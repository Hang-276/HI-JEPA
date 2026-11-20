"""Reveal fresh physical paths and score the frozen A1 window forecasts."""

import argparse
import importlib.util
import itertools
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_PATH = HERE.parent / "lejepa_direction_a1_20261113/controlled/controlled.py"
spec = importlib.util.spec_from_file_location("a1_window_audit_source", SOURCE_PATH)
source_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_module)
digest, paths, read, write = source_module.digest, source_module.paths, source_module.read, source_module.write

FORECAST = HERE / "forecast"
FORECAST_RECEIPT = FORECAST / "RECEIPT.json"
DESIGN_LOCK = HERE / "DESIGN_LOCK.json"
REVEAL = HERE / "reveal.npz"
REVEAL_RECEIPT = HERE / "REVEAL_RECEIPT.json"
SUMMARY = HERE / "SUMMARY.json"
AXIS = 2
CLASSES = ("damage", "no_change", "repair")


def configuration():
    path = HERE / "protocol.json"
    config = read(path)
    receipt = read(FORECAST_RECEIPT)
    if digest(path) != receipt["config_sha256"]:
        raise ValueError("Forecast protocol hash changed.")
    if digest(DESIGN_LOCK) != receipt["design_sha256"]:
        raise ValueError("Forecast does not match the frozen design.")
    design = read(DESIGN_LOCK)
    if design["config"] != config or design["sources"] != receipt["sources"]:
        raise ValueError("Forecast protocol or sources differ from the frozen design.")
    rounds = config["rounds"]
    for relative, expected in receipt["sources"].items():
        if digest(REPO / relative) != expected:
            raise ValueError(f"Forecast source changed: {relative}")
    _, _, model_names, _, _ = source_module.identity()
    models = set(model_names)
    tasks = {
        (task["round"], task["process"], task["model"]): task["filename"]
        for task in receipt["tasks"]
    }
    expected_tasks = {
        (round_name, process, model)
        for round_name in rounds
        for process in config["processes"]
        for model in models
    }
    if len(models) != 14 or len(tasks) != len(receipt["tasks"]) or set(tasks) != expected_tasks:
        raise ValueError("Forecast tasks do not form the complete locked experiment.")
    if len(set(tasks.values())) != len(tasks) or not set(tasks.values()) <= set(receipt["files"]):
        raise ValueError("Forecast task files are repeated or absent from the receipt.")
    return config, receipt, tasks


def check_files(receipt):
    for relative, expected in receipt["files"].items():
        if digest(FORECAST / relative) != expected:
            raise ValueError(f"Forecast artifact changed: {relative}")


def truth_key(round_name, process):
    return f"{round_name}__{process}"


def reveal():
    if REVEAL.exists() or REVEAL_RECEIPT.exists() or SUMMARY.exists():
        raise FileExistsError("The reveal or score archive already exists.")
    config, receipt, _ = configuration()
    check_files(receipt)
    labels = {
        truth_key(round_name, process): paths(
            master, "test", config["n_test"], config["length"], np.asarray(rates)
        )
        for round_name, master in config["rounds"].items()
        for process, rates in config["processes"].items()
    }
    with REVEAL.open("xb") as output:
        np.savez_compressed(output, **labels)
    write(REVEAL_RECEIPT, {
        "forecast_receipt_sha256": digest(FORECAST_RECEIPT),
        "reveal_sha256": digest(REVEAL),
        "config_sha256": receipt["config_sha256"],
        "design_sha256": receipt["design_sha256"],
        "label_keys": sorted(labels),
    })


def top_two(scores):
    return np.sort(np.argsort(scores, axis=-1, kind="stable")[..., :2], axis=-1)


def increment_scores(states, lags):
    numerator = np.zeros((len(states), states.shape[-1]), dtype=np.float64)
    for lag in lags:
        increments = states[:, lag:] - states[:, :-lag]
        numerator += np.square(increments).mean(axis=1)
    return numerator / len(lags)


def normalized_scores(states, reference, lags):
    variance = reference.var(axis=0)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Physical comparison has an undefined reference variance.")
    return increment_scores(states, lags) / variance


def increment_cross(visible, lags):
    cross = np.zeros((len(visible), 4), dtype=np.float64)
    for lag in lags:
        increments = visible[:, lag:] - visible[:, :-lag]
        cross += (increments[..., [AXIS]] * increments).mean(axis=1)
    return cross / len(lags)


def exact_set(first, second):
    return np.all(first == second, axis=-1)


def correct_effect(candidate, baseline, truth):
    return exact_set(candidate, truth).astype(np.int8) - exact_set(baseline, truth).astype(np.int8)


def empirical_recovery(candidate, physical):
    error = float(np.square(candidate - physical).mean())
    return error, float(1 - error / physical.var())


def validate_predictions(source, n, a):
    predictions = {}
    for rule in ("full", "drop"):
        values = source[f"pred_{rule}"]
        if values.shape != (a, n) or not np.issubdtype(values.dtype, np.integer) or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError(f"Invalid effect prediction for {rule}.")
        predictions[rule] = values
    for rule in ("full", "abs"):
        values = source[f"prob_{rule}"]
        if values.shape != (a, n, 3) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid effect probabilities for {rule}.")
        np.testing.assert_allclose(values.sum(axis=-1), 1, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(source["prob_full"], np.eye(3)[predictions["full"] + 1])
    for key, shape in (("proposed_full_top", (a, n, 2)), ("pred_physical_top", (n, 2))):
        values = source[key]
        if values.shape != shape or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Invalid named top-2 shape or dtype: {key}")
        if np.any((values < 0) | (values > 3)) or np.any(values[..., 0] >= values[..., 1]):
            raise ValueError(f"Invalid ordered names: {key}")
    opportunity = source["signed_opportunity"]
    if opportunity.shape != (a, n) or opportunity.dtype != np.bool_:
        raise ValueError("Invalid signed-opportunity mask.")
    return predictions


def add_class_counts(counts, effect, predictions):
    for class_id in range(-1, 2):
        hit = effect == class_id
        counts["truth"][:, class_id + 1] += hit
        for rule, estimate in predictions.items():
            counts[rule][:, class_id + 1] += hit & (estimate == effect)


def empty_counts(n):
    return {key: np.zeros((n, len(CLASSES)), dtype=np.int64) for key in ("truth", "full", "drop")}


def totals(counts):
    truth = counts["truth"].sum(axis=0)
    result = {
        "n_windows_scored": int(truth.sum()),
        "class_counts": dict(zip(CLASSES, truth.tolist(), strict=True)),
        "distinct_trajectory_rows_by_class": dict(
            zip(CLASSES, (counts["truth"] > 0).sum(axis=0).tolist(), strict=True)
        ),
        "rules": {},
    }
    for rule in ("full", "drop"):
        correct = counts[rule].sum(axis=0)
        recall = [float(k / n) if n else None for k, n in zip(correct, truth, strict=True)]
        result["rules"][rule] = {
            "class_correct": dict(zip(CLASSES, correct.tolist(), strict=True)),
            "recall_by_class": dict(zip(CLASSES, recall, strict=True)),
            "macro_recall": float(np.mean(recall)) if all(value is not None for value in recall) else None,
            "accuracy": float(correct.sum() / truth.sum()) if truth.sum() else None,
        }
    return result


def score_task(source, physical, physical_top, config):
    lags = config["lags"]
    n = config["n_test"]
    visible = source["visible"].astype(np.float64)
    reference = source["reference"].astype(np.float64)
    if visible.shape != (n, config["length"], 4) or reference.shape != (config["n_reference"], 4):
        raise ValueError("Incorrect visible path or reference shape.")
    if not np.isfinite(visible).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite path or reference.")
    coefficients, donors, doses, signs, cal_mse = (
        source[key] for key in ("coefficients", "donors", "doses", "signs", "cal_extra_mse")
    )
    a = len(coefficients)
    if not a or any(values.shape != (a,) for values in (coefficients, donors, doses, signs, cal_mse)):
        raise ValueError("Invalid intervention arrays.")
    if not all(np.isfinite(values).all() for values in (coefficients, doses, cal_mse)):
        raise ValueError("Nonfinite intervention metadata.")
    if not np.issubdtype(donors.dtype, np.integer) or not np.isin(donors, [0, 3]).all():
        raise ValueError("Invalid donor identity.")
    planned = list(itertools.product(config["deltas"], (0, 3), (1, -1)))
    observed = list(zip(doses.tolist(), donors.tolist(), signs.tolist(), strict=True))
    if observed != planned or not np.array_equal(signs, np.sign(coefficients)):
        raise ValueError("Each dose and donor must have both prespecified signed arms.")
    np.testing.assert_allclose(cal_mse, doses, atol=1e-12, rtol=0)
    predictions = validate_predictions(source, n, a)
    baseline = top_two(normalized_scores(visible, reference, lags))
    correct_base = exact_set(baseline, physical_top)
    guessed_physical = source["pred_physical_top"]
    full_top = source["proposed_full_top"]
    baseline_numerator = increment_scores(visible, lags)
    cross = increment_cross(visible, lags)
    brier = {rule: np.zeros(n, dtype=np.float64) for rule in ("full", "abs")}
    counts = empty_counts(n)
    active_counts = empty_counts(n)
    opportunity_classes = np.zeros(3, dtype=np.int64)
    arms = []
    reconstruction_matches = 0
    baseline_error, baseline_r2 = empirical_recovery(visible[..., AXIS], physical[..., AXIS])
    for i, (coefficient, donor, dose) in enumerate(zip(coefficients, donors, doses, strict=True)):
        changed = visible.copy()
        changed[..., AXIS] += coefficient * visible[..., donor]
        changed_ref = reference.copy()
        changed_ref[..., AXIS] += coefficient * reference[..., donor]
        denominator = changed_ref.var(axis=0)
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise ValueError("An intervention has an undefined reference variance.")
        candidate = top_two(increment_scores(changed, lags) / denominator)
        reconstructed = exact_set(candidate, full_top[i])
        reconstruction_matches += int(reconstructed.sum())
        effect = correct_effect(candidate, baseline, physical_top)
        predicted_effect = correct_effect(full_top[i], baseline, guessed_physical)
        unsigned_numerator = baseline_numerator.copy()
        unsigned_numerator[:, AXIS] += coefficient**2 * baseline_numerator[:, donor]
        drop_top = top_two(unsigned_numerator / denominator)
        drop_effect = correct_effect(drop_top, baseline, guessed_physical)
        absolute_cross = np.abs(2 * coefficient * cross[:, donor])
        branch_tops, branch_effects = [], []
        for sign in (1, -1):
            branch = unsigned_numerator.copy()
            branch[:, AXIS] += sign * absolute_cross
            branch_top = top_two(branch / denominator)
            branch_tops.append(branch_top)
            branch_effects.append(correct_effect(branch_top, baseline, guessed_physical))
        absolute_probability = np.mean(np.eye(3)[np.asarray(branch_effects) + 1], axis=0)
        opportunity = source["signed_opportunity"][i]
        reconstructed_opportunity = ~exact_set(branch_tops[0], branch_tops[1])
        opportunity_outcomes = np.bincount(effect[opportunity] + 1, minlength=len(CLASSES))
        opportunity_classes += opportunity_outcomes
        active = ~exact_set(candidate, baseline)
        labels = np.eye(3, dtype=np.float64)[effect + 1]
        for rule in brier:
            brier[rule] += np.square(source[f"prob_{rule}"][i] - labels).sum(axis=1)
        row_predictions = {key: value[i] for key, value in predictions.items()}
        add_class_counts(counts, effect, row_predictions)
        for label in range(-1, 2):
            mask = active & (effect == label)
            active_counts["truth"][:, label + 1] += mask
            for rule, estimate in row_predictions.items():
                active_counts[rule][:, label + 1] += mask & (estimate == label)
        mse, r2 = empirical_recovery(changed[..., AXIS], physical[..., AXIS])
        arms.append({
            "index": i,
            "donor": int(donor),
            "dose": float(dose),
            "coefficient": float(coefficient),
            "sign": int(np.sign(coefficient)),
            "cal_extra_mse": float(cal_mse[i]),
            "test_recipient_mse": mse,
            "test_recipient_r2": r2,
            "candidate_top2_correct": int(exact_set(candidate, physical_top).sum()),
            "candidate_changed_windows": int(active.sum()),
            "true_damage_windows": int(np.count_nonzero(effect == -1)),
            "true_repair_windows": int(np.count_nonzero(effect == 1)),
            "changed_but_same_correctness": int(np.count_nonzero(active & (effect == 0))),
            "signed_opportunity_windows": int(opportunity.sum()),
            "signed_opportunity_true_class_counts": dict(zip(CLASSES, opportunity_outcomes.tolist(), strict=True)),
            "signed_opportunity_reconstruction_mismatches": int(np.count_nonzero(
                opportunity != reconstructed_opportunity
            )),
            "direct_candidate_reconstruction_matches": int(reconstructed.sum()),
            "full_effect_vs_saved_candidate_mismatches": int(np.count_nonzero(predicted_effect != predictions["full"][i])),
            "drop_effect_reconstruction_mismatches": int(np.count_nonzero(drop_effect != predictions["drop"][i])),
            "abs_probability_reconstruction_mismatches": int(np.count_nonzero(
                np.any(absolute_probability != source["prob_abs"][i], axis=-1)
            )),
        })
    groups = []
    for dose in np.unique(doses):
        group = [arm for arm in arms if arm["dose"] == dose]
        if len(group) < 2 or len({arm["donor"] for arm in group}) < 2:
            raise ValueError("Recovery matching requires at least two distinct donors per dose.")
        values = [arm["test_recipient_r2"] for arm in group]
        difference = max(values) - min(values)
        groups.append({
            "dose": float(dose), "max_absolute_r2_difference": difference,
            "passes": difference <= config.get("r2_match_tolerance", .005),
        })
    summary = {
        "baseline_top2_correct": int(correct_base.sum()),
        "cal_dose_match_max_abs": float(np.max(np.abs(cal_mse - doses))),
        "baseline_recipient_mse": baseline_error,
        "baseline_recipient_r2": baseline_r2,
        "predicted_physical_top2_correct": int(exact_set(guessed_physical, physical_top).sum()),
        "direct_candidate_reconstruction_matches": reconstruction_matches,
        "primary_full_sample": totals(counts),
        "active_sample_diagnostic": totals(active_counts),
        "test_state_matches_by_dose": groups,
        "signed_opportunity_true_class_counts": dict(zip(CLASSES, opportunity_classes.tolist(), strict=True)),
        "arms": arms,
    }
    return summary, counts, active_counts, brier


def class_contrast(counts, draws, minimum_rows):
    truth = counts["truth"].sum(axis=0)
    difference = counts["full"] - counts["drop"]
    support = (counts["truth"] > 0).sum(axis=0)
    if np.any(support < minimum_rows):
        return {
            "macro_recall_gain": None, "interval_95": None, "class_support_passes": False,
            "passes": False, "bootstrap_usable_draws": 0,
        }
    estimate = float(np.mean(difference.sum(axis=0) / truth))
    boot_truth = draws @ counts["truth"]
    boot_difference = draws @ difference
    valid = np.all(boot_truth > 0, axis=1)
    if not valid.any():
        return {
            "macro_recall_gain": estimate, "interval_95": None, "class_support_passes": True,
            "passes": False, "bootstrap_usable_draws": 0,
        }
    samples = np.mean(boot_difference[valid] / boot_truth[valid], axis=1)
    ci = np.quantile(samples, [0.025, 0.975]).tolist()
    return {
        "macro_recall_gain": estimate, "interval_95": ci, "class_support_passes": True,
        "bootstrap_usable_draws": int(valid.sum()),
    }


def score_round(round_name, tasks, receipt, labels, config, draws):
    n = config["n_test"]
    aggregate = empty_counts(n)
    active = empty_counts(n)
    brier = {rule: np.zeros(n, dtype=np.float64) for rule in ("full", "abs")}
    task_outputs = {}
    for process in config["processes"]:
        physical = labels[truth_key(round_name, process)]
        if physical.shape != (n, config["length"], 4) or not np.isfinite(physical).all():
            raise ValueError(f"Unexpected physical path shape: {round_name}/{process}")
        physical_top = top_two(increment_scores(physical, config["lags"]))
        for stage, proc, model in tasks:
            if stage != round_name or proc != process:
                continue
            relative = tasks[(stage, proc, model)]
            path = FORECAST / relative
            if digest(path) != receipt["files"][relative]:
                raise ValueError(f"Forecast artifact changed: {relative}")
            with np.load(path, allow_pickle=False) as source:
                task, counts, active_counts, errors = score_task(source, physical, physical_top, config)
            task_outputs[f"{process}__{model}"] = task
            for key in aggregate:
                aggregate[key] += counts[key]
                active[key] += active_counts[key]
            for rule in brier:
                brier[rule] += errors[rule]
    overall = totals(aggregate)
    overall["active_sample_diagnostic"] = totals(active)
    primary = class_contrast(aggregate, draws, config.get("minimum_class_rows", 50))
    minimum_gain = config.get("primary_minimum_gain", .01)
    primary["passes"] = bool(
        primary["class_support_passes"]
        and primary["interval_95"] is not None
        and primary["macro_recall_gain"] >= minimum_gain
        and primary["interval_95"][0] > 0
    )
    n_scored = overall["n_windows_scored"]
    error_difference = brier["full"] - brier["abs"]
    error_draws = draws @ error_difference / n_scored
    state_matches = [
        row for task in task_outputs.values() for row in task["test_state_matches_by_dose"]
    ]
    opportunity_counts = {
        name: sum(task["signed_opportunity_true_class_counts"][name] for task in task_outputs.values())
        for name in CLASSES
    }
    return {
        "primary": primary,
        "overall": overall,
        "test_state_match": {
            "passed": all(row["passes"] for row in state_matches),
            "n_matched": sum(row["passes"] for row in state_matches),
            "n_compared": len(state_matches),
            "maximum_absolute_r2_difference": max(row["max_absolute_r2_difference"] for row in state_matches),
        },
        "direct_candidate_reconstruction_matches": sum(
            task["direct_candidate_reconstruction_matches"] for task in task_outputs.values()
        ),
        "signed_opportunity": {
            "n": sum(opportunity_counts.values()),
            "fraction": sum(opportunity_counts.values()) / n_scored,
            "true_class_counts": opportunity_counts,
        },
        "probability_brier_diagnostic": {
            "loss": "sum of three squared class probability errors per scored intervention-window",
            "full_mean": float(brier["full"].sum() / n_scored),
            "abs_mean": float(brier["abs"].sum() / n_scored),
            "full_minus_abs": float(error_difference.sum() / n_scored),
            "full_minus_abs_interval_95": np.quantile(error_draws, [.025, .975]).tolist(),
        },
        "tasks": task_outputs,
    }


def score():
    if SUMMARY.exists():
        raise FileExistsError(SUMMARY)
    config, receipt, tasks = configuration()
    check_files(receipt)
    reveal_receipt = read(REVEAL_RECEIPT)
    if (
        reveal_receipt["forecast_receipt_sha256"] != digest(FORECAST_RECEIPT)
        or reveal_receipt["config_sha256"] != receipt["config_sha256"]
        or reveal_receipt["design_sha256"] != receipt["design_sha256"]
        or reveal_receipt["reveal_sha256"] != digest(REVEAL)
    ):
        raise ValueError("Physical truth was not sealed against these forecasts.")
    rng = np.random.default_rng(config["bootstrap_seed"])
    n = config["n_test"]
    draws = rng.multinomial(n, np.full(n, 1 / n), size=config.get("bootstrap_draws", 4096))
    reports = {}
    with np.load(REVEAL, allow_pickle=False) as labels:
        if sorted(labels.files) != reveal_receipt["label_keys"]:
            raise ValueError("Physical archive labels differ from the reveal receipt.")
        for round_name in config["rounds"]:
            reports[round_name] = score_round(round_name, tasks, receipt, labels, config, draws)
    write(SUMMARY, {
        "forecast_receipt_sha256": reveal_receipt["forecast_receipt_sha256"],
        "reveal_sha256": reveal_receipt["reveal_sha256"],
        "bootstrap_draws": len(draws),
        "rounds": reports,
        "passes_both_primary_rounds": all(row["primary"]["passes"] for row in reports.values()),
        "all_state_matching_passes": all(row["test_state_match"]["passed"] for row in reports.values()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reveal", "score"))
    arguments = parser.parse_args()
    {"reveal": reveal, "score": score}[arguments.stage]()


if __name__ == "__main__":
    main()
