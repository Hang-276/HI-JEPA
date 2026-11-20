"""Score prespecified donor-risk predictions after separately sealed truth reveal."""

from pathlib import Path

import numpy as np

from controlled import HERE, METHODS, PAIR, RECIPIENT, calculate_moments, digest, read, sources, top_two, write


def flip_matrix(visible, truth):
    return (visible < 0) != (truth < 0)


def confusion(estimated, actual):
    return {
        "true_positive": int(np.count_nonzero(estimated & actual)),
        "false_positive": int(np.count_nonzero(estimated & ~actual)),
        "false_negative": int(np.count_nonzero(~estimated & actual)),
        "true_negative": int(np.count_nonzero(~estimated & ~actual)),
        "accuracy": float(np.mean(estimated == actual)),
    }


def paired_changes(correct):
    baseline = correct[0]
    return [
        {
            "accuracy": float(values.mean()),
            "accuracy_minus_base": float(values.mean() - baseline.mean()),
            "repaired": int(np.count_nonzero(values & ~baseline)),
            "new_error": int(np.count_nonzero(~values & baseline)),
        }
        for values in correct
    ]


def recipient_recovery(visible, truth, arms):
    error = visible[..., RECIPIENT].astype(np.float64) - truth[..., RECIPIENT]
    variance = truth[..., RECIPIENT].var()
    recovery = []
    for arm in arms:
        donor = arm["donor"]
        residual = error if donor is None else error + arm["a"] * visible[..., donor].astype(np.float64)
        mse = float(np.mean(residual**2))
        recovery.append({"mse": mse, "r2": float(1 - mse / variance)})
    return recovery


def intervals(matrix, bootstrap_weights):
    point = float(matrix.mean())
    draws = bootstrap_weights @ matrix
    return {"estimate": point, "interval_95": np.quantile(draws, [0.025, 0.975]).tolist()}


def score_panel(source, truth_states, physical, fit, process, config):
    candidate = source["candidate_scores"]
    candidate_top = source["candidate_top2"]
    candidates_from_scores = np.stack([top_two(arm) for arm in candidate])
    np.testing.assert_array_equal(candidate_top, candidates_from_scores)
    true_top = top_two(physical)
    true_gap = physical[:, PAIR[1]] - physical[:, PAIR[0]]
    candidate_gap = candidate[..., PAIR[1]] - candidate[..., PAIR[0]]
    np.testing.assert_array_equal(candidate_gap, source["candidate_pair_gap"])
    wrong_pair = flip_matrix(candidate_gap, true_gap)
    wrong_set = np.any(candidate_top != true_top, axis=-1)
    forecast_gap = source["physical_pair_gap_prediction"]
    predicted_reversal = source["pair_reversal_prediction"]
    predicted_correct_set = source["correct_set_prediction"]
    np.testing.assert_array_equal(predicted_reversal, flip_matrix(candidate_gap[None], forecast_gap[:, None]))
    pair_true_sign_correct = (forecast_gap < 0) == (true_gap < 0)
    np.testing.assert_array_equal(predicted_reversal == wrong_pair, np.broadcast_to(
        pair_true_sign_correct[:, None], predicted_reversal.shape
    ))
    method_data = {
        method: {
            "physical_pair_accuracy": float(pair_true_sign_correct[index].mean()),
            "physical_score_mae": np.abs(source["physical_score_prediction"][index] - physical).mean(axis=0).tolist(),
            "reversal_confusion_by_arm": [confusion(predicted, true) for predicted, true in zip(predicted_reversal[index], wrong_pair)],
            "top2_changes_from_base": paired_changes(predicted_correct_set[index]),
        }
        for index, method in enumerate(METHODS)
    }
    recovery = recipient_recovery(source["visible_named_readout"], truth_states, fit["arms"])
    names = [arm["name"] for arm in fit["arms"]]
    actual_changes = paired_changes(~wrong_set)
    truth_boundary = np.sort(physical, axis=-1)
    margin = truth_boundary[:, 2] - truth_boundary[:, 1]
    bins = np.digitize(margin, [0.01, 0.03], right=True)
    risk_rows = []
    paired = []
    for index, cal in enumerate(fit["calibrated_risks"][process]):
        slow, fast = 1 + 2 * index, 2 + 2 * index
        pair_risk_difference = wrong_pair[slow].astype(float) - wrong_pair[fast]
        set_risk_difference = wrong_set[slow].astype(float) - wrong_set[fast]
        matrix = {
            "pair": cal["cal_pair_direction"] * pair_risk_difference,
            "top2": cal["cal_top2_direction"] * set_risk_difference,
            "pair_full_minus_diag": pair_true_sign_correct[1].astype(float) - pair_true_sign_correct[0],
        }
        paired.append(matrix)
        observed_difference = float(pair_risk_difference.mean())
        observed_set_difference = float(set_risk_difference.mean())
        r2_gap = abs(recovery[slow]["r2"] - recovery[fast]["r2"])
        risk_rows.append({
            **cal,
            "test_pair_slow_minus_fast": observed_difference,
            "test_top2_slow_minus_fast": observed_set_difference,
            "test_pair_direction_correct": cal["cal_pair_direction"] == int(np.sign(observed_difference)),
            "test_top2_direction_correct": cal["cal_top2_direction"] == int(np.sign(observed_set_difference)),
            "oriented_test_pair_difference": float(matrix["pair"].mean()),
            "oriented_test_top2_difference": float(matrix["top2"].mean()),
            "test_r2_absolute_difference": r2_gap,
            "matched": r2_gap <= config["r2_match_tolerance"],
        })
    return {
        "truth_physical_pair_axis3_below_axis2": int(np.count_nonzero(true_gap < 0)),
        "mean_true_pair_gap": float(true_gap.mean()),
        "recovery_by_arm": dict(zip(names, recovery)),
        "candidate_ref_variance_by_arm": dict(zip(names, source["reference_variance"].tolist())),
        "mean_candidate_pair_gap_by_arm": dict(zip(names, candidate_gap.mean(axis=1).tolist())),
        "pair_error_by_arm": dict(zip(names, wrong_pair.mean(axis=1).tolist())),
        "top2_changes_from_base": dict(zip(names, actual_changes)),
        "top2_set_change_count_vs_base_by_arm": dict(zip(names, np.any(candidate_top != candidate_top[:1], axis=-1).sum(axis=1).tolist())),
        "top2_accuracy_by_physical_gap_bin": {
            name: [
                {
                    "n": int(mask.sum()),
                    "accuracy": float((~values[mask]).mean()) if mask.any() else None,
                }
                for mask in (bins == 0, bins == 1, bins == 2)
            ]
            for name, values in zip(names, wrong_set)
        },
        "physical_forecast_methods": method_data,
        "donor_comparisons": risk_rows,
    }, paired


def aggregate(rows, matrices, weights, config):
    all_comparisons = [row for panel in rows.values() for row in panel["donor_comparisons"]]
    weighted = np.stack([row["pair"] for row in matrices])
    primary = intervals(weighted.mean(axis=0), weights)
    primary["passes"] = (
        primary["estimate"] >= config["primary_minimum_gain"] and primary["interval_95"][0] > 0
    )
    output = {
        "primary_oriented_pair_risk_difference": primary,
        "oriented_top2_risk_difference": intervals(np.mean([row["top2"] for row in matrices], axis=0), weights),
        "physical_pair_sign_full_minus_diagonal": intervals(
            np.mean([row["pair_full_minus_diag"] for row in matrices], axis=0), weights
        ),
        "state_match_count": sum(row["matched"] for row in all_comparisons),
        "total_comparisons": len(all_comparisons),
        "state_match_maximum_absolute_r2_gap": max(row["test_r2_absolute_difference"] for row in all_comparisons),
        "all_state_matches": all(row["matched"] for row in all_comparisons),
    }
    mask = [i for i, row in enumerate(all_comparisons) if row["matched"]]
    output["state_matched_only_posthoc"] = intervals(weighted[mask].mean(axis=0), weights) if mask else None
    for task in ("pair", "top2"):
        identified = [row for row in all_comparisons if row[f"cal_{task}_direction_identifiable"]]
        output[f"identifiable_{task}_count"] = len(identified)
        output[f"identifiable_{task}_direction_correct_count"] = sum(row[f"test_{task}_direction_correct"] for row in identified)
        output[f"mean_unsigned_cal_{task}_difference"] = float(np.mean([
            abs(row[f"cal_{task}_slow_minus_fast"]) for row in all_comparisons
        ]))
        output[f"mean_signed_test_{task}_difference"] = float(np.mean([
            row[f"test_{task}_slow_minus_fast"] for row in all_comparisons
        ]))
    return output


def main():
    forecast = HERE / "forecast"
    receipt = read(forecast / "RECEIPT.json")
    reveal = read(HERE / "REVEAL_RECEIPT.json")
    truth_path = HERE / "REVEAL.npz"
    if receipt["sources"] != sources() or reveal["forecast_receipt_sha256"] != digest(forecast / "RECEIPT.json"):
        raise ValueError("Forecast source or receipt changed.")
    if reveal["physical_sha256"] != digest(truth_path):
        raise ValueError("Physical archive changed.")
    config = receipt["config"]
    weights = np.random.default_rng(config["bootstrap_seed"]).multinomial(
        config["n_test"], np.full(config["n_test"], 1 / config["n_test"]), size=config["bootstrap_draws"]
    ) / config["n_test"]
    reports = {}
    with np.load(truth_path, allow_pickle=False) as targets:
        for stage in receipt["round_rng"]:
            physical_states = {process: targets[f"{stage}_{process}"] for process in receipt["process_names"]}
            physical_score = {
                process: calculate_moments(states, config["lags"]).diagonal(axis1=1, axis2=2)
                for process, states in physical_states.items()
            }
            rows, matrices = {}, []
            for model in receipt["model_names"]:
                fit_name = f"{model}_fit.json"
                if digest(forecast / fit_name) != receipt["fit_files"][fit_name]:
                    raise ValueError("CAL fit changed.")
                fit = read(forecast / fit_name)
                for process in receipt["process_names"]:
                    name = f"{stage}_{model}_{process}.npz"
                    if digest(forecast / name) != receipt["forecast_files"][name]:
                        raise ValueError(f"Forecast changed: {name}")
                    with np.load(forecast / name, allow_pickle=False) as source:
                        row, paired = score_panel(source, physical_states[process], physical_score[process], fit, process, config)
                    rows[f"{model}/{process}"] = row
                    matrices.extend(paired)
            reports[stage] = {"summary": aggregate(rows, matrices, weights, config), "panels": rows}
    write(HERE / "SUMMARY.json", {
        "forecast_receipt_sha256": reveal["forecast_receipt_sha256"],
        "physical_sha256": reveal["physical_sha256"],
        "reports": reports,
    })


if __name__ == "__main__":
    main()
