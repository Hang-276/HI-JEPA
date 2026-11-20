"""Score frozen factorial forecasts against a separately generated reveal."""

import itertools
from pathlib import Path

import numpy as np

from factorial import ARMS, HERE, TRAINING, banks, load, seed, sha256, top2, verify_forecast, write_json


FACTORS = ("CAL", "NAME", "REF")


def all_contrasts(correct):
    """Return all factorial effects with each unselected factor averaged."""
    values = {}
    for order in range(1, 4):
        for axes in itertools.combinations(range(3), order):
            signs = np.prod(2 * ARMS[:, axes] - 1, axis=1)
            values["×".join(FACTORS[axis] for axis in axes)] = correct @ signs / (2 ** (3 - order))
    return values


def bank_arm_score(forecast, target, config):
    scores = forecast["scores"]
    names = forecast["permutation"]
    scores_truth = target["physical_scores"]
    order = np.argsort(scores_truth, axis=-1, kind="stable")
    true_set = np.sort(order[..., :2], axis=-1)
    true_ordered = np.take_along_axis(scores_truth, order, axis=-1)
    truth_gap = true_ordered[..., 2] - true_ordered[..., 1]
    gap_bins = np.digitize(truth_gap, config["true_margin_bins"], right=True)
    estimate_at_truth = np.take_along_axis(scores, order[:, None], axis=-1)
    estimate_gap = estimate_at_truth[..., 2] - estimate_at_truth[..., 1]
    pred = forecast["top2"]
    np.testing.assert_array_equal(
        top2(scores.reshape(-1, 4)).reshape(pred.shape), pred
    )
    correct = np.all(pred == true_set[:, None], axis=-1)
    repair = correct & ~correct[:, :1]
    harm = ~correct & correct[:, :1]
    flip = estimate_gap <= 0
    abs_gap_error = np.abs(estimate_gap - truth_gap[:, None])

    # Oracle substitutions are a diagnostic using TEST-fitted conversion slopes.
    slope_squared = target["named_slopes"] ** 2
    physical_num = forecast["numerators"] / slope_squared[:, :, None]
    physical_var = target["reference_physical_scale"]
    oracle_combinations = {
        "physical_numerator_only": scores_truth[:, None] / physical_var[:, :, None],
        "physical_denominator_only": physical_num,
    }
    bank_records = []
    for bank_index, bank_permutations in enumerate(names):
        summary = []
        for arm, (cal, name, ref) in enumerate(ARMS):
            count = correct.shape[2]
            by_margin = []
            for margin_bin in range(3):
                mask = gap_bins[bank_index] == margin_bin
                by_margin.append(
                    {
                        "windows": int(mask.sum()),
                        "correct": int(correct[bank_index, arm, mask].sum()),
                        "boundary_sign_flips": int(flip[bank_index, arm, mask].sum()),
                        "boundary_gap_absolute_error_sum": float(abs_gap_error[bank_index, arm, mask].sum()),
                    }
                )
            substituted = {}
            for label, values in oracle_combinations.items():
                outcome = np.all(top2(values[bank_index, arm]) == true_set[bank_index], axis=1)
                substituted[label] = int(outcome.sum())
            low_name = cal * 4 + ref
            high_name = low_name + 2
            summary.append(
                {
                    "arm": [int(cal), int(name), int(ref)],
                    "windows": count,
                    "correct": int(correct[bank_index, arm].sum()),
                    "repairs_vs_all_low": int(repair[bank_index, arm].sum()),
                    "harms_vs_all_low": int(harm[bank_index, arm].sum()),
                    "boundary_sign_flips": int(flip[bank_index, arm].sum()),
                    "boundary_gap_absolute_error_sum": float(abs_gap_error[bank_index, arm].sum()),
                    "physical_margin_bins": by_margin,
                    "best_minus_second_naming_objective": float(forecast["assignment_gap"][bank_index, arm]),
                    "name_permutation_changed_low_to_high": bool(
                        np.any(bank_permutations[low_name] != bank_permutations[high_name])
                    ),
                    "name_only_select_set_changed_windows": int(
                        np.any(pred[bank_index, low_name] != pred[bank_index, high_name], axis=1).sum()
                    ),
                    "physical_slopes": target["physical_slopes"][bank_index, arm].tolist(),
                    "linear_cross_axis_fraction": target["linear_cross_axis_fraction"][bank_index, arm].tolist(),
                    "residual_fraction": target["residual_fraction"][bank_index, arm].tolist(),
                    "reference_physical_scale": target["reference_physical_scale"][bank_index, arm].tolist(),
                    "oracle_substitution_correct": substituted,
                }
            )
        bank_records.append(summary)
    return correct, bank_records


def per_arm_summary(banks):
    result = []
    for index, condition_arms in enumerate(zip(*banks)):
        aggregate = {}
        for count in (
            "windows", "correct", "repairs_vs_all_low", "harms_vs_all_low", "boundary_sign_flips",
            "boundary_gap_absolute_error_sum", "name_only_select_set_changed_windows",
        ):
            aggregate[count] = sum(entry[count] for entry in condition_arms)
        aggregate["arm"] = ARMS[index].tolist()
        aggregate["accuracy"] = aggregate["correct"] / aggregate["windows"]
        aggregate["boundary_gap_mae"] = aggregate["boundary_gap_absolute_error_sum"] / aggregate["windows"]
        for diagnostic in ("linear_cross_axis_fraction", "residual_fraction", "reference_physical_scale"):
            values = np.asarray([arm[diagnostic] for arm in condition_arms])
            if diagnostic == "reference_physical_scale":
                values = np.abs(values - 1)
            aggregate[f"median_{diagnostic}"] = np.median(values, axis=0).tolist()
        aggregate["name_permutation_changed_banks"] = sum(
            arm["name_permutation_changed_low_to_high"] for arm in condition_arms
        )
        aggregate["median_naming_objective_gap"] = float(
            np.median([arm["best_minus_second_naming_objective"] for arm in condition_arms])
        )
        aggregate["physical_margin_bins"] = [
            {key: sum(entry["physical_margin_bins"][b][key] for entry in condition_arms)
             for key in condition_arms[0]["physical_margin_bins"][b]}
            for b in range(3)
        ]
        aggregate["oracle_substitution_correct"] = {
            key: sum(entry["oracle_substitution_correct"][key] for entry in condition_arms)
            for key in condition_arms[0]["oracle_substitution_correct"]
        }
        result.append(aggregate)
    return result


def inference(correct, config, train):
    n_conditions = len(train["cases"])
    n_seeds = len(train["seeds"])
    n_banks = config["banks_per_checkpoint"]
    accuracy = correct.mean(axis=2).reshape(n_conditions, n_seeds, n_banks, len(ARMS))
    cases = list(train["cases"])
    rng = seed(config, "bank_bootstrap")
    indices = rng.integers(n_banks, size=(config["bank_bootstrap_draws"], n_conditions, n_seeds, n_banks))
    drawn = np.take_along_axis(accuracy[None], indices[..., None], axis=3)
    effects = all_contrasts(accuracy)
    boot_effects = all_contrasts(drawn)
    by_condition = {}
    for ci, case in enumerate(cases):
        simple = np.stack(
            [accuracy[ci, ..., 4 + index] - accuracy[ci, ..., index] for index in range(4)], axis=-1
        )
        by_condition[case] = {
            "effect_point": {label: float(value[ci].mean()) for label, value in effects.items()},
            "effect_by_seed": {label: value[ci].mean(axis=1).tolist() for label, value in effects.items()},
            "effect_bank_interval": {
                label: np.quantile(value[:, ci].mean(axis=(1, 2)), [0.025, 0.975]).tolist()
                for label, value in boot_effects.items()
            },
            "simple_cal_by_name_ref": simple.mean(axis=(0, 1)).tolist(),
            "simple_cal_by_seed_name_ref": simple.mean(axis=1).tolist(),
        }
    separated, crossed = [cases.index(name) for name in config["primary_conditions"]]
    point_seed = (effects["CAL"][separated] - effects["CAL"][crossed]).mean(axis=1)
    bootstrap_difference = (boot_effects["CAL"][:, separated] - boot_effects["CAL"][:, crossed]).mean(axis=(1, 2))
    interval = np.quantile(bootstrap_difference, [0.025, 0.975])
    return {
        "primary": {
            "conditions": config["primary_conditions"],
            "separated_minus_crossed_cal_gain": float(point_seed.mean()),
            "by_seed": dict(zip(train["seeds"], point_seed.tolist())),
            "conditional_bank_bootstrap_95": interval.tolist(),
            "passes": bool(interval[0] > 0 and point_seed.mean() > 0 and (point_seed > 0).sum() >= 3),
        },
        "by_condition": by_condition,
    }


def main():
    config = load(HERE / "protocol.json")
    train = load(TRAINING / "protocol.json")
    expected = verify_forecast(train)
    reveal_lock = load(HERE / "REVEAL_LOCK.json")
    if (reveal_lock["forecast_lock_sha256"] != sha256(HERE / "FORECAST_LOCK.json")
            or reveal_lock["reveal_sha256"] != sha256(HERE / "REVEAL.npz")):
        raise ValueError("Truth reveal integrity mismatch.")
    with np.load(HERE / "FORECAST.npz", allow_pickle=False) as forecast, np.load(HERE / "REVEAL.npz", allow_pickle=False) as target:
        if target["forecasts_sha256"].item() != expected["forecast_sha256"]:
            raise ValueError("Truth is unrelated to the frozen forecasts.")
        correct, bank_records = bank_arm_score(forecast, target, config)
    identities = banks(config, train)
    effects = inference(correct, config, train)
    condition_summary = {
        name: per_arm_summary([record for identity, record in zip(identities, bank_records) if identity[0] == name])
        for name in train["cases"]
    }
    write_json(
        HERE / "SUMMARY.json",
        {
            "window_count": len(identities) * config["test_windows_per_bank"],
            "forecast_sha256": expected["forecast_sha256"],
            "reveal_sha256": reveal_lock["reveal_sha256"],
            **effects,
            "arm_summaries_by_condition": condition_summary,
            "bank_results": [
                {
                    "case": name,
                    "seed": seed_id,
                    "repeat": repeat,
                    "rates": rates.tolist(),
                    "arms": record,
                    "effects": {
                        label: float(value)
                        for label, value in all_contrasts(correct[index].mean(axis=1)).items()
                    },
                }
                for index, ((name, seed_id, repeat, rates), record) in enumerate(zip(identities, bank_records))
            ],
        },
    )


if __name__ == "__main__":
    main()
