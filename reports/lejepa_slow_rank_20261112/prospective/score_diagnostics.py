"""Lock historical R² matching and describe prospective outcomes after the primary reveal."""

import argparse
import json
from pathlib import Path

import numpy as np

from score import METHODS, N_TEST, confidence, risk, sha256, top2, verify_lock, write_json


HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent / "training"
SEPARATED_CASES = ("separated_wide", "separated_narrow")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def match_models():
    seeds = sorted(load(TRAINING / "protocol.json")["seeds"])
    if len(seeds) != 4:
        raise ValueError("The locked matching requires exactly four seeds per condition.")
    matches = []
    wide, narrow = SEPARATED_CASES
    for seed in seeds:
        path_wide = TRAINING / "runs" / f"{wide}_{seed}.json"
        path_narrow = TRAINING / "runs" / f"{narrow}_{seed}.json"
        r2_wide = min(load(path_wide)["oracle_linear_factor_r2"])
        r2_narrow = min(load(path_narrow)["oracle_linear_factor_r2"])
        distance = abs(r2_wide - r2_narrow)
        matches.append(
            {
                "seed": seed,
                "historical_min_axis_r2_wide": r2_wide,
                "historical_min_axis_r2_narrow": r2_narrow,
                "absolute_r2_distance": distance,
                "included": distance <= 0.002,
                "prior_wide_record_sha256": sha256(path_wide),
                "prior_narrow_record_sha256": sha256(path_narrow),
            }
        )
    return matches


def compare_pairs(matches, main_results):
    by_seed = {
        (entry["case"], entry["model_seed"]): entry
        for entry in main_results["nominal_by_case_model_seed"]
    }
    results = []
    for entry in matches:
        if not entry["included"]:
            continue
        wide = by_seed[SEPARATED_CASES[0], entry["seed"]]
        narrow = by_seed[SEPARATED_CASES[1], entry["seed"]]
        results.append(
            {
                **entry,
                "test_wide": wide,
                "test_narrow": narrow,
                "test_error_narrow_minus_wide": narrow["nominal_top2_error"] - wide["nominal_top2_error"],
            }
        )
    return results


def evaluate_slices(ranks, wrong, cases, case_names, physical_margin, frozen):
    results = {method: [] for method in METHODS}
    margin_bins = (
        ("up_to_0.01", physical_margin <= 0.01),
        ("0.01_to_0.03", (physical_margin > 0.01) & (physical_margin <= 0.03)),
        ("over_0.03", physical_margin > 0.03),
    )
    for method, rules in frozen["rules"].items():
        for rule in rules:
            accepted = ranks[method] >= rule["threshold"]
            by_case = {}
            for case_id, case_name in enumerate(case_names):
                mask = cases == case_id
                by_case[case_name] = risk(
                    int((accepted & mask).sum()), int((accepted & wrong & mask).sum()), int(mask.sum())
                )
            by_margin = {}
            for label, mask in margin_bins:
                by_margin[label] = risk(
                    int((accepted & mask).sum()), int((accepted & wrong & mask).sum()), int(mask.sum())
                )
            results[method].append(
                {"dev_target_coverage": rule["dev_target_coverage"], "by_case": by_case, "by_physical_margin": by_margin}
            )
    return results


def diagnostics():
    forecast = HERE / "forecasts.npz"
    reveal = HERE / "reveal.npz"
    match_lock = load(HERE / "R2_MATCH_LOCK.json")
    matches = match_models()
    if match_lock["pairs"] != matches or match_lock["scorer_sha256"] != sha256(Path(__file__)):
        raise ValueError("The historical match lock or source records have changed.")
    frozen = verify_lock(forecast, HERE / "PROTOCOL.md", HERE / "THRESHOLD_LOCK.json")
    main_results = load(HERE / "SCORE_SUMMARY.json")
    if main_results["reveal_sha256"] != sha256(reveal):
        raise ValueError("Diagnostic reveal differs from the primary reveal.")

    with np.load(forecast, allow_pickle=False) as source, np.load(reveal, allow_pickle=False) as target:
        ranks = confidence(source["test_scores"], source["test_uncertainty"], N_TEST)
        wrong = np.any(top2(source["test_scores"]) != top2(target["test_physical_scores"]), axis=1)
        ordered = np.sort(target["test_physical_scores"], axis=1)
        physical_margin = ordered[:, 2] - ordered[:, 1]
        cases = source["bank_case"][source["test_bank"]]
        slices = evaluate_slices(ranks, wrong, cases, source["case_names"].tolist(), physical_margin, frozen)

    write_json(
        HERE / "SCORE_DIAGNOSTICS.json",
        {
            "scope": "descriptive, without simultaneous subgroup risk bounds",
            "matched_historical_r2_comparisons": compare_pairs(matches, main_results),
            "fixed_rule_composition": slices,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "reveal"))
    args = parser.parse_args()
    if args.stage == "freeze":
        write_json(
            HERE / "R2_MATCH_LOCK.json",
            {
                "rule": "Compare separated wide/narrow with identical seeds. Include only prior weakest axis linear R² absolute difference at most 0.002.",
                "scorer_sha256": sha256(Path(__file__)),
                "pairs": match_models(),
            },
        )
    else:
        diagnostics()


if __name__ == "__main__":
    main()
