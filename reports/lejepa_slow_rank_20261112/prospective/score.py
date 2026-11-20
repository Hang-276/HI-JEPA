"""Freeze label-free thresholds, then score a separately sealed prospective reveal."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


TARGET_COVERAGES = (0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8)
METHODS = ("gap_only", "uncertainty_aware")
ERROR_LIMIT = 0.05
FAMILY_ALPHA = 0.05
PER_RULE_ALPHA = FAMILY_ALPHA / (len(TARGET_COVERAGES) * len(METHODS))
N_BANKS = 64
N_DEV_PER_BANK = 128
N_TEST = 8192
SCORER = Path(__file__).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, result):
    with path.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")


def top2(scores):
    return np.sort(np.argsort(scores, axis=1, kind="stable")[:, :2], axis=1)


def confidence(scores, uncertainty, expected_count):
    if scores.shape != (expected_count, 4) or uncertainty.shape != (expected_count,):
        raise ValueError("Unexpected nominal score or uncertainty shape.")
    if not (np.isfinite(scores).all() and np.isfinite(uncertainty).all()):
        raise ValueError("Nominal and uncertainty scores must all be finite.")
    ordered = np.sort(scores, axis=1)
    return {"gap_only": ordered[:, 2] - ordered[:, 1], "uncertainty_aware": uncertainty}


def bank_ids(values, count):
    if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Bank IDs must be a flat integer array of the expected length.")
    if np.any((values < 0) | (values >= N_BANKS)):
        raise ValueError("Bank ID is outside the locked bank range.")
    return values


def freeze(forecasts, protocol, lock):
    with np.load(forecasts, allow_pickle=False) as source:
        dev_bank = bank_ids(source["dev_bank"], N_BANKS * N_DEV_PER_BANK)
        if not np.all(np.bincount(dev_bank, minlength=N_BANKS) == N_DEV_PER_BANK):
            raise ValueError("Development sample is not balanced over the 64 fixed banks.")
        ranks = confidence(source["dev_scores"], source["dev_uncertainty"], len(dev_bank))

    frozen_rules = {}
    for method, values in ranks.items():
        descending = np.sort(values)[::-1]
        rules = []
        for target in TARGET_COVERAGES:
            cutoff_rank = math.ceil(target * len(values))
            threshold = float(descending[cutoff_rank - 1])
            rules.append(
                {
                    "dev_target_coverage": target,
                    "threshold": threshold,
                    "dev_accepted_count": int(np.count_nonzero(values >= threshold)),
                }
            )
        frozen_rules[method] = rules

    write_json(
        lock,
        {
            "forecasts_sha256": sha256(forecasts),
            "protocol_sha256": sha256(protocol),
            "scorer_sha256": sha256(SCORER),
            "dev_count": N_BANKS * N_DEV_PER_BANK,
            "expected_test_count": N_TEST,
            "expected_banks": N_BANKS,
            "accepts_when": "score >= threshold",
            "error_limit": ERROR_LIMIT,
            "family_alpha": FAMILY_ALPHA,
            "per_rule_alpha": PER_RULE_ALPHA,
            "rules": frozen_rules,
        },
    )


def clopper_pearson_upper(errors, accepted, alpha):
    """Invert the exact binomial CDF in log space for a one-sided upper bound."""
    if not 0 <= errors <= accepted or not 0 < alpha < 1:
        raise ValueError("Invalid binomial count or alpha.")
    if accepted == 0:
        return None
    if errors == accepted:
        return 1.0
    if errors == 0:
        return -math.expm1(math.log(alpha) / accepted)

    k = np.arange(errors + 1, dtype=float)
    log_choose = np.zeros(errors + 1)
    log_choose[1:] = np.cumsum(np.log(accepted - k[1:] + 1) - np.log(k[1:]))
    log_alpha = math.log(alpha)
    low, high = 0.0, 1.0
    for _ in range(56):
        p = (low + high) / 2
        log_probability = log_choose + k * math.log(p) + (accepted - k) * math.log1p(-p)
        if np.logaddexp.reduce(log_probability) > log_alpha:
            low = p
        else:
            high = p
    return high


def risk(accepted, errors, test_count):
    return {
        "accepted": accepted,
        "errors": errors,
        "coverage": accepted / test_count,
        "selective_error": errors / accepted if accepted else None,
        "unconditional_wrong_acceptance": errors / test_count,
    }


def evaluate_fixed_rules(values, wrong, rules):
    evaluations = []
    for rule in rules:
        accepted = values >= rule["threshold"]
        n_accepted = int(accepted.sum())
        n_wrong = int(np.count_nonzero(accepted & wrong))
        upper = clopper_pearson_upper(n_wrong, n_accepted, PER_RULE_ALPHA)
        observed = risk(n_accepted, n_wrong, len(wrong))
        if upper is None:
            reason = "no_test_accepts"
        elif upper <= ERROR_LIMIT:
            reason = "certified"
        elif observed["selective_error"] > ERROR_LIMIT:
            reason = "observed_error_exceeds_limit"
        elif clopper_pearson_upper(0, n_accepted, PER_RULE_ALPHA) > ERROR_LIMIT:
            reason = "too_few_accepts_even_with_zero_errors"
        else:
            reason = "observed_error_within_limit_but_upper_bound_exceeds_it"
        evaluations.append({**rule, **observed, "upper": upper, "status": reason})
    return evaluations


def matched_coverage(ranks, wrong):
    rankings = {method: np.argsort(-values, kind="stable") for method, values in ranks.items()}
    results = []
    for coverage in TARGET_COVERAGES:
        count = math.floor(coverage * len(wrong))
        risks = {
            method: risk(count, int(np.count_nonzero(wrong[ordering[:count]])), len(wrong))
            for method, ordering in rankings.items()
        }
        results.append(
            {
                "test_target_coverage": coverage,
                **risks,
                "uncertainty_minus_gap_selective_error": (
                    risks["uncertainty_aware"]["selective_error"] - risks["gap_only"]["selective_error"]
                ),
            }
        )
    return results


def baseline_groups(source, ids, wrong):
    bank_case = source["bank_case"]
    bank_seed = source["bank_model_seed"]
    case_names = source["case_names"].tolist()
    if bank_case.shape != (N_BANKS,) or bank_seed.shape != (N_BANKS,):
        raise ValueError("Bank case and model-seed metadata must be indexed by bank ID.")
    case_rows = bank_case[ids]
    seed_rows = bank_seed[ids]
    return [
        {
            "case": str(case_names[case]),
            "model_seed": int(seed),
            "test_count": int(mask.sum()),
            "nominal_top2_wrong": int(np.count_nonzero(mask & wrong)),
            "nominal_top2_error": float(wrong[mask].mean()) if np.any(mask) else None,
        }
        for case in np.unique(bank_case)
        for seed in np.unique(bank_seed[bank_case == case])
        for mask in [(case_rows == case) & (seed_rows == seed)]
    ]


def verify_lock(forecasts, protocol, lock):
    frozen = json.loads(lock.read_text(encoding="utf-8"))
    for path, field in [
        (forecasts, "forecasts_sha256"),
        (protocol, "protocol_sha256"),
        (SCORER, "scorer_sha256"),
    ]:
        if sha256(path) != frozen[field]:
            raise ValueError(f"Digest no longer matches the lock for {path.name}.")
    if tuple(frozen["rules"]) != METHODS or any(
        tuple(rule["dev_target_coverage"] for rule in frozen["rules"][method]) != TARGET_COVERAGES
        for method in METHODS
    ):
        raise ValueError("Locked rule grid differs from the scorer's fixed family.")
    return frozen


def reveal_score(forecasts, protocol, lock, reveal, output):
    frozen = verify_lock(forecasts, protocol, lock)
    with np.load(forecasts, allow_pickle=False) as source, np.load(reveal, allow_pickle=False) as target:
        if str(target["forecasts_sha256"].item()) != frozen["forecasts_sha256"]:
            raise ValueError("Reveal is not bound to the frozen forecasts.")
        ids = bank_ids(source["test_bank"], N_TEST)
        if not np.array_equal(ids, target["test_bank"]):
            raise ValueError("Forecast and reveal rows have different bank IDs.")

        nominal = source["test_scores"]
        ranks = confidence(nominal, source["test_uncertainty"], N_TEST)
        physical = target["test_physical_scores"]
        if physical.shape != (N_TEST, 4) or not np.isfinite(physical).all():
            raise ValueError("Physical reveal must have a complete, finite 4-axis row per test window.")
        wrong = np.any(top2(nominal) != top2(physical), axis=1)
        groups = baseline_groups(source, ids, wrong)

    fixed = {
        method: evaluate_fixed_rules(values, wrong, frozen["rules"][method])
        for method, values in ranks.items()
    }
    best = {}
    for method, evaluations in fixed.items():
        certified = [entry for entry in evaluations if entry["status"] == "certified"]
        best[method] = max(certified, key=lambda entry: entry["coverage"], default=None)
    certified_coverage = {
        method: best[method]["coverage"] if best[method] else 0.0
        for method in METHODS
    }
    result = {
        "lock_sha256": sha256(lock),
        "reveal_sha256": sha256(reveal),
        "test_count": len(wrong),
        "min_accepts_to_certify_with_zero_errors": math.ceil(
            math.log(PER_RULE_ALPHA) / math.log1p(-ERROR_LIMIT)
        ),
        "nominal_baseline": risk(len(wrong), int(wrong.sum()), len(wrong)),
        "nominal_by_case_model_seed": groups,
        "test_counts_by_bank": np.bincount(ids, minlength=N_BANKS).tolist(),
        "nominal_gap_score_quantiles": np.quantile(ranks["gap_only"], [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "uncertainty_score_quantiles": np.quantile(
            ranks["uncertainty_aware"], [0, 0.1, 0.5, 0.9, 1]
        ).tolist(),
        "uncertainty_score_positive_fraction": float((ranks["uncertainty_aware"] > 0).mean()),
        "primary_max_certified_coverage": certified_coverage,
        "uncertainty_minus_gap_certified_coverage": (
            certified_coverage["uncertainty_aware"] - certified_coverage["gap_only"]
        ),
        "best_certified_rule": best,
        "fixed_rule_results": fixed,
        "matched_realized_coverage_descriptive": matched_coverage(ranks, wrong),
    }
    write_json(output, result)


def main():
    directory = SCORER.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "reveal"))
    parser.add_argument("--forecasts", type=Path, default=directory / "forecasts.npz")
    parser.add_argument("--protocol", type=Path, default=directory / "PROTOCOL.md")
    parser.add_argument("--lock", type=Path, default=directory / "THRESHOLD_LOCK.json")
    parser.add_argument("--reveal", type=Path, default=directory / "reveal.npz")
    parser.add_argument("--output", type=Path, default=directory / "SCORE_SUMMARY.json")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.forecasts, args.protocol, args.lock)
    else:
        reveal_score(args.forecasts, args.protocol, args.lock, args.reveal, args.output)


if __name__ == "__main__":
    main()
