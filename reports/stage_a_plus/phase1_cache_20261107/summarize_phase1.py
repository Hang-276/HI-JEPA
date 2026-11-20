"""Deterministically aggregate four revealed Phase 1 batches.

This module never fits a model or opens array/checkpoint files.  Importing it does
not read results.  Running it requires all four reveal_<slot>.json files and the
frozen PHASE1_CACHE_REVIEW_PROTOCOL.md in the parent directory. Step 6's module is loaded under a private
name; only that in-memory copy's method names are adapted.

Each reveal file has a ``rows`` list.  Required row fields are slot, rep,
episode, target, y, z, train_r2, eval_r2, predictions, physical_boundary_gap,
small_gap, and two block_diagnostics.  Prediction values are finite score4
vectors or null (unavailable, never imputed).  Each block requires
physical_top2 (two zero-based axes or null),
physical_margin_original_fine_set (finite scalar or null), and
predicted_margins (all seven method names, each a finite scalar or null).
Optional fine_top2 is reported when present.  physical_boundary_gap and
small_gap are supplied by the frozen reveal procedure, never re-thresholded.
"""

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
STEP6_PATH = HERE.parent / "step6_fine_20261105" / "summarize.py"
METHODS = (
    "identity", "size", "direction", "persistence",
    "b_diag", "b_fixed", "b_current",
)
OLD_RULES = ("identity", "size", "direction")
CANDIDATE = "b_current"
COMPARATORS = ("b_fixed",) + OLD_RULES
GATE_METHODS = (CANDIDATE,) + COMPARATORS
EXPECTED_BATCHES = 4
EXPECTED_REPS = 3


def load_metrics():
    spec = importlib.util.spec_from_file_location("_phase1_step6_metrics", STEP6_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FORECASTS = METHODS
    module.ORACLE = ()
    module.METHODS = METHODS
    return module


@contextmanager
def methods_for(metrics, methods):
    """Adapt only the private module, restoring it even after an exception."""
    previous = metrics.METHODS
    metrics.METHODS = tuple(methods)
    try:
        yield
    finally:
        metrics.METHODS = previous


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def finite_vector(value):
    return isinstance(value, list) and len(value) == 4 and all(finite_number(v) for v in value)


def valid_set(value, nullable=False):
    if value is None:
        return nullable
    return (isinstance(value, list) and len(value) == 2 and
            all(type(v) is int and 0 <= v < 4 for v in value) and len(set(value)) == 2)


def row_key(row):
    return row["slot"], row["episode"], row["rep"]


def episode_key(row):
    return row["slot"], row["episode"]


def available(row, method):
    return finite_vector(row["predictions"][method])


def scope(rows):
    return {
        "rows": len(rows),
        "physical_episodes": len({episode_key(r) for r in rows}),
        "slots": len({r["slot"] for r in rows}),
        "parents": len({(r["slot"], r["rep"]) for r in rows}),
    }


def validate_rows(rows, metrics):
    if not rows:
        raise ValueError("No revealed rows")
    seen = set()
    families = defaultdict(list)
    for row in rows:
        if not isinstance(row.get("slot"), str) or not row["slot"]:
            raise ValueError("Invalid slot")
        if any(type(row.get(k)) is not int or row[k] < 0 for k in ("rep", "episode")):
            raise ValueError("Invalid rep/episode")
        key = row_key(row)
        if key in seen:
            raise ValueError(f"Duplicate revealed row: {key}")
        seen.add(key)
        if not valid_set(row.get("target")):
            raise ValueError(f"Invalid target: {key}")
        for name in ("y", "z", "train_r2", "eval_r2"):
            if not finite_vector(row.get(name)):
                raise ValueError(f"Invalid {name}: {key}")
        forecasts = row.get("predictions")
        if not isinstance(forecasts, dict) or not set(METHODS).issubset(forecasts):
            raise ValueError(f"Missing forecast methods: {key}")
        for method in METHODS:
            if forecasts[method] is not None and not finite_vector(forecasts[method]):
                raise ValueError(f"Malformed forecast {method}: {key}")
        if not finite_number(row.get("physical_boundary_gap")):
            raise ValueError(f"Invalid physical_boundary_gap: {key}")
        if type(row.get("small_gap")) is not bool:
            raise ValueError(f"small_gap must use the frozen Boolean classification: {key}")
        blocks = row.get("block_diagnostics")
        if not isinstance(blocks, list) or len(blocks) != 2:
            raise ValueError(f"Exactly two block diagnostics required: {key}")
        for index, block in enumerate(blocks):
            if not isinstance(block, dict) or "physical_top2" not in block:
                raise ValueError(f"Missing physical block set: {key}, block {index}")
            if not valid_set(block["physical_top2"], nullable=True):
                raise ValueError(f"Invalid physical block set: {key}, block {index}")
            if "fine_top2" in block and not valid_set(block["fine_top2"], nullable=True):
                raise ValueError(f"Invalid Fine block set: {key}, block {index}")
            if "physical_margin_original_fine_set" not in block:
                raise ValueError(f"Missing fixed-set block margin: {key}, block {index}")
            value = block["physical_margin_original_fine_set"]
            if value is not None and not finite_number(value):
                raise ValueError(f"Invalid fixed-set block margin: {key}, block {index}")
            predictions = block.get("predicted_margins")
            if not isinstance(predictions, dict) or not set(METHODS).issubset(predictions):
                raise ValueError(f"Missing block forecast margins: {key}, block {index}")
            if any(v is not None and not finite_number(v) for v in predictions.values()):
                raise ValueError(f"Invalid block forecast margin: {key}, block {index}")
        families[episode_key(row)].append(row)

    reps_by_slot = defaultdict(set)
    for (slot, ep), family in families.items():
        reps = tuple(sorted(r["rep"] for r in family))
        if len(reps) != EXPECTED_REPS:
            raise ValueError(f"Expected three parents for {slot}, episode {ep}: {reps}")
        reps_by_slot[slot].add(reps)
        first = family[0]
        for other in family[1:]:
            if other["small_gap"] != first["small_gap"]:
                raise ValueError(f"Physical small-gap flag differs between parents: {(slot, ep)}")
            if not math.isclose(other["physical_boundary_gap"], first["physical_boundary_gap"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Physical boundary gap differs between parents: {(slot, ep)}")
            if metrics.top(other["z"]) != metrics.top(first["z"]):
                raise ValueError(f"Physical set differs between parents: {(slot, ep)}")
            for i in range(2):
                left = first["block_diagnostics"][i]["physical_top2"]
                right = other["block_diagnostics"][i]["physical_top2"]
                if (None if left is None else sorted(left)) != (None if right is None else sorted(right)):
                    raise ValueError(f"Physical block set differs between parents: {(slot, ep, i)}")
    if len(reps_by_slot) != EXPECTED_BATCHES:
        raise ValueError(f"Expected four slots, got {sorted(reps_by_slot)}")
    if any(len(reps) != 1 for reps in reps_by_slot.values()):
        raise ValueError("Parent panel changes between episodes")


def per_method_stats(rows, method, metrics):
    usable = [r for r in rows if available(r, method)]
    result = metrics.method_stats(usable, method)
    result["coverage"] = {
        "input": scope(rows), "finite_prediction": scope(usable),
        "missing_prediction_rows": len(rows) - len(usable),
        "note": "Step 6 statistics below use finite predictions; exact top2 ties are counted separately.",
    }
    return result


def metric_bundle(rows, metrics):
    by_slot = defaultdict(list)
    for row in rows:
        by_slot[row["slot"]].append(row)
    per_slot = {
        slot: {method: per_method_stats(items, method, metrics) for method in METHODS}
        for slot, items in sorted(by_slot.items())
    }
    return {
        "scope": scope(rows),
        "micro": {m: per_method_stats(rows, m, metrics) for m in METHODS},
        "macro": {m: metrics.macro(per_slot, m) for m in METHODS},
        "per_slot": per_slot,
    }


def pair_records(rows, metrics):
    # Generate the original matching/eligibility metadata without requiring an
    # available prediction.  Every matched pair remains in the audit output.
    with methods_for(metrics, ()):
        pairs = metrics.match_rows(rows)
    index = {row_key(row): row for row in rows}
    for pair in pairs:
        left, right = [index[(pair["slot"], pair["episode"], rep)] for rep in pair["rep"]]
        pair["small_gap"] = left["small_gap"]
        pair["physical_boundary_gap"] = left["physical_boundary_gap"]
        pair["method_available"] = {}
        for method in METHODS:
            present = available(left, method) and available(right, method)
            pair["method_available"][method] = present
            margins = [
                metrics.margin(row["predictions"][method], metrics.top(row["y"]))
                if available(row, method) else None for row in (left, right)
            ]
            pair["predicted_physical_margins_for_fine_set"][method] = margins
            pair["concordance"][method] = None
            if pair["discordant"] and present:
                good = pair["rep"].index(pair["correct_rep"])
                difference = margins[good] - margins[1 - good]
                pair["concordance"][method] = (
                    "tie" if abs(difference) <= metrics.MARGIN_TIE
                    else "win" if difference > 0 else "loss"
                )
        pair["all_method_common_available"] = all(pair["method_available"].values())
        pair["gate_common_available"] = all(pair["method_available"][m] for m in GATE_METHODS)
    return pairs


def pair_scope(pairs):
    return {
        "pairs": len(pairs),
        "physical_episodes": len({(p["slot"], p["episode"]) for p in pairs}),
        "slots": len({p["slot"] for p in pairs}),
    }


def comparison_bundle(pairs, metrics):
    common = [p for p in pairs if p["all_method_common_available"]]
    result = {
        "all_original_matched": pair_scope(pairs),
        "all_original_discordant": pair_scope([p for p in pairs if p["discordant"]]),
        "common_all_methods": metrics.comparison_stats(common),
        "common_all_methods_excluded": pair_scope([p for p in pairs if not p["all_method_common_available"]]),
        "per_method_available": {},
    }
    for method in METHODS:
        usable = [p for p in pairs if p["method_available"][method]]
        with methods_for(metrics, (method,)):
            result["per_method_available"][method] = metrics.comparison_stats(usable)
    return result


def matched_bundle(pairs, slots, metrics):
    result = {}
    for name, selected in (
        ("train", pairs),
        ("train_and_eval", [p for p in pairs if p["strict_also_eval"]]),
    ):
        result[name] = {
            "overall": comparison_bundle(selected, metrics),
            "per_slot": {
                slot: comparison_bundle([p for p in selected if p["slot"] == slot], metrics)
                for slot in slots
            },
        }
    return result


def common_rows(rows, methods, metrics, classified=False):
    result = []
    for row in rows:
        if not all(available(row, m) for m in methods):
            continue
        if classified and (metrics.top(row["y"]) is None or metrics.top(row["z"]) is None or
                           any(metrics.top(row["predictions"][m]) is None for m in methods)):
            continue
        result.append(row)
    return result


def coverage(rows, metrics):
    both = {
        "all_seven": METHODS,
        "decision_candidate_and_comparators": GATE_METHODS,
    }
    result = {"input": scope(rows), "common": {}, "excluded_rows": []}
    for name, methods in both.items():
        finite = common_rows(rows, methods, metrics)
        classified = common_rows(rows, methods, metrics, classified=True)
        result["common"][name] = {
            "methods": list(methods), "finite_score_vectors": scope(finite),
            "unique_top2_for_every_method_and_outcome": scope(classified),
            "missing_score_rows": len(rows) - len(finite),
            "exact_top2_tie_rows_after_finite_filter": len(finite) - len(classified),
        }
    for row in rows:
        missing = [m for m in METHODS if not available(row, m)]
        ties = [m for m in METHODS if available(row, m) and metrics.top(row["predictions"][m]) is None]
        outcome_tie = metrics.top(row["y"]) is None or metrics.top(row["z"]) is None
        if missing or ties or outcome_tie:
            result["excluded_rows"].append({
                "slot": row["slot"], "episode": row["episode"], "rep": row["rep"],
                "unavailable_methods": missing, "exact_top2_tie_methods": ties,
                "outcome_top2_tie": outcome_tie,
            })
    return result


def wrong_flag_counts(rows, method, metrics):
    stats = metrics.method_stats(rows, method)
    c = stats["correctness_confusion"]
    return {
        "true_wrong_detected": c["TN"], "wrong_cases_missed": c["FP"],
        "correct_cases_falsely_flagged": c["FN"], "correct_cases_not_flagged": c["TP"],
        "wrong_case_detection_rate": stats["wrong_parent_detection"],
        "wrong_flag_false_positive_rate": stats["wrong_flag_false_alarm"],
        "classified_rows": stats["n_classified"],
    }


def decision_bundle(rows, pairs, metrics):
    # Equal coverage is enforced jointly, never by comparing each method's own
    # available subset.  Exact forecast top2 ties are abstentions for this gate.
    usable = common_rows(rows, GATE_METHODS, metrics, classified=True)
    counts = {m: wrong_flag_counts(usable, m, metrics) for m in GATE_METHODS}
    comparison = {}
    for baseline in COMPARATORS:
        comparison[baseline] = {
            "additional_wrong_rows_detected": counts[CANDIDATE]["true_wrong_detected"] - counts[baseline]["true_wrong_detected"],
            "additional_false_positive_rows": counts[CANDIDATE]["correct_cases_falsely_flagged"] - counts[baseline]["correct_cases_falsely_flagged"],
        }
        comparison[baseline]["passes"] = (
            comparison[baseline]["additional_wrong_rows_detected"] > 0 and
            comparison[baseline]["additional_false_positive_rows"] <= 0
        )
    classification_gate = bool(usable) and all(c["passes"] for c in comparison.values())
    classification = {
        "common_coverage": scope(usable), "method_counts": counts,
        "comparisons": comparison, "passes": classification_gate,
        "strongest_old_detection_count": max(counts[m]["true_wrong_detected"] for m in OLD_RULES),
        "strongest_old_detection_rules": [m for m in OLD_RULES if counts[m]["true_wrong_detected"] == max(counts[x]["true_wrong_detected"] for x in OLD_RULES)],
        "rule": "Candidate detects strictly more wrong rows than b_fixed and every old rule, with no more false positives than any comparator, on identical classifiable rows.",
    }
    ranking = {}
    for name, pool in (("train", pairs), ("train_and_eval", [p for p in pairs if p["strict_also_eval"]])):
        common = [p for p in pool if p["gate_common_available"]]
        with methods_for(metrics, GATE_METHODS):
            stats = metrics.comparison_stats(common)
        slots = {}
        for slot in sorted({r["slot"] for r in rows}):
            slot_pairs = [p for p in common if p["slot"] == slot]
            with methods_for(metrics, GATE_METHODS):
                slot_stats = metrics.comparison_stats(slot_pairs)
            scores = {
                m: slot_stats["methods"][m]["equal_slot_episode_pair_concordance"] for m in GATE_METHODS
            }
            complete = all(v is not None for v in scores.values())
            old_best = max(scores[m] for m in OLD_RULES) if complete else None
            passed = complete and scores[CANDIDATE] > scores["b_fixed"] and scores[CANDIDATE] > old_best
            slots[slot] = {
                "concordance": scores, "strongest_old_concordance": old_best,
                "strictly_beats_b_fixed_and_best_old": passed,
                "discordant_scope": pair_scope([p for p in slot_pairs if p["discordant"]]),
            }
        winning_slots = [slot for slot, item in slots.items() if item["strictly_beats_b_fixed_and_best_old"]]
        passed = len(winning_slots) >= 2
        ranking[name] = {
            "common_comparison": stats, "per_slot": slots,
            "winning_slots": winning_slots, "passes_two_slot_gate": passed,
            "status": (
                "candidate_for_independent_confirmation" if passed and classification_gate
                else "ranking_only_development" if passed
                else "development_only_gate_not_passed"
            ),
        }
    return {
        "candidate": CANDIDATE, "comparators": list(COMPARATORS),
        "classification": classification, "ranking": ranking,
        "overall_status": ranking["train"]["status"],
        "primary_matching_status": ranking["train"]["status"],
        "status_basis": "TRAIN-only matching is the primary development gate; strict TRAIN+CAL matching is reported as a post-label stratum.",
        "interpretation": "All CAL is exposed development data. Passing is permission to consider an independent frozen confirmation, not confirmation itself; pairs sharing a trajectory are dependent.",
    }


def scalar_state(values):
    if any(v is None for v in values):
        return "unavailable"
    if any(v == 0 for v in values):
        return "exact_zero_in_one_or_both_blocks"
    left, right = (v > 0 for v in values)
    return ("positive_both" if left and right else "negative_both" if not left and not right
            else "positive_then_negative" if left else "negative_then_positive")


def set_state(values):
    if any(v is None for v in values):
        return "exact_top2_tie_in_one_or_both_blocks"
    return "same_set" if sorted(values[0]) == sorted(values[1]) else "changed_set"


def diagnostic_bundle(rows, metrics):
    physical = {}
    by_row = []
    physical_counts = Counter()
    true_counts = Counter()
    fine_counts = Counter()
    forecast_counts = {m: Counter() for m in METHODS}
    forecast_episodes = {m: defaultdict(set) for m in METHODS}
    true_episodes = defaultdict(set)
    for row in rows:
        blocks = row["block_diagnostics"]
        key = episode_key(row)
        physical_sets = [b["physical_top2"] for b in blocks]
        physical_state = set_state(physical_sets)
        if key not in physical:
            physical[key] = {
                "slot": row["slot"], "episode": row["episode"], "small_gap": row["small_gap"],
                "physical_boundary_gap": row["physical_boundary_gap"],
                "physical_top2_blocks": physical_sets, "physical_set_stability": physical_state,
            }
            physical_counts[physical_state] += 1
        actual = [b["physical_margin_original_fine_set"] for b in blocks]
        actual_state = scalar_state(actual)
        true_counts[actual_state] += 1
        true_episodes[actual_state].add(key)
        forecasts = {m: [b["predicted_margins"][m] for b in blocks] for m in METHODS}
        states = {m: scalar_state(values) for m, values in forecasts.items()}
        for method, state in states.items():
            forecast_counts[method][state] += 1
            forecast_episodes[method][state].add(key)
        fine_state = "not_supplied"
        if all("fine_top2" in b for b in blocks):
            fine_state = set_state([b["fine_top2"] for b in blocks])
        fine_counts[fine_state] += 1
        by_row.append({
            "slot": row["slot"], "episode": row["episode"], "rep": row["rep"],
            "small_gap": row["small_gap"], "original_fine_top2": metrics.top(row["y"]),
            "physical_set_stability": physical_state,
            "actual_original_set_block_margins": actual, "actual_original_set_block_state": actual_state,
            "predicted_block_margins": forecasts, "predicted_block_states": states,
            "block_local_fine_set_stability": fine_state,
        })
    return {
        "scope": scope(rows), "physical_set_stability_by_trajectory": dict(physical_counts),
        "fixed_original_fine_set_margin_states_by_row": dict(true_counts),
        "fixed_original_fine_set_state_trajectory_incidence": {k: len(v) for k, v in true_episodes.items()},
        "block_local_fine_set_stability_by_row": dict(fine_counts),
        "prediction_margin_states_by_row": {m: dict(v) for m, v in forecast_counts.items()},
        "prediction_state_trajectory_incidence": {m: {k: len(v) for k, v in states.items()} for m, states in forecast_episodes.items()},
        "trajectory_incidence_note": "A trajectory with different parents in different states appears in multiple state categories; incidence categories are not additive.",
        "unit_note": "Two blocks, lags, axes, and seed pairs do not add independent trajectories. Block differences do not identify sampling noise versus nonstationarity.",
        "trajectories": [physical[k] for k in sorted(physical)], "rows": by_row,
    }


def near_boundary_bundle(rows, pairs, metrics):
    result = {
        "definition": "Use the supplied frozen small_gap flag; no threshold fitting or reselection.",
        "overall_scope": scope(rows), "groups": {},
    }
    slots = sorted({r["slot"] for r in rows})
    for flag, label in ((True, "small_gap"), (False, "other_gap")):
        group = [r for r in rows if r["small_gap"] is flag]
        selected_pairs = [p for p in pairs if p["small_gap"] is flag]
        result["groups"][label] = {
            "scope": scope(group), "row_fraction": metrics.divide(len(group), len(rows)),
            "physical_episode_fraction": metrics.divide(scope(group)["physical_episodes"], scope(rows)["physical_episodes"]),
            "metrics": metric_bundle(group, metrics), "coverage": coverage(group, metrics),
            "matched": matched_bundle(selected_pairs, slots, metrics),
            "block_diagnostics": diagnostic_bundle(group, metrics),
        }
    return result


def main():
    paths = sorted(HERE.glob("reveal_*.json"))
    if len(paths) != EXPECTED_BATCHES:
        raise ValueError(f"Require all four reveal_<slot>.json batches before aggregation; found {len(paths)}")
    protocol = HERE.parent / "PHASE1_CACHE_REVIEW_PROTOCOL.md"
    if not protocol.is_file():
        raise FileNotFoundError(protocol)
    rows, inputs, slot_batches = [], [], set()
    for path in paths:
        content = path.read_bytes()
        raw = json.loads(content)
        batch = raw["rows"]
        if not isinstance(batch, list) or not batch:
            raise ValueError(f"Empty or invalid batch: {path.name}")
        slots = {row["slot"] for row in batch}
        if len(slots) != 1 or slots & slot_batches:
            raise ValueError(f"Each reveal must contain one distinct slot: {path.name}")
        slot_batches.update(slots)
        if "expected_episode_count" in raw:
            if len({r["episode"] for r in batch}) != raw["expected_episode_count"]:
                raise ValueError(f"Episode completeness check failed: {path.name}")
        rows.extend(batch)
        inputs.append({
            "file": path.name, "sha256": hashlib.sha256(content).hexdigest(),
            "rows": len(batch), "slot": next(iter(slots)),
            "expected_episode_count": raw.get("expected_episode_count"),
        })
    rows.sort(key=row_key)
    metrics = load_metrics()
    validate_rows(rows, metrics)
    pairs = pair_records(rows, metrics)
    slots = sorted(slot_batches)
    stats = metric_bundle(rows, metrics)
    common = common_rows(rows, METHODS, metrics)
    out = {
        "schema": "phase1_summary_v1", "inputs": inputs,
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "reused_step6_metric_script_sha256": hashlib.sha256(STEP6_PATH.read_bytes()).hexdigest(),
        "methods": list(METHODS), "scope": scope(rows),
        "metric_conventions": {
            "score": "lower is slower; six unordered gaps have equal weight",
            "correctness_positive_class": "original Fine top2 equals current physical top2",
            "wrong_flag_confusion_mapping": "Step6 TN = wrong detected; Step6 FN = correct falsely flagged",
            "top2_ties": "exact equality at the second/third score boundary; abstain and count separately",
            "paired_margin_tie_absolute": metrics.MARGIN_TIE,
            "matching_max_axis_r2_difference": metrics.MATCH_LIMIT,
            "weighting": "slot -> physical trajectory -> unordered seed pair; pairs and blocks are dependent within trajectory",
            "coverage": "Per-method summaries use available vectors; all head-to-head gates use identical common coverage.",
        },
        "micro": stats["micro"], "macro": stats["macro"], "per_slot": stats["per_slot"],
        "coverage": coverage(rows, metrics),
        "all_method_common_finite_metrics": metric_bundle(common, metrics),
        "matched": matched_bundle(pairs, slots, metrics), "matched_pairs": pairs,
        "decisions": decision_bundle(rows, pairs, metrics),
        "near_boundary": near_boundary_bundle(rows, pairs, metrics),
        "block_diagnostics": diagnostic_bundle(rows, metrics),
        "target_alignment": {
            "micro": metrics.row_states(rows),
            "per_slot": {s: metrics.row_states([r for r in rows if r["slot"] == s]) for s in slots},
        },
        "limitations": [
            "All results are retrospective development diagnostics on previously exposed CAL.",
            "A passed candidate gate is not independent validation or a deployable certificate.",
            "The root reproduces the old reference separately; this script makes no reproduction assertion.",
            "The producer is responsible for complete episode inventories when expected_episode_count is absent.",
        ],
    }
    destination = HERE / "SUMMARY.json"
    temporary = HERE / "SUMMARY.json.tmp"
    temporary.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(json.dumps({"output": str(destination), "scope": out["scope"], "status": out["decisions"]["overall_status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
