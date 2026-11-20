"""Descriptive reanalysis of archived A1 forecasts; no model loading or fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "reports/lejepa_natural_direction_repair_20261113"
ARMS = ["base", "target", "reverse", "wrong_donor", "joint_linear"]
EDGES = [0, .05, .1, .2, .4, 1, float("inf")]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def top2(values):
    return np.sort(np.argsort(values, axis=-1, kind="stable")[..., :2], axis=-1)


def describe(outcomes, mask):
    selected = outcomes[mask]
    n = len(selected)
    return {"n": n, "repairs": int((selected == 1).sum()),
            "harms": int((selected == -1).sum()), "unchanged": int((selected == 0).sum()),
            "net_pp": 100 * float(selected.mean()) if n else None}


def main():
    summary = read(SOURCE / "SUMMARY.json")
    receipt = read(SOURCE / "forecast/RECEIPT.json")
    config = read(SOURCE / "protocol.json")
    reveal_path = SOURCE / "REVEAL.npz"
    assert digest(reveal_path) == summary["reveal_sha256"]
    used = [SOURCE / "SUMMARY.json", SOURCE / "protocol.json", reveal_path,
            SOURCE / "forecast/RECEIPT.json", HERE / "ANALYSIS_PLAN.md", Path(__file__)]
    physical = {}
    with np.load(reveal_path, allow_pickle=False) as archive:
        for key in archive.files:
            z = archive[key]
            scores = sum(((z[:, lag:] - z[:, :-lag]) ** 2).mean(axis=1)
                         for lag in config["lags"]) / len(config["lags"])
            physical[key] = top2(scores)
    result = {"scope": "exploratory archived TEST reanalysis; no new independent validation",
              "gap_edges": [None if not np.isfinite(x) else x for x in EDGES], "rounds": {}}
    for stage in config["rounds"]:
        cells = summary["results"][stage]["cells"]
        records, outcomes_all, gaps_all, indications_all = [], [], [], []
        correct_all = {arm: [] for arm in ARMS}
        matched_gains = {arm: [] for arm in ["base", "reverse", "wrong_donor", "joint_linear"]}
        group_totals = {k: [0, 0, 0] for k in [-1, 0, 1]}
        strata, within_contrasts = [], []
        for key, row in cells.items():
            if row["cell"]["layer"] != "lag":
                continue
            file = SOURCE / "forecast" / stage / (key + ".npz")
            metadata_path = file.with_suffix(".json")
            for path in [file, metadata_path]:
                relative = str(path.relative_to(SOURCE))
                assert digest(path) == receipt["files"][relative], relative
                used.append(path)
            with np.load(file, allow_pickle=False) as archive:
                scores = archive["scores"]
                chosen = top2(scores)
                np.testing.assert_array_equal(chosen, archive["top2"])
            truth = physical[f'{stage}__{row["cell"]["process"]}']
            correct = np.all(chosen == truth[None], axis=-1)
            for i, arm in enumerate(ARMS):
                assert int(correct[i].sum()) == row["scores"][arm]["correct"]
                correct_all[arm].append(correct[i])
            outcome = correct[1].astype(int) - correct[0].astype(int)
            gap = np.sort(scores[0], axis=1)[:, 2] - np.sort(scores[0], axis=1)[:, 1]
            indication = (np.all(chosen[1] == chosen[4], axis=1).astype(int)
                          - np.all(chosen[0] == chosen[4], axis=1).astype(int))
            if not row["active"]:
                assert not np.any(outcome)
            audit = row["audit_by_arm"]
            r2 = {a: np.asarray(audit[a]["physical_recovery"]["r2"]) for a in ARMS}
            mse = {a: np.asarray(audit[a]["physical_recovery"]["mse"]) for a in ARMS}
            comparisons = {}
            for arm in matched_gains:
                index = ARMS.index(arm)
                distance = float(np.max(np.abs(r2["target"] - r2[arm])))
                gain = correct[1].astype(int) - correct[index].astype(int)
                matched = bool(row["active"] and distance <= .005)
                comparisons[arm] = {"max_axis_r2_difference": distance, "matched": matched,
                                    "target_minus_arm_accuracy_pp": 100 * float(gain.mean())}
                if matched:
                    matched_gains[arm].append(gain)
            rec = {"key": key, "name": row["cell"]["name"], "process": row["cell"]["process"],
                   "active": row["active"], "recipient": row["recipient"],
                   "target_mean_mse_change": float((mse["target"] - mse["base"]).mean()),
                   "target_recipient_r2_change": float((r2["target"] - r2["base"])[row["recipient"]]),
                   "joint_mean_mse_minus_target": float((mse["joint_linear"] - mse["target"]).mean()),
                   "outcome": describe(outcome, np.ones(len(outcome), dtype=bool)), "comparisons": comparisons}
            records.append(rec)
            outcomes_all.append(outcome)
            gaps_all.append(gap)
            indications_all.append(indication)
            for b, (left, right) in enumerate(zip(EDGES[:-1], EDGES[1:])):
                mask = (gap >= left) & (gap < right)
                positive = mask & (indication == 1)
                negative = mask & (indication == -1)
                stat = {"cell": key, "bin": b, **describe(outcome, mask),
                        "joint_supports_target": describe(outcome, positive),
                        "joint_supports_base": describe(outcome, negative)}
                strata.append(stat)
                if positive.any() and negative.any():
                    within_contrasts.append({"cell": key, "bin": b,
                                            "n_positive": int(positive.sum()), "n_negative": int(negative.sum()),
                                            "net_difference_pp": 100 * float(outcome[positive].mean() - outcome[negative].mean())})
        outcomes = np.concatenate(outcomes_all)
        gaps = np.concatenate(gaps_all)
        indications = np.concatenate(indications_all)
        active = [r for r in records if r["active"]]
        result["rounds"][stage] = {
            "n_cells": len(records), "active_cells": len(active), "n_innovation_rows": config["n_test"],
            "total_evaluations": len(outcomes),
            "accuracy_percent": {a: 100 * float(np.concatenate(v).mean()) for a, v in correct_all.items()},
            "overall_target_outcome": describe(outcomes, np.ones(len(outcomes), dtype=bool)),
            "audit": {"active_target_state_mse_improved": sum(r["target_mean_mse_change"] < 0 for r in active),
                      "active_joint_state_mse_better_than_target": sum(r["joint_mean_mse_minus_target"] < 0 for r in active),
                      "active_target_r2_change_range": [min(r["target_recipient_r2_change"] for r in active), max(r["target_recipient_r2_change"] for r in active)],
                      "active_with_repairs_and_harms": sum(r["outcome"]["repairs"] > 0 and r["outcome"]["harms"] > 0 for r in active)},
            "matched_r2_descriptive": {a: {"cells": len(g), "evaluations": len(g) * config["n_test"],
                                           "target_minus_arm_pp": 100 * float(np.mean(g)) if g else None}
                                       for a, g in matched_gains.items()},
            "gap_strata": [{"left": left, "right": right if np.isfinite(right) else None,
                             **describe(outcomes, (gaps >= left) & (gaps < right))}
                            for left, right in zip(EDGES[:-1], EDGES[1:])],
            "joint_reference_indication": {str(k): describe(outcomes, indications == k) for k in [-1, 0, 1]},
            "within_cell_gap_strata": strata,
            "within_cell_contrasts": {"n_strata_both_indications": len(within_contrasts),
                                      "positive_difference": sum(x["net_difference_pp"] > 0 for x in within_contrasts),
                                      "nonpositive_difference": sum(x["net_difference_pp"] <= 0 for x in within_contrasts),
                                      "equal_stratum_mean_difference_pp": float(np.mean([x["net_difference_pp"] for x in within_contrasts])) if within_contrasts else None,
                                      "details": within_contrasts},
            "cells": records,
        }
    result["inputs_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in dict.fromkeys(used)}
    with (HERE / "SUMMARY.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    for stage, values in result["rounds"].items():
        print(stage, json.dumps({k: v for k, v in values.items() if k not in
                               ["within_cell_gap_strata", "within_cell_contrasts", "cells"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
