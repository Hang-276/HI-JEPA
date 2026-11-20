#!/usr/bin/env python3
"""L5: Head-to-head comparison of unlabeled `AUDIT + own_cal` vs. labeled LTT / SGR
(Angelopoulos et al., AoAS 2025; Geifman & El-Yaniv, NeurIPS 2017) under matched label budgets.

Directly addresses LITERATURE_GAPS_AND_POSITIONING_20261116.md line 131 using read-only
sealed predictions from:
1. Low-dim B1/B3 transfer (baseline, coupled, faster, slower) and A2 (strong vs weak);
2. Visual 32x32 sprites formal_v1 (visual_lejepa vs long_horizon_pred);
3. Official FourRoom H-JEPA fourroom_01 (seeds 42/43/44, Levels 1 & 2).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A3_DIR = ROOT / "reports/lejepa_a3_gap_20261115"
A4_DIR = ROOT / "reports/lejepa_a4_20261115"
B_DIR = ROOT / "reports/lejepa_b_transfer_20261115"
SPRITES_DIR = ROOT / "results/visual_sprites_lejepa/formal_v1"
FOURROOM_DIR = ROOT / "results/fourroom_hjepa/fourroom_01"
OUT = ROOT / "results/closeout_20261117/l5_selective_baseline_comparison"


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cp_upper(errors: int, accepted: int, alpha: float) -> float:
    """Exact one-sided Clopper-Pearson upper bound matching A3 score.py."""
    if accepted <= 0 or errors >= accepted:
        return 1.0
    log_alpha = math.log(alpha)
    if errors == 0:
        return -math.expm1(log_alpha / accepted)
    log_choose = math.lgamma(accepted + 1) - math.lgamma(errors + 1) - math.lgamma(accepted - errors + 1)
    lo, hi = errors / accepted, 1.0
    for _ in range(60):
        p = 0.5 * (lo + hi)
        if p == lo or p == hi:
            break
        odds = (1.0 - p) / p
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


def ltt_fixed_sequence_threshold(
    cal_gap: np.ndarray,
    cal_wrong: np.ndarray,
    alpha_cal: float = 0.05,
    max_risk: float = 0.01,
    grid_quantiles: np.ndarray | None = None,
) -> dict:
    """Learn-Then-Test (Angelopoulos et al., 2025, Sec 3.2) fixed-sequence calibration
    from highest threshold (strictest, lowest coverage) down to lowest threshold (highest coverage).

    Stops at the first candidate threshold in the ordered sequence whose exact one-sided
    binomial p-value exceeds alpha_cal (equivalently, exact CP upper bound > max_risk).
    """
    if grid_quantiles is None:
        # Ordered from high threshold (98th quantile -> ~2% coverage) down to 0th quantile (100% coverage)
        grid_quantiles = np.linspace(0.98, 0.00, 50)
    candidates = np.quantile(cal_gap, grid_quantiles, method="higher")
    best_tau = None
    best_cov_cal = 0.0
    best_acc_cal = 0
    best_err_cal = 0

    for tau in candidates:
        mask = cal_gap >= tau
        acc = int(mask.sum())
        err = int(cal_wrong[mask].sum())
        ub = cp_upper(err, acc, alpha_cal)
        if acc > 0 and ub <= max_risk:
            best_tau = float(tau)
            best_cov_cal = acc / len(cal_gap)
            best_acc_cal = acc
            best_err_cal = err
        else:
            # Fixed-sequence testing MUST stop at the first non-rejected null!
            # However, at extremely high quantile (e.g. 0.98), sample size `acc` may be too small
            # to certify even 0 errors (n < ceil(log(alpha)/log(1-max_risk)) = 299 for alpha=0.05, r=0.01)!
            # Therefore, we start fixed-sequence ONLY at the highest threshold that has at least
            # n_min_certifiable = ceil(log(alpha_cal) / log(1 - max_risk)) accepted CAL windows!
            pass

    # Properly anchored fixed-sequence starting at n_min_certifiable (optimal high-power LTT ordering!)
    n_min = math.ceil(math.log(alpha_cal) / math.log(1.0 - max_risk))
    if len(cal_gap) < n_min:
        return {
            "certified_on_cal": False,
            "reason": f"insufficient_labeled_cal_windows_need_at_least_{n_min}_accepted",
            "n_min_required_for_zero_errors": n_min,
            "threshold": None,
            "cal_coverage": 0.0,
        }

    # Sort gaps descending; index n_min - 1 is the highest threshold with >= n_min accepted samples!
    sorted_gaps = np.sort(cal_gap)[::-1]
    start_tau = sorted_gaps[n_min - 1]
    # Construct decreasing sequence from start_tau down to min(cal_gap)
    seq_taus = np.unique(np.concatenate([[start_tau], candidates[candidates <= start_tau]]))[::-1]

    chosen_tau = None
    chosen_stats = {}
    for tau in seq_taus:
        mask = cal_gap >= tau
        acc = int(mask.sum())
        err = int(cal_wrong[mask].sum())
        ub = cp_upper(err, acc, alpha_cal)
        if ub <= max_risk:
            chosen_tau = float(tau)
            chosen_stats = {
                "cal_accepted": acc,
                "cal_errors": err,
                "cal_coverage": acc / len(cal_gap),
                "cal_cp_upper": ub,
            }
        else:
            # Exact fixed-sequence stopping rule (Theorem 1 of Angelopoulos et al., 2025)
            break

    if chosen_tau is None:
        return {
            "certified_on_cal": False,
            "reason": "rejected_at_initial_anchor_threshold",
            "n_min_required_for_zero_errors": n_min,
            "threshold": None,
            "cal_coverage": 0.0,
        }
    return {
        "certified_on_cal": True,
        "n_min_required_for_zero_errors": n_min,
        "threshold": chosen_tau,
        **chosen_stats,
    }


def eval_on_test(
    test_gap: np.ndarray,
    test_wrong: np.ndarray,
    is_flip: np.ndarray | None,
    threshold: float | None,
    alpha_test: float,
) -> dict:
    if threshold is None:
        return {
            "abstained_entirely": True,
            "coverage": 0.0,
            "accepted": 0,
            "errors": 0,
            "selective_error": None,
            "cp_upper": 1.0,
            "passes_35pct_1pct": False,
            "flip_accepted": 0,
            "flip_errors": 0,
            "flip_accuracy_on_accepted": None,
        }
    mask = test_gap >= threshold
    acc = int(mask.sum())
    err = int(test_wrong[mask].sum())
    cov = acc / len(test_gap)
    ub = cp_upper(err, acc, alpha_test)
    res = {
        "abstained_entirely": False,
        "coverage": cov,
        "accepted": acc,
        "errors": err,
        "selective_error": err / acc if acc > 0 else None,
        "cp_upper": ub,
        "empirical_passes_35pct_1pct": (cov >= 0.35 and acc > 0 and (err / acc) <= 0.01),
        "passes_35pct_1pct": (cov >= 0.35 and acc > 0 and ub <= 0.01),
    }
    if is_flip is not None:
        f_mask = mask & is_flip
        f_acc = int(f_mask.sum())
        f_err = int(test_wrong[f_mask].sum())
        res["flip_accepted"] = f_acc
        res["flip_errors"] = f_err
        res["flip_accuracy_on_accepted"] = (1.0 - f_err / f_acc) if f_acc > 0 else None
    return res


def analyze_visual_sprites(metrics) -> dict:
    """Compare unlabeled AUDIT + own_cal vs labeled LTT (budgets 128, 256, 512, 1024) on formal_v1."""
    summary = json.loads((SPRITES_DIR / "SUMMARY.json").read_text())
    ref_truth = np.load(SPRITES_DIR / "reference_truth.npz")
    phys_var = ref_truth["variance"]
    test_truth = np.load(SPRITES_DIR / "test_truth.npz")["states"]

    # Also regenerate CAL physical truth read-only via render.role_labels? Wait: let's check if we can split
    # or regenerate CAL physical coordinates from DESIGN_LOCK.json!
    lock = json.loads((SPRITES_DIR / "DESIGN_LOCK.json").read_text())
    cfg = lock["config"]
    sys.path.insert(0, str(ROOT / "scripts/visual_sprites_lejepa"))
    run_vs = load_mod("ro_vs_run", ROOT / "scripts/visual_sprites_lejepa/run.py")
    cal_truth = run_vs.role_labels(cfg, "cal", False)  # [1024, 193, 4] exact CAL ground truth!

    alpha_test = summary["alpha"]  # 0.05 / 6
    budgets = [128, 256, 512, 1024]
    rows = []

    for saved in summary["rows"]:
        cell = saved["cell"]
        cdir = SPRITES_DIR / cell
        pred_var = np.load(cdir / "reference.npz")["variance"]
        cal_pred = np.load(cdir / "calibration.npz")["prediction"]
        test_pred = np.load(cdir / "test_prediction.npz")["prediction"]
        own_thr = json.loads((cdir / "thresholds.json").read_text())["own_cal"]

        cal_sc = metrics.score_windows(cal_pred, cal_truth, phys_var, pred_var, (4, 16, 32))
        test_sc = metrics.score_windows(test_pred, test_truth, phys_var, pred_var, (4, 16, 32))
        is_flip = np.any(test_sc["true_top2"] != np.array([0, 1]), axis=1)

        own_res = eval_on_test(test_sc["gap"], test_sc["wrong"], is_flip, own_thr, alpha_test)
        ltt_by_budget = {}
        for b in budgets:
            ltt_cal = ltt_fixed_sequence_threshold(
                cal_sc["gap"][:b], cal_sc["wrong"][:b], alpha_cal=0.05, max_risk=0.01
            )
            ltt_test = eval_on_test(test_sc["gap"], test_sc["wrong"], is_flip, ltt_cal["threshold"], alpha_test)
            ltt_by_budget[str(b)] = {"calibration": ltt_cal, "test": ltt_test}

        rows.append({
            "cell": cell,
            "eligible_audit_085": saved["eligible"],
            "minimum_audit_r2": saved["minimum_r2"],
            "unlabeled_own_cal_0_traj_labels": {"threshold": own_thr, "test": own_res},
            "labeled_ltt_without_audit_gate": ltt_by_budget,
        })
    return {"alpha_test": alpha_test, "cells": rows}


def analyze_fourroom_01(metrics) -> dict:
    """Test whether labeled LTT on CAL=1024 could rescue FourRoom fourroom_01 without AUDIT gate."""
    summary = json.loads((FOURROOM_DIR / "summary.json").read_text())
    thresholds = json.loads((FOURROOM_DIR / "thresholds.json").read_text())
    phys_var = np.load(FOURROOM_DIR / "reference_truth.npz")["variance"]
    test_truth = np.load(FOURROOM_DIR / "test_truth.npz")["states"]
    alpha_test = thresholds["per_claim_alpha"]  # 0.05 / 12

    # Note: fourroom_01 did NOT save cal_truth.npz (by strict unlabeled CAL design!).
    # So we perform a split-TEST oracle upper-bound simulation (first 1024 windows calibrate LTT,
    # remaining 1024 windows evaluate, averaged with the reverse 2-fold split) to give LTT every advantage!
    cells = sorted({r["cell"] for r in summary["rows"]})
    rows = []
    for cell in cells:
        cdir = FOURROOM_DIR / cell
        audit = json.loads((cdir / "audit.json").read_text())
        pred_var = np.load(cdir / "reference.npz")["variance"]
        test_pred = np.load(cdir / "test_prediction.npz")["prediction"]
        sc = metrics.score_windows(test_pred, test_truth, phys_var, pred_var, (4, 16, 32))
        is_flip = np.any(sc["true_top2"] != np.array([0, 1]), axis=1)

        own_thr = thresholds["own_cal"][cell]
        own_full = eval_on_test(sc["gap"], sc["wrong"], is_flip, own_thr, alpha_test)

        # Two-fold oracle labeled calibration check on 1024 labeled trajectories
        fold_results = []
        for idx_cal, idx_eval in ((slice(0, 1024), slice(1024, 2048)), (slice(1024, 2048), slice(0, 1024))):
            ltt_c = ltt_fixed_sequence_threshold(sc["gap"][idx_cal], sc["wrong"][idx_cal], alpha_cal=0.05, max_risk=0.01)
            ltt_e = eval_on_test(sc["gap"][idx_eval], sc["wrong"][idx_eval], is_flip[idx_eval], ltt_c["threshold"], alpha_test)
            fold_results.append({"cal": ltt_c, "eval": ltt_e})

        rows.append({
            "cell": cell,
            "eligible_audit_085": audit["eligible"],
            "minimum_audit_r2": float(min(audit["r2"].values())),
            "unlabeled_own_cal": own_full,
            "oracle_2fold_labeled_ltt_1024": fold_results,
        })
    return {"cells": rows}


def analyze_lowdim_b_transfer() -> dict:
    """Compare unlabeled own_cal vs labeled LTT across budgets {128, 256, 512, 1024, 2048} on B1/B3."""
    sys.path.insert(0, str(A4_DIR))
    b_mod = load_mod("ro_b_run_l5", B_DIR / "run.py")
    cfg = json.loads((B_DIR / "config.json").read_text())
    forecast = json.loads((B_DIR / "FORECAST.json").read_text())
    refs = {(c["name"], c["process"]): np.asarray(c["reference_variance"])
            for c in json.loads((A4_DIR / "NEW_FORECAST.json").read_text())["cells"]}

    alpha_test = cfg["alpha"] / 128.0
    budgets = [128, 256, 512, 1024, 2048]
    regime_summaries = {}

    # Pre-load readouts
    readouts = {f"{case}_{seed}": b_mod.panel.load_readout(f"{case}_{seed}")[0]
                for seed in cfg["seeds"] for case in cfg["training_cases"]}

    cell_map = {(c["name"], c["process"], c["regime"]): c for c in forecast["cells"]}

    for regime in ("baseline", "coupled", "faster", "slower"):
        per_budget_emp_pass = {str(b): 0 for b in budgets}
        per_budget_cp_pass = {str(b): 0 for b in budgets}
        per_budget_bonf_cp_pass = {str(b): 0 for b in budgets}
        per_budget_bonf_cov = {str(b): [] for b in budgets}
        per_budget_cov = {str(b): [] for b in budgets}
        per_budget_err = {str(b): [] for b in budgets}
        own_cal_pass = 0
        own_cal_cov = []
        own_cal_err = []

        for seed in cfg["seeds"]:
            for process in cfg["processes"]:
                z_cal = b_mod.paths(cfg, seed, process, regime, cfg["cal_role"], cfg["n_cal"])
                z_test = b_mod.paths(cfg, seed, process, regime, cfg["test_role"], cfg["n_test"])
                true_cal_s = b_mod.old.slow_scores(z_cal, cfg["lags"])
                true_test_s = b_mod.old.slow_scores(z_test, cfg["lags"])
                true_cal_top2 = np.sort(np.argsort(true_cal_s, axis=1, kind="stable")[:, :2], axis=1)
                true_test_top2 = np.sort(np.argsort(true_test_s, axis=1, kind="stable")[:, :2], axis=1)

                for case in cfg["training_cases"]:
                    name = f"{case}_{seed}"
                    ro = readouts[name]
                    ref_v = refs[(name, process)]
                    cal_pred_s = b_mod.old.slow_scores(ro(z_cal), cfg["lags"]) / ref_v[None, :]
                    cal_ord = np.sort(cal_pred_s, axis=1)
                    cal_gap = cal_ord[:, 2] - cal_ord[:, 1]
                    cal_top2 = np.sort(np.argsort(cal_pred_s, axis=1, kind="stable")[:, :2], axis=1)
                    cal_wrong = np.any(cal_top2 != true_cal_top2, axis=1)

                    c_meta = cell_map[(name, process, regime)]
                    test_npz = np.load(B_DIR / c_meta["path"])
                    test_pred_s = test_npz["predicted_scores"]
                    test_ord = np.sort(test_pred_s, axis=1)
                    test_gap = test_ord[:, 2] - test_ord[:, 1]
                    test_top2 = np.sort(np.argsort(test_pred_s, axis=1, kind="stable")[:, :2], axis=1)
                    test_wrong = np.any(test_top2 != true_test_top2, axis=1)

                    # Unlabeled per-cell own_cal (0 trajectory labels)
                    tau_own = float(np.quantile(cal_gap, 0.5, method="higher"))
                    res_own = eval_on_test(test_gap, test_wrong, None, tau_own, alpha_test)
                    own_cal_pass += int(res_own["passes_35pct_1pct"])
                    own_cal_cov.append(res_own["coverage"])
                    own_cal_err.append(res_own["selective_error"] or 0.0)

                    # Labeled LTT across budgets (both marginal alpha=0.05 and familywise alpha=alpha_test)
                    for b in budgets:
                        ltt_c = ltt_fixed_sequence_threshold(cal_gap[:b], cal_wrong[:b], alpha_cal=0.05, max_risk=0.01)
                        res_b = eval_on_test(test_gap, test_wrong, None, ltt_c["threshold"], alpha_test)
                        per_budget_emp_pass[str(b)] += int(res_b.get("empirical_passes_35pct_1pct", False))
                        per_budget_cp_pass[str(b)] += int(res_b["passes_35pct_1pct"])
                        per_budget_cov[str(b)].append(res_b["coverage"])
                        per_budget_err[str(b)].append(res_b["selective_error"] if res_b["selective_error"] is not None else 0.0)

                        ltt_bonf = ltt_fixed_sequence_threshold(cal_gap[:b], cal_wrong[:b], alpha_cal=alpha_test, max_risk=0.01)
                        res_bonf = eval_on_test(test_gap, test_wrong, None, ltt_bonf["threshold"], alpha_test)
                        per_budget_bonf_cp_pass[str(b)] += int(res_bonf["passes_35pct_1pct"])
                        per_budget_bonf_cov[str(b)].append(res_bonf["coverage"])

        regime_summaries[regime] = {
            "total_cells": 16,
            "unlabeled_per_cell_own_cal_0_traj_labels": {
                "cells_passing_bonf_cp_35pct_1pct": own_cal_pass,
                "mean_coverage": float(np.mean(own_cal_cov)),
                "mean_selective_error": float(np.mean(own_cal_err)),
            },
            "labeled_ltt_by_cal_trajectory_budget": {
                str(b): {
                    "labeled_frames_required": b * 193 * 4,
                    "marginal_ltt_empirical_pass_35pct_1pct": per_budget_emp_pass[str(b)],
                    "marginal_ltt_strict_test_bonf_cp_pass": per_budget_cp_pass[str(b)],
                    "marginal_ltt_mean_coverage": float(np.mean(per_budget_cov[str(b)])),
                    "marginal_ltt_mean_selective_error": float(np.mean(per_budget_err[str(b)])),
                    "familywise_ltt_strict_test_bonf_cp_pass": per_budget_bonf_cp_pass[str(b)],
                    "familywise_ltt_mean_coverage": float(np.mean(per_budget_bonf_cov[str(b)])),
                }
                for b in budgets
            },
        }
    return regime_summaries


def main() -> None:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = load_mod("ro_metrics_l5", ROOT / "src/slowrepr/metrics.py")

    n_min_single = math.ceil(math.log(0.05) / math.log(0.99))
    n_min_bonf16 = math.ceil(math.log(0.05 / 16) / math.log(0.99))

    sprites_res = analyze_visual_sprites(metrics)
    fourroom_res = analyze_fourroom_01(metrics)
    lowdim_res = analyze_lowdim_b_transfer()

    payload = {
        "evidence_tier": "read_only_selective_calibration_baseline_comparison",
        "elapsed_seconds": time.perf_counter() - t0,
        "theoretical_sample_complexity_floor_for_1pct_binomial_certification": {
            "min_accepted_labeled_windows_at_alpha_005_zero_errors": n_min_single,
            "min_accepted_labeled_windows_at_bonferroni_16_zero_errors": n_min_bonf16,
        },
        "visual_sprites_formal_v1": sprites_res,
        "fourroom_01_hjepa": fourroom_res,
        "low_dim_b1_b3_transfer": lowdim_res,
    }
    (OUT / "l5_selective_baseline_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )

    print(json.dumps({
        "status": "OK",
        "elapsed_seconds": payload["elapsed_seconds"],
        "min_accepted_labeled_windows_single_alpha005": n_min_single,
        "low_dim_b_transfer": lowdim_res,
        "visual_sprites_summary": [
            {
                "cell": c["cell"],
                "eligible": c["eligible_audit_085"],
                "own_cal_pass": c["unlabeled_own_cal_0_traj_labels"]["test"]["passes_35pct_1pct"],
                "own_cal_cov": round(c["unlabeled_own_cal_0_traj_labels"]["test"]["coverage"], 3),
                "own_cal_err": c["unlabeled_own_cal_0_traj_labels"]["test"]["selective_error"],
                "ltt_256_cov": round(c["labeled_ltt_without_audit_gate"]["256"]["test"]["coverage"], 3),
                "ltt_1024_cov": round(c["labeled_ltt_without_audit_gate"]["1024"]["test"]["coverage"], 3),
                "ltt_1024_pass": c["labeled_ltt_without_audit_gate"]["1024"]["test"]["passes_35pct_1pct"],
                "ltt_1024_flip_acc": c["labeled_ltt_without_audit_gate"]["1024"]["test"]["flip_accuracy_on_accepted"],
            }
            for c in sprites_res["cells"]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
