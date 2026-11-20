#!/usr/bin/env python3
"""L1: Read-only mechanistic closeout of 32x32 visual_sprites_lejepa/formal_v1.

Verifies forecast seal hashes, reproduces SUMMARY.json from raw arrays, and
performs an exact variance-share and multi-lag slowness decomposition of the
fast-axis collapse vs. pseudo-slow hijacking across visual_lejepa and
long_horizon_pred (seeds 101, 102, 103).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "results/visual_sprites_lejepa/formal_v1"
OUT = ROOT / "results/closeout_20261117/l1_visual_sprites"
AXES = ["agent_x", "agent_y", "distractor_x", "distractor_y"]
LAGS = (4, 16, 32)


def load_metrics():
    spec = importlib.util.spec_from_file_location("slowrepr_metrics_ro", ROOT / "src/slowrepr/metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def multi_lag_numerator(paths: np.ndarray, lags: tuple[int, ...] = LAGS) -> np.ndarray:
    """Mean squared increment across lags for [N, T, D] or [N, T] array."""
    acc = np.zeros((paths.shape[0],) + paths.shape[2:], dtype=np.float64)
    for h in lags:
        diff = paths[:, h:] - paths[:, :-h]
        acc += np.mean(diff * diff, axis=1)
    return acc / len(lags)


def build_slow_fast_features(z: np.ndarray) -> dict[str, np.ndarray]:
    """Construct raw feature blocks from flattened physical states z [M, 4]."""
    z0, z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    slow_lin = np.column_stack([z0, z1])
    slow_harm = np.column_stack([
        z0**2, z1**2, z0 * z1,
        z0**3, z1**3, (z0**2) * z1, z0 * (z1**2),
        z0**4, z1**4, (z0**2) * (z1**2),
    ])
    fast_lin = np.column_stack([z2, z3])
    fast_harm = np.column_stack([
        z2**2, z3**2, z2 * z3,
        z2**3, z3**3, (z2**2) * z3, z2 * (z3**2),
    ])
    cross_sf = np.column_stack([
        z0 * z2, z0 * z3, z1 * z2, z1 * z3,
    ])
    return {
        "slow_lin": slow_lin,
        "slow_harm": slow_harm,
        "fast_lin": fast_lin,
        "fast_harm": fast_harm,
        "cross_sf": cross_sf,
    }


def decompose_fast_axis(
    ref_true: np.ndarray,
    ref_pred_axis: np.ndarray,
    test_true: np.ndarray,
    test_pred_axis: np.ndarray,
    own_idx: int,
    other_fast_idx: int,
) -> dict:
    """Orthogonally decompose a predicted fast axis [N, T] into sequential physical subspaces.

    Fits sequential OLS projections on REF [N_ref*T] and evaluates both on REF and
    out-of-sample on TEST [N_test*T], including multi-lag dynamic slowness of each component.
    """
    n_ref, t_len = ref_pred_axis.shape
    n_test, _ = test_pred_axis.shape

    zr = ref_true.reshape(-1, 4).astype(np.float64)
    yr = ref_pred_axis.reshape(-1).astype(np.float64)
    zt = test_true.reshape(-1, 4).astype(np.float64)
    yt = test_pred_axis.reshape(-1).astype(np.float64)

    yr_mean = yr.mean()
    yt_mean = yt.mean()
    yr_c = yr - yr_mean
    yt_c = yt - yt_mean

    var_ref = float(np.mean(yr_c**2))
    var_test = float(np.mean(yt_c**2))

    fr = build_slow_fast_features(zr)
    ft = build_slow_fast_features(zt)

    blocks_ref = [
        ("own_fast_linear", zr[:, [own_idx]]),
        ("other_fast_linear_and_harm", np.hstack([zr[:, [other_fast_idx]], fr["fast_harm"]])),
        ("slow_agent_linear", fr["slow_lin"]),
        ("slow_agent_harmonics", fr["slow_harm"]),
        ("slow_fast_interactions", fr["cross_sf"]),
    ]
    blocks_test = [
        ("own_fast_linear", zt[:, [own_idx]]),
        ("other_fast_linear_and_harm", np.hstack([zt[:, [other_fast_idx]], ft["fast_harm"]])),
        ("slow_agent_linear", ft["slow_lin"]),
        ("slow_agent_harmonics", ft["slow_harm"]),
        ("slow_fast_interactions", ft["cross_sf"]),
    ]

    # Sequential orthogonal projection fitted on REF (centered features)
    rem_ref = yr_c.copy()
    rem_test = yt_c.copy()
    used_ref_cols = []
    used_test_cols = []

    components = {}
    for (name, xr_raw), (_, xt_raw) in zip(blocks_ref, blocks_test):
        xr_m = xr_raw.mean(axis=0, keepdims=True)
        xr = xr_raw - xr_m
        xt = xt_raw - xr_m
        if used_ref_cols:
            prev_r = np.hstack(used_ref_cols)
            prev_t = np.hstack(used_test_cols)
            coef_orth, _, _, _ = np.linalg.lstsq(prev_r, xr, rcond=None)
            xr_orth = xr - prev_r @ coef_orth
            xt_orth = xt - prev_t @ coef_orth
        else:
            xr_orth = xr
            xt_orth = xt
        beta, _, _, _ = np.linalg.lstsq(xr_orth, rem_ref, rcond=None)
        comp_r = xr_orth @ beta
        comp_t = xt_orth @ beta
        rem_ref = rem_ref - comp_r
        rem_test = rem_test - comp_t
        used_ref_cols.append(xr_orth)
        used_test_cols.append(xt_orth)

        comp_r_w = comp_r.reshape(n_ref, t_len)
        comp_t_w = comp_t.reshape(n_test, t_len)
        vr = float(np.mean(comp_r**2))
        vt = float(np.mean(comp_t**2))
        num_r = float(np.mean(multi_lag_numerator(comp_r_w)))
        num_t = float(np.mean(multi_lag_numerator(comp_t_w)))
        components[name] = {
            "ref_variance": vr,
            "ref_variance_share": vr / var_ref if var_ref > 0 else 0.0,
            "ref_effective_slowness": num_r / vr if vr > 1e-18 else None,
            "test_variance": vt,
            "test_variance_share": vt / var_test if var_test > 0 else 0.0,
            "test_effective_slowness": num_t / vt if vt > 1e-18 else None,
        }

    rem_r_w = rem_ref.reshape(n_ref, t_len)
    rem_t_w = rem_test.reshape(n_test, t_len)
    vr_res = float(np.mean(rem_ref**2))
    vt_res = float(np.mean(rem_test**2))
    num_r_res = float(np.mean(multi_lag_numerator(rem_r_w)))
    num_t_res = float(np.mean(multi_lag_numerator(rem_t_w)))
    components["unexplained_residual"] = {
        "ref_variance": vr_res,
        "ref_variance_share": vr_res / var_ref if var_ref > 0 else 0.0,
        "ref_effective_slowness": num_r_res / vr_res if vr_res > 1e-18 else None,
        "test_variance": vt_res,
        "test_variance_share": vt_res / var_test if var_test > 0 else 0.0,
        "test_effective_slowness": num_t_res / vt_res if vt_res > 1e-18 else None,
    }

    # Exact orthogonality sanity check on REF
    ref_share_sum = sum(c["ref_variance_share"] for c in components.values())
    assert abs(ref_share_sum - 1.0) < 1e-10, f"REF orthogonal variance shares sum to {ref_share_sum}"

    slow_total_share_ref = (
        components["slow_agent_linear"]["ref_variance_share"]
        + components["slow_agent_harmonics"]["ref_variance_share"]
    )
    fast_total_share_ref = (
        components["own_fast_linear"]["ref_variance_share"]
        + components["other_fast_linear_and_harm"]["ref_variance_share"]
    )

    return {
        "pred_ref_variance": var_ref,
        "pred_test_variance": var_test,
        "ref_slow_leakage_share_total": slow_total_share_ref,
        "ref_fast_signal_share_total": fast_total_share_ref,
        "ref_unexplained_plus_interaction_share": (
            components["slow_fast_interactions"]["ref_variance_share"]
            + components["unexplained_residual"]["ref_variance_share"]
        ),
        "components": components,
    }


def main() -> None:
    t0 = time.perf_counter()
    metrics = load_metrics()
    OUT.mkdir(parents=True, exist_ok=True)

    lock = json.loads((FORMAL / "DESIGN_LOCK.json").read_text())
    receipt = json.loads((FORMAL / "FORECAST_RECEIPT.json").read_text())
    summary = json.loads((FORMAL / "SUMMARY.json").read_text())

    # 1. Cryptographic seal verification
    assert sha256_file(FORMAL / "DESIGN_LOCK.json") == receipt["design_lock_sha256"]
    for rel, expected_sha in receipt["files_sha256"].items():
        actual_sha = sha256_file(FORMAL / rel)
        assert actual_sha == expected_sha, f"Hash mismatch on {rel}"
    assert sha256_file(FORMAL / "test_truth.npz") == summary["test_truth_sha256"]

    ref_truth_npz = np.load(FORMAL / "reference_truth.npz")
    ref_truth = ref_truth_npz["states"]
    phys_ref_var = ref_truth_npz["variance"]
    test_truth = np.load(FORMAL / "test_truth.npz")["states"]

    # True window properties
    true_scores_probe = metrics.score_windows(test_truth, test_truth, phys_ref_var, phys_ref_var, LAGS)
    true_scores = true_scores_probe["true_scores"]
    true_top2 = true_scores_probe["true_top2"]
    true_gap = true_scores_probe["true_gap"]
    is_regular = np.all(true_top2 == np.array([0, 1]), axis=1)
    is_flip = ~is_regular
    assert int(is_regular.sum()) == 1889 and int(is_flip.sum()) == 159

    # Categorize true flip types among the 159 flip windows
    flip_pair_counts = {}
    for row in true_top2[is_flip]:
        key = f"({AXES[row[0]]}, {AXES[row[1]]})"
        flip_pair_counts[key] = flip_pair_counts.get(key, 0) + 1

    cell_reports = []
    alpha = summary["alpha"]

    for saved_row in summary["rows"]:
        cell = saved_row["cell"]
        cdir = FORMAL / cell
        ref_npz = np.load(cdir / "reference.npz")
        ref_pred = ref_npz["prediction"]
        pred_ref_var = ref_npz["variance"]
        test_pred = np.load(cdir / "test_prediction.npz")["prediction"]
        threshold = json.loads((cdir / "thresholds.json").read_text())["own_cal"]

        scored = metrics.score_windows(test_pred, test_truth, phys_ref_var, pred_ref_var, LAGS)
        risk_res = metrics.risk(scored["gap"], scored["wrong"], threshold, alpha, 0.35, 0.01)

        # Verify exact match with SUMMARY.json
        assert risk_res["accepted"] == saved_row["own_cal"]["accepted"]
        assert risk_res["errors"] == saved_row["own_cal"]["errors"]
        np.testing.assert_allclose(risk_res["error_upper"], saved_row["own_cal"]["error_upper"], rtol=1e-12, atol=1e-12)

        pred_scores = scored["predicted_scores"]
        pred_top2 = scored["predicted_top2"]
        wrong = scored["wrong"]
        accept = scored["gap"] >= threshold

        # TEST R^2 per axis
        test_var_true = test_truth.reshape(-1, 4).var(axis=0, ddof=0)
        test_r2 = 1.0 - np.mean((test_pred.reshape(-1, 4) - test_truth.reshape(-1, 4)) ** 2, axis=0) / test_var_true

        # Selection pattern diagnostics (how often agent [0,1] vs fast-invaded sets are picked)
        def pattern_stats(mask: np.ndarray) -> dict:
            sub_top2 = pred_top2[mask]
            sub_wrong = wrong[mask]
            n_sub = int(mask.sum())
            picks_01 = int(np.all(sub_top2 == np.array([0, 1]), axis=1).sum())
            picks_with_dx = int(np.any(sub_top2 == 2, axis=1).sum())
            picks_with_dy = int(np.any(sub_top2 == 3, axis=1).sum())
            picks_23 = int(np.all(sub_top2 == np.array([2, 3]), axis=1).sum())
            return {
                "windows": n_sub,
                "errors": int(sub_wrong.sum()),
                "accuracy": float(1.0 - sub_wrong.mean()) if n_sub else None,
                "error_rate": float(sub_wrong.mean()) if n_sub else None,
                "picks_agent_01": picks_01,
                "picks_agent_01_rate": picks_01 / n_sub if n_sub else None,
                "picks_containing_distractor_x": picks_with_dx,
                "picks_containing_distractor_y": picks_with_dy,
                "picks_both_distractors_23": picks_23,
            }

        # Proposition 1 perturbation decomposition on TEST
        # s_hat_i - s_i = [K_i + E_i - s_i*(V_hat_i - V_0_i)] / V_hat_i
        cross_term = scored["cross_term"]  # K_i
        error_energy = scored["error_energy"]  # E_i
        denom_shift = true_scores * (pred_ref_var[None, :] - phys_ref_var[None, :])
        total_pert = (cross_term + error_energy - denom_shift) / pred_ref_var[None, :]
        np.testing.assert_allclose(pred_scores - true_scores, total_pert, rtol=1e-11, atol=1e-11)

        # Fast-axis variance & slowness attribution
        fast_decomp = {
            "distractor_x": decompose_fast_axis(ref_truth, ref_pred[:, :, 2], test_truth, test_pred[:, :, 2], 2, 3),
            "distractor_y": decompose_fast_axis(ref_truth, ref_pred[:, :, 3], test_truth, test_pred[:, :, 3], 3, 2),
        }

        # Empirical critical condition check (EXPLORATORY section 6.1):
        # Does mean normalized predicted fast score drop near or below slow agent mean score?
        mean_pred_scores = pred_scores.mean(axis=0)
        mean_true_scores = true_scores.mean(axis=0)
        slow_boundary_mean = float(max(mean_pred_scores[0], mean_pred_scores[1]))

        cell_reports.append({
            "cell": cell,
            "arm": saved_row["arm"],
            "seed": saved_row["seed"],
            "status": saved_row["status"],
            "eligible": saved_row["eligible"],
            "audit_r2": saved_row["r2"],
            "test_r2": dict(zip(AXES, test_r2.tolist())),
            "minimum_audit_r2": saved_row["minimum_r2"],
            "threshold_own_cal": threshold,
            "overall_all_answer": pattern_stats(np.ones(len(wrong), dtype=bool)),
            "overall_own_cal": saved_row["own_cal"],
            "regular_all_answer": pattern_stats(is_regular),
            "regular_own_cal": pattern_stats(is_regular & accept),
            "flip_all_answer": pattern_stats(is_flip),
            "flip_own_cal": pattern_stats(is_flip & accept),
            "variance_diagnostics": {
                "physical_ref_variance": dict(zip(AXES, phys_ref_var.tolist())),
                "predicted_ref_variance": dict(zip(AXES, pred_ref_var.tolist())),
                "ref_variance_ratio_pred_over_true": dict(zip(AXES, (pred_ref_var / phys_ref_var).tolist())),
            },
            "mean_normalized_scores_test": {
                "true": dict(zip(AXES, mean_true_scores.tolist())),
                "predicted": dict(zip(AXES, mean_pred_scores.tolist())),
                "predicted_slow_agent_max": slow_boundary_mean,
                "predicted_fast_min": float(min(mean_pred_scores[2], mean_pred_scores[3])),
                "population_hijack_margin_fast_minus_slow": float(
                    min(mean_pred_scores[2], mean_pred_scores[3]) - slow_boundary_mean
                ),
            },
            "prop1_perturbation_rms_by_axis": {
                "total_score_error_rms": dict(zip(AXES, np.sqrt(np.mean(total_pert**2, axis=0)).tolist())),
                "cross_term_over_vhat_rms": dict(
                    zip(AXES, np.sqrt(np.mean((cross_term / pred_ref_var[None, :]) ** 2, axis=0)).tolist())
                ),
                "error_energy_over_vhat_rms": dict(
                    zip(AXES, np.sqrt(np.mean((error_energy / pred_ref_var[None, :]) ** 2, axis=0)).tolist())
                ),
                "denom_term_over_vhat_rms": dict(
                    zip(AXES, np.sqrt(np.mean((denom_shift / pred_ref_var[None, :]) ** 2, axis=0)).tolist())
                ),
            },
            "fast_axis_orthogonal_decomposition": fast_decomp,
        })

    output_payload = {
        "evidence_tier": "read_only_formal_closeout_diagnostics",
        "source_experiment": "results/visual_sprites_lejepa/formal_v1",
        "verified_forecast_files": len(receipt["files_sha256"]),
        "config_rho": lock["config"]["rho"],
        "pred_lag_long_horizon": lock["config"]["encoder"]["long_lag"],
        "eval_lags": list(LAGS),
        "test_windows": {
            "total": int(len(wrong)),
            "regular_agent_slowest": int(is_regular.sum()),
            "local_velocity_flip": int(is_flip.sum()),
            "flip_rate": float(is_flip.mean()),
            "flip_true_top2_breakdown": flip_pair_counts,
            "true_gap_quantiles_regular": np.quantile(true_gap[is_regular], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
            "true_gap_quantiles_flip": np.quantile(true_gap[is_flip], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        },
        "cells": cell_reports,
        "elapsed_seconds": time.perf_counter() - t0,
    }

    (OUT / "l1_visual_sprites_diagnostics.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "status": "OK",
        "elapsed_seconds": output_payload["elapsed_seconds"],
        "summary": [
            {
                "cell": c["cell"],
                "min_audit_r2": round(c["minimum_audit_r2"], 4),
                "reg_all_err": f"{c['regular_all_answer']['errors']}/{c['regular_all_answer']['windows']}",
                "flip_all_acc": round(c["flip_all_answer"]["accuracy"], 4),
                "own_cal_err": f"{c['overall_own_cal']['errors']}/{c['overall_own_cal']['accepted']}",
                "dx_var_ratio": round(c["variance_diagnostics"]["ref_variance_ratio_pred_over_true"]["distractor_x"], 6),
                "dy_var_ratio": round(c["variance_diagnostics"]["ref_variance_ratio_pred_over_true"]["distractor_y"], 6),
                "dx_slow_leak_share": round(
                    c["fast_axis_orthogonal_decomposition"]["distractor_x"]["ref_slow_leakage_share_total"], 4
                ),
                "dy_slow_leak_share": round(
                    c["fast_axis_orthogonal_decomposition"]["distractor_y"]["ref_slow_leakage_share_total"], 4
                ),
                "pop_hijack_margin": round(
                    c["mean_normalized_scores_test"]["population_hijack_margin_fast_minus_slow"], 4
                ),
            }
            for c in cell_reports
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
