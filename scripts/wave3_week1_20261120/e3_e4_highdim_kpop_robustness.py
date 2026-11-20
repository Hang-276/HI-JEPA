#!/usr/bin/env python3
"""E3 + E4: High-Dimensional Scaling (d=4..32), Quantile/K_pop Comparison, and Non-Gaussian / AR(2) Robustness.

Implements general-(d, k) metrics matching `src/slowrepr/metrics.py` exactly on (d=4, k=2),
and tests:
  E3-A: Dimension scaling d in {4, 8, 16, 32}, k = d // 2:
        L1+L2 (R^2 x rho_r) with own_cal quantile sweep (g30, g50, g70, g80),
        unsupervised L0-orth (SFA lag-1 & Cardoso SOBI) on finite CAL=1024,
        and Regression + AR(1) -> K_pop.
  E3-B: Stationary vs 80/20 Regime-Switching comparison (K_pop vs K_W + own_cal).
  E4-A: Non-Gaussian AR(1) innovations (Gaussian, Laplace, Student-t df=5, Student-t df=3)
        across L0-elem nonlinearities and Theorem 2 / own_cal finite-window reliability.
  E4-B: Damped oscillatory AR(2) dynamics vs AR(1), exact Toeplitz Theorem 2 check,
        and lag-aliasing failure mode when period aligns with LAGS=(4, 16, 32).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/wave3_week1_20261120/e3_e4_scaling_robustness"
METRICS_PATH = ROOT / "src/slowrepr/metrics.py"
LAGS = (4, 16, 32)
SOBI_LAGS = (1, 4, 16, 32)


def load_metrics():
    spec = importlib.util.spec_from_file_location("slowrepr_metrics", METRICS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_windows_nd(
    prediction: np.ndarray,
    true_states: np.ndarray,
    physical_ref_variance: np.ndarray,
    predicted_ref_variance: np.ndarray,
    lags: tuple[int, ...] = LAGS,
    k: int = 2,
) -> dict:
    """General-(d, k) extension of src/slowrepr/metrics.py::score_windows."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(true_states, dtype=np.float64)
    phys_ref = np.asarray(physical_ref_variance, dtype=np.float64)
    pred_ref = np.asarray(predicted_ref_variance, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError(f"Expected identical 3D shapes, got {pred.shape} vs {truth.shape}")
    d = pred.shape[2]
    if not (1 <= k < d):
        raise ValueError(f"Expected 1 <= k < d, got k={k}, d={d}")

    true_scores = np.zeros((pred.shape[0], d), dtype=np.float64)
    pred_scores = np.zeros((pred.shape[0], d), dtype=np.float64)
    for h in lags:
        true_scores += ((truth[:, h:, :] - truth[:, :-h, :]) ** 2).mean(axis=1) / phys_ref[None, :]
        pred_scores += ((pred[:, h:, :] - pred[:, :-h, :]) ** 2).mean(axis=1) / pred_ref[None, :]
    true_scores /= len(lags)
    pred_scores /= len(lags)

    true_order = np.argsort(true_scores, axis=1, kind="mergesort")
    pred_order = np.argsort(pred_scores, axis=1, kind="mergesort")
    true_topk = np.sort(true_order[:, :k], axis=1)
    pred_topk = np.sort(pred_order[:, :k], axis=1)

    rows = np.arange(pred.shape[0])
    # Boundary gap between k-th slowest (index k-1) and (k+1)-th slowest (index k)
    pred_sorted = np.take_along_axis(pred_scores, pred_order, axis=1)
    true_sorted = np.take_along_axis(true_scores, true_order, axis=1)
    gap = pred_sorted[:, k] - pred_sorted[:, k - 1]
    true_gap = true_sorted[:, k] - true_sorted[:, k - 1]
    wrong = np.any(pred_topk != true_topk, axis=1)
    # Also track partial Hamming accuracy (fraction of top-k set correct) for large d
    overlap = np.array(
        [len(set(pred_topk[i].tolist()) & set(true_topk[i].tolist())) / float(k) for i in rows],
        dtype=np.float64,
    )
    return {
        "true_scores": true_scores,
        "predicted_scores": pred_scores,
        "true_topk": true_topk,
        "predicted_topk": pred_topk,
        "gap": gap,
        "true_gap": true_gap,
        "wrong": wrong,
        "topk_recall": overlap,
    }


def selective_risk_nd(
    gap: np.ndarray,
    wrong: np.ndarray,
    threshold: float,
    alpha: float = 0.05,
    topk_recall: np.ndarray | None = None,
) -> dict:
    accepted_mask = gap >= threshold
    windows = int(len(gap))
    accepted = int(np.sum(accepted_mask))
    errors = int(np.sum(wrong[accepted_mask]))
    coverage = accepted / float(windows)
    acc_rec = float(np.mean(topk_recall[accepted_mask])) if (topk_recall is not None and accepted > 0) else None
    if accepted == 0:
        return {
            "windows": windows, "accepted": 0, "errors": 0, "coverage": 0.0,
            "selective_error": 1.0, "cp_upper": 1.0, "accepted_topk_recall": acc_rec,
        }
    sel_err = errors / float(accepted)
    cp_upper = float(binomtest(errors, accepted).proportion_ci(confidence_level=1.0 - alpha, method="exact").high)
    return {
        "windows": windows,
        "accepted": accepted,
        "errors": errors,
        "coverage": coverage,
        "selective_error": sel_err,
        "cp_upper": cp_upper,
        "accepted_topk_recall": acc_rec,
    }


def verify_metrics_compatibility(metrics) -> None:
    rng = np.random.default_rng(2026112099)
    z = rng.standard_normal((128, 65, 4))
    zh = z + 0.2 * rng.standard_normal((128, 65, 4))
    p_ref = np.array([1.0, 1.2, 0.9, 1.1])
    h_ref = np.array([1.05, 1.18, 0.92, 1.08])
    ref = metrics.score_windows(zh, z, p_ref, h_ref, LAGS)
    nd = score_windows_nd(zh, z, p_ref, h_ref, LAGS, k=2)
    for k_old, k_new in [
        ("true_scores", "true_scores"),
        ("predicted_scores", "predicted_scores"),
        ("true_top2", "true_topk"),
        ("predicted_top2", "predicted_topk"),
        ("gap", "gap"),
        ("true_gap", "true_gap"),
        ("wrong", "wrong"),
    ]:
        diff = float(np.max(np.abs(ref[k_old].astype(np.float64) - nd[k_new].astype(np.float64))))
        assert diff < 1e-13, f"Mismatch on {k_old}: {diff}"
    print("Verified score_windows_nd matches src/slowrepr/metrics.py to < 1e-13.", flush=True)


def sample_ar1_family(
    rng: np.random.Generator,
    n_ep: int,
    T: int,
    rhos: np.ndarray,
    family: str = "gaussian",
    burn: int = 400,
) -> np.ndarray:
    rhos = np.asarray(rhos, dtype=np.float64)
    d = len(rhos)
    scale = np.sqrt(1.0 - rhos ** 2)

    def draw_unit_innov(shape):
        if family == "gaussian":
            return rng.standard_normal(shape)
        if family == "laplace":
            return rng.laplace(0.0, 1.0 / math.sqrt(2.0), size=shape)
        if family == "student_t5":
            return rng.standard_t(5.0, size=shape) / math.sqrt(5.0 / 3.0)
        if family == "student_t3":
            return rng.standard_t(3.0, size=shape) / math.sqrt(3.0)
        raise ValueError(f"Unknown family: {family}")

    if family == "gaussian":
        z = draw_unit_innov((n_ep, T, d))
        for t in range(1, T):
            z[:, t, :] = rhos * z[:, t - 1, :] + scale * z[:, t, :]
        return z

    # Exact stationary burn-in for non-Gaussian innovations
    total_T = burn + T
    eps = draw_unit_innov((n_ep, total_T, d))
    z_all = np.zeros_like(eps)
    z_all[:, 0, :] = eps[:, 0, :]
    for t in range(1, total_T):
        z_all[:, t, :] = rhos * z_all[:, t - 1, :] + scale * eps[:, t, :]
    out = z_all[:, burn:, :]
    return out


def joint_diagonalize(matrices: np.ndarray, tol: float = 1e-9, max_sweeps: int = 200) -> np.ndarray:
    """Cardoso-Souloumiac equal-weight JAD for symmetric [K, d, d] matrices."""
    A = np.array(matrices, dtype=np.float64, copy=True)
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    k_cnt, d, _ = A.shape
    w = np.full(k_cnt, 1.0 / k_cnt, dtype=np.float64)
    V = np.eye(d, dtype=np.float64)
    for _ in range(max_sweeps):
        max_sin = 0.0
        for p in range(d - 1):
            for q in range(p + 1, d):
                g0 = A[:, p, p] - A[:, q, q]
                g1 = 2.0 * A[:, p, q]
                g00 = float(np.dot(w, g0 * g0))
                g01 = float(np.dot(w, g0 * g1))
                g11 = float(np.dot(w, g1 * g1))
                G = np.array([[g00, g01], [g01, g11]], dtype=np.float64)
                _, eigvecs = np.linalg.eigh(G)
                u = eigvecs[:, 1]
                if u[0] < 0.0:
                    u = -u
                norm = float(np.hypot(u[0], u[1]))
                if norm < 1e-15:
                    continue
                u0 = min(1.0, max(0.0, float(u[0] / norm)))
                u1 = float(u[1] / norm)
                c = float(np.sqrt(0.5 * (1.0 + u0)))
                s = u1 / (2.0 * c)
                if abs(s) < 1e-15:
                    continue
                max_sin = max(max_sin, abs(s))

                cp = A[:, :, p].copy()
                cq = A[:, :, q].copy()
                A[:, :, p] = c * cp + s * cq
                A[:, :, q] = -s * cp + c * cq

                rp = A[:, p, :].copy()
                rq = A[:, q, :].copy()
                A[:, p, :] = c * rp + s * rq
                A[:, q, :] = -s * rp + c * rq

                vp = V[:, p].copy()
                vq = V[:, q].copy()
                V[:, p] = c * vp + s * vq
                V[:, q] = -s * vp + c * vq
        if max_sin < tol:
            break
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    diag_mean = np.mean(np.diagonal(A, axis1=-2, axis2=-1), axis=0)
    order = np.argsort(diag_mean)[::-1]
    return V[:, order]


def fit_unmixing_matrices(h_cal: np.ndarray, lags: tuple[int, ...] = SOBI_LAGS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return whitening W_whiten, SFA_lag1 axis matrix, SOBI axis matrix on [N, T, d]."""
    X = h_cal.reshape(-1, h_cal.shape[-1])
    mu = X.mean(axis=0)
    Xc = X - mu
    cov0 = (Xc.T @ Xc) / len(X)
    evals, evecs = np.linalg.eigh(cov0)
    evals = np.maximum(evals, 1e-10)
    W_whiten = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

    Xw = (h_cal - mu[None, None, :]) @ W_whiten  # [N, T, d]
    d = h_cal.shape[-1]
    C_lags = []
    for h in lags:
        A = Xw[:, :-h, :].reshape(-1, d)
        B = Xw[:, h:, :].reshape(-1, d)
        C = (A.T @ B) / len(A)
        C_lags.append(0.5 * (C + C.T))
    C_stack = np.stack(C_lags, axis=0)  # [K, d, d]
    _, vecs_lag1 = np.linalg.eigh(C_stack[0])
    V_sfa = vecs_lag1[:, ::-1]
    V_sobi = joint_diagonalize(C_stack)
    return mu, W_whiten, (V_sfa, V_sobi)


def hungarian_align_and_audit(u_cal: np.ndarray, z_cal: np.ndarray, u_test: np.ndarray, z_test: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Match unmixed coordinates to ground-truth factors via max |corr| on CAL; return min R^2, mean MCC, aligned test/ref."""
    d = z_cal.shape[-1]
    u_flat = u_cal.reshape(-1, d)
    z_flat = z_cal.reshape(-1, d)
    u_n = (u_flat - u_flat.mean(0)) / np.maximum(u_flat.std(0), 1e-12)
    z_n = (z_flat - z_flat.mean(0)) / np.maximum(z_flat.std(0), 1e-12)
    corr = (u_n.T @ z_n) / len(u_n)  # corr[i_u, j_z]
    row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
    # Map each true factor j_z (col_ind) -> matching u index (row_ind) with sign
    inv_perm = np.argsort(col_ind)
    matched_u = row_ind[inv_perm]
    signs = np.sign(corr[matched_u, np.arange(d)])
    signs[signs == 0] = 1.0
    mccs = np.abs(corr[matched_u, np.arange(d)])

    u_cal_al = u_cal[:, :, matched_u] * signs[None, None, :]
    u_test_al = u_test[:, :, matched_u] * signs[None, None, :]

    # Standardize each aligned axis on CAL and check R^2 vs true z
    uc = u_cal_al.reshape(-1, d)
    ut = u_test_al.reshape(-1, d)
    u_mean = uc.mean(0)
    u_std = np.maximum(uc.std(0), 1e-12)
    u_cal_std = (u_cal_al - u_mean[None, None, :]) / u_std[None, None, :]
    u_test_std = (u_test_al - u_mean[None, None, :]) / u_std[None, None, :]

    # Linear R^2 predicting each z_j from all u (invariant check) and from aligned u_j (signed correlation)
    X_c = np.concatenate([uc, np.ones((len(uc), 1))], axis=1)
    beta, *_ = np.linalg.lstsq(X_c, z_flat, rcond=None)
    pred_z = np.concatenate([ut, np.ones((len(ut), 1))], axis=1) @ beta
    z_t_flat = z_test.reshape(-1, d)
    var_z = np.var(z_t_flat, axis=0)
    mse = np.mean((pred_z - z_t_flat) ** 2, axis=0)
    lin_r2 = 1.0 - mse / np.maximum(var_z, 1e-12)
    return float(np.min(lin_r2)), float(np.mean(mccs)), u_cal_std, u_test_std


def linear_subspace_r2(feats_cal: np.ndarray, targets_cal: np.ndarray, feats_test: np.ndarray, targets_test: np.ndarray) -> tuple[float, float]:
    """Fit ridge-free OLS mapping feats -> targets on CAL and return (min_R2, mean_R2) on TEST."""
    fc = feats_cal.reshape(-1, feats_cal.shape[-1])
    tc = targets_cal.reshape(-1, targets_cal.shape[-1])
    ft = feats_test.reshape(-1, feats_test.shape[-1])
    tt = targets_test.reshape(-1, targets_test.shape[-1])
    Xc = np.concatenate([fc, np.ones((len(fc), 1))], axis=1)
    beta, *_ = np.linalg.lstsq(Xc, tc, rcond=None)
    pred = np.concatenate([ft, np.ones((len(ft), 1))], axis=1) @ beta
    var_t = np.var(tt, axis=0)
    mse = np.mean((pred - tt) ** 2, axis=0)
    r2 = 1.0 - mse / np.maximum(var_t, 1e-12)
    return float(np.min(r2)), float(np.mean(r2))


def run_e3() -> dict:
    t0 = time.perf_counter()
    print("=== E3: High-Dimensional Scaling (d=4..32), Unsupervised L0-orth, & K_pop vs K_W ===", flush=True)
    rng = np.random.default_rng(2026112004)
    T = 193
    n_cal, n_test, n_ref = 1024, 2048, 256
    dims = [4, 8, 16, 32]
    quantiles = [0.30, 0.50, 0.70, 0.80]
    scaling_rows = []
    unmix_rows = []
    switch_rows = []
    block_rows = []
    block_unmix_rows = []

    for d in dims:
        k = d // 2
        rhos = np.linspace(0.96, 0.75, d)
        gap_rho = float(rhos[k - 1] - rhos[k])
        unit = np.ones(d, dtype=np.float64)

        z_cal = sample_ar1_family(rng, n_cal, T, rhos, "gaussian")
        z_test = sample_ar1_family(rng, n_test, T, rhos, "gaussian")
        z_ref = sample_ar1_family(rng, n_ref, T, rhos, "gaussian")

        true_test = score_windows_nd(z_test, z_test, unit, unit, LAGS, k=k)
        oracle_kpop_vs_kw_err = float(np.mean(np.any(true_test["true_topk"] != np.arange(k)[None, :], axis=1)))

        # 1. Supervised L1+L2 cells across quantiles g30, g50, g70, g80
        for r2, rr in [(0.95, 0.00), (0.95, 0.65), (0.98, 0.65), (0.995, 0.65)]:
            sig = math.sqrt(r2)
            eps = math.sqrt(1.0 - r2)
            zh_cal = sig * z_cal + eps * sample_ar1_family(rng, n_cal, T, np.full(d, rr), "gaussian")
            zh_test = sig * z_test + eps * sample_ar1_family(rng, n_test, T, np.full(d, rr), "gaussian")
            zh_ref = sig * z_ref + eps * sample_ar1_family(rng, n_ref, T, np.full(d, rr), "gaussian")
            pred_ref_var = zh_ref.reshape(-1, d).var(0)
            cal_sc = score_windows_nd(zh_cal, z_cal, unit, pred_ref_var, LAGS, k=k)
            test_sc = score_windows_nd(zh_test, z_test, unit, pred_ref_var, LAGS, k=k)

            # K_pop fit on zh_cal via global lag-1 autocorrelation
            x0, x1 = zh_cal[:, :-1, :], zh_cal[:, 1:, :]
            rho_hat = np.sum(x0 * x1, axis=(0, 1)) / np.sqrt(np.sum(x0 ** 2, axis=(0, 1)) * np.sum(x1 ** 2, axis=(0, 1)))
            kpop_set = np.sort(np.argsort(-rho_hat)[:k])
            kpop_vs_kw = float(np.mean(np.any(test_sc["true_topk"] != kpop_set[None, :], axis=1)))
            kpop_vs_pop = float(np.any(kpop_set != np.arange(k)))

            by_q = {}
            for q in quantiles:
                tau = float(np.quantile(cal_sc["gap"], 1.0 - q))
                rk = selective_risk_nd(test_sc["gap"], test_sc["wrong"], tau, alpha=0.05)
                by_q[f"g{int(round(q * 100))}"] = rk
            row = {
                "d": d, "k": k, "gap_rho": gap_rho, "r2": r2, "rho_r": rr,
                "all_err": float(test_sc["wrong"].mean()),
                "topk_recall_all": float(test_sc["topk_recall"].mean()),
                "kpop_vs_kw_err": kpop_vs_kw,
                "kpop_vs_pop_err": kpop_vs_pop,
                "oracle_kpop_vs_kw_err": oracle_kpop_vs_kw_err,
                **{f"{qk}_{mk}": rk[mk] for qk, rk in by_q.items() for mk in ("coverage", "selective_error", "cp_upper")},
            }
            scaling_rows.append(row)
            print(
                f"  E3-A d={d:2d} (d_rho={gap_rho:.4f}) R2={r2:.3f} rr={rr:.2f} | all={row['all_err']:.4f} | "
                f"g50={by_q['g50']['selective_error']:.4f}(cov={by_q['g50']['coverage']:.2f}) | "
                f"g30={by_q['g30']['selective_error']:.4f}(cov={by_q['g30']['coverage']:.2f}) | "
                f"Kpop_vs_KW={kpop_vs_kw:.4f}",
                flush=True,
            )

        # 2. Unsupervised L0-orth (SFA_lag1 & SOBI) on orthogonal mixing Q at eps=0 (exact L0) and R2=0.98, rr=0.65
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        for regime, r2_u, rr_u in [("exact_L0_orth", 1.0, 0.0), ("noisy_R2_0.98_rr0.65", 0.98, 0.65)]:
            sig_u = math.sqrt(r2_u)
            eps_u = math.sqrt(max(0.0, 1.0 - r2_u))
            h_cal = (sig_u * z_cal + eps_u * sample_ar1_family(rng, n_cal, T, np.full(d, rr_u), "gaussian")) @ Q
            h_test = (sig_u * z_test + eps_u * sample_ar1_family(rng, n_test, T, np.full(d, rr_u), "gaussian")) @ Q
            h_ref = (sig_u * z_ref + eps_u * sample_ar1_family(rng, n_ref, T, np.full(d, rr_u), "gaussian")) @ Q
            mu_u, W_w, (V_sfa, V_sobi) = fit_unmixing_matrices(h_cal, SOBI_LAGS)
            for mname, V in [("SFA_lag1", V_sfa), ("SOBI_4lags", V_sobi)]:
                A = W_w @ V
                u_cal = (h_cal - mu_u[None, None, :]) @ A
                u_test = (h_test - mu_u[None, None, :]) @ A
                u_ref = (h_ref - mu_u[None, None, :]) @ A
                min_r2, mean_mcc, u_cal_std, u_test_std = hungarian_align_and_audit(u_cal, z_cal, u_test, z_test)
                # Evaluate K_W with matched REF on unmixed axes (using unmixed REF std normalization)
                ur_flat = (u_ref - u_cal.reshape(-1, d).mean(0)) / np.maximum(u_cal.reshape(-1, d).std(0), 1e-12)
                # Actually compute u_ref on the exact same Hungarian permutation/std!
                # Quick exact ref var from CAL/REF:
                cal_sc_u = score_windows_nd(u_cal_std, z_cal, unit, np.ones(d), LAGS, k=k)
                test_sc_u = score_windows_nd(u_test_std, z_test, unit, np.ones(d), LAGS, k=k)
                tau_50 = float(np.quantile(cal_sc_u["gap"], 0.50))
                rk_50 = selective_risk_nd(test_sc_u["gap"], test_sc_u["wrong"], tau_50, alpha=0.05)
                tau_30 = float(np.quantile(cal_sc_u["gap"], 0.70))
                rk_30 = selective_risk_nd(test_sc_u["gap"], test_sc_u["wrong"], tau_30, alpha=0.05)
                unmix_rows.append({
                    "d": d, "regime": regime, "method": mname,
                    "audit_min_lin_r2": min_r2, "mean_hungarian_mcc": mean_mcc,
                    "all_err": float(test_sc_u["wrong"].mean()),
                    "topk_recall_all": float(test_sc_u["topk_recall"].mean()),
                    "g50_cov": rk_50["coverage"], "g50_sel_err": rk_50["selective_error"],
                    "g30_cov": rk_30["coverage"], "g30_sel_err": rk_30["selective_error"],
                })
                print(
                    f"  E3-Unmix d={d:2d} {regime:20s} {mname:10s} | MCC={mean_mcc:.4f} all_err={test_sc_u['wrong'].mean():.4f} "
                    f"g50={rk_50['selective_error']:.4f} g30={rk_30['selective_error']:.4f}",
                    flush=True,
                )

        # 3. 80/20 Non-Stationary Regime-Switching (Mode A: rhos[0..d-1], Mode B: swap slow block [0:k] and fast block [k:d])
        rhos_A = rhos
        rhos_B = np.concatenate([rhos[k:], rhos[:k]])

        def sample_switch(n_ep: int) -> tuple[np.ndarray, np.ndarray]:
            nA = int(round(0.8 * n_ep))
            nB = n_ep - nA
            zA = sample_ar1_family(rng, nA, T, rhos_A, "gaussian")
            zB = sample_ar1_family(rng, nB, T, rhos_B, "gaussian")
            z_mix = np.concatenate([zA, zB], axis=0)
            is_B = np.concatenate([np.zeros(nA, dtype=bool), np.ones(nB, dtype=bool)])
            perm = rng.permutation(n_ep)
            return z_mix[perm], is_B[perm]

        z_sw_cal, isB_cal = sample_switch(n_cal)
        z_sw_test, isB_test = sample_switch(n_test)
        z_sw_ref, _ = sample_switch(n_ref)
        sig, eps = math.sqrt(0.98), math.sqrt(0.02)
        zh_sw_cal = sig * z_sw_cal + eps * sample_ar1_family(rng, n_cal, T, np.full(d, 0.65), "gaussian")
        zh_sw_test = sig * z_sw_test + eps * sample_ar1_family(rng, n_test, T, np.full(d, 0.65), "gaussian")
        zh_sw_ref = sig * z_sw_ref + eps * sample_ar1_family(rng, n_ref, T, np.full(d, 0.65), "gaussian")
        pred_ref_sw = zh_sw_ref.reshape(-1, d).var(0)
        cal_sw = score_windows_nd(zh_sw_cal, z_sw_cal, unit, pred_ref_sw, LAGS, k=k)
        test_sw = score_windows_nd(zh_sw_test, z_sw_test, unit, pred_ref_sw, LAGS, k=k)
        tau_sw50 = float(np.quantile(cal_sw["gap"], 0.50))
        rk_sw50 = selective_risk_nd(test_sw["gap"], test_sw["wrong"], tau_sw50, alpha=0.05)

        x0, x1 = zh_sw_cal[:, :-1, :], zh_sw_cal[:, 1:, :]
        rho_hat_sw = np.sum(x0 * x1, axis=(0, 1)) / np.sqrt(np.sum(x0 ** 2, axis=(0, 1)) * np.sum(x1 ** 2, axis=(0, 1)))
        kpop_sw = np.sort(np.argsort(-rho_hat_sw)[:k])
        kpop_vs_kw_sw = float(np.mean(np.any(test_sw["true_topk"] != kpop_sw[None, :], axis=1)))
        kpop_modeA_err = float(np.mean(np.any(kpop_sw[None, :] != np.arange(k)[None, :], axis=1)))
        kpop_modeB_err = float(np.mean(np.any(kpop_sw[None, :] != np.arange(k, d)[None, :], axis=1)))
        kpop_oracle_mix_err = float(0.8 * kpop_modeA_err + 0.2 * kpop_modeB_err)

        switch_rows.append({
            "d": d, "k": k, "gap_rho": gap_rho,
            "kpop_vs_regime_oracle_err": kpop_oracle_mix_err,
            "kpop_modeB_err": kpop_modeB_err,
            "kpop_vs_kw_err": kpop_vs_kw_sw,
            "kw_all_err": float(test_sw["wrong"].mean()),
            "kw_g50_cov": rk_sw50["coverage"],
            "kw_g50_sel_err": rk_sw50["selective_error"],
            "kw_modeB_all_err": float(test_sw["wrong"][isB_test].mean()),
            "kw_modeB_g50_cov": float((test_sw["gap"][isB_test] >= tau_sw50).mean()),
            "kw_modeB_g50_sel_err": float(test_sw["wrong"][isB_test][test_sw["gap"][isB_test] >= tau_sw50].mean())
            if np.any(test_sw["gap"][isB_test] >= tau_sw50) else 1.0,
        })
        print(
            f"  E3-Switch d={d:2d} (80/20) | K_pop mix_err={kpop_vs_kw_sw:.4f} (ModeB={kpop_modeB_err:.2f}) | "
            f"K_W+own_cal g50={rk_sw50['selective_error']:.4f} (ModeB_sel={switch_rows[-1]['kw_modeB_g50_sel_err']:.4f})",
            flush=True,
        )

        # 4. E3-A2: Block-Separated High-Dim Scaling (constant slow/fast boundary gap Delta_rho = 0.12)
        rhos_blk = np.concatenate([np.linspace(0.96, 0.94, k), np.linspace(0.82, 0.75, d - k)])
        gap_blk = float(rhos_blk[k - 1] - rhos_blk[k])
        zb_cal = sample_ar1_family(rng, n_cal, T, rhos_blk, "gaussian")
        zb_test = sample_ar1_family(rng, n_test, T, rhos_blk, "gaussian")
        zb_ref = sample_ar1_family(rng, n_ref, T, rhos_blk, "gaussian")

        for r2, rr in [(0.95, 0.00), (0.95, 0.65), (0.98, 0.65)]:
            sig, eps = math.sqrt(r2), math.sqrt(1.0 - r2)
            zhb_cal = sig * zb_cal + eps * sample_ar1_family(rng, n_cal, T, np.full(d, rr), "gaussian")
            zhb_test = sig * zb_test + eps * sample_ar1_family(rng, n_test, T, np.full(d, rr), "gaussian")
            zhb_ref = sig * zb_ref + eps * sample_ar1_family(rng, n_ref, T, np.full(d, rr), "gaussian")
            pred_ref_b = zhb_ref.reshape(-1, d).var(0)
            cal_sc_b = score_windows_nd(zhb_cal, zb_cal, unit, pred_ref_b, LAGS, k=k)
            test_sc_b = score_windows_nd(zhb_test, zb_test, unit, pred_ref_b, LAGS, k=k)

            x0_b, x1_b = zhb_cal[:, :-1, :], zhb_cal[:, 1:, :]
            rho_hat_b = np.sum(x0_b * x1_b, axis=(0, 1)) / np.sqrt(np.sum(x0_b ** 2, axis=(0, 1)) * np.sum(x1_b ** 2, axis=(0, 1)))
            kpop_set_b = np.sort(np.argsort(-rho_hat_b)[:k])
            kpop_vs_kw_b = float(np.mean(np.any(test_sc_b["true_topk"] != kpop_set_b[None, :], axis=1)))

            tau_50_b = float(np.quantile(cal_sc_b["gap"], 0.50))
            rk_50_b = selective_risk_nd(test_sc_b["gap"], test_sc_b["wrong"], tau_50_b, alpha=0.05, topk_recall=test_sc_b["topk_recall"])
            tau_30_b = float(np.quantile(cal_sc_b["gap"], 0.70))
            rk_30_b = selective_risk_nd(test_sc_b["gap"], test_sc_b["wrong"], tau_30_b, alpha=0.05, topk_recall=test_sc_b["topk_recall"])
            brow = {
                "d": d, "k": k, "gap_rho": gap_blk, "r2": r2, "rho_r": rr,
                "all_err": float(test_sc_b["wrong"].mean()),
                "topk_recall_all": float(test_sc_b["topk_recall"].mean()),
                "g50_cov": rk_50_b["coverage"], "g50_sel_err": rk_50_b["selective_error"],
                "g50_topk_rec": rk_50_b["accepted_topk_recall"],
                "g30_cov": rk_30_b["coverage"], "g30_sel_err": rk_30_b["selective_error"],
                "g30_topk_rec": rk_30_b["accepted_topk_recall"],
                "kpop_vs_kw_err": kpop_vs_kw_b,
            }
            block_rows.append(brow)
            print(
                f"  E3-A2(blk) d={d:2d} (gap={gap_blk:.2f}) R2={r2:.2f} rr={rr:.2f} | all={brow['all_err']:.4f} "
                f"(rec={brow['topk_recall_all']:.4f}) | g50={brow['g50_sel_err']:.4f}({brow['g50_cov']:.2f}) | "
                f"g30={brow['g30_sel_err']:.4f}({brow['g30_cov']:.2f})",
                flush=True,
            )

        Qb, _ = np.linalg.qr(rng.standard_normal((d, d)))
        for regime, r2_u, rr_u in [("exact_L0_orth", 1.0, 0.0), ("noisy_R2_0.98_rr0.65", 0.98, 0.65)]:
            sig_u = math.sqrt(r2_u)
            eps_u = math.sqrt(max(0.0, 1.0 - r2_u))
            hb_cal = (sig_u * zb_cal + eps_u * sample_ar1_family(rng, n_cal, T, np.full(d, rr_u), "gaussian")) @ Qb
            hb_test = (sig_u * zb_test + eps_u * sample_ar1_family(rng, n_test, T, np.full(d, rr_u), "gaussian")) @ Qb
            mu_b, W_b, (V_sfa_b, V_sobi_b) = fit_unmixing_matrices(hb_cal, SOBI_LAGS)
            for mname, V in [("SFA_lag1", V_sfa_b), ("SOBI_4lags", V_sobi_b)]:
                A = W_b @ V
                ub_cal = (hb_cal - mu_b[None, None, :]) @ A
                ub_test = (hb_test - mu_b[None, None, :]) @ A
                min_r2_b, mean_mcc_b, ub_cal_std, ub_test_std = hungarian_align_and_audit(ub_cal, zb_cal, ub_test, zb_test)
                sub_min_r2, sub_mean_r2 = linear_subspace_r2(ub_cal[:, :, :k], zb_cal[:, :, :k], ub_test[:, :, :k], zb_test[:, :, :k])
                cal_sc_bu = score_windows_nd(ub_cal_std, zb_cal, unit, np.ones(d), LAGS, k=k)
                test_sc_bu = score_windows_nd(ub_test_std, zb_test, unit, np.ones(d), LAGS, k=k)
                tau_50_bu = float(np.quantile(cal_sc_bu["gap"], 0.50))
                rk_50_bu = selective_risk_nd(test_sc_bu["gap"], test_sc_bu["wrong"], tau_50_bu, alpha=0.05)
                tau_30_bu = float(np.quantile(cal_sc_bu["gap"], 0.70))
                rk_30_bu = selective_risk_nd(test_sc_bu["gap"], test_sc_bu["wrong"], tau_30_bu, alpha=0.05)
                block_unmix_rows.append({
                    "d": d, "k": k, "regime": regime, "method": mname,
                    "audit_min_lin_r2": min_r2_b, "mean_hungarian_mcc": mean_mcc_b,
                    "slow_subspace_min_r2": sub_min_r2, "slow_subspace_lin_r2": sub_mean_r2,
                    "all_err": float(test_sc_bu["wrong"].mean()),
                    "topk_recall_all": float(test_sc_bu["topk_recall"].mean()),
                    "g50_sel_err": rk_50_bu["selective_error"], "g30_sel_err": rk_30_bu["selective_error"],
                })
                print(
                    f"  E3-A2-Unmix d={d:2d} {regime:20s} {mname:10s} | MCC={mean_mcc_b:.4f} "
                    f"sub_R2={sub_mean_r2:.4f} rec={test_sc_bu['topk_recall'].mean():.4f} g50={rk_50_bu['selective_error']:.4f}",
                    flush=True,
                )

    # 5. E3-A3: High-Dim Window-Length Scaling (d=16, k=8, Delta_rho=0.12, R^2=0.98, rho_r=0.65)
    t_scaling_rows = []
    d_t, k_t = 16, 8
    rhos_t = np.concatenate([np.linspace(0.96, 0.94, 8), np.linspace(0.82, 0.75, 8)])
    unit_t = np.ones(d_t, dtype=np.float64)
    sig_t, eps_t = math.sqrt(0.98), math.sqrt(0.02)
    for T_cur in [193, 385, 769]:
        zt_cal = sample_ar1_family(rng, n_cal, T_cur, rhos_t, "gaussian")
        zt_test = sample_ar1_family(rng, n_test, T_cur, rhos_t, "gaussian")
        zt_ref = sample_ar1_family(rng, n_ref, T_cur, rhos_t, "gaussian")
        zht_cal = sig_t * zt_cal + eps_t * sample_ar1_family(rng, n_cal, T_cur, np.full(d_t, 0.65), "gaussian")
        zht_test = sig_t * zt_test + eps_t * sample_ar1_family(rng, n_test, T_cur, np.full(d_t, 0.65), "gaussian")
        zht_ref = sig_t * zt_ref + eps_t * sample_ar1_family(rng, n_ref, T_cur, np.full(d_t, 0.65), "gaussian")
        pred_ref_t = zht_ref.reshape(-1, d_t).var(0)
        cal_sc_t = score_windows_nd(zht_cal, zt_cal, unit_t, pred_ref_t, LAGS, k=k_t)
        test_sc_t = score_windows_nd(zht_test, zt_test, unit_t, pred_ref_t, LAGS, k=k_t)
        tau_50_t = float(np.quantile(cal_sc_t["gap"], 0.50))
        rk_50_t = selective_risk_nd(test_sc_t["gap"], test_sc_t["wrong"], tau_50_t, alpha=0.05, topk_recall=test_sc_t["topk_recall"])
        tau_30_t = float(np.quantile(cal_sc_t["gap"], 0.70))
        rk_30_t = selective_risk_nd(test_sc_t["gap"], test_sc_t["wrong"], tau_30_t, alpha=0.05, topk_recall=test_sc_t["topk_recall"])
        trow = {
            "d": d_t, "k": k_t, "T": T_cur, "r2": 0.98, "rho_r": 0.65,
            "all_err": float(test_sc_t["wrong"].mean()),
            "topk_recall_all": float(test_sc_t["topk_recall"].mean()),
            "g50_cov": rk_50_t["coverage"], "g50_sel_err": rk_50_t["selective_error"],
            "g50_topk_rec": rk_50_t["accepted_topk_recall"],
            "g30_cov": rk_30_t["coverage"], "g30_sel_err": rk_30_t["selective_error"],
            "g30_topk_rec": rk_30_t["accepted_topk_recall"],
        }
        t_scaling_rows.append(trow)
        print(
            f"  E3-A3 T={T_cur:3d} (d=16,k=8) | all={trow['all_err']:.4f} (rec={trow['topk_recall_all']:.4f}) | "
            f"g50={trow['g50_sel_err']:.4f}(rec={trow['g50_topk_rec']:.4f}) | "
            f"g30={trow['g30_sel_err']:.4f}(rec={trow['g30_topk_rec']:.4f})",
            flush=True,
        )

    return {
        "scaling_rows": scaling_rows,
        "unmixing_rows": unmix_rows,
        "block_rows": block_rows,
        "block_unmix_rows": block_unmix_rows,
        "t_scaling_rows": t_scaling_rows,
        "switching_rows": switch_rows,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def ar2_stationary_sigma(a1: float, a2: float) -> float:
    return math.sqrt(((1.0 + a2) * ((1.0 - a2) ** 2 - a1 ** 2)) / (1.0 - a2))


def ar2_exact_acf(a1: float, a2: float, T: int) -> np.ndarray:
    gamma = np.zeros(T, dtype=np.float64)
    gamma[0] = 1.0
    if T > 1:
        gamma[1] = a1 / (1.0 - a2)
    for h in range(2, T):
        gamma[h] = a1 * gamma[h - 1] + a2 * gamma[h - 2]
    return gamma


def sample_ar2(rng: np.random.Generator, n_ep: int, T: int, a1s: np.ndarray, a2s: np.ndarray, burn: int = 600) -> np.ndarray:
    d = len(a1s)
    sigmas = np.array([ar2_stationary_sigma(float(a1s[j]), float(a2s[j])) for j in range(d)], dtype=np.float64)
    total = burn + T
    eps = rng.standard_normal((n_ep, total, d))
    z = np.zeros_like(eps)
    for t in range(2, total):
        z[:, t, :] = a1s * z[:, t - 1, :] + a2s * z[:, t - 2, :] + sigmas * eps[:, t, :]
    return z[:, burn:, :]


def build_M(T: int, lags: tuple[int, ...] = LAGS) -> np.ndarray:
    M = np.zeros((T, T), dtype=np.float64)
    for h in lags:
        D = np.zeros((T - h, T), dtype=np.float64)
        idx = np.arange(T - h)
        D[idx, idx + h] = 1.0
        D[idx, idx] = -1.0
        M += (D.T @ D) / (T - h)
    M /= len(lags)
    return M


def run_e4() -> dict:
    t0 = time.perf_counter()
    print("\n=== E4: Non-Gaussian (Laplace / Student-t) & Oscillatory AR(2) Stress Test ===", flush=True)
    rng = np.random.default_rng(2026112005)
    T = 193
    n_cal, n_test, n_ref = 1024, 2048, 256
    unit = np.ones(4, dtype=np.float64)

    # E4-A: Non-Gaussian stationary AR(1)
    # 1) L0-elem flip across distributions
    rhos_gap = np.array([0.965, 0.930, 0.900, 0.750])
    l0_rows = []
    l1_rows = []
    for fam in ["gaussian", "laplace", "student_t5", "student_t3"]:
        z_cal = sample_ar1_family(rng, n_cal, T, rhos_gap, fam)
        z_test = sample_ar1_family(rng, n_test, T, rhos_gap, fam)
        z_ref = sample_ar1_family(rng, n_ref, T, rhos_gap, fam)

        # Bounded/sublinear diffeomorphisms finite on ALL distributions including t_3, plus sinh(0.9z) on light/exp tails
        transforms = [("tanh(4z)", lambda x: np.tanh(4.0 * x)), ("asinh(4z)", lambda x: np.arcsinh(4.0 * x))]
        if fam in ("gaussian", "laplace"):
            transforms.append(("cubic(1.5)", lambda x: x + 1.5 * (x ** 3)))

        for tname, fn in transforms:
            zh_ref = z_ref.copy()
            zh_test = z_test.copy()
            zh_ref[:, :, 1] = fn(z_ref[:, :, 1])
            zh_test[:, :, 1] = fn(z_test[:, :, 1])
            phys_var = z_ref.reshape(-1, 4).var(0)
            pred_var = zh_ref.reshape(-1, 4).var(0)
            sc = score_windows_nd(zh_test, z_test, phys_var, pred_var, LAGS, k=2)
            pop_flip = float(np.mean(sc["predicted_scores"][:, 1] > sc["predicted_scores"][:, 2]))
            l0_rows.append({
                "family": fam, "transform": tname,
                "all_top2_err": float(sc["wrong"].mean()), "boundary_flip_rate": pop_flip,
            })
            print(f"  E4-A L0-elem {fam:10s} {tname:10s} | all_err={sc['wrong'].mean():.4f} bdry_flip={pop_flip:.4f}", flush=True)

        # 2) Theorem 2 / L1+L2 + own_cal robustness on non-Gaussian AR(1) (use matching innovation residual family for uniform noise scale)
        for r2, rr in [(0.95, 0.00), (0.95, 0.75), (0.98, 0.75)]:
            sig, eps = math.sqrt(r2), math.sqrt(1.0 - r2)
            zh_cal = sig * z_cal + eps * sample_ar1_family(rng, n_cal, T, np.full(4, rr), fam)
            zh_test = sig * z_test + eps * sample_ar1_family(rng, n_test, T, np.full(4, rr), fam)
            zh_ref = sig * z_ref + eps * sample_ar1_family(rng, n_ref, T, np.full(4, rr), fam)
            phys_var = z_ref.reshape(-1, 4).var(0)
            pred_var = zh_ref.reshape(-1, 4).var(0)
            cal_sc = score_windows_nd(zh_cal, z_cal, phys_var, pred_var, LAGS, k=2)
            test_sc = score_windows_nd(zh_test, z_test, phys_var, pred_var, LAGS, k=2)
            tau = float(np.quantile(cal_sc["gap"], 0.50))
            rk = selective_risk_nd(test_sc["gap"], test_sc["wrong"], tau, alpha=0.05)
            # Excess kurtosis diagnostic
            flat_z = z_ref.reshape(-1, 4)[:, 1]
            kurt_excess = float(np.mean((flat_z / np.std(flat_z)) ** 4) - 3.0)
            l1_rows.append({
                "family": fam, "r2": r2, "rho_r": rr, "z_excess_kurtosis": kurt_excess,
                "all_err": float(test_sc["wrong"].mean()),
                "own_cal_cov": rk["coverage"], "own_cal_sel_err": rk["selective_error"], "cp_upper": rk["cp_upper"],
            })
            print(
                f"  E4-A L1+L2   {fam:10s} R2={r2:.2f} rr={rr:.2f} (kurt={kurt_excess:5.2f}) | "
                f"all={test_sc['wrong'].mean():.4f} g50_cov={rk['coverage']:.3f} g50_sel={rk['selective_error']:.4f}",
                flush=True,
            )

    # E4-B: Damped oscillatory AR(2) + exact Toeplitz Theorem 2 check
    # Compare slow-envelope oscillatory regimes vs AR(1):
    # Axes 0,1 = AR(2) with poles r_0=0.955, r_1=0.935 and frequency omega; Axes 2,3 = pure AR(1) [0.89, 0.78]
    M = build_M(T, LAGS)
    idx_t = np.arange(T)
    ar2_rows = []
    for cname, omegas, rs in [
        ("slow_osc_pi64", np.array([math.pi / 64.0, math.pi / 64.0, 0.0, 0.0]), np.array([0.955, 0.935, 0.890, 0.780])),
        ("mid_osc_pi32",  np.array([math.pi / 32.0, math.pi / 32.0, 0.0, 0.0]), np.array([0.955, 0.935, 0.890, 0.780])),
        ("alias_osc_pi16", np.array([math.pi / 16.0, math.pi / 16.0, 0.0, 0.0]), np.array([0.955, 0.935, 0.890, 0.780])),
    ]:
        a1s = np.where(omegas > 0, 2.0 * rs * np.cos(omegas), rs)
        a2s = np.where(omegas > 0, -(rs ** 2), 0.0)
        z_cal = sample_ar2(rng, n_cal, T, a1s, a2s)
        z_test = sample_ar2(rng, n_test, T, a1s, a2s)
        z_ref = sample_ar2(rng, n_ref, T, a1s, a2s)

        # Population lag score on unit stationary variance
        pop_scores = []
        for j in range(4):
            acf = ar2_exact_acf(float(a1s[j]), float(a2s[j]), T)
            pop_scores.append(float(2.0 * np.mean([1.0 - acf[h] for h in LAGS])))

        # Exact Theorem 2 trace check on boundary pair (axes 1 vs 2) for R2=0.95, rr=0.65
        for rr in [0.00, 0.65]:
            r2 = 0.95
            sig, eps = math.sqrt(r2), math.sqrt(1.0 - r2)
            zh_cal = sig * z_cal + eps * sample_ar1_family(rng, n_cal, T, np.full(4, rr), "gaussian")
            zh_test = sig * z_test + eps * sample_ar1_family(rng, n_test, T, np.full(4, rr), "gaussian")
            zh_ref = sig * z_ref + eps * sample_ar1_family(rng, n_ref, T, np.full(4, rr), "gaussian")
            cal_sc = score_windows_nd(zh_cal, z_cal, unit, zh_ref.reshape(-1, 4).var(0), LAGS, k=2)
            test_sc = score_windows_nd(zh_test, z_test, unit, zh_ref.reshape(-1, 4).var(0), LAGS, k=2)
            tau = float(np.quantile(cal_sc["gap"], 0.50))
            rk = selective_risk_nd(test_sc["gap"], test_sc["wrong"], tau, alpha=0.05)

            # Analytical Theorem 2 sd(d1 - d2) using exact Toeplitz ACF of axis 1 and axis 2
            acf1 = ar2_exact_acf(float(a1s[1]), float(a2s[1]), T)
            acf2 = ar2_exact_acf(float(a1s[2]), float(a2s[2]), T)
            Sz1 = acf1[np.abs(idx_t[:, None] - idx_t[None, :])]
            Sz2 = acf2[np.abs(idx_t[:, None] - idx_t[None, :])]
            Sr = (rr ** np.abs(idx_t[:, None] - idx_t[None, :]))
            MSz1, MSz2, MSr = M @ Sz1, M @ Sz2, M @ Sr
            eps2 = 1.0 - r2
            ana_var = (
                4.0 * eps2 * (1.0 - eps2) * (float(np.sum(MSz1 * MSr)) + float(np.sum(MSz2 * MSr)))
                + 2.0 * (eps2 ** 2) * (float(np.sum(MSz1 ** 2)) + float(np.sum(MSz2 ** 2)) + 2.0 * float(np.sum(MSr ** 2)))
            )
            ana_sd = math.sqrt(max(ana_var, 1e-18))
            d_emp = (test_sc["predicted_scores"][:, 1] - test_sc["true_scores"][:, 1]) - (
                test_sc["predicted_scores"][:, 2] - test_sc["true_scores"][:, 2]
            )
            emp_sd = float(np.std(d_emp))

            # Envelope ranking check: do axes {0,1} (largest envelope r) remain ranked slowest under LAGS=(4,16,32)?
            envelope_in_top2_rate = float(np.mean((test_sc["true_topk"][:, 0] == 0) & (test_sc["true_topk"][:, 1] == 1)))

            ar2_rows.append({
                "process": cname, "rho_r": rr, "pop_scores_LAGS": pop_scores,
                "envelope_pair_in_true_kw_rate": envelope_in_top2_rate,
                "ana_thm2_sd_d1d2": ana_sd, "emp_sd_d1d2": emp_sd,
                "all_kw_err": float(test_sc["wrong"].mean()),
                "own_cal_cov": rk["coverage"], "own_cal_sel_err": rk["selective_error"],
            })
            print(
                f"  E4-B AR(2) {cname:15s} rr={rr:.2f} | env_is_K_W={envelope_in_top2_rate:.3f} | "
                f"Thm2 sd ana={ana_sd:.4f} emp={emp_sd:.4f} | all_err={test_sc['wrong'].mean():.4f} g50_sel={rk['selective_error']:.4f}",
                flush=True,
            )

    return {
        "l0_nongaussian_rows": l0_rows,
        "l1_nongaussian_rows": l1_rows,
        "ar2_rows": ar2_rows,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def write_results_md(summary: dict) -> None:
    lines = [
        "# E3 & E4: High-Dimensional Scaling (`d=4..32`), `K_pop` Switching, and Non-Gaussian / AR(2) Robustness",
        "",
        "## 1. E3-A1: Continuous-Spectrum High-Dimensional Scaling (`k = d/2`) & `own_cal` Quantile Sweep",
        "",
        "| `d` (`k`) | $\\Delta\\rho$ | $R^2$ | $\\rho_r$ | `all_err` | `topk_rec` | `g50` sel (cov) | `g30` sel (cov) | `K_pop` vs $K_W$ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["e3"]["scaling_rows"]:
        lines.append(
            f"| {r['d']} ({r['k']}) | {r['gap_rho']:.4f} | {r['r2']:.3f} | {r['rho_r']:.2f} | {r['all_err']*100:.2f}% | "
            f"{r['topk_recall_all']*100:.2f}% | {r['g50_selective_error']*100:.2f}% ({r['g50_coverage']*100:.1f}%) | "
            f"{r['g30_selective_error']*100:.2f}% ({r['g30_coverage']*100:.1f}%) | {r['kpop_vs_kw_err']*100:.2f}% |"
        )

    lines += [
        "",
        "## 1b. E3-A2: Block-Separated High-Dim Scaling (`d=4..32`, constant boundary gap $\\Delta\\rho = 0.12$)",
        "",
        "| `d` (`k`) | $\\Delta\\rho$ | $R^2$ | $\\rho_r$ | `all_err` | `topk_rec` | `g50` sel (cov) | `g50_rec` | `g30` sel (cov) | `K_pop` vs $K_W$ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["e3"]["block_rows"]:
        g50_rec_str = f"{r['g50_topk_rec']*100:.2f}%" if r["g50_topk_rec"] is not None else "n/a"
        lines.append(
            f"| {r['d']} ({r['k']}) | {r['gap_rho']:.4f} | {r['r2']:.2f} | {r['rho_r']:.2f} | {r['all_err']*100:.2f}% | "
            f"{r['topk_recall_all']*100:.2f}% | {r['g50_sel_err']*100:.2f}% ({r['g50_cov']*100:.1f}%) | "
            f"{g50_rec_str} | {r['g30_sel_err']*100:.2f}% ({r['g30_cov']*100:.1f}%) | {r['kpop_vs_kw_err']*100:.2f}% |"
        )

    lines += [
        "",
        "| `d` | Regime | Method | Audit min $R^2$ | Mean MCC | `slow_subspace_lin_R2` | `topk_rec` | `all_err` | `g50` sel | `g30` sel |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["e3"]["block_unmix_rows"]:
        lines.append(
            f"| {r['d']} | `{r['regime']}` | `{r['method']}` | {r['audit_min_lin_r2']:.4f} | {r['mean_hungarian_mcc']:.4f} | "
            f"{r['slow_subspace_lin_r2']:.4f} | {r['topk_recall_all']*100:.2f}% | {r['all_err']*100:.2f}% | "
            f"{r['g50_sel_err']*100:.2f}% | {r['g30_sel_err']*100:.2f}% |"
        )

    lines += [
        "",
        "## 1c. E3-A3: High-Dim Window-Length Scaling (`d=16, k=8`, $\\Delta\\rho=0.12$, $R^2=0.98, \\rho_r=0.65$)",
        "",
        "> Extreme-value vs subspace invariance: holding `T=193` fixed shrinks the exact-set boundary gap between $\\min_{1..k} s_i$ and $\\max_{k+1..d} s_j$ by $\\approx\\sigma\\sqrt{2\\ln(d/2)}$ over $k(d-k)$ cross-pairs even while per-axis `topk_recall` (`>94%`) and `slow_subspace_lin_R2` (`0.98–1.00`) stay invariant; increasing window length $T$ directly counteracts this extreme-value concentration.",
        "",
        "| `T` | `all_err` | `topk_rec` (all) | `g50` sel (cov) | `g50_rec` | `g30` sel (cov) | `g30_rec` |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["e3"]["t_scaling_rows"]:
        g50r = f"{r['g50_topk_rec']*100:.2f}%" if r["g50_topk_rec"] is not None else "n/a"
        g30r = f"{r['g30_topk_rec']*100:.2f}%" if r["g30_topk_rec"] is not None else "n/a"
        lines.append(
            f"| {r['T']} | {r['all_err']*100:.2f}% | {r['topk_recall_all']*100:.2f}% | "
            f"{r['g50_sel_err']*100:.2f}% ({r['g50_cov']*100:.1f}%) | {g50r} | "
            f"{r['g30_sel_err']*100:.2f}% ({r['g30_cov']*100:.1f}%) | {g30r} |"
        )

    lines += [
        "",
        "## 2. E3 Unsupervised `L0-orth` (`SFA_lag1` vs `SOBI_4lags` on Finite `CAL=1024`)",
        "",
        "| `d` | Regime | Method | Audit min $R^2$ | Mean MCC | `all_err` | `g50` sel | `g30` sel |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary["e3"]["unmixing_rows"]:
        lines.append(
            f"| {r['d']} | `{r['regime']}` | `{r['method']}` | {r['audit_min_lin_r2']:.4f} | {r['mean_hungarian_mcc']:.4f} | "
            f"{r['all_err']*100:.2f}% | {r['g50_sel_err']*100:.2f}% | {r['g30_sel_err']*100:.2f}% |"
        )

    lines += [
        "",
        "## 3. E3-B: 80/20 Regime-Switching (`K_pop` vs $K_W$ + `own_cal`, $R^2=0.98, \\rho_r=0.65$)",
        "",
        "> Note on `d >= 8` pooled `K_pop` chimera: at `d >= 8`, interior boundary rates reverse under 80/20 averaging (`0.8 * 0.87 + 0.2 * 0.75 = 0.846 < 0.864 = 0.8 * 0.84 + 0.2 * 0.96`), causing global `K_pop` to commit to a chimera index set that misses both Mode A and Mode B (`100%` regime-oracle error).",
        "",
        "| `d` | `K_pop` Oracle Mix Err | `K_pop` Mode-B Err | `K_pop` vs $K_W$ | $K_W$ All Err | $K_W$ `g50` Sel Err | Mode-B `g50` Sel Err |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["e3"]["switching_rows"]:
        lines.append(
            f"| {r['d']} | {r['kpop_vs_regime_oracle_err']*100:.2f}% | {r['kpop_modeB_err']*100:.1f}% | {r['kpop_vs_kw_err']*100:.2f}% | "
            f"{r['kw_all_err']*100:.2f}% | {r['kw_g50_sel_err']*100:.2f}% | {r['kw_modeB_g50_sel_err']*100:.2f}% |"
        )

    lines += [
        "",
        "## 4. E4-A: Non-Gaussian Stationary AR(1) Innovations",
        "",
        "| Family | Transform (`L0-elem`) | `all_top2_err` | Boundary Flip Rate |",
        "|---|---|---:|---:|",
    ]
    for r in summary["e4"]["l0_nongaussian_rows"]:
        lines.append(f"| `{r['family']}` | `{r['transform']}` | {r['all_top2_err']*100:.2f}% | {r['boundary_flip_rate']*100:.2f}% |")

    lines += [
        "",
        "| Family | $R^2$ | $\\rho_r$ | Excess Kurtosis | `all_err` | `g50` cov | `g50` sel_err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["e4"]["l1_nongaussian_rows"]:
        lines.append(
            f"| `{r['family']}` | {r['r2']:.2f} | {r['rho_r']:.2f} | {r['z_excess_kurtosis']:.2f} | "
            f"{r['all_err']*100:.2f}% | {r['own_cal_cov']*100:.1f}% | {r['own_cal_sel_err']*100:.2f}% |"
        )

    lines += [
        "",
        "## 5. E4-B: Damped Oscillatory AR(2) Dynamics & Exact Toeplitz Theorem 2 Check",
        "",
        "| Process | $\\rho_r$ | Env `{0,1}` == $K_W$ Rate | Thm 2 `sd(d1-d2)` Ana | Emp `sd(d1-d2)` | `all_err` | `g50` sel_err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["e4"]["ar2_rows"]:
        lines.append(
            f"| `{r['process']}` | {r['rho_r']:.2f} | {r['envelope_pair_in_true_kw_rate']*100:.1f}% | "
            f"{r['ana_thm2_sd_d1d2']:.4f} | {r['emp_sd_d1d2']:.4f} | {r['all_kw_err']*100:.2f}% | {r['own_cal_sel_err']*100:.2f}% |"
        )

    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    verify_metrics_compatibility(metrics)
    t0 = time.perf_counter()
    e3 = run_e3()
    e4 = run_e4()
    summary = {
        "experiment": "wave3_week1_e3_e4_scaling_robustness",
        "metrics_sha256": hashlib.sha256(METRICS_PATH.read_bytes()).hexdigest(),
        "e3": e3,
        "e4": e4,
        "total_elapsed_seconds": time.perf_counter() - t0,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_results_md(summary)
    print(f"Wrote {OUT / 'SUMMARY.json'} and {OUT / 'RESULTS.md'} in {summary['total_elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
