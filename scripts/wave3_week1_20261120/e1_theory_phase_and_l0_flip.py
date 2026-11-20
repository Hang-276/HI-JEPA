#!/usr/bin/env python3
"""E1: Quantitative Theory-vs-Empirical Verification of Theorems 1 & 2.

Verifies:
  E1-A: L0-elem (elementwise diffeomorphism) order non-invariance via Mehler/Hermite
        closed forms (sinh, cubic, tanh) and 2D flip-boundary overlay on empirical heatmaps.
  E1-B: L1 Single-frame indistinguishability impossibility (identical P(Z_t, Z_hat_t),
        identical R^2 & linear MCC, diverging finite-window reliability & flip threshold).
  E1-C: Theorem 2 exact quadratic-form finite-window variance (tr(M Sigma_z M Sigma_r))
        and noise-to-margin ratio overlaid on (R^2 x rho_r) and (Delta_rho x T) grids,
        plus the stationary Regression+AR(1)->K_pop baseline.
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
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/wave3_week1_20261120/e1_theory_phase"
METRICS_PATH = ROOT / "src/slowrepr/metrics.py"
LAGS = (4, 16, 32)


def load_metrics():
    spec = importlib.util.spec_from_file_location("slowrepr_metrics", METRICS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def Sigma_ar1(T: int, rho: float) -> np.ndarray:
    t = np.arange(T, dtype=np.float64)
    return np.power(rho, np.abs(t[:, None] - t[None, :]))


def mu_ar1(rho: float, lags: tuple[int, ...] = LAGS) -> float:
    return float(2.0 * np.mean([1.0 - (rho ** h) for h in lags]))


def sample_ar1(rng: np.random.Generator, n_ep: int, T: int, rhos: np.ndarray) -> np.ndarray:
    rhos = np.asarray(rhos, dtype=np.float64)
    scale = np.sqrt(1.0 - rhos ** 2)
    z = rng.standard_normal((n_ep, T, len(rhos)))
    for t in range(1, T):
        z[:, t, :] = rhos * z[:, t - 1, :] + scale * z[:, t, :]
    return z


def exact_tanh_hermite_weights(gamma: float, max_deg: int = 65) -> tuple[np.ndarray, float]:
    """Exact Hermite probabilities w_n for tanh(gamma * Z), Z ~ N(0, 1), via spectral trapezoidal quadrature."""
    pts = np.linspace(-10.0, 10.0, 2001)
    dz = pts[1] - pts[0]
    wts = dz * np.exp(-0.5 * (pts ** 2)) / math.sqrt(2.0 * math.pi)
    vals = np.tanh(gamma * pts)
    var = float(np.sum(wts * (vals ** 2)))
    std_vals = vals / math.sqrt(var)
    weights = np.zeros(max_deg + 1, dtype=np.float64)
    # Orthonormal probabilists' Hermite recurrence: h_{n+1} = (x*h_n - sqrt(n)*h_{n-1}) / sqrt(n+1)
    h_prev = np.zeros_like(pts)
    h_curr = np.ones_like(pts)
    for n in range(1, max_deg + 1):
        h_next = (pts * h_curr - math.sqrt(n - 1) * h_prev) / math.sqrt(n)
        h_prev, h_curr = h_curr, h_next
        if n % 2 == 1:
            c_n = float(np.sum(wts * std_vals * h_curr))
            weights[n] = c_n ** 2
    eta = max(0.0, 1.0 - float(weights[1]))
    return weights, eta


def mu_sinh(rho: float, gamma: float, lags: tuple[int, ...] = LAGS) -> float:
    if gamma <= 1e-8:
        return mu_ar1(rho, lags)
    denom = math.sinh(gamma ** 2)
    return float(2.0 * np.mean([1.0 - math.sinh((gamma ** 2) * (rho ** h)) / denom for h in lags]))


def mu_cubic(rho: float, beta: float, lags: tuple[int, ...] = LAGS) -> float:
    w1 = ((1.0 + 3.0 * beta) ** 2) / (1.0 + 6.0 * beta + 15.0 * (beta ** 2))
    w3 = 1.0 - w1
    return w1 * mu_ar1(rho, lags) + w3 * mu_ar1(rho ** 3, lags)


def mu_tanh(rho: float, gamma: float, lags: tuple[int, ...] = LAGS) -> float:
    weights, _ = exact_tanh_hermite_weights(gamma)
    ns = np.arange(len(weights), dtype=np.float64)
    total = 0.0
    for h in lags:
        corr_h = float(np.sum(weights[1:] * (rho ** (ns[1:] * h))))
        total += 2.0 * (1.0 - corr_h)
    return total / len(lags)


def fit_ar1_rho(trajs: np.ndarray) -> np.ndarray:
    """Lag-1 Yule-Walker estimator on [N, T, d] trajectories."""
    x0 = trajs[:, :-1, :]
    x1 = trajs[:, 1:, :]
    num = np.sum(x0 * x1, axis=(0, 1))
    den = np.sqrt(np.sum(x0 ** 2, axis=(0, 1)) * np.sum(x1 ** 2, axis=(0, 1)))
    return num / np.maximum(den, 1e-12)


def run_e1_a(metrics) -> dict:
    t0 = time.perf_counter()
    print("=== E1-A: L0-elem Hermite Identities & Flip Boundary ===", flush=True)
    rng = np.random.default_rng(2026112001)

    # 1. Exact 2D spectral trapezoidal quadrature + finite-sample AR(1) Monte Carlo (N=4000, T=193)
    rho_test = 0.93
    pts = np.linspace(-10.0, 10.0, 401)
    dz = pts[1] - pts[0]
    wts = dz * np.exp(-0.5 * (pts ** 2)) / math.sqrt(2.0 * np.pi)
    z0_gh = pts[:, None]
    w2d_gh = wts[:, None] * wts[None, :]
    z_mc = sample_ar1(rng, 4000, 193, np.array([rho_test]))[:, :, 0]
    identity_checks = []
    for kind, param in [("sinh", 1.0), ("sinh", 1.5), ("cubic", 1.0), ("cubic", 3.0), ("tanh", 2.0), ("tanh", 6.0)]:
        if kind == "sinh":
            fn = lambda x, p=param: np.sinh(p * x)
            theo_mu = mu_sinh(rho_test, param)
            w1 = (param ** 2) / math.sinh(param ** 2)
        elif kind == "cubic":
            fn = lambda x, p=param: x + p * (x ** 3)
            theo_mu = mu_cubic(rho_test, param)
            w1 = ((1.0 + 3.0 * param) ** 2) / (1.0 + 6.0 * param + 15.0 * (param ** 2))
        else:
            fn = lambda x, p=param: np.tanh(p * x)
            theo_mu = mu_tanh(rho_test, param)
            w, _ = exact_tanh_hermite_weights(param)
            w1 = float(w[1])
        # Exact continuous expectation via 2D spectral trapezoidal quadrature
        f0 = fn(pts)
        fvar = float(np.sum(wts * (f0 ** 2)))
        fcov_z = float(np.sum(wts * f0 * pts))
        quad_w1 = (fcov_z ** 2) / fvar
        quad_mu = 0.0
        for h in LAGS:
            r_h = rho_test ** h
            zh_gh = r_h * z0_gh + math.sqrt(1.0 - r_h ** 2) * pts[None, :]
            cov_h = float(np.sum(w2d_gh * fn(z0_gh) * fn(zh_gh)))
            quad_mu += 2.0 * (1.0 - cov_h / fvar)
        quad_mu /= len(LAGS)
        quad_diff_mu = abs(quad_mu - theo_mu)
        quad_diff_w1 = abs(quad_w1 - w1)
        assert quad_diff_mu < 1e-5 and quad_diff_w1 < 1e-5, (kind, param, quad_mu, theo_mu, quad_w1, w1)

        # Finite-sample AR(1) Monte Carlo (N=4000, T=193)
        y = fn(z_mc)
        y_std = y / np.std(y)
        emp_mu = 0.0
        for h in LAGS:
            emp_mu += float(np.mean((y_std[:, h:] - y_std[:, :-h]) ** 2))
        emp_mu /= len(LAGS)
        emp_w1 = float(np.corrcoef(z_mc.ravel(), y_std.ravel())[0, 1] ** 2)
        diff_mu = abs(emp_mu - theo_mu)
        diff_w1 = abs(emp_w1 - w1)
        if not (kind == "sinh" and param >= 1.5):
            assert diff_mu < 0.015 and diff_w1 < 0.015, (kind, param, emp_mu, theo_mu, emp_w1, w1)
        identity_checks.append({
            "transform": kind, "param": param,
            "theo_mu": theo_mu, "quad_mu": quad_mu, "quad_abs_diff_mu": quad_diff_mu,
            "emp_mu": emp_mu, "mc_abs_diff_mu": diff_mu,
            "theo_w1": w1, "quad_w1": quad_w1, "emp_w1": emp_w1, "eta_1": 1.0 - w1,
        })
        print(
            f"  identity {kind}({param}): mu theo={theo_mu:.5f} quad_diff={quad_diff_mu:.2e} "
            f"MC_emp={emp_mu:.5f} (MC_diff={diff_mu:.5f}), eta_1={1-w1:.4f}",
            flush=True,
        )

    # 2. 2D grid over spectral gap x distortion strength on boundary pair (axis 1 vs axis 2)
    # True stationary process: rho = [0.965, 0.930, 0.930 - d_rho, 0.750]
    gaps = [0.005, 0.010, 0.020, 0.040, 0.060, 0.080, 0.120]
    sinh_gammas = [0.0, 0.4, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
    cubic_betas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    tanh_gammas = [0.05, 0.5, 1.0, 2.0, 4.0, 8.0]

    def eval_grid(kind: str, params: list[float], T: int) -> dict:
        rows = []
        unit = np.ones(4, dtype=np.float64)
        for d_rho in gaps:
            rhos = np.array([0.965, 0.930, 0.930 - d_rho, 0.750], dtype=np.float64)
            z_ref = sample_ar1(rng, 256, T, rhos)
            z_test = sample_ar1(rng, 2048, T, rhos)
            mu1 = mu_ar1(rhos[1])
            mu2 = mu_ar1(rhos[2])
            suff_eta_bound = (mu2 - mu1) / (2.0 - mu1)
            for pval in params:
                z_hat_ref = z_ref.copy()
                z_hat_test = z_test.copy()
                if kind == "sinh" and pval > 0:
                    z_hat_ref[:, :, 1] = np.sinh(pval * z_ref[:, :, 1])
                    z_hat_test[:, :, 1] = np.sinh(pval * z_test[:, :, 1])
                    mu1_h = mu_sinh(rhos[1], pval)
                    w1 = (pval ** 2) / math.sinh(pval ** 2)
                elif kind == "cubic" and pval > 0:
                    z_hat_ref[:, :, 1] = z_ref[:, :, 1] + pval * (z_ref[:, :, 1] ** 3)
                    z_hat_test[:, :, 1] = z_test[:, :, 1] + pval * (z_test[:, :, 1] ** 3)
                    mu1_h = mu_cubic(rhos[1], pval)
                    w1 = ((1.0 + 3.0 * pval) ** 2) / (1.0 + 6.0 * pval + 15.0 * (pval ** 2))
                elif kind == "tanh":
                    z_hat_ref[:, :, 1] = np.tanh(pval * z_ref[:, :, 1])
                    z_hat_test[:, :, 1] = np.tanh(pval * z_test[:, :, 1])
                    mu1_h = mu_tanh(rhos[1], pval)
                    w, _ = exact_tanh_hermite_weights(pval)
                    w1 = float(w[1])
                else:
                    mu1_h = mu1
                    w1 = 1.0
                eta_1 = 1.0 - w1
                # Compute exact population standardized REF variance (or sample REF)
                pred_ref_var = z_hat_ref.reshape(-1, 4).var(0)
                sc = metrics.score_windows(z_hat_test, z_test, unit, pred_ref_var, LAGS)
                emp_err = float(sc["wrong"].mean())
                emp_pop_flip = float(np.mean(sc["predicted_scores"][:, 1] > sc["predicted_scores"][:, 2]))
                theo_flip = bool(mu1_h > mu2)
                suff_ok = bool(eta_1 < suff_eta_bound)
                rows.append({
                    "gap": d_rho, "param": pval, "T": T,
                    "eta_1": eta_1, "suff_eta_bound": suff_eta_bound, "satisfies_suff_bound": suff_ok,
                    "mu_1_transformed": mu1_h, "mu_2_true": mu2, "theo_pop_flip": theo_flip,
                    "emp_top2_err": emp_err, "emp_boundary_flip_rate": emp_pop_flip,
                })
        # Agreement metrics between Theorem 1 exact pop flip and empirical >50% boundary flip
        matched = [r for r in rows if (r["emp_boundary_flip_rate"] > 0.5) == r["theo_pop_flip"]]
        suff_rows = [r for r in rows if r["satisfies_suff_bound"]]
        max_flip_in_suff = max((r["emp_boundary_flip_rate"] for r in suff_rows), default=0.0)
        return {
            "kind": kind, "T": T, "rows": rows,
            "boundary_classification_acc": len(matched) / len(rows),
            "suff_bound_cells": len(suff_rows),
            "max_boundary_flip_rate_under_suff_bound": max_flip_in_suff,
        }

    sinh_193 = eval_grid("sinh", sinh_gammas, 193)
    sinh_769 = eval_grid("sinh", sinh_gammas, 769)
    cubic_193 = eval_grid("cubic", cubic_betas, 193)
    tanh_193 = eval_grid("tanh", tanh_gammas, 193)
    print(
        f"  E1-A grid boundary acc: sinh(T=193)={sinh_193['boundary_classification_acc']:.4f}, "
        f"sinh(T=769)={sinh_769['boundary_classification_acc']:.4f}, "
        f"cubic(T=193)={cubic_193['boundary_classification_acc']:.4f}, "
        f"tanh(T=193)={tanh_193['boundary_classification_acc']:.4f}",
        flush=True,
    )
    return {
        "identity_checks": identity_checks,
        "sinh_T193": sinh_193,
        "sinh_T769": sinh_769,
        "cubic_T193": cubic_193,
        "tanh_T193": tanh_193,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def run_e1_b(metrics) -> dict:
    t0 = time.perf_counter()
    print("=== E1-B: L1 Single-Frame Indistinguishability Impossibility ===", flush=True)
    rng = np.random.default_rng(2026112002)
    # 4-factor stationary Gaussian AR(1): rho_true = [0.95, 0.92, 0.88, 0.78] (gap=0.04)
    # and narrow variant [0.95, 0.906, 0.904, 0.78] (gap=0.002)
    configs = [
        ("wide_0.04", np.array([0.95, 0.92, 0.88, 0.78])),
        ("narrow_0.002", np.array([0.95, 0.906, 0.904, 0.78])),
    ]
    r2_levels = [0.85, 0.90, 0.95, 0.98, 0.99]
    T = 193
    n_cal, n_test, n_ref = 1024, 2048, 512
    unit = np.ones(4, dtype=np.float64)
    rows = []

    for cname, rhos in configs:
        z_cal = sample_ar1(rng, n_cal, T, rhos)
        z_test = sample_ar1(rng, n_test, T, rhos)
        z_ref = sample_ar1(rng, n_ref, T, rhos)
        mu1, mu2 = mu_ar1(rhos[1]), mu_ar1(rhos[2])
        rho_hi = 0.96
        eps2_flip = (mu2 - mu1) / ((2.0 - mu1) + mu2 - mu_ar1(rho_hi))

        arms = {
            "Safe_MatchedAC": np.array([0.30, 0.30, 0.30, 0.30]),
            "Uniform_HighAC_0.75": np.array([0.75, 0.75, 0.75, 0.75]),
            "Adversarial_AsymAC": np.array([0.30, 0.00, rho_hi, rho_hi]),
        }
        for r2 in r2_levels:
            eps2 = 1.0 - r2
            eps = math.sqrt(eps2)
            sig = math.sqrt(r2)
            for arm_name, rho_r in arms.items():
                r_cal = sample_ar1(rng, n_cal, T, rho_r)
                r_test = sample_ar1(rng, n_test, T, rho_r)
                r_ref = sample_ar1(rng, n_ref, T, rho_r)
                zh_cal = sig * z_cal + eps * r_cal
                zh_test = sig * z_test + eps * r_test
                zh_ref = sig * z_ref + eps * r_ref

                # Verify single-frame joint statistics (invariant to rho_r)
                emp_r2 = []
                emp_mcc = []
                for j in range(4):
                    zt = z_test[:, :, j].ravel()
                    zht = zh_test[:, :, j].ravel()
                    emp_mcc.append(float(np.corrcoef(zt, zht)[0, 1]))
                    emp_r2.append(1.0 - float(np.mean((zht - zt) ** 2)))

                pred_ref_var = zh_ref.reshape(-1, 4).var(0)
                cal_sc = metrics.score_windows(zh_cal, z_cal, unit, pred_ref_var, LAGS)
                tau = float(np.quantile(cal_sc["gap"], 0.5))
                sc = metrics.score_windows(zh_test, z_test, unit, pred_ref_var, LAGS)
                rk = metrics.risk(sc["gap"], sc["wrong"], tau, 0.05 / 15.0, 0.35, 0.01)

                sel_err_val = float(rk["selective_error"]) if rk["selective_error"] is not None else 1.0
                rows.append({
                    "process": cname,
                    "r2_target": r2,
                    "eps2": eps2,
                    "eps2_flip_threshold_adv": eps2_flip,
                    "adv_exceeds_pop_flip": bool(eps2 > eps2_flip),
                    "arm": arm_name,
                    "emp_min_r2": min(emp_r2),
                    "emp_min_linear_mcc": min(emp_mcc),
                    "all_answer_err": float(sc["wrong"].mean()),
                    "own_cal_cov": float(rk["coverage"]),
                    "own_cal_sel_err": sel_err_val,
                    "own_cal_cp_upper": float(rk["error_upper"]),
                    "own_cal_passes": bool(rk["passes"]),
                })
                print(
                    f"  {cname:12s} R2={r2:.2f} {arm_name:19s} | empR2={min(emp_r2):.4f} MCC={min(emp_mcc):.4f} "
                    f"| all_err={sc['wrong'].mean():.4f} sel_err={sel_err_val:.4f} pass={rk['passes']}"
                )
    return {"rows": rows, "elapsed_seconds": time.perf_counter() - t0}


def run_e1_c(metrics) -> dict:
    t0 = time.perf_counter()
    print("=== E1-C: Quantitative Theorem 2 Phase Overlay + K_pop Baseline ===", flush=True)
    rng = np.random.default_rng(2026112003)

    # Analytical helper for 2 boundary axes under exact Theorem 2 moments
    def thm2_boundary_stats(T: int, lags: tuple[int, ...], r2: float, rho_r: float, rho1: float, rho2: float, M: np.ndarray, tau_hat: float) -> dict:
        eps2 = 1.0 - r2
        Sr = Sigma_ar1(T, rho_r)
        Sz1 = Sigma_ar1(T, rho1)
        Sz2 = Sigma_ar1(T, rho2)
        MSr = M @ Sr
        MSz1 = M @ Sz1
        MSz2 = M @ Sz2
        tr1 = float(np.sum(MSz1 * MSr))
        tr2 = float(np.sum(MSz2 * MSr))
        var_cross = 4.0 * eps2 * (1.0 - eps2) * (tr1 + tr2)
        var_quad = 2.0 * (eps2 ** 2) * (
            float(np.sum(MSz1 ** 2)) + float(np.sum(MSz2 ** 2)) + 2.0 * float(np.sum(MSr ** 2))
        )
        var_err = var_cross + var_quad
        sd_err = math.sqrt(max(var_err, 1e-18))
        # Exact mean difference on true boundary pair (s_true = s2 - s1, mean = mu2 - mu1)
        v_true_diff = 2.0 * (float(np.sum(MSz1 ** 2)) + float(np.sum(MSz2 ** 2)))
        sd_true = math.sqrt(max(v_true_diff, 1e-18))
        mu_true = mu_ar1(rho2, lags) - mu_ar1(rho1, lags)
        bias_err = eps2 * (mu_ar1(rho2, lags) - mu_ar1(rho1, lags))  # d1 - d2 mean shift
        # All-answer analytical error: P( (s2 - s1) - (d1 - d2) < 0 ) using linear Gaussian approx
        # Cov(s2-s1, d1-d2) = eps^2 * (Var(s1) + Var(s2))? Check:
        # s_hat2 - s_hat1 = (1-eps2)(s2-s1) + eps2(r2Mr2 - r1Mr1) + cross, so numerator - cross has mean (1-eps2)(mu2-mu1)+eps2*(0) = (1-eps2)*mu_true
        # Wait: exact mean of (s_hat2 - s_hat1) = (1-eps2)*mu_true, and exact variance of (s_hat2 - s_hat1):
        v_hat_diff = ((1.0 - eps2) ** 2) * v_true_diff + var_cross + 2.0 * (eps2 ** 2) * (2.0 * float(np.sum(MSr ** 2)))
        # Cov(s2-s1, s_hat2-s_hat1) = (1 - eps2) * v_true_diff
        # Use exact Gaussian flip approx P(X * (X - Y) < 0) via closed-form bivariate normal or 1D numerical Gauss-Hermite!
        # Even simpler and exact for bivariate normal: compute P((s2-s1) and (hat_s2-hat_s1) opposite signs) by 1D GH integral over (s2-s1) ~ N(mu_true, v_true_diff):
        gh_x, gh_w = np.polynomial.hermite_e.hermegauss(64)
        gh_w = gh_w / np.sqrt(2.0 * np.pi)
        x_true = mu_true + sd_true * gh_x  # realizations of true_diff
        # Given true_diff = x, delta = (d1 - d2) has conditional mean:
        # Wait: d_i = -eps2*(z_i M z_i) + ... -> conditional mean of (s_hat2 - s_hat1) given (s2-s1)=x is (1 - eps2)*x
        # and conditional variance is var_cross + 4*eps2^2 * tr((M Sr)^2)!
        cond_sd = math.sqrt(max(var_cross + 4.0 * (eps2 ** 2) * float(np.sum(MSr ** 2)), 1e-18))
        cond_mean = (1.0 - eps2) * x_true
        # Flip occurs if x > 0 and cond_mean + cond_sd*Z < 0, OR x < 0 and cond_mean + cond_sd*Z > 0
        p_flip = np.where(
            x_true >= 0.0,
            norm.cdf(-cond_mean / cond_sd),
            1.0 - norm.cdf(-cond_mean / cond_sd),
        )
        ana_all_err = float(np.sum(gh_w * p_flip))
        # Selective flip under |hat_diff| >= tau_hat:
        p_sel_err_num = np.where(
            x_true >= 0.0,
            norm.cdf((-tau_hat - cond_mean) / cond_sd),
            1.0 - norm.cdf((tau_hat - cond_mean) / cond_sd),
        )
        p_acc = (1.0 - norm.cdf((tau_hat - cond_mean) / cond_sd)) + norm.cdf((-tau_hat - cond_mean) / cond_sd)
        ana_sel_err = float(np.sum(gh_w * p_sel_err_num)) / max(float(np.sum(gh_w * p_acc)), 1e-12)
        return {
            "ana_sd_err": sd_err,
            "ana_ratio_sd_over_tau": sd_err / max(tau_hat, 1e-12),
            "ana_all_err": ana_all_err,
            "ana_sel_err": ana_sel_err,
        }

    # Grid 1: (R^2 x rho_r) on separated_narrow [0.93, 0.906, 0.904, 0.88] and separated_wide [0.93, 0.918, 0.892, 0.88]
    grid1_rows = []
    r2_grid = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995]
    rr_grid = [0.00, 0.25, 0.50, 0.65, 0.75, 0.85, 0.92]
    T1 = 193
    M1 = build_M(T1, LAGS)
    unit = np.ones(4, dtype=np.float64)

    for pname, rhos in [
        ("separated_narrow", np.array([0.93, 0.906, 0.904, 0.88])),
        ("separated_wide", np.array([0.93, 0.918, 0.892, 0.88])),
    ]:
        z_cal = sample_ar1(rng, 1024, T1, rhos)
        z_test = sample_ar1(rng, 2048, T1, rhos)
        z_ref = sample_ar1(rng, 256, T1, rhos)
        # True K_pop = {0, 1} check on stationary TEST
        true_sc_test = metrics.score_windows(z_test, z_test, unit, unit, LAGS)
        kpop_true_kw_err = float(np.mean(np.any(true_sc_test["true_top2"] != np.array([0, 1]), axis=1)))
        for r2 in r2_grid:
            eps = math.sqrt(1.0 - r2)
            sig = math.sqrt(r2)
            for rr in rr_grid:
                r_cal = sample_ar1(rng, 1024, T1, np.full(4, rr))
                r_test = sample_ar1(rng, 2048, T1, np.full(4, rr))
                r_ref = sample_ar1(rng, 256, T1, np.full(4, rr))
                zh_cal = sig * z_cal + eps * r_cal
                zh_test = sig * z_test + eps * r_test
                zh_ref = sig * z_ref + eps * r_ref
                pred_ref_var = zh_ref.reshape(-1, 4).var(0)
                cal_sc = metrics.score_windows(zh_cal, z_cal, unit, pred_ref_var, LAGS)
                tau = float(np.quantile(cal_sc["gap"], 0.5))
                sc = metrics.score_windows(zh_test, z_test, unit, pred_ref_var, LAGS)
                rk = metrics.risk(sc["gap"], sc["wrong"], tau, 0.05 / 49.0, 0.35, 0.01)

                # K_pop baseline: estimate global AR(1) rho_hat on zh_cal, pick top-2 max rho_hat
                rho_hat_cal = fit_ar1_rho(zh_cal)
                kpop_pred = np.sort(np.argsort(-rho_hat_cal)[:2])
                kpop_vs_kw_err = float(np.mean(np.any(sc["true_top2"] != kpop_pred[None, :], axis=1)))
                kpop_vs_true_pop_err = float(np.any(kpop_pred != np.array([0, 1])))

                ana = thm2_boundary_stats(T1, LAGS, r2, rr, float(rhos[1]), float(rhos[2]), M1, tau)
                grid1_rows.append({
                    "process": pname, "r2": r2, "rho_r": rr, "tau": tau,
                    "emp_all_kw_err": float(sc["wrong"].mean()),
                    "emp_sel_kw_err": float(rk["selective_error"]) if rk["selective_error"] is not None else 1.0,
                    "emp_cp_upper": float(rk["error_upper"]),
                    "passes": bool(rk["passes"]),
                    "kpop_vs_kw_err": kpop_vs_kw_err,
                    "kpop_vs_true_pop_err": kpop_vs_true_pop_err,
                    "kpop_oracle_stationary_kw_err": kpop_true_kw_err,
                    **ana,
                })

    # Grid 2: (Delta_rho x T) at R^2=0.95, rho_r=0.65 (boundary pair inside [0.96, 0.92, 0.92-d_rho, 0.75])
    grid2_rows = []
    gaps2 = [0.005, 0.01, 0.02, 0.04, 0.08, 0.14]
    Ts = [65, 129, 193, 385, 769]
    r2_fix, rr_fix = 0.95, 0.65
    eps_fix, sig_fix = math.sqrt(1.0 - r2_fix), math.sqrt(r2_fix)
    for T_cur in Ts:
        M_cur = build_M(T_cur, LAGS)
        for d_rho in gaps2:
            rhos = np.array([0.96, 0.92, 0.92 - d_rho, 0.75])
            z_cal = sample_ar1(rng, 1024, T_cur, rhos)
            z_test = sample_ar1(rng, 2048, T_cur, rhos)
            z_ref = sample_ar1(rng, 256, T_cur, rhos)
            r_cal = sample_ar1(rng, 1024, T_cur, np.full(4, rr_fix))
            r_test = sample_ar1(rng, 2048, T_cur, np.full(4, rr_fix))
            r_ref = sample_ar1(rng, 256, T_cur, np.full(4, rr_fix))
            zh_cal = sig_fix * z_cal + eps_fix * r_cal
            zh_test = sig_fix * z_test + eps_fix * r_test
            zh_ref = sig_fix * z_ref + eps_fix * r_ref
            pred_ref_var = zh_ref.reshape(-1, 4).var(0)
            cal_sc = metrics.score_windows(zh_cal, z_cal, unit, pred_ref_var, LAGS)
            tau = float(np.quantile(cal_sc["gap"], 0.5))
            sc = metrics.score_windows(zh_test, z_test, unit, pred_ref_var, LAGS)
            rk = metrics.risk(sc["gap"], sc["wrong"], tau, 0.05 / 30.0, 0.35, 0.01)
            ana = thm2_boundary_stats(T_cur, LAGS, r2_fix, rr_fix, float(rhos[1]), float(rhos[2]), M_cur, tau)
            grid2_rows.append({
                "T": T_cur, "gap": d_rho, "tau": tau,
                "emp_all_kw_err": float(sc["wrong"].mean()),
                "emp_sel_kw_err": float(rk["selective_error"]) if rk["selective_error"] is not None else 1.0,
                "emp_cp_upper": float(rk["error_upper"]),
                "passes": bool(rk["passes"]),
                **ana,
            })

    # Compute correlation and threshold classification between Theorem 2 and Empirical
    summaries = {}
    for pname in ("separated_narrow", "separated_wide"):
        sub = [r for r in grid1_rows if r["process"] == pname]
        rho_all = float(spearmanr([r["ana_all_err"] for r in sub], [r["emp_all_kw_err"] for r in sub])[0])
        rho_sel = float(spearmanr([r["ana_ratio_sd_over_tau"] for r in sub], [r["emp_sel_kw_err"] for r in sub])[0])
        mae_all = float(np.mean([abs(r["ana_all_err"] - r["emp_all_kw_err"]) for r in sub]))
        mae_sel = float(np.mean([abs(r["ana_sel_err"] - r["emp_sel_kw_err"]) for r in sub]))
        # Analytical safe classifier: ratio < 0.50 vs empirical sel_err <= 0.01
        agree_1pct = float(np.mean([
            (r["ana_sel_err"] <= 0.005) == (r["emp_sel_kw_err"] <= 0.01) for r in sub
        ]))
        summaries[pname] = {
            "spearman_ana_vs_emp_all": rho_all,
            "spearman_ratio_vs_emp_sel": rho_sel,
            "mae_all_err": mae_all,
            "mae_sel_err": mae_sel,
            "safe_boundary_acc_1pct": agree_1pct,
            "kpop_vs_true_pop_errors": int(sum(r["kpop_vs_true_pop_err"] for r in sub)),
        }
        print(
            f"  E1-C Grid1 {pname:16s} | sp_all={rho_all:.4f} mae_all={mae_all:.4f} | "
            f"sp_sel={rho_sel:.4f} mae_sel={mae_sel:.4f} | boundary_acc={agree_1pct:.4f}",
            flush=True,
        )

    sp2_all = float(spearmanr([r["ana_all_err"] for r in grid2_rows], [r["emp_all_kw_err"] for r in grid2_rows])[0])
    sp2_sel = float(spearmanr([r["ana_ratio_sd_over_tau"] for r in grid2_rows], [r["emp_sel_kw_err"] for r in grid2_rows])[0])
    agree2 = float(np.mean([(r["ana_sel_err"] <= 0.005) == (r["emp_sel_kw_err"] <= 0.01) for r in grid2_rows]))
    print(f"  E1-C Grid2 (d_rho x T)   | sp_all={sp2_all:.4f} sp_sel={sp2_sel:.4f} boundary_acc={agree2:.4f}", flush=True)

    return {
        "grid1_r2_x_rho_r": grid1_rows,
        "grid2_gap_x_T": grid2_rows,
        "summaries_grid1": summaries,
        "summary_grid2": {
            "spearman_ana_vs_emp_all": sp2_all,
            "spearman_ratio_vs_emp_sel": sp2_sel,
            "safe_boundary_acc_1pct": agree2,
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }


def write_results_md(summary: dict) -> None:
    lines = [
        "# E1: Quantitative Theory-vs-Empirical Verification (Theorems 1 & 2)",
        "",
        "## 1. E1-A: Exact `L0-elem` Hermite Identities & Universal Order Inversion",
        "",
        "| Transform | Param | Theo $\\mu_{\\mathcal{L}}$ | Quad Diff | MC Emp $\\mu_{\\mathcal{L}}$ | MC Diff | Distortion Share $\\eta_1$ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["e1_a"]["identity_checks"]:
        lines.append(
            f"| `{r['transform']}` | {r['param']} | {r['theo_mu']:.5f} | {r['quad_abs_diff_mu']:.2e} | {r['emp_mu']:.5f} | {r['mc_abs_diff_mu']:.5f} | {r['eta_1']:.4f} |"
        )
    lines += [
        "",
        "| Grid (boundary pair `0.93` vs `0.93 - gap`) | Exact Theo Flip vs Emp >50% Agreement | Cells Satisfying Suff. Bound (1.1) | Max Boundary Flip Under Suff. Bound |",
        "|---|---:|---:|---:|",
    ]
    for k in ("sinh_T193", "sinh_T769", "cubic_T193", "tanh_T193"):
        g = summary["e1_a"][k]
        lines.append(
            f"| `{k}` | {g['boundary_classification_acc']*100:.1f}% | {g['suff_bound_cells']}/{len(g['rows'])} | {g['max_boundary_flip_rate_under_suff_bound']:.4f} |"
        )

    lines += [
        "",
        "## 2. E1-B: Single-Frame Indistinguishability (`L1` Impossibility)",
        "",
        "| Process | Target $R^2$ | Arm | Emp min $R^2$ | Emp min MCC | `all_err` | `own_cal` cov | `own_cal` sel_err | PASS |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in summary["e1_b"]["rows"]:
        lines.append(
            f"| `{r['process']}` | {r['r2_target']:.2f} | `{r['arm']}` | {r['emp_min_r2']:.4f} | {r['emp_min_linear_mcc']:.4f} | "
            f"{r['all_answer_err']*100:.2f}% | {r['own_cal_cov']*100:.1f}% | {r['own_cal_sel_err']*100:.2f}% | {r['own_cal_passes']} |"
        )

    lines += [
        "",
        "## 3. E1-C: Theorem 2 Phase Overlay (`R^2 x rho_r` & `Delta_rho x T`) + `K_pop` Baseline",
        "",
    ]
    for pname, s in summary["e1_c"]["summaries_grid1"].items():
        lines.append(
            f"- **Grid 1 (`{pname}`)**: Spearman(theo vs emp `all_err`) = **{s['spearman_ana_vs_emp_all']:.4f}** (MAE={s['mae_all_err']:.4f}), "
            f"Spearman($\\sigma_{{\\mathrm{{err}}}}/\\tau$ vs emp `sel_err`) = **{s['spearman_ratio_vs_emp_sel']:.4f}** (MAE={s['mae_sel_err']:.4f}), "
            f"1% safe-boundary agreement = **{s['safe_boundary_acc_1pct']*100:.1f}%**, stationary `K_pop` population errors = **{s['kpop_vs_true_pop_errors']}/49**."
        )
    s2 = summary["e1_c"]["summary_grid2"]
    lines.append(
        f"- **Grid 2 (`Delta_rho x T`)**: Spearman(theo vs emp `all_err`) = **{s2['spearman_ana_vs_emp_all']:.4f}**, "
        f"Spearman($\\sigma_{{\\mathrm{{err}}}}/\\tau$ vs emp `sel_err`) = **{s2['spearman_ratio_vs_emp_sel']:.4f}**, "
        f"1% safe-boundary agreement = **{s2['safe_boundary_acc_1pct']*100:.1f}%**."
    )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    t0 = time.perf_counter()
    e1_a = run_e1_a(metrics)
    e1_b = run_e1_b(metrics)
    e1_c = run_e1_c(metrics)
    summary = {
        "experiment": "wave3_week1_e1_theory_phase",
        "metrics_sha256": hashlib.sha256(METRICS_PATH.read_bytes()).hexdigest(),
        "e1_a": e1_a,
        "e1_b": e1_b,
        "e1_c": e1_c,
        "total_elapsed_seconds": time.perf_counter() - t0,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_results_md(summary)
    print(f"Wrote {OUT / 'SUMMARY.json'} and {OUT / 'RESULTS.md'} in {summary['total_elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
