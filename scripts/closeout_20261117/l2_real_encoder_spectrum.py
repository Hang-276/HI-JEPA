#!/usr/bin/env python3
"""L2: Real frozen LeJEPA encoder residual spectrum, (a, b) ratios, and excess-error attribution.

Part A (Supervised Readouts A4 / A2 strong vs weak):
  - Measures real neural residual variance, lag autocovariance, cross-axis leakage,
    and effective residual dynamic slowness mu(r_j) across all 8 frozen LeJEPA encoders.
  - Runs a 5-level counterfactual ladder on the exact sealed A2 TEST trajectories to
    attribute the excess error over the isotropic scalar-R^2 white-noise sandbox.

Part B (Unlabeled Lag Unmixing B2, lags 1/4/16/32/multi):
  - Decomposes real frozen LeJEPA embeddings h(z) = A z + e_nl(z) on a large independent
    stationary panel to measure the real population nonlinear spectral bias ratio b_real(l)
    vs. the finite-CAL (N_cal=1024) sampling fluctuation ratio a_real(l), linking both to
    sealed B2 axis recovery and TEST slow-selection accuracy across all 80 cells.
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
from scipy.stats import spearmanr
import torch

ROOT = Path(__file__).resolve().parents[2]
A4_DIR = ROOT / "reports/lejepa_a4_20261115"
A2_DIR = ROOT / "reports/lejepa_a2_readout_20261115"
B2_DIR = ROOT / "reports/lejepa_b2_lag_20261115"
OUT = ROOT / "results/closeout_20261117/l2_real_encoder_mechanism"


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def top2_set(scores: np.ndarray) -> np.ndarray:
    return np.sort(np.argsort(scores, axis=1, kind="stable")[:, :2], axis=1)


def slow_scores_np(paths: np.ndarray, lags: list[int] | tuple[int, ...]) -> np.ndarray:
    return sum(np.mean((paths[:, h:] - paths[:, :-h]) ** 2, axis=1) for h in lags) / len(lags)


def whitened_lag_ops_single(latents: np.ndarray, lags: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute whitened symmetric lag operators [len(lags)+1, D, D] matching B2 white_operators."""
    mean = latents.mean(axis=(0, 1))
    c0 = np.einsum("nti,ntj->ij", latents, latents, optimize=True) / (latents.shape[0] * latents.shape[1]) - np.outer(mean, mean)
    vals, vecs = np.linalg.eigh(c0)
    whitening = (vecs / np.sqrt(vals[None, :])) @ vecs.T
    ops = []
    for h in lags:
        left, right = latents[:, :-h], latents[:, h:]
        raw = np.einsum("nti,ntj->ij", left, right, optimize=True) / (left.shape[0] * left.shape[1])
        raw = 0.5 * (raw + raw.T)
        end_mean = 0.5 * (left.mean(axis=(0, 1)) + right.mean(axis=(0, 1)))
        shift = np.outer(end_mean, mean)
        centered = raw - shift - shift.T + np.outer(mean, mean)
        op = whitening @ centered @ whitening
        ops.append(0.5 * (op + op.T))
    ops.append(sum(ops) / len(ops))
    return mean, whitening, np.stack(ops, axis=0)


def run_part_a(a4, panel, weak_mod) -> dict:
    """Part A: Supervised strong vs weak readouts residual spectrum & excess-error ladder."""
    a2_cfg = json.loads((A2_DIR / "config.json").read_text())
    a2_forecast = json.loads((A2_DIR / "FORECAST.json").read_text())
    a4_forecast = json.loads((A4_DIR / "NEW_FORECAST.json").read_text())
    weak_receipt = json.loads((A2_DIR / "weak/RECEIPT.json").read_text())["members"]
    lags = a2_cfg["lags"]

    models = sorted({f"{c}_{s}" for s in a2_cfg["seeds"] for c in a2_cfg["training_cases"]})
    readouts = {}
    audit_r2_map = {}
    for name in models:
        readouts[(name, "strong")] = panel.load_readout(name)[0]
        readouts[(name, "weak")] = weak_mod.load_readout(name)[0]
        audit_r2_map[(name, "strong")] = np.asarray(weak_receipt[name]["audit"]["strong_r2"], dtype=np.float64)
        audit_r2_map[(name, "weak")] = np.asarray(weak_receipt[name]["audit"]["weak_r2"], dtype=np.float64)

    cell_lookup = {(c["name"], c["process"], c["arm"]): c for c in a2_forecast["cells"]}
    rng_mc = np.random.default_rng(2026111702)

    ladder_rows = []
    spectrum_rows = []

    for seed in a2_cfg["seeds"]:
        for offset, (process, rates) in enumerate(a2_cfg["processes"].items()):
            master = a4_forecast["config"]["new_master_base"] + seed * 10 + offset
            # Regenerate exact A2 TEST physical trajectories (read-only deterministic seed)
            z_test = a4.trajectories(a2_cfg, master, a2_cfg["test_role"], a2_cfg["n_test"], rates)
            true_scores = slow_scores_np(z_test, lags)  # low-dim known unit variance = 1
            true_top2 = top2_set(true_scores)

            # Independent diagnostic trajectory pool to measure population residual spectrum cleanly
            z_diag = rng_mc.normal(size=(2048, a2_cfg["length"], 4))
            rho = np.asarray(rates, dtype=np.float64)
            sc = np.sqrt(1.0 - rho**2)
            for t in range(1, a2_cfg["length"]):
                z_diag[:, t] = rho * z_diag[:, t - 1] + sc * z_diag[:, t]

            for tcase in a2_cfg["training_cases"]:
                name = f"{tcase}_{seed}"
                for arm in ("strong", "weak"):
                    meta = cell_lookup[(name, process, arm)]
                    npz_path = A2_DIR / meta["path"]
                    assert sha256_file(npz_path) == meta["sha256"]
                    sealed_pred_scores = np.load(npz_path)["predicted_scores"]
                    ref_var = np.asarray(meta["reference_variance"], dtype=np.float64)
                    r2_vec = audit_r2_map[(name, arm)]
                    r2_mean = float(r2_vec.mean())
                    r2_min = float(r2_vec.min())

                    # Verify our float32 PyTorch forward readout matches sealed scores within float32 precision
                    ro = readouts[(name, arm)]
                    z_hat_test = np.concatenate([ro(z_test[i : i + 256]) for i in range(0, len(z_test), 256)], axis=0)
                    recomp_scores = slow_scores_np(z_hat_test, lags) / ref_var[None, :]
                    np.testing.assert_allclose(recomp_scores, sealed_pred_scores, rtol=1e-5, atol=1e-5)

                    # Measure real residual r = z_hat - z on independent diagnostic stream z_diag
                    z_hat_diag = np.concatenate([ro(z_diag[i : i + 256]) for i in range(0, len(z_diag), 256)], axis=0)
                    res_diag = z_hat_diag - z_diag
                    res_centered = res_diag - res_diag.mean(axis=(0, 1), keepdims=True)
                    res_var = res_centered.var(axis=(0, 1), ddof=0)
                    res_mse = np.mean(res_diag**2, axis=(0, 1))
                    cov_zr = np.mean(
                        (z_diag - z_diag.mean(axis=(0, 1), keepdims=True)) * res_centered,
                        axis=(0, 1),
                    )
                    res_num = slow_scores_np(res_centered, lags).mean(axis=0)
                    res_eff_slowness = res_num / np.maximum(res_var, 1e-15)
                    sig_eff_slowness = np.mean([2.0 * (1.0 - rho**h) for h in lags], axis=0)

                    # Off-diagonal linear cross-axis mixing matrix M_mix: r_t approx z_t @ B
                    zf = z_diag.reshape(-1, 4)
                    rf = res_centered.reshape(-1, 4)
                    b_mix = (zf.T @ rf) / len(zf)  # since Cov(z) = I_4 in population
                    b_off = b_mix - np.diag(np.diag(b_mix))
                    rem_nl = (rf - zf @ b_mix).reshape(z_diag.shape)
                    rem_nl_var = rem_nl.var(axis=(0, 1), ddof=0)
                    rem_nl_slowness = slow_scores_np(rem_nl, lags).mean(axis=0) / np.maximum(rem_nl_var, 1e-15)

                    # Lag-1 autocorrelation of residual per axis
                    r_lag1_corr = np.mean(res_centered[:, 1:] * res_centered[:, :-1], axis=(0, 1)) / np.maximum(res_var, 1e-15)

                    # 5-Level Counterfactual Error Attribution Ladder on exact z_test (averaged over 4 MC draws for noise steps)
                    n_mc = 4
                    err_l0, err_l1, err_l1_min, err_l2, err_l3 = [], [], [], [], []
                    # Precompute empirical block-Toeplitz temporal Cholesky for residual per axis + instantaneous cross-cov?
                    # Even cleaner: generate exact AR(1)-matched + linear cross-axis mixing Gaussian residual!
                    # Specifically: r_gauss_t = z_test_t @ b_mix + u_t, where u_t is 4D VAR/AR(1) matching rem_nl lag-0 cov & lag-1 autocorr!
                    u_flat = rem_nl.reshape(-1, 4)
                    cov_u0 = (u_flat.T @ u_flat) / len(u_flat)
                    phi_u = np.clip(
                        np.mean(rem_nl[:, 1:] * rem_nl[:, :-1], axis=(0, 1)) / np.maximum(rem_nl_var, 1e-15),
                        -0.995,
                        0.995,
                    )
                    a_u = np.diag(phi_u)
                    q_u = cov_u0 - a_u @ cov_u0 @ a_u.T
                    # Project q_u to PSD if tiny numerical asymmetry occurs
                    evals_q, evecs_q = np.linalg.eigh(0.5 * (q_u + q_u.T))
                    chol_u0 = np.linalg.cholesky(cov_u0 + 1e-12 * np.eye(4))
                    chol_qu = evecs_q @ np.diag(np.sqrt(np.maximum(evals_q, 0.0)))

                    for _ in range(n_mc):
                        w = rng_mc.normal(size=z_test.shape)
                        # L0: scalar mean R^2 white noise, population denom (1 + (1 - r2_mean))
                        v0 = max(1.0 - r2_mean, 0.0)
                        s0 = slow_scores_np(z_test + np.sqrt(v0) * w, lags) / (1.0 + v0)
                        err_l0.append(float(np.mean(np.any(top2_set(s0) != true_top2, axis=1))))

                        # L0b: scalar min R^2 white noise (matches EXPLORATORY min-R^2 anchor column)
                        v0m = max(1.0 - r2_min, 0.0)
                        s0m = slow_scores_np(z_test + np.sqrt(v0m) * w, lags) / (1.0 + v0m)
                        err_l1_min.append(float(np.mean(np.any(top2_set(s0m) != true_top2, axis=1))))

                        # L1: heterogeneous per-axis R^2_j white noise, population denom (1 + v_j)
                        vj = np.maximum(1.0 - r2_vec, 0.0)
                        s1 = slow_scores_np(z_test + w * np.sqrt(vj)[None, None, :], lags) / (1.0 + vj)[None, :]
                        err_l1.append(float(np.mean(np.any(top2_set(s1) != true_top2, axis=1))))

                        # L2: heterogeneous per-axis R^2_j white noise + actual sealed REF variance denominator!
                        # Match total output variance by keeping z_test + w*sqrt(vj) and dividing by actual ref_var * (1+vj)/actual_pop_var?
                        # Wait: to isolate purely the finite-REF sampling shift eta_i = ref_var - pop_var_actual,
                        # scale white-noise output so its population variance equals pop_var_actual = 1 + 2*cov_zr + res_var, then divide by ref_var!
                        pop_var_actual = 1.0 + 2.0 * cov_zr + res_var
                        s2 = (
                            slow_scores_np(z_test + w * np.sqrt(vj)[None, None, :], lags)
                            * (pop_var_actual / (1.0 + vj))[None, :]
                            / ref_var[None, :]
                        )
                        err_l2.append(float(np.mean(np.any(top2_set(s2) != true_top2, axis=1))))

                        # L3: Gaussian residual matching linear cross-mixing b_mix + temporal AR(1) autocorrelation phi_u + actual ref_var
                        u_sim = np.empty_like(z_test)
                        u_sim[:, 0] = rng_mc.normal(size=(len(z_test), 4)) @ chol_u0.T
                        innov_u = rng_mc.normal(size=(len(z_test), z_test.shape[1] - 1, 4))
                        for t in range(1, z_test.shape[1]):
                            u_sim[:, t] = u_sim[:, t - 1] @ a_u.T + innov_u[:, t - 1] @ chol_qu.T
                        z_hat_g = z_test + z_test @ b_mix + u_sim
                        s3 = slow_scores_np(z_hat_g, lags) / ref_var[None, :]
                        err_l3.append(float(np.mean(np.any(top2_set(s3) != true_top2, axis=1))))

                    err_actual = float(np.mean(np.any(top2_set(sealed_pred_scores) != true_top2, axis=1)))

                    spectrum_rows.append({
                        "name": name,
                        "seed": seed,
                        "training_case": tcase,
                        "process": process,
                        "arm": arm,
                        "audit_r2": r2_vec.tolist(),
                        "residual_mse": res_mse.tolist(),
                        "residual_var": res_var.tolist(),
                        "cov_z_r": cov_zr.tolist(),
                        "residual_lag1_autocorr": r_lag1_corr.tolist(),
                        "residual_effective_slowness": res_eff_slowness.tolist(),
                        "signal_effective_slowness": sig_eff_slowness.tolist(),
                        "nonlinear_rem_effective_slowness": rem_nl_slowness.tolist(),
                        "off_diag_linear_mixing_frob": float(np.linalg.norm(b_off, ord="fro")),
                        "ref_var_sealed": ref_var.tolist(),
                        "pop_var_estimated": (1.0 + 2.0 * cov_zr + res_var).tolist(),
                    })
                    ladder_rows.append({
                        "name": name,
                        "process": process,
                        "arm": arm,
                        "err_l0_iso_mean_r2": float(np.mean(err_l0)),
                        "err_l0b_iso_min_r2": float(np.mean(err_l1_min)),
                        "err_l1_hetero_axis_r2": float(np.mean(err_l1)),
                        "err_l2_plus_sealed_ref": float(np.mean(err_l2)),
                        "err_l3_plus_temporal_and_cross_cov": float(np.mean(err_l3)),
                        "err_l4_actual_neural": err_actual,
                    })

    # Group summaries by (arm, process)
    grouped_ladder = {}
    for arm in ("strong", "weak"):
        for process in ("separated_narrow", "separated_wide"):
            sub = [r for r in ladder_rows if r["arm"] == arm and r["process"] == process]
            grouped_ladder[f"{arm}__{process}"] = {
                k: float(np.mean([r[k] for r in sub]))
                for k in (
                    "err_l0_iso_mean_r2",
                    "err_l0b_iso_min_r2",
                    "err_l1_hetero_axis_r2",
                    "err_l2_plus_sealed_ref",
                    "err_l3_plus_temporal_and_cross_cov",
                    "err_l4_actual_neural",
                )
            }

    grouped_spectrum = {}
    for arm in ("strong", "weak"):
        for process in ("separated_narrow", "separated_wide"):
            sub = [r for r in spectrum_rows if r["arm"] == arm and r["process"] == process]
            grouped_spectrum[f"{arm}__{process}"] = {
                "mean_residual_lag1_autocorr_by_axis": np.mean([r["residual_lag1_autocorr"] for r in sub], axis=0).tolist(),
                "mean_residual_effective_slowness_by_axis": np.mean([r["residual_effective_slowness"] for r in sub], axis=0).tolist(),
                "mean_signal_effective_slowness_by_axis": sub[0]["signal_effective_slowness"],
                "mean_off_diag_mixing_frob": float(np.mean([r["off_diag_linear_mixing_frob"] for r in sub])),
            }

    return {
        "grouped_excess_error_ladder": grouped_ladder,
        "grouped_residual_spectrum": grouped_spectrum,
        "cells_ladder": ladder_rows,
        "cells_spectrum": spectrum_rows,
    }


def run_part_b(panel, b2_mod) -> dict:
    """Part B: Real frozen LeJEPA encoder lag-unmixing spectral bias ratio b_real vs sampling ratio a_real."""
    b2_cfg = json.loads((B2_DIR / "config.json").read_text())
    b2_forecast = json.loads((B2_DIR / "FORECAST.json").read_text())
    b2_summary = json.loads((B2_DIR / "SUMMARY.json").read_text())

    panel_cfg, train_cfg, _ = panel.configs()
    encoders = {}
    for name in panel.panel_names(panel_cfg):
        model = panel.encoder_source.make_mlp_encoder(4, train_cfg["hidden"], train_cfg["layers"], device="cpu").eval()
        model.load_state_dict(torch.load(A4_DIR / "panel/runs" / f"{name}.pt", weights_only=True, map_location="cpu"))
        encoders[name] = model

    # Compute high-precision population operator C_pop^(l) using M_pairs=524,288 INDEPENDENT
    # stationary transition pairs (Z_0, Z_l) -> eliminates trajectory correlation variance (kappa_rho ~ 28x)!
    rng_pop = np.random.default_rng(2026111703)
    m_pairs = 524288
    cal_lags = b2_cfg["cal_lags"]  # [1, 4, 16, 32]
    eval_lags = b2_cfg["eval_lags"]  # [4, 16, 32]
    arms = b2_cfg["arms"]  # ["lag1", "lag4", "lag16", "lag32", "multi"]

    z0_iid = rng_pop.normal(size=(m_pairs, 4))
    innov_iid = rng_pop.normal(size=(m_pairs, 4))

    ideal_eigs_map = {}
    ideal_gaps_map = {}
    for process, rates in b2_cfg["processes"].items():
        rho = np.asarray(rates, dtype=np.float64)
        ideal = rho[None, :] ** np.asarray(cal_lags)[:, None]
        ideal = np.concatenate([ideal, ideal.mean(axis=0, keepdims=True)], axis=0)
        diffs = np.abs(ideal[:, :, None] - ideal[:, None, :])
        diffs[:, np.arange(4), np.arange(4)] = np.inf
        rel_gap = diffs[:, 1:3, :].min(axis=(1, 2))
        ideal_eigs_map[process] = ideal
        ideal_gaps_map[process] = rel_gap

    pop_ops_cache = {}
    bias_same_coord = {}
    encoder_linearity_stats = {}

    for name, model in encoders.items():
        # Marginal moments on z0_iid (shared across processes since stationary marginal is always N(0, I_4)!)
        h0 = panel.encoder_source.embed(model, z0_iid)
        mean_h = h0.mean(axis=0)
        h0_c = h0 - mean_h[None, :]
        c0 = (h0_c.T @ h0_c) / m_pairs
        vals, vecs = np.linalg.eigh(c0)
        w_mat = (vecs / np.sqrt(vals[None, :])) @ vecs.T
        u0 = h0_c @ w_mat  # whitened embedding at t=0 [M, 4]

        # Orthogonal decomposition u0 = z0 @ Q_sig + e_nl(z0)
        q_sig = (z0_iid.T @ u0) / m_pairs
        e_nl0 = u0 - z0_iid @ q_sig
        nl_frac = float(np.mean(np.sum(e_nl0**2, axis=1)) / 4.0)
        u_svd, s_svd, vt_svd = np.linalg.svd(q_sig)
        q_orth = u_svd @ vt_svd
        encoder_linearity_stats[name] = {
            "linear_signal_variance_fraction": 1.0 - nl_frac,
            "nonlinear_distortion_fraction_eps": nl_frac,
            "whitened_signal_singular_values": s_svd.tolist(),
        }

        for process, rates in b2_cfg["processes"].items():
            rho = np.asarray(rates, dtype=np.float64)
            ops_u = []
            for h in cal_lags:
                rh = rho**h
                zl = z0_iid * rh[None, :] + innov_iid * np.sqrt(1.0 - rh**2)[None, :]
                hl = panel.encoder_source.embed(model, zl)
                ul = (hl - mean_h[None, :]) @ w_mat
                op = (u0.T @ ul) / m_pairs
                ops_u.append(0.5 * (op + op.T))
            ops_u.append(sum(ops_u) / len(ops_u))
            ops_pop = np.stack(ops_u, axis=0)
            pop_ops_cache[(name, process)] = ops_pop

            ideal_eigs = ideal_eigs_map[process]
            rel_gaps = ideal_gaps_map[process]
            b_ops = []
            sin_angles = []
            for a_idx in range(len(arms)):
                # Note: in whitened embedding u, linear signal alone has lag operator q_sig.T @ diag(ideal) @ q_sig;
                # ideal pure O(4) operator is q_orth.T @ diag(ideal) @ q_orth. Difference captures exact nonlinear distortion bias!
                c_ideal_in_u = q_orth.T @ np.diag(ideal_eigs[a_idx]) @ q_orth
                b_op = float(np.linalg.norm(ops_pop[a_idx] - c_ideal_in_u, ord=2))
                b_ops.append(b_op / float(rel_gaps[a_idx]))
                _, evecs_u = np.linalg.eigh(ops_pop[a_idx])
                v_top2 = evecs_u[:, -2:]
                target_top2 = q_orth[:2, :].T
                svals = np.linalg.svd(v_top2.T @ target_top2, compute_uv=False)
                sin_angles.append(float(np.sqrt(max(0.0, 1.0 - np.min(svals) ** 2))))
            bias_same_coord[(name, process)] = (b_ops, sin_angles)

    # Now evaluate every sealed B2 cell (16 model-process pairs x 5 lag arms = 80 records)
    records = []
    for cell in b2_forecast["cells"]:
        name = cell["name"]
        seed = cell["seed"]
        process = cell["process"]
        p_idx = cell["process_index"]
        rates = b2_cfg["processes"][process]

        z_cal = b2_mod.trajectories(b2_cfg, "cal", seed, p_idx, rates, b2_cfg["cal_windows"])
        z_test = b2_mod.trajectories(b2_cfg, "test", seed, p_idx, rates, b2_cfg["test_windows"])
        true_scores = slow_scores_np(z_test, eval_lags)
        true_top2 = top2_set(true_scores)

        h_cal = panel.encoder_source.embed(encoders[name], z_cal)
        _, _, ops_cal = whitened_lag_ops_single(h_cal, cal_lags)
        ops_pop = pop_ops_cache[(name, process)]
        b_ops, sin_angles = bias_same_coord[(name, process)]

        npz_path = B2_DIR / cell["path"]
        assert sha256_file(npz_path) == cell["sha256"]
        sealed_scores = np.load(npz_path)["scores"]  # [5, 2048, 4]

        rel_gaps = ideal_gaps_map[process]
        ideal_eigs = ideal_eigs_map[process]

        for a_idx, arm_label in enumerate(arms):
            # Sampling fluctuation ||C_cal - C_pop||_2 (aligning symmetric whitening gauge C0_cal^{-1/2} vs C0_pop^{-1/2}
            # Note: both symmetric square roots live in R^4 embedding space so C0_cal^{-1/2} ~ C0_pop^{-1/2} directly!)
            sampling_op_norm = float(np.linalg.norm(ops_cal[a_idx] - ops_pop[a_idx], ord=2))
            eigs_pop = np.linalg.eigvalsh(ops_pop[a_idx])[::-1]
            eigs_ideal = np.sort(ideal_eigs[a_idx])[::-1]
            eig_bias_inf = float(np.max(np.abs(eigs_pop - eigs_ideal)))

            gap_ideal = float(rel_gaps[a_idx])
            a_ratio = sampling_op_norm / gap_ideal
            b_eig_ratio = eig_bias_inf / gap_ideal

            arm_diag = cell["calibration"]["arms"][arm_label]
            corr2_axes23 = float(np.mean(arm_diag["state_audit"]["axis_correlation_squared"][1:3]))
            r2_axes23 = float(np.mean(arm_diag["state_audit"]["axis_r2_named_scale"][1:3]))
            off_diag_rms = float(arm_diag["state_audit"]["off_diagonal_loading_rms_axes2_3"])
            boot_radius = float(arm_diag["operator_radius_q90"])

            pred_top2 = top2_set(sealed_scores[a_idx])
            test_err = float(np.mean(np.any(pred_top2 != true_top2, axis=1)))

            records.append({
                "name": name,
                "seed": seed,
                "training_case": cell["training_case"],
                "process": process,
                "arm": arm_label,
                "ideal_boundary_eigengap": gap_ideal,
                "sampling_op_error_cal_vs_pop": sampling_op_norm,
                "bootstrap_radius_q90_sealed": boot_radius,
                "pop_eigenvalue_bias_max": eig_bias_inf,
                "a_sampling_over_gap": a_ratio,
                "b_eigbias_over_gap": b_eig_ratio,
                "b_operator_bias_over_gap": b_ops[a_idx],
                "a_plus_b_operator": a_ratio + b_ops[a_idx],
                "pop_top2_subspace_sin_angle": sin_angles[a_idx],
                "boundary_axes23_corr2": corr2_axes23,
                "boundary_axes23_named_r2": r2_axes23,
                "off_diagonal_loading_rms_23": off_diag_rms,
                "test_all_answer_error": test_err,
                "test_all_answer_accuracy": 1.0 - test_err,
            })

    # Aggregate table by (process, arm)
    by_proc_arm = {}
    for process in ("separated_narrow", "separated_wide"):
        for arm_label in arms:
            sub = [r for r in records if r["process"] == process and r["arm"] == arm_label]
            by_proc_arm[f"{process}__{arm_label}"] = {
                "ideal_eigengap": sub[0]["ideal_boundary_eigengap"],
                "mean_sampling_op_error": float(np.mean([r["sampling_op_error_cal_vs_pop"] for r in sub])),
                "mean_a_sampling_over_gap": float(np.mean([r["a_sampling_over_gap"] for r in sub])),
                "mean_b_operator_bias_over_gap": float(np.mean([r["b_operator_bias_over_gap"] for r in sub])),
                "mean_a_plus_b_operator": float(np.mean([r["a_plus_b_operator"] for r in sub])),
                "mean_pop_subspace_sin_angle": float(np.mean([r["pop_top2_subspace_sin_angle"] for r in sub])),
                "mean_boundary_corr2": float(np.mean([r["boundary_axes23_corr2"] for r in sub])),
                "mean_off_diag_rms": float(np.mean([r["off_diagonal_loading_rms_23"] for r in sub])),
                "mean_test_accuracy": float(np.mean([r["test_all_answer_accuracy"] for r in sub])),
                "mean_test_error": float(np.mean([r["test_all_answer_error"] for r in sub])),
            }

    # Spearman correlations across all 80 real encoder cells (and within narrow=40)
    def corr_stats(subset: list[dict]) -> dict:
        a_arr = np.array([r["a_sampling_over_gap"] for r in subset])
        b_arr = np.array([r["b_operator_bias_over_gap"] for r in subset])
        ab_arr = np.array([r["a_plus_b_operator"] for r in subset])
        off_arr = np.array([r["off_diagonal_loading_rms_23"] for r in subset])
        err_arr = np.array([r["test_all_answer_error"] for r in subset])
        c2_arr = np.array([r["boundary_axes23_corr2"] for r in subset])
        return {
            "spearman_a_vs_off_diag_rms": float(spearmanr(a_arr, off_arr).statistic),
            "spearman_b_vs_off_diag_rms": float(spearmanr(b_arr, off_arr).statistic),
            "spearman_a_plus_b_vs_off_diag_rms": float(spearmanr(ab_arr, off_arr).statistic),
            "spearman_a_plus_b_vs_boundary_corr2": float(spearmanr(ab_arr, c2_arr).statistic),
            "spearman_a_plus_b_vs_test_error": float(spearmanr(ab_arr, err_arr).statistic),
            "spearman_a_vs_test_error": float(spearmanr(a_arr, err_arr).statistic),
            "spearman_b_vs_test_error": float(spearmanr(b_arr, err_arr).statistic),
        }

    # Verify EVERY cell's accuracy matches sealed B2 SUMMARY.json to machine precision!
    for r in records:
        cell_key = f"{r['name']}__{r['process']}"
        expected_acc = b2_summary["by_cell"][cell_key]["test"][r["arm"]]["accuracy"]
        assert abs(r["test_all_answer_accuracy"] - expected_acc) < 1e-12, (cell_key, r["arm"])

    return {
        "encoder_linearity_stats": encoder_linearity_stats,
        "by_process_and_lag_arm": by_proc_arm,
        "correlations_all_80_cells": corr_stats(records),
        "correlations_narrow_40_cells": corr_stats([r for r in records if r["process"] == "separated_narrow"]),
        "correlations_wide_40_cells": corr_stats([r for r in records if r["process"] == "separated_wide"]),
        "cells": records,
    }


def main() -> None:
    t0 = time.perf_counter()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(A4_DIR))
    a4 = load_mod("ro_a4_forecast", A4_DIR / "forecast.py")
    panel = a4.panel
    weak_mod = load_mod("ro_a2_weak", A2_DIR / "weak/run.py")
    b2_mod = load_mod("ro_b2_run", B2_DIR / "run.py")

    part_a = run_part_a(a4, panel, weak_mod)
    part_b = run_part_b(panel, b2_mod)

    payload = {
        "evidence_tier": "read_only_real_encoder_mechanism_verification",
        "elapsed_seconds": time.perf_counter() - t0,
        "part_a_supervised_readout_excess_error_and_spectrum": part_a,
        "part_b_unlabeled_lag_unmixing_real_a_and_b_ratios": part_b,
    }
    (OUT / "l2_real_encoder_mechanism.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "status": "OK",
        "elapsed_seconds": payload["elapsed_seconds"],
        "part_a_ladder": part_a["grouped_excess_error_ladder"],
        "part_a_spectrum": part_a["grouped_residual_spectrum"],
        "part_b_narrow": {k: v for k, v in part_b["by_process_and_lag_arm"].items() if k.startswith("separated_narrow")},
        "part_b_correlations_all80": part_b["correlations_all_80_cells"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
