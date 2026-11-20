#!/usr/bin/env python3
"""B1-2 (P3-A): Unsupervised Cardoso-Souloumiac SOBI and CAL-Bootstrap eSOBI.

Evaluates equal-weight Joint Approximate Diagonalization (SOBI) and CAL-only
inverse-variance weighted eSOBI against single-lag SFA and unweighted averaging:
  1. Sub-Experiment 1: Exact head-to-head comparison on Low-Dim B2 encoders
     (verifying exact numerical match to sealed B2 SUMMARY.json on baseline arms).
  2. Sub-Experiment 2: Zero-label slow-subspace discovery and quadratic-harmonic
     occupation on frozen 8-D 32x32 visual encoders from formal_v1.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/wave2_local_20261119/p3a_unmixing"
B2_DIR = ROOT / "reports/lejepa_b2_lag_20261115"
VIS_DIR = ROOT / "results/visual_sprites_lejepa/formal_v1"
VIS_SCRIPTS = ROOT / "scripts/visual_sprites_lejepa"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def off_diag_energy(matrices: np.ndarray, weights: np.ndarray) -> float:
    diag = np.diagonal(matrices, axis1=-2, axis2=-1)
    total = np.sum(matrices ** 2, axis=(-2, -1))
    diag_sum = np.sum(diag ** 2, axis=-1)
    return float(np.sum(weights * (total - diag_sum)))


def joint_diagonalize(matrices: np.ndarray, weights: np.ndarray | None = None,
                      tol: float = 1e-9, max_sweeps: int = 200):
    """Cardoso-Souloumiac Givens-rotation JAD for real symmetric matrices [K, d, d]."""
    A = np.array(matrices, dtype=np.float64, copy=True)
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    k_cnt, d, d2 = A.shape
    if d != d2:
        raise ValueError(f"Expected square matrices, got {A.shape}")
    if weights is None:
        w = np.full(k_cnt, 1.0 / k_cnt, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (k_cnt,) or np.any(w <= 0) or not np.isfinite(w).all():
            raise ValueError(f"Invalid weights: {w}")
        w = w / w.sum()

    V = np.eye(d, dtype=np.float64)
    sweeps = 0
    for sweep in range(1, max_sweeps + 1):
        sweeps = sweep
        max_sin = 0.0
        for p in range(d - 1):
            for q in range(p + 1, d):
                g0 = A[:, p, p] - A[:, q, q]
                g1 = 2.0 * A[:, p, q]
                g00 = float(np.dot(w, g0 * g0))
                g01 = float(np.dot(w, g0 * g1))
                g11 = float(np.dot(w, g1 * g1))
                G = np.array([[g00, g01], [g01, g11]], dtype=np.float64)
                eigvals, eigvecs = np.linalg.eigh(G)
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

                # Column rotation A <- A @ R, where R = [[c, -s], [s, c]]
                cp = A[:, :, p].copy()
                cq = A[:, :, q].copy()
                A[:, :, p] = c * cp + s * cq
                A[:, :, q] = -s * cp + c * cq

                # Row rotation A <- R^T @ A
                rp = A[:, p, :].copy()
                rq = A[:, q, :].copy()
                A[:, p, :] = c * rp + s * rq
                A[:, q, :] = -s * rp + c * rq

                # Accumulate orthogonal eigenvectors V <- V @ R
                vp = V[:, p].copy()
                vq = V[:, q].copy()
                V[:, p] = c * vp + s * vq
                V[:, q] = -s * vp + c * vq

        if max_sin < tol:
            break

    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    # Sort columns deterministically by descending weighted mean eigenvalue.
    weighted_diag = np.sum(w[:, None] * np.diagonal(A, axis1=-2, axis2=-1), axis=0)
    order = np.argsort(weighted_diag)[::-1]
    V = V[:, order]
    A = A[:, order, :][:, :, order]
    return V, sweeps, off_diag_energy(A, w)


def verify_jad_unit_test():
    rng = np.random.default_rng(2026111999)
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    mats = np.stack([Q @ np.diag(rng.uniform(-1.0, 2.0, 8)) @ Q.T for _ in range(4)])
    V, sweeps, off_e = joint_diagonalize(mats)
    ortho_err = float(np.max(np.abs(V.T @ V - np.eye(8))))
    if ortho_err > 1e-12 or off_e > 1e-11:
        raise AssertionError(f"JAD unit test failed: ortho_err={ortho_err}, off_e={off_e}")
    print(f"JAD unit test PASSED: sweeps={sweeps}, ortho_err={ortho_err:.2e}, off_e={off_e:.2e}", flush=True)


def run_subexp1_lowdim_b2() -> dict:
    t0 = time.perf_counter()
    print("\n=== B1-2 Sub-Experiment 1: Low-Dim B2 Head-to-Head (SFA vs SOBI vs eSOBI) ===", flush=True)
    b2 = load_module("p3a_b2_run", B2_DIR / "run.py")
    config = json.loads((B2_DIR / "config.json").read_text())
    sealed_summary = json.loads((B2_DIR / "SUMMARY.json").read_text())

    panel_config, train_config, _ = b2.panel.configs()
    encoders = {}
    for name in b2.panel.panel_names(panel_config):
        model = b2.encoder.make_mlp_encoder(4, train_config["hidden"], train_config["layers"], device="cpu").eval()
        model.load_state_dict(torch.load(B2_DIR / f"../lejepa_a4_20261115/panel/runs/{name}.pt", weights_only=True, map_location="cpu"))
        encoders[name] = model

    old_arms = list(config["arms"])  # ['lag1', 'lag4', 'lag16', 'lag32', 'multi']
    new_arms = ["sobi_equal", "esobi_inv", "esobi_snr"]
    all_arms = old_arms + new_arms
    dose = config["name_dose"]
    deltas = np.eye(4)[:, None, :] * dose
    n_cal = config["cal_windows"]
    base_weights = np.full(n_cal, 1.0 / n_cal)

    by_cell = {}
    max_baseline_diff = 0.0
    results_grid = {}

    for seed in config["seeds"]:
        for process_index, (process, rates) in enumerate(config["processes"].items()):
            cal = b2.trajectories(config, "cal", seed, process_index, rates, n_cal)
            anchors = b2.state_pool(config, "name", seed, process_index, config["name_anchors"])
            reference = b2.state_pool(config, "ref", seed, process_index, config["reference_states"])
            audit = b2.state_pool(config, "audit", seed, process_index, config["audit_states"])
            test = b2.trajectories(config, "test", seed, process_index, rates, config["test_windows"])
            true_scores = np.diagonal(b2.dynamic_gram(test, config["eval_lags"]), axis1=1, axis2=2)
            true_order = b2.encoder.top_two(true_scores)

            bootstrap = b2.rng(config, "cal_bootstrap", seed, process_index)
            weights = np.vstack([
                base_weights,
                bootstrap.multinomial(n_cal, base_weights, size=config["cal_bootstrap_draws"]) / n_cal,
            ])

            for training in config["training_cases"]:
                name = f"{training}_{seed}"
                model = encoders[name]
                pos = b2.encoder.embed(model, anchors[None] + deltas)
                neg = b2.encoder.embed(model, anchors[None] - deltas)
                response = ((pos - neg) / (2.0 * dose)).mean(axis=1)
                latent = b2.encoder.embed(model, cal)

                means, _, _, whitening, operators = b2.white_operators(latent, config["cal_lags"], weights)
                # Baseline 5 arms on index 0 (full CAL)
                _, vecs_old = np.linalg.eigh(operators[0])
                axes_old = whitening[0][None] @ vecs_old[..., ::-1]  # (5, 4, 4)

                # CAL-only whitened lag operators and bootstrap variances
                M = operators[0, :4]
                M_boot = operators[1:, :4]
                sigma2 = np.mean(np.sum((M_boot - M[None]) ** 2, axis=(-2, -1)), axis=0)
                if np.any(sigma2 <= 0) or not np.isfinite(sigma2).all():
                    raise ValueError(f"Invalid CAL bootstrap sigma2: {sigma2}")

                eigvals_sorted = np.sort(np.linalg.eigvalsh(M), axis=-1)[:, ::-1]
                emp_boundary_gap = np.maximum(eigvals_sorted[:, 1] - eigvals_sorted[:, 2], 1e-6)

                V_sobi, sw_sobi, off_sobi = joint_diagonalize(M, np.full(4, 0.25))
                w_inv = (1.0 / sigma2)
                w_inv /= w_inv.sum()
                V_inv, sw_inv, off_inv = joint_diagonalize(M, w_inv)
                w_snr = (emp_boundary_gap ** 2) / sigma2
                w_snr /= w_snr.sum()
                V_snr, sw_snr, off_snr = joint_diagonalize(M, w_snr)

                axes_new = np.stack([
                    whitening[0] @ V_sobi,
                    whitening[0] @ V_inv,
                    whitening[0] @ V_snr,
                ], axis=0)
                all_axes = np.concatenate([axes_old, axes_new], axis=0)  # (8, 4, 4)
                named, _, _, _ = b2.name_axes(all_axes[None], response)
                transform = named[0]  # (8, 4, 4)

                ref_h = b2.encoder.embed(model, reference)
                ref_var = np.einsum("ni,aij->anj", ref_h, transform).var(axis=1)
                state_audits = b2.named_audit(b2.encoder.embed(model, audit), audit, means[0], transform)
                test_gram = b2.dynamic_gram(b2.encoder.embed(model, test), config["eval_lags"])
                scores = np.einsum("aij,nik,akj->anj", transform, test_gram, transform, optimize=True) / ref_var[:, None]
                if not np.isfinite(scores).all():
                    raise ValueError(f"Nonfinite TEST scores for {name} / {process}")

                correct = np.stack([(b2.encoder.top_two(s) == true_order).all(axis=1) for s in scores])
                results_grid[(seed, process, training)] = correct

                key = f"{name}__{process}"
                sealed_cell = sealed_summary["by_cell"][key]["test"]
                arms_out = {}
                for a_idx, arm in enumerate(all_arms):
                    acc = float(correct[a_idx].mean())
                    if arm in old_arms:
                        diff = abs(acc - sealed_cell[arm]["accuracy"])
                        max_baseline_diff = max(max_baseline_diff, diff)
                    corr2 = state_audits[a_idx]["axis_correlation_squared"]
                    arms_out[arm] = {
                        "accuracy": acc,
                        "correct": int(correct[a_idx].sum()),
                        "gain_vs_lag1": float(acc - float(correct[0].mean())),
                        "gain_vs_multi": float(acc - float(correct[4].mean())),
                        "boundary_corr2_mean": float(0.5 * (corr2[1] + corr2[2])),
                        "boundary_corr2_axes12": [float(corr2[1]), float(corr2[2])],
                        "all_axes_corr2": [float(v) for v in corr2],
                        "off_diag_loading_rms_23": float(state_audits[a_idx]["off_diagonal_loading_rms_axes2_3"]),
                    }
                by_cell[key] = {
                    "seed": seed,
                    "process": process,
                    "training_case": training,
                    "cal_sigma2": sigma2.tolist(),
                    "cal_emp_boundary_gap": emp_boundary_gap.tolist(),
                    "weights_inv": w_inv.tolist(),
                    "weights_snr": w_snr.tolist(),
                    "jad_sweeps": {"sobi_equal": sw_sobi, "esobi_inv": sw_inv, "esobi_snr": sw_snr},
                    "jad_off_diag": {"sobi_equal": off_sobi, "esobi_inv": off_inv, "esobi_snr": off_snr},
                    "arms": arms_out,
                }

    print(f"Verified recomputed B2 baseline arms against sealed SUMMARY.json: max_abs_diff = {max_baseline_diff:.2e}", flush=True)
    if max_baseline_diff > 1e-12:
        raise AssertionError(f"Recomputed B2 baseline diverged from sealed summary: {max_baseline_diff}")

    # Aggregate & paired window bootstrap contrasts per process
    process_summaries = {}
    for process in config["processes"]:
        arm_aggs = {}
        for arm in all_arms:
            cells = [v for v in by_cell.values() if v["process"] == process]
            arm_aggs[arm] = {
                "accuracy": float(np.mean([c["arms"][arm]["accuracy"] for c in cells])),
                "gain_vs_lag1": float(np.mean([c["arms"][arm]["gain_vs_lag1"] for c in cells])),
                "gain_vs_multi": float(np.mean([c["arms"][arm]["gain_vs_multi"] for c in cells])),
                "boundary_corr2_mean": float(np.mean([c["arms"][arm]["boundary_corr2_mean"] for c in cells])),
                "off_diag_loading_rms_23": float(np.mean([c["arms"][arm]["off_diag_loading_rms_23"] for c in cells])),
            }

        # Paired window bootstrap across seeds matching b2.bootstrap_primary
        contrasts = {}
        n_draws = config["window_bootstrap_draws"]
        n_win = config["test_windows"]
        for comp_name, ref_idx in (("vs_lag1", 0), ("vs_multi", 4)):
            draws = np.zeros((len(new_arms), n_draws))
            for seed in config["seeds"]:
                rng_boot = b2.rng(config, "window_bootstrap", seed)
                stacked = np.stack([results_grid[(seed, process, case)] for case in config["training_cases"]])
                means_c = stacked.mean(axis=0)  # (8, n_win)
                new_idx = [all_arms.index(a) for a in new_arms]
                gain = means_c[new_idx] - means_c[ref_idx]
                for start in range(0, n_draws, 128):
                    stop = min(start + 128, n_draws)
                    rows = rng_boot.integers(0, n_win, size=(stop - start, n_win))
                    draws[:, start:stop] += gain[:, rows].mean(axis=2) / len(config["seeds"])
            alpha_bonf = config["familywise_alpha"] / len(new_arms)
            lower = np.quantile(draws, alpha_bonf, axis=1)
            upper = np.quantile(draws, 1.0 - alpha_bonf, axis=1)
            contrasts[comp_name] = {
                arm: {
                    "mean_gain": float(arm_aggs[arm][f"gain_{comp_name}"]),
                    "one_sided_bonf_lower_95": float(lower[i]),
                    "upper": float(upper[i]),
                    "significant_positive": bool(lower[i] > 0.0),
                }
                for i, arm in enumerate(new_arms)
            }

        process_summaries[process] = {"arms": arm_aggs, "contrasts": contrasts}
        print(f"  Process={process}:", flush=True)
        for arm in all_arms:
            g = arm_aggs[arm]
            print(
                f"    {arm:10s} | acc={g['accuracy']*100:6.2f}% | bnd_corr2={g['boundary_corr2_mean']:.4f} "
                f"| off_rms={g['off_diag_loading_rms_23']:.4f} | vs_lag1={g['gain_vs_lag1']*100:+5.2f}pp "
                f"| vs_multi={g['gain_vs_multi']*100:+5.2f}pp",
                flush=True,
            )

    return {
        "max_sealed_baseline_diff": float(max_baseline_diff),
        "process_summaries": process_summaries,
        "by_cell": by_cell,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    var_y = float(np.var(y))
    if var_y <= 1e-15:
        return 0.0
    design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    return float(max(0.0, min(1.0, np.var(pred) / var_y)))


def principal_sin_max(u_slow2: np.ndarray, z_all4: np.ndarray) -> float:
    """Principal angle max sin between linear map of (u0, u1)->z and slow subspace (z0, z1)."""
    u_std = (u_slow2 - u_slow2.mean(0)) / np.maximum(u_slow2.std(0), 1e-8)
    z_std = (z_all4 - z_all4.mean(0)) / np.maximum(z_all4.std(0), 1e-8)
    design = np.column_stack([np.ones(len(u_std)), u_std])
    beta, *_ = np.linalg.lstsq(design, z_std, rcond=None)
    B = beta[1:].T  # (4, 2)
    if np.linalg.matrix_rank(B) < 2:
        return 1.0
    Q, _ = np.linalg.qr(B)
    s = np.linalg.svd(Q[:2, :], compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - min(1.0, float(s[-1])) ** 2)))


def cal_whitened_operators_vis(cal_lat: np.ndarray, cal_lags=(1, 4, 16, 32), n_boot=64, seed=2026111941):
    n_ep, t_len, d = cal_lat.shape
    mu = cal_lat.reshape(-1, d).mean(axis=0, keepdims=True)
    zc = (cal_lat.reshape(-1, d) - mu).reshape(n_ep, t_len, d).astype(np.float64)
    c0_ep = np.einsum("nti,ntj->nij", zc, zc) / t_len

    def build_ops(w_vec):
        c0 = np.einsum("n,nij->ij", w_vec, c0_ep)
        vals, vecs = np.linalg.eigh(c0)
        keep = vals > 1e-8
        r = int(keep.sum())
        if r < 2:
            raise ValueError(f"Degenerate latent covariance rank {r}: {vals}")
        W = (vecs[:, keep] / np.sqrt(vals[keep])) @ vecs[:, keep].T
        ops = []
        for h in cal_lags:
            raw = np.einsum("nti,ntj->nij", zc[:, :-h], zc[:, h:]) / (t_len - h)
            raw = 0.5 * (raw + np.swapaxes(raw, -1, -2))
            c_h = np.einsum("n,nij->ij", w_vec, raw)
            # Project onto whitened subspace (vecs[:, keep] / sqrt(vals[keep]))^T c_h (...)
            P = vecs[:, keep] / np.sqrt(vals[keep])
            ops.append(P.T @ c_h @ P)
        ops = np.stack(ops, axis=0)
        return mu.reshape(d), vals, P_mat(vecs[:, keep], vals[keep]), ops

    def P_mat(vk, ek):
        return vk / np.sqrt(ek)

    w0 = np.full(n_ep, 1.0 / n_ep)
    # Build full whitener once on full CAL so M and M_boot share the exact same basis P!
    c0_full = np.einsum("n,nij->ij", w0, c0_ep)
    vals, vecs = np.linalg.eigh(c0_full)
    keep = vals > 1e-8
    rank = int(keep.sum())
    if rank < 2:
        raise ValueError(f"Degenerate CAL covariance rank {rank}: {vals}")
    P = vecs[:, keep] / np.sqrt(vals[keep])  # (d, rank)

    lag_raws = []
    for h in cal_lags:
        raw = np.einsum("nti,ntj->nij", zc[:, :-h], zc[:, h:]) / (t_len - h)
        raw = 0.5 * (raw + np.swapaxes(raw, -1, -2))
        lag_raws.append(raw)  # each (n_ep, d, d)

    def ops_from_weights(w_vec):
        out = []
        ep_mean = np.einsum("n,ntd->td", w_vec, zc)
        mu_w = ep_mean.mean(axis=0)
        for h_idx, h in enumerate(cal_lags):
            c_h = np.einsum("n,nij->ij", w_vec, lag_raws[h_idx])
            m_left = ep_mean[:-h].mean(axis=0)
            m_right = ep_mean[h:].mean(axis=0)
            c_h = c_h - 0.5 * (np.outer(m_left + m_right, mu_w) + np.outer(mu_w, m_left + m_right)) + np.outer(mu_w, mu_w)
            m_sym = P.T @ c_h @ P
            out.append(0.5 * (m_sym + m_sym.T))
        return np.stack(out, axis=0)

    M = ops_from_weights(w0)
    rng = np.random.default_rng(seed)
    boot_weights = rng.multinomial(n_ep, w0, size=n_boot) / n_ep
    M_boot = np.stack([ops_from_weights(wb) for wb in boot_weights], axis=0)
    return mu.reshape(d), vals, rank, P, M, M_boot


def run_subexp2_visual32_unsupervised() -> dict:
    t0 = time.perf_counter()
    print("\n=== B1-2 Sub-Experiment 2: 32x32 Visual 8-D Unsupervised Slow-Subspace & Harmonics ===", flush=True)
    lock = json.loads((VIS_DIR / "DESIGN_LOCK.json").read_text())
    cfg = lock["config"]
    render = load_module("p3a_vis_render", VIS_SCRIPTS / "render.py")
    models = load_module("p3a_vis_models", VIS_SCRIPTS / "models.py")
    device = models.configure_device("mps", cpu_threads=4) if torch.backends.mps.is_available() else torch.device("cpu")

    cell_names = [f"visual_lejepa_seed{s}" for s in (101, 102, 103)] + [
        f"long_horizon_pred_seed{s}" for s in (101, 102, 103)
    ]
    encoders = {}
    for name in cell_names:
        enc = models.make_encoder(cfg["encoder"]["latent_dim"], device)
        enc.load_state_dict(torch.load(VIS_DIR / name / "encoder.pt", map_location=device, weights_only=True))
        enc.eval()
        encoders[name] = enc

    role_latents, role_truth = {}, {}
    for role in ("cal", "test"):
        tr = time.perf_counter()
        truth = render.sample_states(cfg, role, development=False)
        batch_sz = cfg["runtime"].get("render_batch_size", cfg["runtime"]["encode_batch_size"])
        chunks = {name: [] for name in cell_names}
        for imgs in render.iter_role_images(cfg, role, batch_sz, development=False):
            for name, enc in encoders.items():
                chunks[name].append(models.encode(enc, imgs, device, cfg["runtime"]["encode_batch_size"]))
        role_latents[role] = {name: np.concatenate(chunks[name], axis=0).reshape(truth.shape[0], truth.shape[1], -1) for name in cell_names}
        role_truth[role] = truth
        print(f"  Encoded formal_v1 {role}: {truth.shape} ({time.perf_counter()-tr:.1f}s)", flush=True)

    test_z = role_truth["test"].reshape(-1, 4).astype(np.float64)
    z0, z1 = test_z[:, 0], test_z[:, 1]
    Z_slow = test_z[:, [0, 1]]
    Z_fast = test_z[:, [2, 3]]
    H_quad = np.column_stack([z0**2 - 1.0, z1**2 - 1.0, z0 * z1])
    Z_slow_plus_quad = np.column_stack([Z_slow, H_quad])

    cal_lags = (1, 4, 16, 32)
    eval_lags = tuple(cfg["lags"])
    methods = ("sfa_lag1", "sfa_lag4", "sfa_lag16", "sobi_equal", "esobi_inv")
    cells_out = {}

    for name in cell_names:
        cal_lat = role_latents["cal"][name].astype(np.float64)
        test_lat = role_latents["test"][name].astype(np.float64)
        mu, c0_eigs, rank, P, M, M_boot = cal_whitened_operators_vis(cal_lat, cal_lags=cal_lags)
        sigma2 = np.mean(np.sum((M_boot - M[None]) ** 2, axis=(-2, -1)), axis=0)
        w_inv = (1.0 / np.maximum(sigma2, 1e-18))
        w_inv /= w_inv.sum()

        V_map = {
            "sfa_lag1": np.linalg.eigh(M[0])[1][:, ::-1],
            "sfa_lag4": np.linalg.eigh(M[1])[1][:, ::-1],
            "sfa_lag16": np.linalg.eigh(M[2])[1][:, ::-1],
            "sobi_equal": joint_diagonalize(M, np.full(4, 0.25))[0],
            "esobi_inv": joint_diagonalize(M, w_inv)[0],
        }

        cell_methods = {}
        cal_centered = cal_lat - mu
        test_centered = (test_lat - mu).reshape(-1, test_lat.shape[-1])
        for mname in methods:
            axes = P @ V_map[mname]  # (8, rank)
            cal_u = cal_centered @ axes
            cal_scores = np.mean([
                np.mean((cal_u[:, h:, :] - cal_u[:, :-h, :]) ** 2, axis=(0, 1)) / np.maximum(cal_u.var(axis=(0, 1)), 1e-12)
                for h in eval_lags
            ], axis=0)
            order = np.argsort(cal_scores)  # ascending slowness score (slowest first)
            axes_sorted = axes[:, order]
            cal_scores_sorted = cal_scores[order]

            test_u_seq = (test_lat - mu) @ axes_sorted
            test_u = test_u_seq.reshape(-1, rank)
            r2_z0 = ols_r2(test_z[:, 0], test_u[:, :2])
            r2_z1 = ols_r2(test_z[:, 1], test_u[:, :2])
            sin_t = principal_sin_max(test_u[:, :2], test_z)

            mode_stats = []
            for k in range(rank):
                uk = test_u[:, k]
                r2_fast = ols_r2(uk, Z_fast)
                r2_slow_lin = ols_r2(uk, Z_slow)
                r2_slow_both = ols_r2(uk, Z_slow_plus_quad)
                r2_quad_only = ols_r2(uk, H_quad)
                r2_quad_inc = max(0.0, r2_slow_both - r2_slow_lin)
                # Lag-1 autocorrelation on TEST
                u_ep = test_u_seq[:, :, k]
                u0, u1 = u_ep[:, :-1].ravel(), u_ep[:, 1:].ravel()
                ac1 = float(np.corrcoef(u0, u1)[0, 1]) if np.std(u0) > 1e-12 and np.std(u1) > 1e-12 else 0.0
                mode_stats.append({
                    "mode_idx": k,
                    "cal_score": float(cal_scores_sorted[k]),
                    "test_lag1_ac": ac1,
                    "r2_fast_linear": r2_fast,
                    "r2_slow_linear": r2_slow_lin,
                    "r2_slow_quad_inc": r2_quad_inc,
                    "r2_slow_quad_raw": r2_quad_only,
                })

            cell_methods[mname] = {
                "slow_z0_r2": r2_z0,
                "slow_z1_r2": r2_z1,
                "slow_mean_r2": 0.5 * (r2_z0 + r2_z1),
                "slow_min_r2": min(r2_z0, r2_z1),
                "slow_principal_sin_max": sin_t,
                "modes_2_3_fast_r2_mean": float(0.5 * (mode_stats[2]["r2_fast_linear"] + mode_stats[3]["r2_fast_linear"])) if rank >= 4 else None,
                "modes_2_3_quad_r2_mean": float(0.5 * (mode_stats[2]["r2_slow_quad_inc"] + mode_stats[3]["r2_slow_quad_inc"])) if rank >= 4 else None,
                "max_fast_r2_modes_2_to_7": float(max(m["r2_fast_linear"] for m in mode_stats[2:])) if rank >= 4 else None,
                "fast_top_mode_idx": int(max(mode_stats, key=lambda m: m["r2_fast_linear"])["mode_idx"]),
                "modes": mode_stats,
            }

        cells_out[name] = {
            "c0_eigenvalues": c0_eigs.tolist(),
            "whitened_rank": rank,
            "cal_sigma2": sigma2.tolist(),
            "weights_inv": w_inv.tolist(),
            "methods": cell_methods,
        }

    # Summarize across visual_lejepa vs long_horizon_pred
    arm_summaries = {}
    for arm in ("visual_lejepa", "long_horizon_pred"):
        arm_cells = [v for k, v in cells_out.items() if k.startswith(arm)]
        arm_summaries[arm] = {}
        for mname in methods:
            ms = [c["methods"][mname] for c in arm_cells]
            m23_fast = [m["modes_2_3_fast_r2_mean"] for m in ms if m["modes_2_3_fast_r2_mean"] is not None]
            m23_quad = [m["modes_2_3_quad_r2_mean"] for m in ms if m["modes_2_3_quad_r2_mean"] is not None]
            arm_summaries[arm][mname] = {
                "slow_mean_r2": float(np.mean([m["slow_mean_r2"] for m in ms])),
                "slow_min_r2": float(np.mean([m["slow_min_r2"] for m in ms])),
                "slow_sin_theta_max": float(np.mean([m["slow_principal_sin_max"] for m in ms])),
                "modes_2_3_fast_r2_mean": float(np.mean(m23_fast)) if m23_fast else None,
                "modes_2_3_quad_r2_mean": float(np.mean(m23_quad)) if m23_quad else None,
                "best_fast_mode_r2": float(np.mean([max(x["r2_fast_linear"] for x in m["modes"]) for m in ms])),
                "best_fast_mode_idxs": [m["fast_top_mode_idx"] for m in ms],
            }
            a = arm_summaries[arm][mname]
            print(
                f"  {arm:17s} | {mname:10s} | slow_R2={a['slow_mean_r2']:.4f} | sinT={a['slow_sin_theta_max']:.4f} "
                f"| u23_fast={a['modes_2_3_fast_r2_mean'] or 0:.4f} | u23_quad={a['modes_2_3_quad_r2_mean'] or 0:.4f} "
                f"| best_fast_idx={a['best_fast_mode_idxs']} ({a['best_fast_mode_r2']:.4f})",
                flush=True,
            )

    return {
        "cells": cells_out,
        "arm_summaries": arm_summaries,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def write_markdown(summary: dict):
    s1 = summary["subexp1_lowdim_b2"]
    lines = [
        "# B1-2 (P3-A): Unsupervised Lag Unmixing (`SOBI` & `eSOBI` vs. Single-Lag SFA)",
        "",
        f"JAD verified (Cardoso–Souloumiac Givens rotations, `tol=1e-9`). Recomputed B2 baselines match sealed `SUMMARY.json` with max diff `{s1['max_sealed_baseline_diff']:.2e}`.",
        "",
        "## Sub-Experiment 1: Low-Dim `B2` Exact Head-to-Head (`CAL` = 1,024 trajectories, `TEST` = 4,096 windows x 8 encoders)",
        "",
        "| Process | Arm | TEST Accuracy | Gain vs `lag1` | Gain vs `multi` | Boundary Axes `corr²` | Off-Diag Loading RMS | Bonf. LB vs `lag1` | Bonf. LB vs `multi` |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for proc_name, pdata in s1["process_summaries"].items():
        contrasts = pdata["contrasts"]
        for arm, stats in pdata["arms"].items():
            lb1 = contrasts["vs_lag1"][arm]["one_sided_bonf_lower_95"] * 100 if arm in contrasts["vs_lag1"] else None
            lbm = contrasts["vs_multi"][arm]["one_sided_bonf_lower_95"] * 100 if arm in contrasts["vs_multi"] else None
            s_lb1 = f"{lb1:+.2f}pp" if lb1 is not None else "—"
            s_lbm = f"{lbm:+.2f}pp" if lbm is not None else "—"
            lines.append(
                f"| `{proc_name}` | `{arm}` | {stats['accuracy']*100:.2f}% | "
                f"{stats['gain_vs_lag1']*100:+.2f}pp | {stats['gain_vs_multi']*100:+.2f}pp | "
                f"{stats['boundary_corr2_mean']:.4f} | {stats['off_diag_loading_rms_23']:.4f} | "
                f"{s_lb1} | {s_lbm} |"
            )

    s2 = summary["subexp2_visual32_unsupervised"]
    lines += [
        "",
        "## Sub-Experiment 2: `32x32` Frozen 8-D Visual Representations (`latent_dim = 8 > 4`, zero labels on `CAL`)",
        "",
        "| Model Arm | Method | Slow Subspace `R²` (`z0,z1`) | Principal Angle `sin Θ_max` | Modes `(2,3)` Fast `R²` | Modes `(2,3)` Slow-Quad `R²` | Peak Fast-Axis Mode Index (`R²`) |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for arm, methods in s2["arm_summaries"].items():
        for mname, a in methods.items():
            m23f = f"{a['modes_2_3_fast_r2_mean']:.4f}" if a["modes_2_3_fast_r2_mean"] is not None else "n/a"
            m23q = f"{a['modes_2_3_quad_r2_mean']:.4f}" if a["modes_2_3_quad_r2_mean"] is not None else "n/a"
            lines.append(
                f"| `{arm}` | `{mname}` | {a['slow_mean_r2']:.4f} | {a['slow_sin_theta_max']:.4f} | "
                f"{m23f} | {m23q} | `{a['best_fast_mode_idxs']}` ({a['best_fast_mode_r2']:.4f}) |"
            )
    lines.append("")
    (OUT / "RESULTS.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    verify_jad_unit_test()
    summary = {
        "experiment": "b1_2_p3a_sobi_esobi_unmixing",
        "subexp1_lowdim_b2": run_subexp1_lowdim_b2(),
        "subexp2_visual32_unsupervised": run_subexp2_visual32_unsupervised(),
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_markdown(summary)
    print(f"Wrote {OUT / 'SUMMARY.json'} and {OUT / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
