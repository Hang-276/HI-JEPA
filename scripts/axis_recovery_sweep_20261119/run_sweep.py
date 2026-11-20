#!/usr/bin/env python3
"""Controlled Axis-Recovery (R^2) Sweep vs. Slow-Factor Selection Accuracy.

Directly tests the question:
  "Does more accurate axis recovery (higher R^2) make slow-factor selection more accurate,
   and where does single-frame R^2 hit a plateau dominated by residual temporal autocorrelation?"

Two complementary paper-grade experiments (zero encoder retraining, zero mutation of sealed assets):
  Part 1 (Low-Dim AR(1), 8 frozen A4 LeJEPA encoders x 2 processes = 16 cells):
    (1a) Real neural MLP readout sweep across 11 nested fitting budgets n_train in [12..8192].
    (1b) Matched-R^2 counterfactual controls on identical TEST/CAL/REF trajectories:
         isotropic white noise (lag-1 = 0.0) vs. temporally persistent AR(1) noise (lag-1 = 0.65).
  Part 2 (32x32 Visual Two-Sprite, 3 frozen visual_lejepa encoders from formal_v1):
    Real neural MLP readout sweep across 9 nested fitting budgets n_train in [16..8192],
    reporting regular vs. local-flip accuracy, residual autocorrelation, and own_cal selective risk.
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
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/axis_recovery_sweep_20261119"
METRICS_PATH = ROOT / "src/slowrepr/metrics.py"
A4_PANEL_PATH = ROOT / "reports/lejepa_a4_20261115/panel/run.py"
A4_FORECAST_PATH = ROOT / "reports/lejepa_a4_20261115/forecast.py"
VIS_DIR = ROOT / "results/visual_sprites_lejepa/formal_v1"
VIS_SCRIPTS = ROOT / "scripts/visual_sprites_lejepa"


def load_module(name: str, path: Path):
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


def ar1_trajectories(rng: np.random.Generator, n_ep: int, length: int, rhos: list[float]) -> np.ndarray:
    rhos_arr = np.asarray(rhos, dtype=np.float64)
    scale = np.sqrt(1.0 - rhos_arr ** 2)
    z = np.empty((n_ep, length, len(rhos_arr)), dtype=np.float64)
    z[:, 0] = rng.normal(size=(n_ep, len(rhos_arr)))
    eps = rng.normal(size=(n_ep, length - 1, len(rhos_arr)))
    for t in range(1, length):
        z[:, t] = rhos_arr * z[:, t - 1] + scale * eps[:, t - 1]
    return z


def residual_lag1_autocorr(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Compute per-axis lag-1 autocorrelation of residuals (pred - truth) on [N, T, 4]."""
    res = pred - truth
    r0 = res[:, :-1, :].reshape(-1, 4)
    r1 = res[:, 1:, :].reshape(-1, 4)
    r0 = r0 - r0.mean(axis=0, keepdims=True)
    r1 = r1 - r1.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.mean(r0 ** 2, axis=0) * np.mean(r1 ** 2, axis=0))
    return np.where(denom > 1e-12, np.mean(r0 * r1, axis=0) / denom, 0.0)


def eval_scores_summary(metrics, pred_test, truth_test, pred_cal, phys_ref_var, pred_ref_var, lags, alpha):
    cal_sc = metrics.score_windows(pred_cal, pred_cal, pred_ref_var, pred_ref_var, lags)
    thresh = float(np.quantile(cal_sc["gap"], 0.5))
    sc = metrics.score_windows(pred_test, truth_test, phys_ref_var, pred_ref_var, lags)
    rk = metrics.risk(sc["gap"], sc["wrong"], thresh, alpha, 0.35, 0.01)

    tt2 = sc["true_top2"]
    pt2 = sc["predicted_top2"]
    acc = sc["gap"] >= thresh
    wrong = sc["wrong"]
    tg = sc["true_gap"]

    kpop_ok = (pt2[:, 0] == 0) & (pt2[:, 1] == 1)
    reg_mask = (tt2[:, 0] == 0) & (tt2[:, 1] == 1)
    flp_mask = ~reg_mask

    iz05 = acc & (tg >= 0.05)
    iz10 = acc & (tg >= 0.10)
    res_ac = residual_lag1_autocorr(pred_test, truth_test)

    return {
        "threshold": thresh,
        "all_kw_error": float(wrong.mean()),
        "all_kpop_error": float((~kpop_ok).mean()),
        "own_cal_coverage": float(rk["coverage"]),
        "own_cal_accepted": int(rk["accepted"]),
        "own_cal_errors": int(rk["errors"]),
        "own_cal_kw_error": float(rk["selective_error"]) if rk["selective_error"] is not None else 1.0,
        "own_cal_kpop_error": float((~kpop_ok[acc]).mean()) if acc.any() else 1.0,
        "own_cal_cp_upper": float(rk["error_upper"]),
        "own_cal_passes": bool(rk["passes"]),
        "iz05_errors": int((wrong & iz05).sum()),
        "iz05_accepted": int(iz05.sum()),
        "iz10_errors": int((wrong & iz10).sum()),
        "iz10_accepted": int(iz10.sum()),
        "reg_windows": int(reg_mask.sum()),
        "reg_all_errors": int((wrong & reg_mask).sum()),
        "reg_own_errors": int((wrong & acc & reg_mask).sum()),
        "reg_own_accepted": int((acc & reg_mask).sum()),
        "flp_windows": int(flp_mask.sum()),
        "flp_all_acc": float(1.0 - wrong[flp_mask].mean()) if flp_mask.any() else None,
        "flp_own_errors": int((wrong & acc & flp_mask).sum()),
        "flp_own_accepted": int((acc & flp_mask).sum()),
        "residual_lag1_autocorr_mean": float(res_ac.mean()),
        "residual_lag1_autocorr_per_axis": res_ac.tolist(),
    }


def run_part1_lowdim(metrics) -> dict:
    t0 = time.perf_counter()
    print("=== Part 1: Low-Dim Frozen Real LeJEPA Axis-Recovery Sweep ===", flush=True)
    torch.set_num_threads(4)
    panel = load_module("sweep_a4_panel", A4_PANEL_PATH)
    a4_cfg, enc_cfg, probe_cfg = panel.configs()
    lags = (4, 16, 32)
    length = 193
    processes = {
        "separated_wide": [0.93, 0.918, 0.892, 0.88],
        "separated_narrow": [0.93, 0.906, 0.904, 0.88],
    }
    n_ref = 32768
    n_cal = 1024
    n_test = 2048
    n_audit = 16384
    budgets = [12, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 8192]
    alpha_bonf = 0.05 / 16.0
    unit_var = np.ones(4, dtype=np.float64)

    # Fresh independent evaluation streams shared across budgets per process
    rng_eval = np.random.default_rng(2026111901)
    ref_z = rng_eval.normal(size=(n_ref, 4))
    eval_data = {}
    for proc_name, rhos in processes.items():
        cal_z = ar1_trajectories(rng_eval, n_cal, length, rhos)
        test_z = ar1_trajectories(rng_eval, n_test, length, rhos)
        eval_data[proc_name] = (cal_z, test_z)

    names = panel.panel_names(a4_cfg)
    old_receipt = panel.read(A4_PANEL_PATH.parent / "RECEIPT.json")
    enc_hashes = {}

    # Pre-encode all static & trajectory streams once per frozen encoder
    enc_cache = {}
    for name in names:
        member = old_receipt["members"][name]
        case, seed = member["case"], member["seed"]
        ckpt_path = A4_PANEL_PATH.parent / "runs" / f"{name}.pt"
        enc_hashes[name] = sha256_file(ckpt_path)
        encoder = panel.encoder_source.make_mlp_encoder(4, enc_cfg["hidden"], enc_cfg["layers"], device="cpu").eval()
        encoder.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu"))

        train_full = panel.data(a4_cfg, case, seed, 1, probe_cfg["n_calibration"])
        val_z = panel.data(a4_cfg, case, seed, 2, probe_cfg["n_validation"])
        audit_z = panel.data(a4_cfg, case, seed, 721, n_audit)

        h_train = panel.encoder_source.embed(encoder, train_full)
        h_val = panel.encoder_source.embed(encoder, val_z)
        h_audit = panel.encoder_source.embed(encoder, audit_z)
        h_ref = panel.encoder_source.embed(encoder, ref_z)

        h_procs = {}
        for proc_name, (cal_z, test_z) in eval_data.items():
            h_cal = panel.encoder_source.embed(encoder, cal_z.reshape(-1, 4)).reshape(n_cal, length, 4)
            h_test = panel.encoder_source.embed(encoder, test_z.reshape(-1, 4)).reshape(n_test, length, 4)
            h_procs[proc_name] = (h_cal, h_test)

        enc_cache[name] = {
            "case": case,
            "seed": seed,
            "probe_master": member["probe_master"],
            "train_z": train_full,
            "val_z": val_z,
            "audit_z": audit_z,
            "h_train": h_train,
            "h_val": h_val,
            "h_audit": h_audit,
            "h_ref": h_ref,
            "h_procs": h_procs,
        }
    print(f"Encoded all {len(names)} frozen A4 encoders in {time.perf_counter() - t0:.1f}s", flush=True)

    # 1A. Real MLP readout sweep across nested training budgets
    budget_summaries = []
    for n_tr in budgets:
        tb = time.perf_counter()
        cell_rows = []
        for name in names:
            c = enc_cache[name]
            h_tr = c["h_train"][:n_tr]
            y_tr = c["train_z"][:n_tr]
            mean, scale = panel.probe_source.standardizer(h_tr)
            probe, _ = panel.probe_source.fit_mlp(
                (h_tr - mean) / scale, y_tr,
                (c["h_val"] - mean) / scale, c["val_z"],
                {**probe_cfg, "master": c["probe_master"]},
            )
            pred_audit = panel.probe_source.predict_mlp((c["h_audit"] - mean) / scale, probe)
            aud_r2 = panel.probe_source.r2(c["audit_z"], pred_audit)
            pred_ref = panel.probe_source.predict_mlp((c["h_ref"] - mean) / scale, probe)
            pred_ref_var = pred_ref.var(axis=0)

            for proc_name, (cal_z, test_z) in eval_data.items():
                h_cal, h_test = c["h_procs"][proc_name]
                pred_cal = panel.probe_source.predict_mlp(
                    ((h_cal.reshape(-1, 4) - mean) / scale), probe
                ).reshape(n_cal, length, 4)
                pred_test = panel.probe_source.predict_mlp(
                    ((h_test.reshape(-1, 4) - mean) / scale), probe
                ).reshape(n_test, length, 4)
                ev = eval_scores_summary(
                    metrics, pred_test, test_z, pred_cal, unit_var, pred_ref_var, lags, alpha_bonf
                )
                cell_rows.append({
                    "encoder": name,
                    "process": proc_name,
                    "n_train": n_tr,
                    "audit_r2_per_axis": aud_r2.tolist(),
                    "audit_min_r2": float(aud_r2.min()),
                    "audit_mean_r2": float(aud_r2.mean()),
                    "eligible_085": bool(aud_r2.min() >= 0.85),
                    **ev,
                })

        # Aggregate across all 16 cells and by process
        def agg(rows):
            return {
                "cells": len(rows),
                "audit_min_r2_mean": float(np.mean([r["audit_min_r2"] for r in rows])),
                "audit_min_r2_min": float(np.min([r["audit_min_r2"] for r in rows])),
                "audit_mean_r2": float(np.mean([r["audit_mean_r2"] for r in rows])),
                "eligible_085_count": int(sum(r["eligible_085"] for r in rows)),
                "residual_lag1_autocorr": float(np.mean([r["residual_lag1_autocorr_mean"] for r in rows])),
                "all_kw_error": float(np.mean([r["all_kw_error"] for r in rows])),
                "all_kpop_error": float(np.mean([r["all_kpop_error"] for r in rows])),
                "own_cal_coverage": float(np.mean([r["own_cal_coverage"] for r in rows])),
                "own_cal_kw_error": float(np.mean([r["own_cal_kw_error"] for r in rows])),
                "own_cal_kpop_error": float(np.mean([r["own_cal_kpop_error"] for r in rows])),
                "own_cal_worst_cp_upper": float(np.max([r["own_cal_cp_upper"] for r in rows])),
                "own_cal_pass_count": int(sum(r["own_cal_passes"] for r in rows)),
                "iz05_kw_error": float(
                    sum(r["iz05_errors"] for r in rows) / max(1, sum(r["iz05_accepted"] for r in rows))
                ),
                "iz10_kw_error": float(
                    sum(r["iz10_errors"] for r in rows) / max(1, sum(r["iz10_accepted"] for r in rows))
                ),
            }

        summary = {
            "n_train": n_tr,
            "all_16": agg(cell_rows),
            "wide_8": agg([r for r in cell_rows if r["process"] == "separated_wide"]),
            "narrow_8": agg([r for r in cell_rows if r["process"] == "separated_narrow"]),
            "cells": cell_rows,
        }
        budget_summaries.append(summary)
        a = summary["all_16"]
        print(
            f"  n_train={n_tr:4d} ({time.perf_counter()-tb:4.1f}s) | minR2={a['audit_min_r2_mean']:.4f} "
            f"| resAC1={a['residual_lag1_autocorr']:.3f} | all_KW={a['all_kw_error']*100:5.2f}% "
            f"| own_KW={a['own_cal_kw_error']*100:5.2f}% | iz05={a['iz05_kw_error']*100:5.2f}% "
            f"| worstCP={a['own_cal_worst_cp_upper']*100:5.2f}% | pass={a['own_cal_pass_count']:2d}/16",
            flush=True,
        )

    # 1B. Matched-R^2 Counterfactual Control (white noise vs. AR(1) persistent residual lag-1=0.65)
    print("\n--- Part 1B: Matched-R^2 Counterfactual Control (White vs. Persistent AR(1) lag-1=0.65) ---", flush=True)
    target_r2s = [0.60, 0.75, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995]
    cf_rows = []
    rng_cf = np.random.default_rng(2026111902)
    for r2_target in target_r2s:
        noise_std = float(np.sqrt(1.0 - r2_target))
        for regime, res_rho in (("white_ac0", 0.0), ("ar1_ac065", 0.65)):
            by_proc = {}
            for proc_name, (cal_z, test_z) in eval_data.items():
                e_ref = rng_cf.normal(size=(n_ref, 4))
                e_cal = ar1_trajectories(rng_cf, n_cal, length, [res_rho] * 4)
                e_test = ar1_trajectories(rng_cf, n_test, length, [res_rho] * 4)
                pred_ref = ref_z + noise_std * e_ref
                pred_cal = cal_z + noise_std * e_cal
                pred_test = test_z + noise_std * e_test
                pred_ref_var = pred_ref.var(axis=0)
                ev = eval_scores_summary(
                    metrics, pred_test, test_z, pred_cal, unit_var, pred_ref_var, lags, alpha_bonf
                )
                by_proc[proc_name] = ev
            row = {
                "target_r2": r2_target,
                "regime": regime,
                "res_rho": res_rho,
                "wide": by_proc["separated_wide"],
                "narrow": by_proc["separated_narrow"],
                "mean_all_kw_error": 0.5 * (by_proc["separated_wide"]["all_kw_error"] + by_proc["separated_narrow"]["all_kw_error"]),
                "mean_own_kw_error": 0.5 * (by_proc["separated_wide"]["own_cal_kw_error"] + by_proc["separated_narrow"]["own_cal_kw_error"]),
            }
            cf_rows.append(row)
        w_row = cf_rows[-2]
        p_row = cf_rows[-1]
        print(
            f"  R2={r2_target:.3f} | white: all_KW={w_row['mean_all_kw_error']*100:5.2f}%, own_KW={w_row['mean_own_kw_error']*100:5.2f}% "
            f"| ar1(0.65): all_KW={p_row['mean_all_kw_error']*100:5.2f}%, own_KW={p_row['mean_own_kw_error']*100:5.2f}%",
            flush=True,
        )

    return {
        "encoder_sha256": enc_hashes,
        "budgets": budgets,
        "real_readout_sweep": budget_summaries,
        "counterfactual_matched_r2": cf_rows,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def run_part2_visual32(metrics) -> dict:
    t0 = time.perf_counter()
    print("\n=== Part 2: 32x32 Visual Frozen LeJEPA Axis-Recovery Sweep (3 seeds) ===", flush=True)
    lock = json.loads((VIS_DIR / "DESIGN_LOCK.json").read_text())
    cfg = lock["config"]
    lags = tuple(cfg["lags"])
    seeds = list(cfg["seeds"])
    budgets = [16, 32, 64, 128, 256, 512, 1024, 2048, 8192]
    alpha_bonf = 0.05 / len(seeds)

    render = load_module("vis_sweep_render", VIS_SCRIPTS / "render.py")
    models = load_module("vis_sweep_models", VIS_SCRIPTS / "models.py")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device={device} for frozen 32x32 visual encoding", flush=True)

    encoders = {}
    enc_hashes = {}
    for seed in seeds:
        name = f"visual_lejepa_seed{seed}"
        ckpt = VIS_DIR / name / "encoder.pt"
        enc_hashes[name] = sha256_file(ckpt)
        enc = models.make_encoder(cfg["encoder"]["latent_dim"], device)
        enc.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        enc.eval()
        encoders[name] = enc

    def extract_role(role: str):
        truth = render.sample_states(cfg, role, development=False)
        truth = render.sample_frames(truth, cfg, role, development=False)
        batch_sz = cfg["runtime"].get("render_batch_size", cfg["runtime"]["encode_batch_size"])
        feats = {name: [] for name in encoders}
        for imgs in render.iter_role_images(cfg, role, batch_sz, development=False):
            for name, enc in encoders.items():
                feats[name].append(models.encode(enc, imgs, device, cfg["runtime"]["encode_batch_size"]))
        return {name: np.concatenate(v, axis=0) for name, v in feats.items()}, truth

    role_feats = {}
    role_truth = {}
    for role in ("train", "val", "audit", "ref", "cal", "test"):
        tr = time.perf_counter()
        role_feats[role], role_truth[role] = extract_role(role)
        print(f"  Encoded role={role:5s} shape={role_truth[role].shape} ({time.perf_counter()-tr:.1f}s)", flush=True)

    phys_ref_var = role_truth["ref"].reshape(-1, 4).var(axis=0, ddof=0)
    audit_y = role_truth["audit"].reshape(-1, 4)
    audit_var = audit_y.var(axis=0, ddof=0)
    train_y_full = role_truth["train"].reshape(-1, 4)
    val_y = role_truth["val"].reshape(-1, 4)

    # Probe fitting on CPU is faster for tiny 8->128->128->4 MLPs than MPS dispatch overhead
    probe_dev = torch.device("cpu")
    torch.set_num_threads(4)

    budget_summaries = []
    for n_tr in budgets:
        tb = time.perf_counter()
        seed_rows = []
        for seed in seeds:
            name = f"visual_lejepa_seed{seed}"
            # Nested random permutation fixed by seed so subset spans all episodes cleanly
            perm = np.random.default_rng(2026111900 + seed).permutation(len(train_y_full))[:n_tr]
            probe, _ = models.fit_probe(
                role_feats["train"][name][perm], train_y_full[perm],
                role_feats["val"][name], val_y,
                cfg, seed, probe_dev,
            )
            pred_audit = models.predict(probe, role_feats["audit"][name], probe_dev)
            aud_r2 = 1.0 - np.mean((pred_audit - audit_y) ** 2, axis=0) / audit_var

            pred_ref = models.predict(probe, role_feats["ref"][name], probe_dev)
            pred_ref_var = pred_ref.var(axis=0, ddof=0)

            cal_shape = role_truth["cal"].shape
            test_shape = role_truth["test"].shape
            pred_cal = models.predict(probe, role_feats["cal"][name], probe_dev).reshape(cal_shape)
            pred_test = models.predict(probe, role_feats["test"][name], probe_dev).reshape(test_shape)

            ev = eval_scores_summary(
                metrics, pred_test, role_truth["test"], pred_cal, phys_ref_var, pred_ref_var, lags, alpha_bonf
            )
            seed_rows.append({
                "cell": name,
                "seed": seed,
                "n_train": n_tr,
                "audit_r2_per_axis": aud_r2.tolist(),
                "audit_min_r2": float(aud_r2.min()),
                "eligible_085": bool(aud_r2.min() >= 0.85),
                **ev,
            })

        summary = {
            "n_train": n_tr,
            "audit_min_r2_mean": float(np.mean([r["audit_min_r2"] for r in seed_rows])),
            "audit_min_r2_min": float(np.min([r["audit_min_r2"] for r in seed_rows])),
            "residual_lag1_autocorr": float(np.mean([r["residual_lag1_autocorr_mean"] for r in seed_rows])),
            "all_kw_error": float(np.mean([r["all_kw_error"] for r in seed_rows])),
            "reg_all_error": float(np.mean([r["reg_all_errors"] / r["reg_windows"] for r in seed_rows])),
            "reg_own_error": float(np.mean([r["reg_own_errors"] / max(1, r["reg_own_accepted"]) for r in seed_rows])),
            "flp_all_acc": float(np.mean([r["flp_all_acc"] for r in seed_rows])),
            "flp_own_errors_sum": int(sum(r["flp_own_errors"] for r in seed_rows)),
            "flp_own_accepted_sum": int(sum(r["flp_own_accepted"] for r in seed_rows)),
            "own_cal_coverage": float(np.mean([r["own_cal_coverage"] for r in seed_rows])),
            "own_cal_kw_error": float(np.mean([r["own_cal_kw_error"] for r in seed_rows])),
            "own_cal_worst_cp_upper": float(np.max([r["own_cal_cp_upper"] for r in seed_rows])),
            "own_cal_pass_count": int(sum(r["own_cal_passes"] for r in seed_rows)),
            "seeds": seed_rows,
        }
        budget_summaries.append(summary)
        print(
            f"  n_train={n_tr:4d} ({time.perf_counter()-tb:4.1f}s) | minR2={summary['audit_min_r2_mean']:.4f} "
            f"| resAC1={summary['residual_lag1_autocorr']:.3f} | reg_all={summary['reg_all_error']*100:5.2f}% "
            f"| flp_acc={summary['flp_all_acc']*100:5.1f}% | own_KW={summary['own_cal_kw_error']*100:5.2f}% "
            f"| worstCP={summary['own_cal_worst_cp_upper']*100:5.2f}% | pass={summary['own_cal_pass_count']}/3",
            flush=True,
        )

    return {
        "encoder_sha256": enc_hashes,
        "budgets": budgets,
        "visual_sweep": budget_summaries,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def write_markdown_report(part1: dict, part2: dict, out_path: Path):
    lines = [
        "# Controlled Axis-Recovery ($R^2$) Sweep vs. Slow-Factor Selection Accuracy",
        "",
        "Date: 2026-11-19. Zero encoder retraining; zero modification of sealed assets.",
        "",
        "## 1. Low-Dim Frozen Real LeJEPA (8 Encoders × 2 Processes = 16 Cells)",
        "",
        "| n_train | Mean min AUDIT $R^2$ | Res lag-1 AC | All-Answer $K_W$ Err | All-Answer $K_{\\mathrm{pop}}$ Err | own_cal Cov | own_cal $K_W$ Err | IZ ($\\Delta\\ge 0.05$) Err | Worst CP Upper (Bonf-16) | PASS / 16 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in part1["real_readout_sweep"]:
        a = s["all_16"]
        lines.append(
            f"| {s['n_train']} | {a['audit_min_r2_mean']:.4f} | {a['residual_lag1_autocorr']:.3f} | "
            f"{a['all_kw_error']*100:.2f}% | {a['all_kpop_error']*100:.2f}% | "
            f"{a['own_cal_coverage']*100:.1f}% | {a['own_cal_kw_error']*100:.2f}% | "
            f"{a['iz05_kw_error']*100:.2f}% | {a['own_cal_worst_cp_upper']*100:.2f}% | {a['own_cal_pass_count']}/16 |"
        )

    lines += [
        "",
        "### Matched-$R^2$ Counterfactual Control (Isotropic White Noise vs. Persistent AR(1) Residual lag-1=0.65)",
        "",
        "| Target $R^2$ | White All $K_W$ Err | AR(1) 0.65 All $K_W$ Err | White own_cal $K_W$ Err | AR(1) 0.65 own_cal $K_W$ Err | Ratio (AR1 / White own_cal) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    cf = part1["counterfactual_matched_r2"]
    for i in range(0, len(cf), 2):
        w, p = cf[i], cf[i + 1]
        ratio = (p["mean_own_kw_error"] / w["mean_own_kw_error"]) if w["mean_own_kw_error"] > 1e-9 else float("inf")
        ratio_s = f"{ratio:.2f}x" if np.isfinite(ratio) else "inf"
        lines.append(
            f"| {w['target_r2']:.3f} | {w['mean_all_kw_error']*100:.2f}% | {p['mean_all_kw_error']*100:.2f}% | "
            f"{w['mean_own_kw_error']*100:.2f}% | {p['mean_own_kw_error']*100:.2f}% | {ratio_s} |"
        )

    lines += [
        "",
        "## 2. 32×32 Visual Frozen LeJEPA (3 Seeds from formal_v1, 2,048 TEST Episodes)",
        "",
        "| n_train | Mean min AUDIT $R^2$ | Res lag-1 AC | Regular All Err | Flip All Acc | own_cal Cov | own_cal $K_W$ Err | Flip own_cal Err/Acc | Worst CP Upper (Bonf-3) | PASS / 3 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in part2["visual_sweep"]:
        lines.append(
            f"| {s['n_train']} | {s['audit_min_r2_mean']:.4f} | {s['residual_lag1_autocorr']:.3f} | "
            f"{s['reg_all_error']*100:.2f}% | {s['flp_all_acc']*100:.1f}% | "
            f"{s['own_cal_coverage']*100:.1f}% | {s['own_cal_kw_error']*100:.2f}% | "
            f"{s['flp_own_errors_sum']}/{s['flp_own_accepted_sum']} | {s['own_cal_worst_cp_upper']*100:.2f}% | {s['own_cal_pass_count']}/3 |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = load_module("sweep_metrics", METRICS_PATH)
    t_all = time.perf_counter()
    part1 = run_part1_lowdim(metrics)
    part2 = run_part2_visual32(metrics)
    payload = {
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script_sha256": sha256_file(Path(__file__)),
        "metrics_sha256": sha256_file(METRICS_PATH),
        "total_elapsed_seconds": time.perf_counter() - t_all,
        "part1_lowdim": part1,
        "part2_visual32": part2,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    write_markdown_report(part1, part2, OUT / "RESULTS.md")
    print(f"\nDONE in {payload['total_elapsed_seconds']:.1f}s -> {OUT / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
