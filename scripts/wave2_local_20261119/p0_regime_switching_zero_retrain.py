#!/usr/bin/env python3
"""B1-1 (P0-ZeroRetrain): Zero-retrain regime-switching evaluation on frozen models.

Evaluates whether frozen encoders and static per-frame probes dynamically track
true episode-level slow-factor role switches (80% Mode A <-> 20% Mode B) with
identical N(0, I_4) single-frame marginals, achieving high own_cal coverage and
near-zero selective error on Mode B rather than regressing to a constant {0, 1}
prior.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import beta as beta_dist
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/wave2_local_20261119/p0_regime_switching"
METRICS_PATH = ROOT / "src/slowrepr/metrics.py"
VIS_DIR = ROOT / "results/visual_sprites_lejepa/formal_v1"
VIS_SCRIPTS = ROOT / "scripts/visual_sprites_lejepa"
A4_PANEL_PATH = ROOT / "reports/lejepa_a4_20261115/panel/run.py"
A2_WEAK_PATH = ROOT / "reports/lejepa_a2_readout_20261115/weak/run.py"
A2_RECEIPT_PATH = ROOT / "reports/lejepa_a2_readout_20261115/weak/RECEIPT.json"

LAGS = (4, 16, 32)
LENGTH = 193
MIN_COV = 0.35
MAX_ERR = 0.01
AUDIT_GATE = 0.85


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mixture_ar1(rng: np.random.Generator, n_ep: int, length: int,
                rho_a: list[float], rho_b: list[float], p_b: float = 0.20):
    """Draw stationary AR(1) episodes switching between Mode A (0) and Mode B (1)."""
    modes = (rng.random(n_ep) < float(p_b)).astype(np.int8)
    rho_a_arr = np.asarray(rho_a, dtype=np.float64)
    rho_b_arr = np.asarray(rho_b, dtype=np.float64)
    rho = np.where(modes[:, None] == 1, rho_b_arr[None, :], rho_a_arr[None, :])
    scale = np.sqrt(1.0 - rho ** 2)
    z = np.empty((n_ep, length, 4), dtype=np.float64)
    z[:, 0] = rng.normal(size=(n_ep, 4))
    eps = rng.normal(size=(n_ep, length - 1, 4))
    for t in range(1, length):
        z[:, t] = rho * z[:, t - 1] + scale * eps[:, t - 1]
    return z, modes


def cp_upper(errors: int, accepted: int, alpha: float) -> float | None:
    if accepted <= 0:
        return None
    if errors == accepted:
        return 1.0
    return float(beta_dist.ppf(1.0 - alpha, errors + 1, accepted - errors))


def stratum_metrics(scores: dict, threshold: float, mask: np.ndarray, alpha: float) -> dict:
    count = int(mask.sum())
    if count == 0:
        return {
            "windows": 0, "all_answer_error": None, "accepted": 0, "coverage": 0.0,
            "errors": 0, "selective_error": None, "cp_upper": None,
            "constant_01_error": None, "pred_01_rate_accepted": None,
            "pred_23_rate_accepted": None, "pred_23_rate_all": None,
        }
    accept = (scores["gap"] >= threshold) & mask
    accepted = int(accept.sum())
    wrong = scores["wrong"]
    errors = int((wrong & accept).sum())
    pt2 = scores["predicted_top2"]
    tt2 = scores["true_top2"]
    is_01 = (pt2[:, 0] == 0) & (pt2[:, 1] == 1)
    is_23 = (pt2[:, 0] == 2) & (pt2[:, 1] == 3)
    const01_wrong = (tt2[:, 0] != 0) | (tt2[:, 1] != 1)
    return {
        "windows": count,
        "all_answer_error": float(wrong[mask].mean()),
        "accepted": accepted,
        "coverage": float(accepted / count),
        "errors": errors,
        "selective_error": float(errors / accepted) if accepted > 0 else None,
        "cp_upper": cp_upper(errors, accepted, alpha),
        "constant_01_error": float(const01_wrong[mask].mean()),
        "pred_01_rate_accepted": float(is_01[accept].mean()) if accepted > 0 else None,
        "pred_23_rate_accepted": float(is_23[accept].mean()) if accepted > 0 else None,
        "pred_23_rate_all": float(is_23[mask].mean()),
    }


def evaluate_cell(metrics, pred_ref: np.ndarray, true_ref: np.ndarray,
                  pred_cal: np.ndarray, pred_test: np.ndarray, true_test: np.ndarray,
                  test_modes: np.ndarray, alpha: float) -> dict:
    phys_var = true_ref.reshape(-1, 4).var(axis=0, ddof=0)
    pred_var = pred_ref.reshape(-1, 4).var(axis=0, ddof=0)
    if np.any(phys_var <= 0) or np.any(pred_var <= 0) or not np.isfinite(pred_var).all():
        raise ValueError("Invalid reference variances")

    # Stage (a/b): freeze CAL threshold without touching TEST labels.
    cal_sc = metrics.score_windows(pred_cal, pred_cal, pred_var, pred_var, LAGS)
    threshold = float(np.quantile(cal_sc["gap"], 0.5))

    # Stage (c): score TEST.
    sc = metrics.score_windows(pred_test, true_test, phys_var, pred_var, LAGS)
    rk = metrics.risk(sc["gap"], sc["wrong"], threshold, alpha, MIN_COV, MAX_ERR)
    test_r2 = 1.0 - np.mean((pred_test - true_test) ** 2, axis=(0, 1)) / true_test.reshape(-1, 4).var(axis=0, ddof=0)

    mode_a = test_modes == 0
    mode_b = test_modes == 1
    tg = sc["true_gap"]
    strata = {
        "overall": stratum_metrics(sc, threshold, np.ones_like(mode_a, dtype=bool), alpha),
        "mode_a": stratum_metrics(sc, threshold, mode_a, alpha),
        "mode_b": stratum_metrics(sc, threshold, mode_b, alpha),
        "indifference_ge_0.25": stratum_metrics(sc, threshold, tg >= 0.25, alpha),
        "indifference_mode_b_ge_0.25": stratum_metrics(sc, threshold, (tg >= 0.25) & mode_b, alpha),
        "mode_b_true_gap": {
            "count": int(mode_b.sum()),
            "mean": float(tg[mode_b].mean()) if mode_b.any() else None,
            "median": float(np.median(tg[mode_b])) if mode_b.any() else None,
            "frac_ge_0.25": float((tg[mode_b] >= 0.25).mean()) if mode_b.any() else None,
        },
    }
    return {
        "threshold": threshold,
        "risk": rk,
        "test_r2_per_axis": test_r2.tolist(),
        "test_min_r2": float(np.min(test_r2)),
        "pred_over_phys_var": (pred_var / phys_var).tolist(),
        "strata": strata,
    }


def select_visual_device(models, torch_mod):
    if torch_mod.backends.mps.is_available():
        try:
            return models.configure_device("mps", cpu_threads=4)
        except Exception as exc:
            print(f"MPS preflight failed ({exc}); falling back to CPU", flush=True)
    return torch_mod.device("cpu")


def run_panel_a_visual32(metrics) -> dict:
    t0 = time.perf_counter()
    print("=== B1-1 Panel A: 32x32 Visual Two-Sprite Zero-Retrain Switch ===", flush=True)
    lock = json.loads((VIS_DIR / "DESIGN_LOCK.json").read_text())
    cfg = lock["config"]
    render = load_module("p0_vis_render", VIS_SCRIPTS / "render.py")
    models = load_module("p0_vis_models", VIS_SCRIPTS / "models.py")
    device = select_visual_device(models, torch)
    probe_dev = torch.device("cpu")
    torch.set_num_threads(4)
    print(f"Panel A device={device}", flush=True)

    rho_a = [0.964, 0.961, 0.78, 0.76]
    rho_b = [0.78, 0.76, 0.964, 0.961]
    cell_names = [f"visual_lejepa_seed{s}" for s in (101, 102, 103)] + [
        f"long_horizon_pred_seed{s}" for s in (101, 102, 103)
    ]
    encoders, probes, sealed_audits = {}, {}, {}
    for name in cell_names:
        cell_dir = VIS_DIR / name
        enc = models.make_encoder(cfg["encoder"]["latent_dim"], device)
        enc.load_state_dict(torch.load(cell_dir / "encoder.pt", map_location=device, weights_only=True))
        enc.eval()
        encoders[name] = enc
        probes[name] = torch.load(cell_dir / "probe.pt", map_location="cpu", weights_only=False)
        sealed_audits[name] = json.loads((cell_dir / "audit.json").read_text())

    rng = np.random.default_rng(2026111911)
    role_specs = [("ref", 256), ("cal", 1024), ("test", 2048)]
    truth_states, truth_modes, latents = {}, {}, {name: {} for name in cell_names}
    batch_sz = int(cfg["runtime"].get("render_batch_size", cfg["runtime"]["encode_batch_size"]))
    for role_idx, (role, n_ep) in enumerate(role_specs):
        tr = time.perf_counter()
        states, modes = mixture_ar1(rng, n_ep, LENGTH, rho_a, rho_b, p_b=0.20)
        pixel_rng = np.random.default_rng([2026111911, 100 + role_idx])
        flat_states = states.reshape(-1, 4)
        chunks = {name: [] for name in cell_names}
        for start in range(0, len(flat_states), batch_sz):
            imgs = render.render_frames(flat_states[start:start + batch_sz], cfg, pixel_rng)
            for name, enc in encoders.items():
                chunks[name].append(models.encode(enc, imgs, device, cfg["runtime"]["encode_batch_size"]))
        for name in cell_names:
            latents[name][role] = np.concatenate(chunks[name], axis=0)
        truth_states[role] = states
        truth_modes[role] = modes
        print(f"  Encoded mixture {role}: {states.shape}, mode_b={int(modes.sum())}/{n_ep} ({time.perf_counter()-tr:.1f}s)", flush=True)

    alpha = 0.05 / len(cell_names)
    rows = []
    for name in cell_names:
        arm = "visual_lejepa" if name.startswith("visual_lejepa") else "long_horizon_pred"
        pred = {
            role: models.predict(probes[name], latents[name][role], probe_dev).reshape(truth_states[role].shape)
            for role, _ in role_specs
        }
        ev = evaluate_cell(
            metrics,
            pred["ref"], truth_states["ref"],
            pred["cal"],
            pred["test"], truth_states["test"], truth_modes["test"],
            alpha,
        )
        aud = sealed_audits[name]
        status = (
            "RECOVERY_NOT_VERIFIED" if not aud["eligible"]
            else "PASS" if ev["risk"]["passes"] else "CRITERIA_NOT_MET"
        )
        rows.append({
            "cell": name,
            "arm": arm,
            "seed": aud["seed"],
            "audit_min_r2": float(aud["minimum_r2"]),
            "eligible": bool(aud["eligible"]),
            "status": status,
            "alpha": alpha,
            **ev,
        })
        mb = ev["strata"]["mode_b"]
        ma = ev["strata"]["mode_a"]
        ov = ev["strata"]["overall"]
        print(
            f"  {name:28s} | audit={aud['minimum_r2']:.4f} | status={status:21s} "
            f"| all_err={ov['all_answer_error']*100:5.2f}% (const01={ov['constant_01_error']*100:5.2f}%) "
            f"| ModeB: cov={mb['coverage']*100:5.1f}% acc={mb['accepted']:3d}/{mb['windows']} "
            f"err={mb['errors']:3d} pred23={(mb['pred_23_rate_accepted'] or 0.0)*100:5.1f}%",
            flush=True,
        )

    return {
        "data_seed": 2026111911,
        "rho_a": rho_a,
        "rho_b": rho_b,
        "p_b": 0.20,
        "family_size": len(cell_names),
        "alpha": alpha,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def _agg_panel_b_group(rows: list[dict]) -> dict:
    def mean_or_none(vals):
        clean = [v for v in vals if v is not None]
        return float(np.mean(clean)) if clean else None

    return {
        "cells": len(rows),
        "audit_min_r2_mean": float(np.mean([r["audit_min_r2"] for r in rows])),
        "eligible_count": int(sum(r["eligible"] for r in rows)),
        "pass_count": int(sum(r["status"] == "PASS" for r in rows)),
        "overall_all_err": float(np.mean([r["strata"]["overall"]["all_answer_error"] for r in rows])),
        "overall_const01_err": float(np.mean([r["strata"]["overall"]["constant_01_error"] for r in rows])),
        "overall_cov": float(np.mean([r["risk"]["coverage"] for r in rows])),
        "overall_sel_err": mean_or_none([r["risk"]["selective_error"] for r in rows]),
        "overall_worst_cp": float(np.max([r["risk"]["error_upper"] for r in rows])),
        "mode_a_all_err": float(np.mean([r["strata"]["mode_a"]["all_answer_error"] for r in rows])),
        "mode_a_cov": float(np.mean([r["strata"]["mode_a"]["coverage"] for r in rows])),
        "mode_a_sel_err": mean_or_none([r["strata"]["mode_a"]["selective_error"] for r in rows]),
        "mode_b_all_err": float(np.mean([r["strata"]["mode_b"]["all_answer_error"] for r in rows])),
        "mode_b_cov": float(np.mean([r["strata"]["mode_b"]["coverage"] for r in rows])),
        "mode_b_accepted_total": int(sum(r["strata"]["mode_b"]["accepted"] for r in rows)),
        "mode_b_windows_total": int(sum(r["strata"]["mode_b"]["windows"] for r in rows)),
        "mode_b_errors_total": int(sum(r["strata"]["mode_b"]["errors"] for r in rows)),
        "mode_b_sel_err": mean_or_none([r["strata"]["mode_b"]["selective_error"] for r in rows]),
        "mode_b_pred23_acc_mean": mean_or_none([r["strata"]["mode_b"]["pred_23_rate_accepted"] for r in rows]),
        "mode_b_const01_err": float(np.mean([r["strata"]["mode_b"]["constant_01_error"] for r in rows])),
    }


def run_panel_b_lowdim(metrics) -> dict:
    t0 = time.perf_counter()
    print("\n=== B1-1 Panel B: Low-Dim Frozen A4 Strong vs A2 Weak Zero-Retrain Switch ===", flush=True)
    panel = load_module("p0_a4_panel", A4_PANEL_PATH)
    weak = load_module("p0_a2_weak", A2_WEAK_PATH)
    a4_cfg, _, _ = panel.configs()
    a2_receipt = json.loads(A2_RECEIPT_PATH.read_text())["members"]
    names = panel.panel_names(a4_cfg)

    torch.set_num_threads(4)
    readouts = {
        (name, arm): (panel.load_readout(name)[0] if arm == "strong" else weak.load_readout(name)[0])
        for name in names for arm in ("strong", "weak")
    }

    processes = {
        "clear_switch": ([0.95, 0.94, 0.78, 0.76], [0.78, 0.76, 0.95, 0.94]),
        "narrow_switch": ([0.93, 0.918, 0.892, 0.88], [0.892, 0.88, 0.93, 0.918]),
    }
    rng = np.random.default_rng(2026111912)
    ref_z = rng.normal(size=(32768, 4))
    alpha = 0.05 / (len(names) * 2)

    process_results = {}
    for proc_name, (rho_a, rho_b) in processes.items():
        cal_z, _ = mixture_ar1(rng, 1024, LENGTH, rho_a, rho_b, p_b=0.20)
        test_z, test_modes = mixture_ar1(rng, 2048, LENGTH, rho_a, rho_b, p_b=0.20)
        rows = []
        for arm in ("strong", "weak"):
            for name in names:
                fn = readouts[(name, arm)]
                aud_r2 = a2_receipt[name]["audit"][f"{arm}_r2"]
                min_aud = float(min(aud_r2))
                eligible = bool(min_aud >= AUDIT_GATE)
                pred_ref = fn(ref_z)
                pred_cal = fn(cal_z)
                pred_test = fn(test_z)
                ev = evaluate_cell(
                    metrics, pred_ref, ref_z, pred_cal, pred_test, test_z, test_modes, alpha
                )
                status = (
                    "RECOVERY_NOT_VERIFIED" if not eligible
                    else "PASS" if ev["risk"]["passes"] else "CRITERIA_NOT_MET"
                )
                rows.append({
                    "encoder": name,
                    "arm": arm,
                    "audit_min_r2": min_aud,
                    "eligible": eligible,
                    "status": status,
                    "alpha": alpha,
                    **ev,
                })
        strong_agg = _agg_panel_b_group([r for r in rows if r["arm"] == "strong"])
        weak_agg = _agg_panel_b_group([r for r in rows if r["arm"] == "weak"])
        process_results[proc_name] = {
            "rho_a": rho_a,
            "rho_b": rho_b,
            "p_b": 0.20,
            "family_size": len(names) * 2,
            "alpha": alpha,
            "summary": {"strong": strong_agg, "weak": weak_agg},
            "rows": rows,
        }
        for label, g in (("strong", strong_agg), ("weak  ", weak_agg)):
            print(
                f"  {proc_name:13s} | {label} | audit={g['audit_min_r2_mean']:.4f} "
                f"| PASS={g['pass_count']}/8 | ov_sel={g['overall_sel_err']*100:5.2f}% "
                f"| ModeB: cov={g['mode_b_cov']*100:5.1f}% "
                f"err={g['mode_b_errors_total']}/{g['mode_b_accepted_total']} "
                f"(const01_ov={g['overall_const01_err']*100:5.2f}%)",
                flush=True,
            )

    return {
        "data_seed": 2026111912,
        "n_ref": 32768,
        "processes": process_results,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def write_markdown(summary: dict):
    pa = summary["panel_a_visual32"]
    lines = [
        "# B1-1 (P0-ZeroRetrain): Zero-Retrain Episode-Level Regime Switching",
        "",
        "Each split mixes **80% Mode A** (slow {0,1}) and **20% Mode B** (slow {2,3}) episodes with identical $N(0, I_4)$ single-frame marginals. Frozen encoders and static probes are reused with zero retraining; thresholds are frozen on unlabeled CAL before touching TEST.",
        "",
        "## Panel A: 32x32 Visual Two-Sprite (`formal_v1`, 6 frozen models, `M=6`, `alpha=0.05/6`)",
        "",
        f"Rho A = `{pa['rho_a']}`, Rho B = `{pa['rho_b']}`, TEST windows = `2048`.",
        "",
        "| Cell | Audit min R² | Overall all_err | Const-{0,1} err | Overall own_cal cov | Overall own_cal err (CP upper) | Mode-B all_err | **Mode-B own_cal cov** | **Mode-B err / acc** | Mode-B pred {2,3} | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in pa["rows"]:
        ov = r["strata"]["overall"]
        mb = r["strata"]["mode_b"]
        rk = r["risk"]
        p23 = f"{mb['pred_23_rate_accepted']*100:.1f}%" if mb["pred_23_rate_accepted"] is not None else "n/a"
        sel = f"{rk['selective_error']*100:.2f}%" if rk["selective_error"] is not None else "n/a"
        lines.append(
            f"| `{r['cell']}` | {r['audit_min_r2']:.4f} | {ov['all_answer_error']*100:.2f}% | "
            f"{ov['constant_01_error']*100:.2f}% | {rk['coverage']*100:.2f}% | "
            f"{sel} ({rk['error_upper']*100:.2f}%) | {mb['all_answer_error']*100:.2f}% | "
            f"**{mb['coverage']*100:.2f}%** | **{mb['errors']}/{mb['accepted']}** | {p23} | `{r['status']}` |"
        )

    pb = summary["panel_b_lowdim"]
    lines += [
        "",
        "## Panel B: Low-Dim Frozen A4 Strong vs. A2 Weak Readouts (`M=16` per process, `alpha=0.05/16`)",
        "",
        "| Process | Arm | Audit min R² | PASS | Const-{0,1} err | Overall all_err | Overall own_cal cov | Overall own_cal err (worst CP) | **Mode-B own_cal cov** | **Mode-B err / acc** | Mode-B pred {2,3} |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for proc_name, pdata in pb["processes"].items():
        for arm in ("strong", "weak"):
            g = pdata["summary"][arm]
            sel = f"{g['overall_sel_err']*100:.3f}%" if g["overall_sel_err"] is not None else "n/a"
            p23 = f"{g['mode_b_pred23_acc_mean']*100:.1f}%" if g["mode_b_pred23_acc_mean"] is not None else "n/a"
            lines.append(
                f"| `{proc_name}` | `{arm}` | {g['audit_min_r2_mean']:.4f} | {g['pass_count']}/8 | "
                f"{g['overall_const01_err']*100:.2f}% | {g['overall_all_err']*100:.2f}% | "
                f"{g['overall_cov']*100:.2f}% | {sel} ({g['overall_worst_cp']*100:.2f}%) | "
                f"**{g['mode_b_cov']*100:.2f}%** | **{g['mode_b_errors_total']}/{g['mode_b_accepted_total']}** | {p23} |"
            )
    lines.append("")
    (OUT / "RESULTS.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = load_module("p0_metrics", METRICS_PATH)
    summary = {
        "experiment": "b1_1_p0_regime_switching_zero_retrain",
        "panel_a_visual32": run_panel_a_visual32(metrics),
        "panel_b_lowdim": run_panel_b_lowdim(metrics),
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_markdown(summary)
    print(f"Wrote {OUT / 'SUMMARY.json'} and {OUT / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
