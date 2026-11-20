#!/usr/bin/env python3
"""B1-3 (P3-B): Parameter-Free Temporal Margin Audit Gate vs. Static R^2 >= 0.85.

Compares the static zero-lag admission rule (min AUDIT R^2 >= 0.85) against the
parameter-free Temporal Margin Audit Gate motivated by the triangle inequality
2 * ||s_hat - s||_infty < tau_cal on 128 independent AUDIT trajectories across:
  - Pool 1: Low-Dim frozen A4 budget sweep (11 budgets x 16 cells = 176 units)
  - Pool 2: 32x32 Visual formal_v1 (6) + frozen budget sweep (27) = 33 units
  - Pool 3: 224x224 FourRoom Wave-1 (Family A = 24, Family B = 51; 45 unique own_cal)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/wave2_local_20261119/p3b_temporal_audit"
METRICS_PATH = ROOT / "src/slowrepr/metrics.py"
A4_PANEL_PATH = ROOT / "reports/lejepa_a4_20261115/panel/run.py"
VIS_DIR = ROOT / "results/visual_sprites_lejepa/formal_v1"
VIS_SCRIPTS = ROOT / "scripts/visual_sprites_lejepa"
FOURROOM_ROOT = ROOT / "results/fourroom_hjepa_wave1_20261118/extracted/results"

LAGS = (4, 16, 32)
LENGTH = 193


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ar1_trajectories(rng: np.random.Generator, n_ep: int, length: int, rhos: list[float]) -> np.ndarray:
    rhos_arr = np.asarray(rhos, dtype=np.float64)
    scale = np.sqrt(1.0 - rhos_arr ** 2)
    z = np.empty((n_ep, length, len(rhos_arr)), dtype=np.float64)
    z[:, 0] = rng.normal(size=(n_ep, len(rhos_arr)))
    eps = rng.normal(size=(n_ep, length - 1, len(rhos_arr)))
    for t in range(1, length):
        z[:, t] = rhos_arr * z[:, t - 1] + scale * eps[:, t - 1]
    return z


def eval_gates(metrics, audit_pred: np.ndarray, audit_truth: np.ndarray,
               phys_ref_var: np.ndarray, pred_ref_var: np.ndarray,
               threshold: float, test_result: dict,
               audit_min_r2: float | None = None, lags: tuple[int, ...] = LAGS) -> dict:
    aud_p = np.asarray(audit_pred, dtype=np.float64)
    aud_t = np.asarray(audit_truth, dtype=np.float64)
    if aud_p.ndim == 2:
        n_total = aud_p.shape[0]
        if n_total % LENGTH != 0:
            raise ValueError(f"Cannot reshape audit arrays of length {n_total} by T={LENGTH}")
        aud_p = aud_p.reshape(n_total // LENGTH, LENGTH, 4)
        aud_t = aud_t.reshape(n_total // LENGTH, LENGTH, 4)

    if audit_min_r2 is None:
        aud_var = aud_t.reshape(-1, 4).var(axis=0, ddof=0)
        r2 = 1.0 - np.mean((aud_p - aud_t) ** 2, axis=(0, 1)) / aud_var
        audit_min_r2 = float(np.min(r2))

    aud_sc = metrics.score_windows(aud_p, aud_t, phys_ref_var, pred_ref_var, lags)
    err_bound_2x = 2.0 * np.max(np.abs(aud_sc["predicted_scores"] - aud_sc["true_scores"]), axis=1)
    audit_accept = aud_sc["gap"] >= threshold
    audit_cov = float(audit_accept.mean())
    audit_sel_errors = int(np.count_nonzero(aud_sc["wrong"][audit_accept]))
    audit_all_errors = int(np.count_nonzero(aud_sc["wrong"]))
    q95_2err = float(np.quantile(err_bound_2x, 0.95))
    q90_2err = float(np.quantile(err_bound_2x, 0.90))
    margin_ratio = float(q95_2err / threshold) if threshold > 1e-12 else None

    gate_static = bool(audit_min_r2 >= 0.85)
    gate_zero_err = bool(audit_sel_errors == 0 and audit_cov >= 0.35)
    gate_margin = bool(gate_zero_err and q95_2err < threshold)
    gate_combined = bool(gate_static and gate_margin)

    sel_err = test_result.get("selective_error")
    test_safe_cp = bool(test_result["passes"])
    test_safe_emp = bool(test_result["coverage"] >= 0.35 and sel_err is not None and sel_err <= 0.01)

    return {
        "audit_min_r2": float(audit_min_r2),
        "threshold": float(threshold),
        "audit_windows": int(aud_p.shape[0]),
        "audit_cov": audit_cov,
        "audit_all_errors": audit_all_errors,
        "audit_sel_errors": audit_sel_errors,
        "q90_2err": q90_2err,
        "q95_2err": q95_2err,
        "margin_ratio_q95": margin_ratio,
        "gates": {
            "static_r2_085": gate_static,
            "temporal_audit_zero_err": gate_zero_err,
            "temporal_margin_strict": gate_margin,
            "combined_r2_and_temporal": gate_combined,
        },
        "test_coverage": float(test_result["coverage"]),
        "test_selective_error": float(sel_err) if sel_err is not None else None,
        "test_error_upper": float(test_result["error_upper"]),
        "test_safe_cp": test_safe_cp,
        "test_safe_emp": test_safe_emp,
    }


def summarize_confusion(records: list[dict], target_key: str = "test_safe_cp") -> dict:
    gate_names = ("static_r2_085", "temporal_audit_zero_err", "temporal_margin_strict", "combined_r2_and_temporal")
    out = {}
    n = len(records)
    pos = sum(1 for r in records if r[target_key])
    neg = n - pos
    for g in gate_names:
        tp = sum(1 for r in records if r["gates"][g] and r[target_key])
        fp = sum(1 for r in records if r["gates"][g] and not r[target_key])
        tn = sum(1 for r in records if (not r["gates"][g]) and not r[target_key])
        fn = sum(1 for r in records if (not r["gates"][g]) and r[target_key])
        out[g] = {
            "n": n, "pos_safe": pos, "neg_unsafe": neg,
            "admitted": tp + fp,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "false_pass_rate_neg": float(fp / neg) if neg > 0 else 0.0,
            "fdr_admitted": float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "true_pass_recall": float(tp / pos) if pos > 0 else None,
            "accuracy": float((tp + tn) / n) if n > 0 else None,
        }
    return out


def run_pool1_lowdim(metrics) -> list[dict]:
    t0 = time.perf_counter()
    print("=== Pool 1: Low-Dim Frozen A4 Budget Sweep (176 units) ===", flush=True)
    torch.set_num_threads(4)
    panel = load_module("p3b_a4_panel", A4_PANEL_PATH)
    a4_cfg, enc_cfg, probe_cfg = panel.configs()
    old_receipt = panel.read(A4_PANEL_PATH.parent / "RECEIPT.json")
    names = panel.panel_names(a4_cfg)

    processes = {
        "separated_wide": [0.93, 0.918, 0.892, 0.88],
        "separated_narrow": [0.93, 0.906, 0.904, 0.88],
    }
    budgets = [12, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 8192]
    alpha = 0.05 / 16.0

    rng_eval = np.random.default_rng(2026111901)
    ref_z = rng_eval.normal(size=(32768, 4))
    phys_var = ref_z.var(axis=0, ddof=0)
    rng_aud = np.random.default_rng(2026111931)
    eval_data = {}
    for proc_name, rhos in processes.items():
        cal_z = ar1_trajectories(rng_eval, 1024, LENGTH, rhos)
        test_z = ar1_trajectories(rng_eval, 2048, LENGTH, rhos)
        aud_z = ar1_trajectories(rng_aud, 128, LENGTH, rhos)
        eval_data[proc_name] = (aud_z, cal_z, test_z)

    enc_cache = {}
    for name in names:
        member = old_receipt["members"][name]
        case, seed = member["case"], member["seed"]
        ckpt = A4_PANEL_PATH.parent / "runs" / f"{name}.pt"
        encoder = panel.encoder_source.make_mlp_encoder(4, enc_cfg["hidden"], enc_cfg["layers"], device="cpu").eval()
        encoder.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
        train_full = panel.data(a4_cfg, case, seed, 1, probe_cfg["n_calibration"])
        val_z = panel.data(a4_cfg, case, seed, 2, probe_cfg["n_validation"])
        h_procs = {
            p: tuple(panel.encoder_source.embed(encoder, arr.reshape(-1, 4)).reshape(arr.shape) for arr in splits)
            for p, splits in eval_data.items()
        }
        enc_cache[name] = {
            "probe_master": member["probe_master"],
            "train_z": train_full,
            "val_z": val_z,
            "h_train": panel.encoder_source.embed(encoder, train_full),
            "h_val": panel.encoder_source.embed(encoder, val_z),
            "h_ref": panel.encoder_source.embed(encoder, ref_z),
            "h_procs": h_procs,
        }

    records = []
    for n_tr in budgets:
        for name in names:
            c = enc_cache[name]
            h_tr, y_tr = c["h_train"][:n_tr], c["train_z"][:n_tr]
            mean, scale = panel.probe_source.standardizer(h_tr)
            probe, _ = panel.probe_source.fit_mlp(
                (h_tr - mean) / scale, y_tr,
                (c["h_val"] - mean) / scale, c["val_z"],
                {**probe_cfg, "master": c["probe_master"]},
            )
            pred_ref = panel.probe_source.predict_mlp((c["h_ref"] - mean) / scale, probe)
            pred_var = pred_ref.var(axis=0, ddof=0)

            for proc_name, (aud_z, cal_z, test_z) in eval_data.items():
                h_aud, h_cal, h_test = c["h_procs"][proc_name]
                pred_aud = panel.probe_source.predict_mlp((h_aud.reshape(-1, 4) - mean) / scale, probe).reshape(aud_z.shape)
                pred_cal = panel.probe_source.predict_mlp((h_cal.reshape(-1, 4) - mean) / scale, probe).reshape(cal_z.shape)
                pred_test = panel.probe_source.predict_mlp((h_test.reshape(-1, 4) - mean) / scale, probe).reshape(test_z.shape)

                cal_sc = metrics.score_windows(pred_cal, pred_cal, pred_var, pred_var, LAGS)
                thresh = float(np.quantile(cal_sc["gap"], 0.5))
                test_sc = metrics.score_windows(pred_test, test_z, phys_var, pred_var, LAGS)
                test_rk = metrics.risk(test_sc["gap"], test_sc["wrong"], thresh, alpha, 0.35, 0.01)

                rec = eval_gates(metrics, pred_aud, aud_z, phys_var, pred_var, thresh, test_rk)
                rec.update({
                    "pool": "pool1_lowdim_sweep",
                    "unit_id": f"{name}__{proc_name}__n{n_tr}",
                    "n_train": n_tr,
                    "encoder": name,
                    "process": proc_name,
                })
                records.append(rec)

    print(f"Pool 1 finished ({len(records)} units, {time.perf_counter()-t0:.1f}s)", flush=True)
    return records


def run_pool2_visual32(metrics) -> list[dict]:
    t0 = time.perf_counter()
    print("\n=== Pool 2: 32x32 Visual formal_v1 (6) + Budget Sweep (27) ===", flush=True)
    records = []

    # 2A: 6 sealed formal_v1 cells directly from disk
    vis_summary = json.loads((VIS_DIR / "SUMMARY.json").read_text())
    phys_ref_var = np.load(VIS_DIR / "reference_truth.npz")["variance"]
    for row in vis_summary["rows"]:
        cell = row["cell"]
        cdir = VIS_DIR / cell
        aud = np.load(cdir / "audit.npz")
        pred_var = np.load(cdir / "reference.npz")["variance"]
        thresh = float(json.loads((cdir / "thresholds.json").read_text())["own_cal"])
        rec = eval_gates(
            metrics, aud["prediction"], aud["truth"],
            phys_ref_var, pred_var, thresh, row["own_cal"],
            audit_min_r2=float(row["minimum_r2"]),
        )
        rec.update({"pool": "pool2_visual_formal", "unit_id": f"formal_v1/{cell}"})
        records.append(rec)

    # 2B: 9 x 3 visual_lejepa budget sweep
    lock = json.loads((VIS_DIR / "DESIGN_LOCK.json").read_text())
    cfg = lock["config"]
    lags = tuple(cfg["lags"])
    seeds = list(cfg["seeds"])
    budgets = [16, 32, 64, 128, 256, 512, 1024, 2048, 8192]
    alpha = 0.05 / len(seeds)

    render = load_module("p3b_vis_render", VIS_SCRIPTS / "render.py")
    models = load_module("p3b_vis_models", VIS_SCRIPTS / "models.py")
    device = models.configure_device("mps", cpu_threads=4) if torch.backends.mps.is_available() else torch.device("cpu")

    encoders = {}
    for s in seeds:
        name = f"visual_lejepa_seed{s}"
        enc = models.make_encoder(cfg["encoder"]["latent_dim"], device)
        enc.load_state_dict(torch.load(VIS_DIR / name / "encoder.pt", map_location=device, weights_only=True))
        enc.eval()
        encoders[name] = enc

    role_feats, role_truth = {}, {}
    batch_sz = cfg["runtime"].get("render_batch_size", cfg["runtime"]["encode_batch_size"])
    for role in ("train", "val", "audit", "ref", "cal", "test"):
        truth = render.sample_frames(render.sample_states(cfg, role, development=False), cfg, role, development=False)
        chunks = {name: [] for name in encoders}
        for imgs in render.iter_role_images(cfg, role, batch_sz, development=False):
            for name, enc in encoders.items():
                chunks[name].append(models.encode(enc, imgs, device, cfg["runtime"]["encode_batch_size"]))
        role_feats[role] = {name: np.concatenate(v, axis=0) for name, v in chunks.items()}
        role_truth[role] = truth

    phys_var = role_truth["ref"].reshape(-1, 4).var(axis=0, ddof=0)
    train_y_full = role_truth["train"].reshape(-1, 4)
    val_y = role_truth["val"].reshape(-1, 4)
    probe_dev = torch.device("cpu")
    torch.set_num_threads(4)

    for n_tr in budgets:
        for s in seeds:
            name = f"visual_lejepa_seed{s}"
            perm = np.random.default_rng(2026111900 + s).permutation(len(train_y_full))[:n_tr]
            probe, _ = models.fit_probe(
                role_feats["train"][name][perm], train_y_full[perm],
                role_feats["val"][name], val_y, cfg, s, probe_dev,
            )
            pred_aud = models.predict(probe, role_feats["audit"][name], probe_dev).reshape(role_truth["audit"].shape)
            pred_ref = models.predict(probe, role_feats["ref"][name], probe_dev)
            pred_var = pred_ref.var(axis=0, ddof=0)
            pred_cal = models.predict(probe, role_feats["cal"][name], probe_dev).reshape(role_truth["cal"].shape)
            pred_test = models.predict(probe, role_feats["test"][name], probe_dev).reshape(role_truth["test"].shape)

            cal_sc = metrics.score_windows(pred_cal, pred_cal, pred_var, pred_var, lags)
            thresh = float(np.quantile(cal_sc["gap"], 0.5))
            test_sc = metrics.score_windows(pred_test, role_truth["test"], phys_var, pred_var, lags)
            test_rk = metrics.risk(test_sc["gap"], test_sc["wrong"], thresh, alpha, 0.35, 0.01)

            rec = eval_gates(metrics, pred_aud, role_truth["audit"], phys_var, pred_var, thresh, test_rk, lags=lags)
            rec.update({"pool": "pool2_visual_sweep", "unit_id": f"{name}__n{n_tr}", "n_train": n_tr})
            records.append(rec)

    print(f"Pool 2 finished ({len(records)} units, {time.perf_counter()-t0:.1f}s)", flush=True)
    return records


def run_pool3_fourroom(metrics) -> dict:
    t0 = time.perf_counter()
    print("\n=== Pool 3: 224x224 FourRoom Wave-1 Sealed Results ===", flush=True)

    def eval_family_rows(rows: list[dict], src_dir_fn, pool_tag: str) -> list[dict]:
        out = []
        ref_truth_cache = {}
        for r in rows:
            exp_dir = src_dir_fn(r)
            if exp_dir not in ref_truth_cache:
                ref_truth_cache[exp_dir] = np.load(exp_dir / "reference_truth.npz")["variance"]
            phys_var = ref_truth_cache[exp_dir]
            cdir = exp_dir / r["cell"]
            aud = np.load(cdir / "audit.npz")
            pred_var = np.load(cdir / "reference.npz")["variance"]
            rec = eval_gates(
                metrics, aud["prediction"], aud["truth"],
                phys_var, pred_var, float(r["threshold"]), r,
                audit_min_r2=float(r["minimum_r2"]),
            )
            rec.update({
                "pool": pool_tag,
                "unit_id": f"{exp_dir.name}/{r['cell']}/{r['rule']}",
                "rule": r["rule"],
                "readout": r["readout"],
                "level": r["level"],
            })
            out.append(rec)
        return out

    p2_rows = json.loads((FOURROOM_ROOT / "fourroom_probe_v2/summary.json").read_text())["rows"]
    fam_a = eval_family_rows(p2_rows, lambda r: FOURROOM_ROOT / "fourroom_probe_v2", "pool3_fourroom_familyA")

    comb = json.loads((FOURROOM_ROOT / "fourroom_depth_v2_combined/summary.json").read_text())
    src_map = {"lewm": "fourroom_lewm_v2", "l2": "fourroom_probe_v2", "l4": "fourroom_hjepa_l4_v2"}
    fam_b = eval_family_rows(comb["rows"], lambda r: FOURROOM_ROOT / src_map[r["model"]], "pool3_fourroom_familyB")

    # Also build unique own_cal set (45 unique cell folders across lewm(9) + probe_v2(18) + l4(18))
    unique_rows = []
    for fam_name in ("fourroom_lewm_v2", "fourroom_probe_v2", "fourroom_hjepa_l4_v2"):
        s = json.loads((FOURROOM_ROOT / fam_name / "summary.json").read_text())
        for r in s["rows"]:
            if r["rule"] == "own_cal":
                unique_rows.append((fam_name, r))
    # Re-evaluate CP upper bound cleanly under M=45 for the deduplicated 45 own_cal pool
    alpha_45 = 0.05 / len(unique_rows)
    unique_45 = []
    ref_truth_cache = {}
    for fam_name, r in unique_rows:
        exp_dir = FOURROOM_ROOT / fam_name
        if exp_dir not in ref_truth_cache:
            ref_truth_cache[exp_dir] = np.load(exp_dir / "reference_truth.npz")["variance"]
        cdir = exp_dir / r["cell"]
        aud = np.load(cdir / "audit.npz")
        pred_var = np.load(cdir / "reference.npz")["variance"]
        ts = np.load(cdir / "test_scores.npz")
        rk45 = metrics.risk(ts["gap"], ts["wrong"], float(r["threshold"]), alpha_45, 0.35, 0.01)
        rec = eval_gates(
            metrics, aud["prediction"], aud["truth"],
            ref_truth_cache[exp_dir], pred_var, float(r["threshold"]), rk45,
            audit_min_r2=float(r["minimum_r2"]),
        )
        rec.update({
            "pool": "pool3_fourroom_unique45_own_cal",
            "unit_id": f"{fam_name}/{r['cell']}",
            "readout": r["readout"],
            "level": r["level"],
        })
        unique_45.append(rec)

    print(f"Pool 3 finished: famA={len(fam_a)}, famB={len(fam_b)}, unique45={len(unique_45)} ({time.perf_counter()-t0:.1f}s)", flush=True)
    return {"family_a_24": fam_a, "family_b_51": fam_b, "unique_45": unique_45}


def write_markdown(summary: dict):
    lines = [
        "# B1-3 (P3-B): Parameter-Free Temporal Margin Audit Gate vs. Static `min R² >= 0.85`",
        "",
        "Compares admission rules computed **strictly before accessing TEST** using independent `AUDIT` windows (`128 x 193` = 128 windows):",
        "- `static_r2_085`: $\\min_i R^2_{\\mathrm{AUDIT},i} \\ge 0.85$",
        "- `temporal_audit_zero_err`: 0 selective errors on accepted `AUDIT` windows and `AUDIT` coverage $\\ge 0.35$",
        "- `temporal_margin_strict` (**parameter-free triangle-inequality gate**): `temporal_audit_zero_err` AND $Q_{0.95,\\mathrm{AUDIT}}(2\\|\\hat{s}-s\\|_\\infty) < \\tau_{\\mathrm{cal}}$",
        "- `combined_r2_and_temporal`: both `static_r2_085` and `temporal_margin_strict`",
        "",
        "## Confusion Matrix Comparison (`target = TEST passes` at Bonferroni CP upper <= 1% & cov >= 35%)",
        "",
        "| Evaluation Pool | Total (Safe / Unsafe) | Gate Rule | Admitted (TP+FP) | TP | **False-Pass (FP)** | TN | FN | FPR (`FP/Unsafe`) | FDR (`FP/Adm`) | True-Pass Recall |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pool_title, conf in summary["confusion_by_pool"].items():
        for g in ("static_r2_085", "temporal_audit_zero_err", "temporal_margin_strict", "combined_r2_and_temporal"):
            c = conf[g]
            rec = f"{c['true_pass_recall']*100:.1f}%" if c["true_pass_recall"] is not None else "n/a"
            lines.append(
                f"| `{pool_title}` | {c['n']} ({c['pos_safe']}/{c['neg_unsafe']}) | `{g}` | {c['admitted']} | "
                f"{c['TP']} | **{c['FP']}** | {c['TN']} | {c['FN']} | "
                f"{c['false_pass_rate_neg']*100:.1f}% | {c['fdr_admitted']*100:.1f}% | {rec} |"
            )

    lines += [
        "",
        "## Empirical-Safety Confusion Matrix Comparison (`target = TEST cov >= 35% & emp_sel_err <= 1%`)",
        "",
        "| Evaluation Pool | Total (EmpSafe / Unsafe) | Gate Rule | Admitted | TP | **False-Pass (FP)** | Recall |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for pool_title, conf in summary["confusion_emp_by_pool"].items():
        for g in ("static_r2_085", "temporal_audit_zero_err", "temporal_margin_strict", "combined_r2_and_temporal"):
            c = conf[g]
            rec = f"{c['true_pass_recall']*100:.1f}%" if c["true_pass_recall"] is not None else "n/a"
            lines.append(
                f"| `{pool_title}` | {c['n']} ({c['pos_safe']}/{c['neg_unsafe']}) | `{g}` | {c['admitted']} | "
                f"{c['TP']} | **{c['FP']}** | {rec} |"
            )

    lines += ["", "## Notable False-Pass Examples Under Static `R² >= 0.85` Caught by Temporal Margin Gate", ""]
    fp_cases = summary["static_false_passes_caught"]
    if not fp_cases:
        lines.append("None.")
    else:
        lines += [
            "| Pool | Unit | Audit min R² | AUDIT sel err / cov | `Q95(2||s_hat-s||_inf)` vs `tau_cal` | `TEST` sel err (CP upper) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in fp_cases[:20]:
            sel = f"{r['test_selective_error']*100:.2f}%" if r["test_selective_error"] is not None else "n/a"
            lines.append(
                f"| `{r['pool']}` | `{r['unit_id']}` | {r['audit_min_r2']:.4f} | "
                f"{r['audit_sel_errors']} / {r['audit_cov']*100:.1f}% | "
                f"{r['q95_2err']:.4f} vs {r['threshold']:.4f} | "
                f"{sel} ({r['test_error_upper']*100:.2f}%) |"
            )
    lines.append("")
    (OUT / "RESULTS.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = load_module("p3b_metrics", METRICS_PATH)
    pool1 = run_pool1_lowdim(metrics)
    pool2 = run_pool2_visual32(metrics)
    pool3 = run_pool3_fourroom(metrics)

    unique_all = pool1 + pool2 + pool3["unique_45"]
    pools = {
        "Pool 1: Low-Dim Sweep (176)": pool1,
        "Pool 2: Visual 32x32 (33)": pool2,
        "Pool 3A: FourRoom Family A (24)": pool3["family_a_24"],
        "Pool 3B: FourRoom Family B (51)": pool3["family_b_51"],
        "Pool 3-U: FourRoom Unique own_cal (45)": pool3["unique_45"],
        "ALL UNIQUE POOLS COMBINED (254)": unique_all,
    }
    confusion_cp = {k: summarize_confusion(v, "test_safe_cp") for k, v in pools.items()}
    confusion_emp = {k: summarize_confusion(v, "test_safe_emp") for k, v in pools.items()}

    static_fp_caught = [
        r for r in unique_all
        if r["gates"]["static_r2_085"] and (not r["test_safe_cp"]) and (not r["gates"]["temporal_margin_strict"])
    ]
    static_fp_caught.sort(key=lambda r: -r["test_error_upper"])

    summary = {
        "experiment": "b1_3_p3b_temporal_audit_gate",
        "confusion_by_pool": confusion_cp,
        "confusion_emp_by_pool": confusion_emp,
        "static_false_passes_caught": static_fp_caught,
        "records": {
            "pool1_lowdim_sweep": pool1,
            "pool2_visual32": pool2,
            "pool3_fourroom_family_a": pool3["family_a_24"],
            "pool3_fourroom_family_b": pool3["family_b_51"],
            "pool3_fourroom_unique45": pool3["unique_45"],
        },
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_markdown(summary)

    print("\n=== B1-3 Summary (ALL UNIQUE POOLS COMBINED = 254 units) ===", flush=True)
    for g, c in confusion_cp["ALL UNIQUE POOLS COMBINED (254)"].items():
        print(
            f"  {g:24s} | Adm={c['admitted']:3d} | TP={c['TP']:3d} | FP={c['FP']:3d} "
            f"| TN={c['TN']:3d} | FN={c['FN']:3d} | FPR={c['false_pass_rate_neg']*100:5.1f}% "
            f"| Recall={c['true_pass_recall']*100:5.1f}%",
            flush=True,
        )
    print(f"Wrote {OUT / 'SUMMARY.json'} and {OUT / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
