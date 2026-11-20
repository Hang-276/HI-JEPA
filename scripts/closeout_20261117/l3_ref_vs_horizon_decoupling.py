#!/usr/bin/env python3
"""L3: Decoupling finite-REF denominator sampling bias from trajectory horizon T
on the 8 real frozen A4 LeJEPA models (strong and weak readouts).

Directly addresses ICML_RESEARCH_TODO.md line 132 & EXPLORATORY line 340:
- Repeats R=16 independent REF draws across N_ref in {256, 1024, 4096, 32768} + population nu_i;
- Evaluates across horizons T in {65, 193, 513, 1025};
- Compares known unit physical variance (low-dim protocol) vs. shared physical REF variance
  (visual protocol), and tracks Proposition 1 numerator vs. denominator RMS scaling.
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
A4_DIR = ROOT / "reports/lejepa_a4_20261115"
A2_DIR = ROOT / "reports/lejepa_a2_readout_20261115"
OUT = ROOT / "results/closeout_20261117/l3_ref_horizon_decoupling"
LAGS = (4, 16, 32)


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def top2_and_gap(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores, axis=1, kind="stable")
    ordered = np.take_along_axis(scores, order, axis=1)
    return np.sort(order[:, :2], axis=1), ordered[:, 2] - ordered[:, 1]


def multi_lag_num(paths: np.ndarray, lags: tuple[int, ...] = LAGS) -> np.ndarray:
    return sum(np.mean((paths[:, h:] - paths[:, :-h]) ** 2, axis=1) for h in lags) / len(lags)


def gen_ar1(rng: np.random.Generator, count: int, length: int, rates: list[float]) -> np.ndarray:
    draws = rng.normal(size=(count, length, 4))
    vals = draws.copy()
    rho = np.asarray(rates, dtype=np.float64)
    sc = np.sqrt(1.0 - rho**2)
    for t in range(1, length):
        vals[:, t] = rho * vals[:, t - 1] + sc * draws[:, t]
    return vals


def eval_readout_chunked(readout, arr: np.ndarray, chunk: int = 512) -> np.ndarray:
    if arr.ndim == 2:
        parts = [readout(arr[i : i + 8192]) for i in range(0, len(arr), 8192)]
        return np.concatenate(parts, axis=0)
    parts = [readout(arr[i : i + chunk]) for i in range(0, len(arr), chunk)]
    return np.concatenate(parts, axis=0)


def main() -> None:
    t0 = time.perf_counter()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(A4_DIR))
    a4 = load_mod("ro_a4_l3", A4_DIR / "forecast.py")
    panel = a4.panel
    weak_mod = load_mod("ro_a2_weak_l3", A2_DIR / "weak/run.py")

    a4_cfg = json.loads((A4_DIR / "config.json").read_text())
    processes = a4_cfg["processes"]
    models = sorted(f"{c}_{s}" for s in a4_cfg["seeds"] for c in a4_cfg["training_cases"])

    readouts = {}
    for name in models:
        readouts[(name, "strong")] = panel.load_readout(name)[0]
        readouts[(name, "weak")] = weak_mod.load_readout(name)[0]

    rng = np.random.default_rng(2026111704)

    n_ref_grid = [256, 1024, 4096, 32768]
    r_draws = 16
    t_grid = [65, 193, 513, 1025]
    n_test = 1024
    n_cal = 512
    max_t = max(t_grid)

    # 1. Estimate high-precision population output variance nu_i on 262,144 iid N(0, I_4) states
    z_pop_ref = rng.normal(size=(262144, 4))
    pop_var_map = {}
    for key, ro in readouts.items():
        pred_pop = eval_readout_chunked(ro, z_pop_ref)
        pop_var_map[key] = pred_pop.var(axis=0, ddof=0)

    # 2. Pre-draw R=16 independent maximum-size (32,768) iid REF pools; nested prefixes give N_ref grid cleanly
    ref_pools_z = rng.normal(size=(r_draws, max(n_ref_grid), 4))
    ref_vars_pred = {}  # (name, arm, n_ref) -> [R, 4]
    ref_vars_true = {}  # n_ref -> [R, 4]
    for n_ref in n_ref_grid:
        sub_z = ref_pools_z[:, :n_ref, :]
        ref_vars_true[n_ref] = sub_z.var(axis=1, ddof=0)
    for key, ro in readouts.items():
        pred_all = eval_readout_chunked(ro, ref_pools_z.reshape(-1, 4)).reshape(r_draws, max(n_ref_grid), 4)
        for n_ref in n_ref_grid:
            ref_vars_pred[(key[0], key[1], n_ref)] = pred_all[:, :n_ref, :].var(axis=1, ddof=0)

    # 3. Generate one independent master trajectory panel of length max_t=1025 per process for TEST and CAL,
    #    then slice prefixes [:T] across T in {65, 193, 513, 1025} and score every frozen readout!
    grid_rows = []
    for process, rates in processes.items():
        z_test_full = gen_ar1(rng, n_test, max_t, rates)
        z_cal_full = gen_ar1(rng, n_cal, max_t, rates)

        # Precompute numerator per (model, arm, T) so evaluating 16 REF draws is instantaneous!
        true_num_test = {T: multi_lag_num(z_test_full[:, :T, :]) for T in t_grid}

        for arm in ("strong", "weak"):
            # Accumulate across all 8 models per (process, arm, T, N_ref)
            agg = {
                (T, n_ref_tag): {
                    "unit_all_err_mean": [],
                    "unit_all_err_p90_across_ref": [],
                    "unit_own_cal_err_mean": [],
                    "shared_ref_all_err_mean": [],
                    "shared_ref_own_cal_err_mean": [],
                    "num_pert_rms": [],
                    "pop_denom_shift_rms": [],
                    "finite_ref_sampling_rms": [],
                }
                for T in t_grid
                for n_ref_tag in ([str(n) for n in n_ref_grid] + ["pop_inf"])
            }

            for name in models:
                ro = readouts[(name, arm)]
                pred_test_full = eval_readout_chunked(ro, z_test_full)
                pred_cal_full = eval_readout_chunked(ro, z_cal_full)
                nu = pop_var_map[(name, arm)]

                for T in t_grid:
                    z_t = z_test_full[:, :T, :]
                    p_t = pred_test_full[:, :T, :]
                    p_c = pred_cal_full[:, :T, :]

                    s_true_unit = true_num_test[T]  # known V_0 = 1
                    top2_true_unit, _ = top2_and_gap(s_true_unit)

                    num_pred_t = multi_lag_num(p_t)
                    num_pred_c = multi_lag_num(p_c)

                    # Proposition 1 numerator perturbation K + E = num_pred_t - s_true_unit
                    ke_t = num_pred_t - s_true_unit

                    # (a) Infinite REF (nu)
                    s_pred_pop = num_pred_t / nu[None, :]
                    s_cal_pop = num_pred_c / nu[None, :]
                    top2_pred_pop, gap_pred_pop = top2_and_gap(s_pred_pop)
                    _, gap_cal_pop = top2_and_gap(s_cal_pop)
                    thr_pop = float(np.quantile(gap_cal_pop, 0.5, method="higher"))
                    wrong_pop = np.any(top2_pred_pop != top2_true_unit, axis=1)
                    acc_mask_pop = gap_pred_pop >= thr_pop
                    own_err_pop = float(wrong_pop[acc_mask_pop].mean()) if acc_mask_pop.any() else 0.0

                    agg[(T, "pop_inf")]["unit_all_err_mean"].append(float(wrong_pop.mean()))
                    agg[(T, "pop_inf")]["unit_all_err_p90_across_ref"].append(float(wrong_pop.mean()))
                    agg[(T, "pop_inf")]["unit_own_cal_err_mean"].append(own_err_pop)
                    agg[(T, "pop_inf")]["shared_ref_all_err_mean"].append(float(wrong_pop.mean()))
                    agg[(T, "pop_inf")]["shared_ref_own_cal_err_mean"].append(own_err_pop)
                    agg[(T, "pop_inf")]["num_pert_rms"].append(float(np.sqrt(np.mean((ke_t / nu[None, :]) ** 2))))
                    agg[(T, "pop_inf")]["pop_denom_shift_rms"].append(
                        float(np.sqrt(np.mean((s_true_unit * (nu - 1.0)[None, :] / nu[None, :]) ** 2)))
                    )
                    agg[(T, "pop_inf")]["finite_ref_sampling_rms"].append(0.0)

                    # (b) Finite REF draws R=16
                    for n_ref in n_ref_grid:
                        v_hats = ref_vars_pred[(name, arm, n_ref)]  # [R, 4]
                        v_trues = ref_vars_true[n_ref]  # [R, 4]
                        u_errs, u_own_errs, sh_errs, sh_own_errs = [], [], [], []
                        num_rms_l, pop_rms_l, ref_rms_l = [], [], []
                        for r_i in range(r_draws):
                            vh = v_hats[r_i]
                            vt = v_trues[r_i]
                            sp = num_pred_t / vh[None, :]
                            sc = num_pred_c / vh[None, :]
                            top2_p, gap_p = top2_and_gap(sp)
                            _, gap_c = top2_and_gap(sc)
                            thr = float(np.quantile(gap_c, 0.5, method="higher"))
                            acc_m = gap_p >= thr

                            # Unit truth V_0 = 1
                            w_u = np.any(top2_p != top2_true_unit, axis=1)
                            u_errs.append(float(w_u.mean()))
                            u_own_errs.append(float(w_u[acc_m].mean()) if acc_m.any() else 0.0)

                            # Shared physical REF truth V_0 = vt
                            st_sh = s_true_unit / vt[None, :]
                            top2_t_sh, _ = top2_and_gap(st_sh)
                            w_sh = np.any(top2_p != top2_t_sh, axis=1)
                            sh_errs.append(float(w_sh.mean()))
                            sh_own_errs.append(float(w_sh[acc_m].mean()) if acc_m.any() else 0.0)

                            eta = vh - nu
                            num_rms_l.append(float(np.sqrt(np.mean((ke_t / vh[None, :]) ** 2))))
                            pop_rms_l.append(float(np.sqrt(np.mean((s_true_unit * (nu - 1.0)[None, :] / vh[None, :]) ** 2))))
                            ref_rms_l.append(float(np.sqrt(np.mean((s_true_unit * eta[None, :] / vh[None, :]) ** 2))))

                        tag = str(n_ref)
                        agg[(T, tag)]["unit_all_err_mean"].append(float(np.mean(u_errs)))
                        agg[(T, tag)]["unit_all_err_p90_across_ref"].append(float(np.quantile(u_errs, 0.9)))
                        agg[(T, tag)]["unit_own_cal_err_mean"].append(float(np.mean(u_own_errs)))
                        agg[(T, tag)]["shared_ref_all_err_mean"].append(float(np.mean(sh_errs)))
                        agg[(T, tag)]["shared_ref_own_cal_err_mean"].append(float(np.mean(sh_own_errs)))
                        agg[(T, tag)]["num_pert_rms"].append(float(np.mean(num_rms_l)))
                        agg[(T, tag)]["pop_denom_shift_rms"].append(float(np.mean(pop_rms_l)))
                        agg[(T, tag)]["finite_ref_sampling_rms"].append(float(np.mean(ref_rms_l)))

            for (T, n_ref_tag), metrics_dict in agg.items():
                grid_rows.append({
                    "process": process,
                    "arm": arm,
                    "T": T,
                    "n_ref": n_ref_tag,
                    **{k: float(np.mean(v)) for k, v in metrics_dict.items()},
                })

    payload = {
        "evidence_tier": "read_only_real_model_ref_horizon_decoupling",
        "elapsed_seconds": time.perf_counter() - t0,
        "design": {
            "models": models,
            "arms": ["strong", "weak"],
            "processes": list(processes.keys()),
            "n_ref_grid": n_ref_grid + ["pop_inf"],
            "independent_ref_draws_per_cell": r_draws,
            "horizon_t_grid": t_grid,
            "n_test_windows": n_test,
            "n_cal_windows": n_cal,
        },
        "grid_rows": grid_rows,
    }
    (OUT / "l3_ref_horizon_decoupling.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )

    # Print concise summary for strong/narrow across T and N_ref
    strong_narrow = [
        r for r in grid_rows if r["arm"] == "strong" and r["process"] == "separated_narrow"
    ]
    print(json.dumps({
        "status": "OK",
        "elapsed_seconds": payload["elapsed_seconds"],
        "strong_narrow_summary": [
            {
                "T": r["T"],
                "n_ref": r["n_ref"],
                "unit_all": round(r["unit_all_err_mean"] * 100, 3),
                "unit_p90_ref": round(r["unit_all_err_p90_across_ref"] * 100, 3),
                "shared_all": round(r["shared_ref_all_err_mean"] * 100, 3),
                "unit_own_cal": round(r["unit_own_cal_err_mean"] * 100, 3),
                "shared_own_cal": round(r["shared_ref_own_cal_err_mean"] * 100, 3),
                "num_rms": round(r["num_pert_rms"], 5),
                "ref_rms": round(r["finite_ref_sampling_rms"], 5),
            }
            for r in strong_narrow
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
