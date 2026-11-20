#!/usr/bin/env python3
"""Deterministic Gaussian moment checks against sealed JSON summaries.

Python 3.11+ and NumPy only. No trajectory generation, model loading, training,
prediction reads, evaluation reruns, or imports of archived experiment code.
Only config.json / SUMMARY.json files are read from reports. Outputs are confined
to the repository's results directory. The ideal population-ranking diagnostic
is deliberately distinguished from the archived same-window prediction error.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def score_matrix(length, lags):
    """Equal lag weights, then 1/(T-h) endpoint weights, without centering."""
    matrix = np.zeros((length, length), dtype=np.float64)
    for lag in lags:
        if not 0 < lag < length:
            raise ValueError("Every lag must be in (0, T)")
        left = np.arange(length - lag)
        right = left + lag
        weight = 1.0 / (len(lags) * (length - lag))
        matrix[left, left] += weight
        matrix[right, right] += weight
        matrix[left, right] -= weight
        matrix[right, left] -= weight
    return matrix


def trajectory_covariance(transition, length, axis_i, axis_j):
    """Cov(Z[t,i],Z[s,j]), stationary covariance I; no samples generated."""
    powers = [np.eye(len(transition))]
    for _ in range(1, length):
        powers.append(powers[-1] @ transition)
    powers = np.asarray(powers)
    distance = np.arange(length)[:, None] - np.arange(length)[None, :]
    return np.where(distance >= 0, powers[np.abs(distance), axis_i, axis_j],
                    powers[np.abs(distance), axis_j, axis_i])


def boundary_kl_lower(rho_i, rho_j, length):
    """Pinsker lower bound on P(S_j <= S_i), independent physical axes only."""
    variance_ratio = (1 - rho_j**2) / (1 - rho_i**2)
    divergence = (length - 1) / 2 * (
        variance_ratio - 1 - math.log(variance_ratio)
        + (rho_j - rho_i)**2 / (1 - rho_i**2))
    return {"kl_to_equal_rate_pair": divergence,
            "population_pair_inversion_lower_bound": max(0., .5 - math.sqrt(divergence / 2))}


def exact_moments(rates, exponent, coupling, length, lags, matrix, matrix_root):
    rho = np.asarray(rates, dtype=float)**exponent
    transition = np.diag(rho)
    transition[1, 2] = coupling  # exactly B run.py's zero-based indices
    innovation = np.eye(4) - transition @ transition.T
    q_min = float(np.linalg.eigvalsh(innovation).min())
    if q_min <= 0:
        raise ValueError("Archived Cholesky generator requires positive definite Q")
    score_means = []
    covariance = np.zeros((4, 4))
    norm_rows = []
    for i in range(4):
        sigma = trajectory_covariance(transition, length, i, i)
        b = matrix_root @ sigma @ matrix_root
        b = (b + b.T) / 2
        eigenvalues = np.linalg.eigvalsh(b)
        mean = float(np.trace(matrix @ sigma))
        mean_formula = float(np.mean([2 * (1 - rho[i]**h) for h in lags]))
        variance = float(2 * np.sum(b*b))
        variance_trace = float(2 * np.trace(matrix @ sigma @ matrix @ sigma))
        np.testing.assert_allclose(mean, mean_formula, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(variance, variance_trace, atol=2e-13, rtol=2e-13)
        score_means.append(mean)
        norm_rows.append({"axis_zero_based": i, "mean": mean, "variance": variance,
                          "B_frobenius": float(np.linalg.norm(b)),
                          "B_operator": float(eigenvalues[-1]),
                          "state_effective_length_proxy_T_one_minus_rho": float(length*(1-rho[i]))})
        for j in range(i, 4):
            cross = trajectory_covariance(transition, length, i, j)
            # Gaussian Isserlis: Cov(X'MX,Y'MY)=2 tr(M K M K').
            value = float(2 * np.trace(matrix @ cross @ matrix @ cross.T))
            covariance[i, j] = covariance[j, i] = value
    gap = score_means[2] - score_means[1]
    diff_variance = float(covariance[1, 1] + covariance[2, 2] - 2*covariance[1, 2])
    sigma_diff = math.sqrt(diff_variance)
    snr = gap / sigma_diff
    result = {"rho": rho.tolist(), "transition": transition.tolist(),
              "innovation_covariance": innovation.tolist(), "Q_min_eigenvalue": q_min,
              "coupling_max_psd": math.sqrt((1-rho[1]**2)*(1-rho[2]**2)),
              "score_means": score_means, "score_covariance": covariance.tolist(),
              "boundary_axes_zero_based": [1, 2], "delta_mu": gap,
              "variance_diff": diff_variance, "sigma_diff": sigma_diff, "snr": snr,
              "axis_norms": norm_rows,
              "normal_approx_population_pair_inversion_NOT_task_risk": .5*math.erfc(snr/math.sqrt(2)),
              "ideal_same_window_risk_with_exact_unit_denominators": 0.0}
    if coupling == 0:
        result.update(boundary_kl_lower(rho[1], rho[2], length))
    return result


def aggregate_cells(cells):
    total = sum(c["all_answers"]["n"] for c in cells)
    errors = sum(c["all_answers"]["errors"] for c in cells)
    rules = {}
    for label in ("fixed", "recalibrated"):
        accepted = sum(c["rules"][label]["accepted"] for c in cells)
        wrong = sum(c["rules"][label]["errors"] for c in cells)
        rules[label] = {"accepted": accepted, "errors": wrong,
                        "coverage": accepted/total, "risk": wrong/accepted,
                        "threshold": cells[0]["rules"][label]["threshold"],
                        "worst_upper": max(c["rules"][label]["upper"] for c in cells),
                        "passing_cells": sum(c["rules"][label]["pass"] for c in cells)}
    return {"model_window_evaluations": total, "independent_physical_windows": total//2,
            "all_errors": errors, "all_risk": errors/total,
            "mean_of_cell_predicted_gap_medians_NOT_pooled_median":
                float(np.mean([c["predicted_gap_quantiles"][2] for c in cells])),
            "mean_of_cell_physical_gap_medians_NOT_pooled_median":
                float(np.mean([c["physical_gap_quantiles"][2] for c in cells])),
            "predicted_gap_below_006_fraction": sum(c["by_predicted_gap"][0]["accepted"] for c in cells)/total,
            "rules": rules}


def order_check(rows):
    """Descriptive only: higher ideal SNR would predict lower empirical risk."""
    comparisons = []
    for a, b in itertools.combinations(rows, 2):
        ds = a["ideal"]["snr"] - b["ideal"]["snr"]
        dr = a["empirical"]["all_risk"] - b["empirical"]["all_risk"]
        comparisons.append({"a": a["id"], "b": b["id"],
                            "violates_decreasing_risk_with_snr": bool(ds*dr > 0),
                            "snr_difference": ds, "risk_difference_pp": 100*dr})
    return {"interpretation": "Descriptive order check, not a fitted risk law or independent statistical test",
            "pairs": len(comparisons),
            "violations": sum(c["violates_decreasing_risk_with_snr"] for c in comparisons),
            "comparisons": comparisons}


def algebra_checks(rows, length, lags, matrix, matrix_root):
    """Independent increment-covariance route and analytic limit checks."""
    ast.parse(Path(__file__).read_text(), feature_version=(3, 11))
    checks = {"python_311_grammar": True,
              "M_nullity_at_1e_12": int(np.sum(np.linalg.eigvalsh(matrix) < 1e-12))}
    assert checks["M_nullity_at_1e_12"] == 4
    max_error = 0.
    for rho in (.906, .918, .93**.25, .88**4):
        total = 0.
        for h in lags:
            for ell in lags:
                u = np.arange(length-h)[:, None] - np.arange(length-ell)[None, :]
                cov = (rho**np.abs(u+h-ell) - rho**np.abs(u+h)
                       - rho**np.abs(u-ell) + rho**np.abs(u))
                total += 2*np.sum(cov**2)/((length-h)*(length-ell)*len(lags)**2)
        distance = np.abs(np.arange(length)[:, None]-np.arange(length)[None, :])
        sigma = rho**distance
        direct = 2*np.trace(matrix @ sigma @ matrix @ sigma)
        np.testing.assert_allclose(total, direct, rtol=1e-12, atol=1e-13)
        max_error = max(max_error, abs(float(total-direct)))
    checks["independent_increment_covariance_variance_max_abs_error"] = max_error
    distance = np.abs(np.arange(length)[:, None]-np.arange(length)[None, :])
    c = -matrix_root @ distance @ matrix_root
    checks["slow_limit_C_min_eigenvalue_roundoff_allowed"] = float(np.linalg.eigvalsh(c).min())
    checks["slow_limit_trace_C"] = float(np.trace(c))
    np.testing.assert_allclose(np.trace(c), 2*np.mean(lags), atol=1e-10)
    by_id = {r["id"]: r for r in rows}
    for process in sorted({r["process"] for r in rows}):
        base = by_id[process+"/baseline"]["ideal"]
        coupled = by_id[process+"/coupled"]["ideal"]
        np.testing.assert_allclose(base["score_means"], coupled["score_means"], atol=1e-13)
        np.testing.assert_allclose(np.diag(base["score_covariance"]), np.diag(coupled["score_covariance"]), atol=1e-13)
        assert coupled["variance_diff"] < base["variance_diff"]
        rho = np.array(base["rho"])
        lam = -np.log(rho)
        limit = 2*np.mean(lags)*(lam[2]-lam[1])/math.sqrt(2*(lam[1]**2+lam[2]**2)*np.sum(c*c))
        exact = exact_moments(rho, 1e-5, 0, length, lags, matrix, matrix_root)["snr"]
        np.testing.assert_allclose(exact, limit, rtol=5e-5)
        checks[process+"_alpha_to_zero_snr_limit"] = {"limit": float(limit), "alpha_1e_5": exact}
    checks["sealed_summary_count_consistency"] = "passed by main verification script"
    return checks


def render(result):
    lines = ["# 理论解析矩与封存汇总核对", "",
             "只读配置/JSON汇总；没有采样、模型前向或重跑评测。总体边界对倒序与同窗模型错选是不同事件。",
             "所有理论分母为精确总体方差1；实测预测分母为冻结、ddof=0的有限REF。", "",
             "## B1/B3：两个基础过程 × 四个条件", "",
             "分差中位数列是8个单元各自中位数的算术平均，不是合池中位数；只凭SUMMARY无法恢复后者。", "",
             "|过程/条件|Δμ|σ差|SNR|实测全答错率|预测分差中位数均值|固定覆盖|50%校准错/答（错率）|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["dynamics"]:
        t, e = row["ideal"], row["empirical"]
        r = e["rules"]["recalibrated"]
        lines.append(f'|{row["id"]}|{t["delta_mu"]:.9f}|{t["sigma_diff"]:.9f}|{t["snr"]:.7f}|'
                     f'{e["all_risk"]*100:.4f}%|{e["mean_of_cell_predicted_gap_medians_NOT_pooled_median"]:.6f}|'
                     f'{e["rules"]["fixed"]["coverage"]*100:.3f}%|{r["errors"]}/{r["accepted"]} ({r["risk"]*100:.4f}%)|')
    check = result["snr_order_check"]
    lines += ["", f'在8条件的{check["pairs"]}对描述性比较中，{check["violations"]}对违背“SNR越高、同窗模型错率越低”。',
              "该比较不是显著性检验。理想精确读出对同窗真值的错误率为0，全部实测错误均与此理想条件有偏离；",
              "不能将实测误差全部归因于非线性残差，有限REF也不同，且总体SNR不描述模型误差方向。", "",
              "## A4同窗任务：总体边界倒序代理不等于模型风险", "",
              "|过程|精确SNR|总体倒序正态近似（非保证）|总体倒序严格下界|实测模型错率|", "|---|---:|---:|---:|---:|"]
    for row in result["a4_comparison"]:
        lines.append(f'|{row["process"]}|{row["snr"]:.7f}|{100*row["normal_proxy"]:.3f}%|'
                     f'{100*row["inversion_lower_bound"]:.3f}%|{100*row["empirical_all_risk"]:.4f}%|')
    lines += ["", "正态尾仅供展示错误目标混用的数量级；未用它拟合、认证或解释A4错误。", "",
              "## B2封存诊断（模型均值）", "",
              "|过程|校准lag|准确率|理想最近间隔|经验矩阵半径q90|间隔/半径均值|旋转损失|", "|---|---|---:|---:|---:|---:|---:|"]
    for r in result["b2"]:
        lines.append(f'|{r["process"]}|{r["arm"]}|{100*r["accuracy"]:.3f}%|{r["ideal_nearest_gap"]:.7f}|'
                     f'{r["mean_radius"]:.7f}|{r["mean_gap_to_radius"]:.4f}|{r["mean_rotation_loss"]:.4f}|')
    lines += ["", "bootstrap经验半径不是已证明的总体算子误差上界；multi不是联合对角化。", "",
              "## A1 local：必须区分MSE参照、净准确率与修复/损害", "",
              "|轮次|MSE相对mean下降|准确率相对raw变化|准确率相对mean变化|修复/损害|", "|---|---:|---:|---:|---:|"]
    for r in result["a1_local"]:
        lines.append(f'|{r["round"]}|{100*r["mse_gain_vs_mean"]:.4f}%|{r["accuracy_gain_vs_raw_pp"]:+.5f}pp|'
                     f'{r["accuracy_gain_vs_mean_pp"]:+.5f}pp|{r["repaired"]}/{r["harmed"]}|')
    lines += ["", "全部输入路径、SHA256、精确矩、统计口径与局限见SUMMARY.json。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/theory_checks_20261116",
                        help="Output directory under this repository's results/ (default: %(default)s)")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT/"results").resolve()):
        parser.error("Outputs must stay within this repository's results/ directory")
    if any(output.is_relative_to((ROOT/p).resolve()) for p in ("reports", "archive", "work")):
        parser.error("Protected experiment directories are read-only")
    provenance = {}

    def read(relative):
        path = ROOT / relative
        data = path.read_bytes()
        provenance[relative] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    cfg = read("reports/lejepa_b_transfer_20261115/config.json")
    summary = read("reports/lejepa_b_transfer_20261115/SUMMARY.json")
    a4cfg = read("reports/lejepa_a4_20261115/config.json")
    a4 = read("reports/lejepa_a4_20261115/SUMMARY.json")
    a3 = read("reports/lejepa_a3_gap_20261115/SUMMARY.json")
    a2 = read("reports/lejepa_a2_readout_20261115/SUMMARY.json")
    b2cfg = read("reports/lejepa_b2_lag_20261115/config.json")
    b2 = read("reports/lejepa_b2_lag_20261115/SUMMARY.json")
    local = read("reports/lejepa_a1_joint_20261115/local/SUMMARY.json")
    reference = read("reports/lejepa_a1_joint_20261115/reference/SUMMARY.json")
    close_b3 = read("reports/lejepa_local_closeout_20261116/b3/SUMMARY.json")
    close_a1 = read("reports/lejepa_local_closeout_20261116/a1/SUMMARY.json")
    for name in ("processes", "length", "lags"):
        assert cfg[name] == a4cfg[name], name
    length, lags = cfg["length"], cfg["lags"]
    matrix = score_matrix(length, lags)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    assert eigenvalues.min() > -1e-14
    root = (eigenvectors*np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    # Deterministic algebra checks, not evaluation data: constant paths vanish.
    np.testing.assert_allclose(matrix @ np.ones(length), 0, atol=1e-16)
    probe = np.sin(np.arange(length) * .17) + np.arange(length) * .001
    direct = np.mean([np.mean((probe[h:]-probe[:-h])**2) for h in lags])
    np.testing.assert_allclose(probe @ matrix @ probe, direct, rtol=2e-14)
    rows = []
    for process, rates in cfg["processes"].items():
        for regime, setting in cfg["regimes"].items():
            theory = exact_moments(rates, setting["exponent"], setting["coupling"], length, lags, matrix, root)
            cells = [c for c in summary["cells"] if c["process"] == process and c["regime"] == regime]
            empirical = aggregate_cells(cells)
            rows.append({"id": process+"/"+regime, "process": process, "regime": regime,
                         "ideal": theory, "empirical": empirical,
                         "ideal_task_risk_shortfall_pp": 100*empirical["all_risk"]})
    for regime in cfg["regimes"]:
        cells = [c for c in summary["cells"] if c["regime"] == regime]
        computed = aggregate_cells(cells)
        np.testing.assert_allclose(computed["all_risk"], summary["aggregate"][regime]["all_error_rate"], atol=0)
        for rule in ("fixed", "recalibrated"):
            for key in ("accepted", "errors", "coverage", "risk"):
                assert computed["rules"][rule][key] == summary["aggregate"][regime]["rules"][rule][key]
        if regime != "coupled":
            assert close_b3["summary"][regime]["coverages"]["0.5"]["errors"] == computed["rules"]["recalibrated"]["errors"]
    a4rows = []
    for process in cfg["processes"]:
        t = next(r["ideal"] for r in rows if r["process"] == process and r["regime"] == "baseline")
        a4rows.append({"process": process, "snr": t["snr"],
                       "normal_proxy": t["normal_approx_population_pair_inversion_NOT_task_risk"],
                       "inversion_lower_bound": t["population_pair_inversion_lower_bound"],
                       "empirical_all_risk": a4["all"]["by_process"][process]["all_window_error"],
                       "a4_archived": a4["all"]["by_process"][process],
                       "a3_g50_archived": a3["thresholds"]["g50"]["by_process"][process]})
    b2rows = []
    for process in cfg["processes"]:
        cells = [c for c in b2["by_cell"].values() if c["process"] == process]
        for arm in b2cfg["arms"]:
            ds = [c["calibration"]["arms"][arm] for c in cells]
            b2rows.append({"process": process, "arm": arm,
                           "accuracy": float(np.mean([c["test"][arm]["accuracy"] for c in cells])),
                           "ideal_nearest_gap": ds[0]["ideal_nearest_eigengap_axes2_3"],
                           "mean_radius": float(np.mean([d["operator_radius_q90"] for d in ds])),
                           "mean_gap_to_radius": float(np.mean([d["ideal_gap_to_radius"] for d in ds])),
                           "mean_rotation_loss": float(np.mean([d["cal_bootstrap_rotation_loss_axes2_3"] for d in ds]))})
    localrows = []
    for name, r in local["rounds"].items():
        v = r["pooled"]
        localrows.append({"round": name, "mse_gain_vs_mean": v["degree2"]["vs_mean_relative_mse_gain"],
                          "accuracy_gain_vs_raw_pp": 100*(v["degree2"]["top2_accuracy"]-v["zero"]["top2_accuracy"]),
                          "accuracy_gain_vs_mean_pp": 100*(v["degree2"]["top2_accuracy"]-v["mean"]["top2_accuracy"]),
                          "repaired": v["degree2"]["top2_repaired"], "harmed": v["degree2"]["top2_harmed"],
                          "windows": v["degree2"]["top2_opportunities"]})
    result = {"status": "deterministic_analytic_checks_only", "runtime": {"python": platform.python_version(), "numpy": np.__version__},
              "operator": {"T": length, "lags": lags, "lag_weights": [1/len(lags)]*len(lags),
                           "reference_ddof": 0, "a4_reference_states": a4cfg["n_reference"], "M_trace": float(np.trace(matrix)),
                           "M_frobenius": float(np.linalg.norm(matrix)), "M_operator": float(eigenvalues[-1])},
              "dynamics": rows, "snr_order_check": order_check(rows), "a4_comparison": a4rows,
              "b_transfer_aggregate_archived": summary["aggregate"], "b2": b2rows, "a1_local": localrows,
              "a1_reference_archived": reference["archived_window_diagnostic"],
              "a1_closeout_scope": close_a1["scope"],
              "a2_archived": {a: {k: v[k] for k in ("overall", "worst_error_upper", "n_cells_passing_both")} for a,v in a2["arms"].items()},
              "source_sha256": provenance,
              "limitations": ["Population pair inversion is not same-window prediction error.",
                              "Normal proxy has no asserted Gaussian approximation error bound.",
                              "Only means of archived cell medians are available; pooled medians cannot be recovered.",
                              "No nonlinear readout or finite-REF risk is predicted by physical trajectory SNR alone.",
                              "Pooled counts are descriptive; shared trajectories are not independent model evaluations."]}
    output.mkdir(parents=True, exist_ok=True)
    validation = algebra_checks(rows, length, lags, matrix, root)
    (output/"SUMMARY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (output/"RESULTS.md").write_text(render(result))
    (output/"VALIDATION.json").write_text(json.dumps(validation, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output_dir": str(output), "conditions": len(rows), "input_files": len(provenance),
                      "snr_order_violations": result["snr_order_check"]["violations"],
                      "no_model_or_trajectory_execution": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
