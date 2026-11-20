#!/usr/bin/env python3
"""Export existing exploratory summaries into complete Markdown and CSV tables.

No sampling, training, model loading, or mutation of the source summary occurs.
Missing values remain NA; CSV numeric risks and coverages are fractions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

EVIDENCE = "exploratory_numerical_sandbox"
ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "results/exploratory_20261116"


def number(value, digits=6):
    return "NA" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "NA" if value is None else f"{100*value:.4f}%"


def integer(value):
    return "NA" if value is None else str(int(value))


def yesno(value):
    return "是" if value else "否"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(header, rows):
    return ["| " + " | ".join(map(cell, header)) + " |",
            "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(map(cell, row)) + " |" for row in rows]


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def count_columns(row, prefix, record, ci_key="ci95_wilson"):
    for key in ("n", "accepted", "errors", "coverage", "risk", "threshold"):
        row[prefix + key] = record.get(key)
    interval = record.get(ci_key)
    row[prefix + "wilson_lower"] = interval[0] if interval else None
    row[prefix + "wilson_upper"] = interval[1] if interval else None


def array_columns(row, prefix, values):
    for i, value in enumerate(values):
        row[f"{prefix}{i}"] = value


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if fields[0] != "evidence_tier":
        raise AssertionError("Every exported CSV must start with evidence_tier")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            # NA is deliberately not zero and not an ambiguous empty numeric cell.
            writer.writerow({key: "NA" if row.get(key) is None else row[key] for key in fields})
    with path.open(newline="", encoding="utf-8") as stream:
        parsed = list(csv.DictReader(stream))
    if len(parsed) != len(rows) or any(r["evidence_tier"] != EVIDENCE for r in parsed):
        raise AssertionError("CSV round-trip validation failed")


def anchor_exports(data, digest):
    anchors = data["anchors"]["cells"]
    md = ["evidence_tier: " + EVIDENCE, "", "# 封存锚点逐项对照", "",
          f"来源：`probe2_summary.json`，SHA256 `{digest}`。本表只展开既有输出，不重新运行实验。", "",
          "128 个单元包括 A4 16、A3 16、A2 strong/weak 32、B1/B3 64。AUDIT R² 是各阶段实际审计数组的均值或最小值；",
          "两者各自投射到同过程、同 α、T=193 的网格。近邻距离为 AUDIT R² 与网格 R² 的绝对差。插值是相邻网格风险的线性描述插值，未经重新仿真或对锚点拟合。", "",
          "误差率及覆盖率显示为百分数；差值单位为百分点。NA 表示无匹配预测，不表示零。各单元共享模型、REF 或物理轨迹，不是128个独立统计重复。",
          "coupled 单元实际 c=.12，而相变表只有 c=0；这些对照明确标记为动力学代理，不参加112单元的主拟合指标。", "",
          "## 全量作答：全部128个单元", ""]
    all_rows, rule_rows, csv_rows = [], [], []
    for index, anchor in enumerate(anchors, 1):
        ident = f"A{index:03d}"
        mean = anchor["projections"]["mean"]
        minimum = anchor["projections"]["min"]
        all_rows.append([ident, anchor["experiment"] + "/" + anchor["arm"], anchor["name"],
                         anchor["process"] + "/" + anchor["regime"],
                         f"{anchor['alpha']:g}/{anchor['coupling']:g}",
                         number(anchor["audit_r2"]["mean"]), number(anchor["audit_r2"]["min"]),
                         f"{anchor['all']['errors']}/{anchor['all']['n']}", percent(anchor["all"]["risk"]),
                         number(mean["nearest"]["r2_population"], 3), number(mean["nearest_grid_r2_distance"]),
                         percent(mean["nearest"]["all_risk"]), percent(mean["interpolation"]["prediction"]["all_risk"]),
                         number(minimum["nearest"]["r2_population"], 3), number(minimum["nearest_grid_r2_distance"]),
                         percent(minimum["nearest"]["all_risk"]), percent(minimum["interpolation"]["prediction"]["all_risk"]),
                         "c=0代理" if anchor["coupling"] else "对角动力学匹配"])
        row = {"evidence_tier": EVIDENCE, "anchor_id": ident,
               **{k: anchor[k] for k in ("experiment", "name", "process", "regime", "arm", "T", "alpha", "coupling")},
               "audit_r2_mean": anchor["audit_r2"]["mean"], "audit_r2_min": anchor["audit_r2"]["min"],
               "audit_r2_max": anchor["audit_r2"]["max"], "audit_r2_range": anchor["audit_r2"]["range"],
               "audit_source": anchor["audit_source"], "selection_design": anchor["selection_design"],
               "n_reference": anchor["n_reference"]}
        for label in ("rho_base", "rho_diagonal_effective", "predicted_reference_variance", "physical_reference_variance"):
            array_columns(row, label + "_", anchor[label])
        array_columns(row, "audit_r2_axis_", anchor["audit_r2"]["by_axis"])
        count_columns(row, "actual_all_", anchor["all"])
        for mode in ("mean", "min"):
            p = anchor["projections"][mode]
            row[mode + "_nearest_r2_distance"] = p["nearest_grid_r2_distance"]
            row[mode + "_coupling_exactly_matched"] = p["coupling_exactly_matched"]
            for key, value in p["nearest"].items():
                row[mode + "_nearest_" + key] = value
            for key, value in p["interpolation"]["prediction"].items():
                row[mode + "_interpolated_" + key] = value
            for key in ("lower_r2", "upper_r2", "upper_weight", "outside_grid"):
                row[mode + "_interpolation_" + key] = p["interpolation"][key]
            for key, value in p["differences"].items():
                row[mode + "_" + key] = value
            row[mode + "_actual_ref_minus_ideal_variance_rms"] = p["actual_ref_minus_ideal_variance_rms"]
            row[mode + "_selective_comparisons_json"] = json_cell(p["selective_comparisons"])
        row["actual_selection_rules_json"] = json_cell(anchor["rules"])
        csv_rows.append(row)
        for rule, observed in anchor["rules"].items():
            pm = mean["selective_comparisons"][rule]
            pl = minimum["selective_comparisons"][rule]
            if "phase_rule" not in pm:
                state = "无匹配规则"
                phase_rule = "NA"
                mean_coverage = min_coverage = mean_risk = min_risk = None
                caveat = "A4批次409/2048或A3 g20/g80在相变表没有同规则输出"
            else:
                phase_rule = pm["phase_rule"]
                mean_coverage, mean_risk = pm["interpolated_coverage"], pm["interpolated_risk"]
                min_coverage, min_risk = pl["interpolated_coverage"], pl["interpolated_risk"]
                if pm["numeric_threshold_matches"]:
                    state = "仅数值阈值相同"
                    caveat = "CAL/REF/读出残差仍不同；不是模型风险定律"
                else:
                    state = "50%意图代理"
                    caveat = "封存CAL按regime跨16单元混池；沙盒按单条件取中位数"
                if anchor["coupling"]:
                    state += ";动力学代理"
                    caveat += ";封存c=.12，沙盒c=0"
            rule_rows.append([ident, rule, integer(observed.get("accepted")), integer(observed.get("errors")),
                              percent(observed.get("coverage")), percent(observed.get("risk")),
                              number(observed.get("threshold")), state, phase_rule,
                              percent(mean_coverage), percent(mean_risk), percent(min_coverage), percent(min_risk), caveat])
    md += table(["ID", "实验/读出", "模型", "过程/条件", "α/c", "AUDIT均值R²", "AUDIT最小R²",
                 "实测错/总", "实测all", "均值近邻R²", "均值距离", "均值近邻all", "均值插值all",
                 "最小近邻R²", "最小距离", "最小近邻all", "最小插值all", "动力学范围"], all_rows)
    md += ["", "## 全部选择规则：匹配、代理及无对应项", "",
           "下表保留全部240个封存选择规则行。A3 g50、A2 fixed、B fixed 的数值阈值可匹配；这不表示残差分布或参考样本匹配。",
           "A4 nominal/adjusted 是TEST批次排序规则，A3 g20/g80也不在本次相变表中，因此预测列为NA。",
           "B recalibrated 与沙盒 own_cal 均面向约50%覆盖，但CAL混池单位不同，仅作代理。均值/最小两列均使用透明R²插值；具体近邻及全部原字段保存在CSV与JSON。", ""]
    md += table(["锚点ID", "封存规则", "作答数", "错误数", "实测覆盖", "实测错率", "封存阈值",
                 "比较状态", "沙盒规则", "均值代理覆盖", "均值代理错率", "最小代理覆盖", "最小代理错率", "范围说明"], rule_rows)
    md += ["", "数值解释：这些投射只检验指定的各向同性白噪声代理。AUDIT R²均值/最小值不能保留逐轴误差谱、跨轴泄漏、共享REF偏差和非线性结构，",
           "实际与代理的差不能唯一归因于误差方向。正态总体轴对倒序概率没有被用作同窗预测错误率。", ""]
    if len(all_rows) != 128 or len(rule_rows) != 240:
        raise AssertionError(f"Unexpected anchor counts: {len(all_rows)}, {len(rule_rows)}")
    return "\n".join(md), csv_rows, len(rule_rows)


def tmin_export(data, digest):
    phase = data["phase"]
    grouped = {}
    for c in phase["cells"]:
        grouped.setdefault((c["process"], c["alpha"], c["r2_population"]), []).append(c)
    rows = []
    reversals = {"all": 0, "own_cal": 0}
    censored = {"all": 0, "own_cal": 0}
    for item in phase["t_min"]:
        key = (item["process"], item["alpha"], item["r2_population"])
        series = sorted(grouped[key], key=lambda c: c["T"])
        max_t = series[-1]["T"]
        row = [item["process"], number(item["alpha"], 2), number(item["r2_population"], 3)]
        for mode in ("all", "own_cal"):
            entry = item[mode]
            first = entry["first_observed_below_1pct"]
            first_upper = entry["first_pointwise_wilson_upper_below_1pct"]
            risk = [c[mode]["risk"] for c in series]
            any_increase = any(risk[i+1] > risk[i] + 1e-15 for i in range(len(risk)-1))
            reversal = entry["below_then_above_on_grid"]
            # Recheck, rather than silently inheriting an ambiguous censor label.
            passing = [c["T"] for c in series if c[mode]["risk"] < .01]
            upper_passing = [c["T"] for c in series if c[mode]["ci95_wilson"][1] < .01]
            if first != (passing[0] if passing else None) or first_upper != (upper_passing[0] if upper_passing else None):
                raise AssertionError("T-min summary differs from exported phase cells")
            crossing = "无" if first is not None else f"观测截至{max_t}未达"
            if first_upper is None:
                crossing = ("" if crossing == "无" else crossing + ";") + f"上界截至{max_t}未达"
            row += [integer(first), integer(first_upper), yesno(reversal), yesno(any_increase), crossing]
            reversals[mode] += int(reversal)
            censored[mode] += int(first is None)
        rows.append(row)
    if len(rows) != 80:
        raise AssertionError("Expected all 80 process/alpha/R2 trajectories")
    md = ["evidence_tier: " + EVIDENCE, "", "# 全部80格窗口长度扫描", "",
          f"来源：`probe2_summary.json`，SHA256 `{digest}`。只读导出，不重新采样或选阈值。", "",
          "每行固定一个过程、α与总体状态R²，扫描 T={65,97,129,193,289,385,577,769,1153,1537}。",
          "“首观测”是首次观察到错误率严格小于1%的网格长度；“首上界”是首次点态近似Wilson 95%上界严格小于1%的长度。",
          "两者都不是已证明的最小窗口定律，不是跨80行或800格同时认证，不能据此断言所有更长窗口都会通过。首次为65也不排除未测试的更短窗口。", "",
          "“回穿1%”指曾低于1%后在更长网格点再次达到或高于1%；“任意上升”还记录未必跨越1%的局部风险上升。",
          "“截至1537未达”只是当前有限网格的右删失记录，不是物理总体T_min>1537的证明。NA保持未观察到，绝不当成0。",
          "own_cal 每格独立CAL定中位阈值，实际作答率约50%而非强制正好50%；真值分母为已知1，预测分母为共享的有限REF经验方差。", "",
          f"全答首观测未达：{censored['all']}/80；own_cal未达：{censored['own_cal']}/80。回穿1%分别为{reversals['all']}/80与{reversals['own_cal']}/80。", ""]
    md += table(["过程", "α", "总体R²", "all首观测T", "all首上界T", "all回穿1%", "all任意上升", "all删失",
                 "own首观测T", "own首上界T", "own回穿1%", "own任意上升", "own删失"], rows)
    md += ["", "完整800格的风险、作答数、点态区间、分差、SNR和双参考口径见 `phase_cells.csv`；",
           "长窗口下的非单调现象须结合共享有限REF、真实同窗排序和噪声归一化理解，不能预先归结为运行错误或普遍时间尺度定律。", ""]
    return "\n".join(md)


def phase_csv(data):
    rows = []
    for c in data["phase"]["cells"]:
        row = {"evidence_tier": EVIDENCE, **{k: c[k] for k in ("process", "alpha", "T", "r2_population", "residual_variance",
               "delta_mu", "physical_delta_mu", "sigma_diff", "snr_eff", "snr_requested_physical_gap_over_predicted_sigma",
               "normal_population_pair_proxy", "population_pair_inversion_observed", "predicted_gap_median")}}
        for key in ("rho", "r2_ref_empirical_by_axis", "physical_reference_variance", "predicted_reference_variance",
                    "empirical_physical_reference_variance", "predicted_population_means", "predicted_population_variances", "physical_score_variances"):
            array_columns(row, key+"_", c[key])
        for key, value in c["conditioned_on_ref"].items():
            row["conditioned_ref_"+key] = value
        for mode in ("all", "own_cal", "fixed"):
            count_columns(row, mode+"_", c[mode])
            count_columns(row, "empirical_true_ref_"+mode+"_", c["empirical_physical_ref_sensitivity"][mode])
        rows.append(row)
    if len(rows) != 800:
        raise AssertionError("Expected full 800-cell phase grid")
    return rows


def stress_csv(data):
    rows = []
    for c in data["stress"]["requested_cells"] + data["stress"]["additional_limit_controls"]:
        row = {"evidence_tier": EVIDENCE, **{k: c[k] for k in ("branch", "extra_limit_control", "fast_signal_fraction", "leakage_amplitude", "noise_sd",
               "audit_gate_min_r2_ge_085", "gate_action", "population_fast_axis_enters", "median_predicted_gap", "any_predicted_fast_axis_frequency",
               "raw_unnormalized_score_wrong_rate_diagnostic_only")},
               "audit_r2_mean": c["audit_r2"]["mean"], "audit_r2_min": c["audit_r2"]["min"],
               "population_top2": " ".join(map(str, c["population_top2"]))}
        array_columns(row, "audit_r2_axis_", c["audit_r2"]["by_axis"])
        for key in ("population_state_r2_by_axis", "predicted_reference_variance", "audit_predicted_variance", "population_predicted_variance", "population_scores"):
            array_columns(row, key+"_", c[key])
        for j, value in enumerate(c["critical_signal_fraction_for_population_fast_entry_by_axis"], 2):
            row[f"critical_signal_fraction_axis_{j}"] = value
        for key, value in c["fast_variance_fractions"].items():
            row["fast_variance_fraction_"+key] = value
        for mode in ("all", "own_cal", "false_fast_inclusion", "false_fast_inclusion_own_cal"):
            count_columns(row, mode+"_", c[mode], ci_key="ci95_wilson_descriptive")
        for mode in ("all", "own_cal"):
            count_columns(row, "empirical_true_ref_"+mode+"_", c["empirical_physical_ref_sensitivity"][mode], ci_key="ci95_wilson_descriptive")
        rows.append(row)
    if len(rows) != 72 or sum(bool(r["extra_limit_control"]) for r in rows) != 16:
        raise AssertionError("Expected 56 requested stress cells and 16 separate controls")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DESTINATION / "probe2_summary.json", help="Existing read-only probe2 summary")
    parser.add_argument("--output-dir", type=Path, default=DESTINATION, help="Output directory inside results/exploratory_20261116")
    args = parser.parse_args()
    started = time.perf_counter()
    out = args.output_dir.resolve()
    if not out.is_relative_to(DESTINATION.resolve()):
        parser.error("Output must remain within results/exploratory_20261116")
    raw = args.summary.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    if data["evidence_tier"] != EVIDENCE:
        raise ValueError("Source evidence tier does not match the exploratory export")
    anchor_md, anchor_rows, rule_count = anchor_exports(data, digest)
    tmin_md = tmin_export(data, digest)
    phase_rows, stress_rows = phase_csv(data), stress_csv(data)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {"ANCHOR_COMPARISON.md": anchor_md, "TMIN_FULL.md": tmin_md}
    for name, content in outputs.items():
        if content.splitlines()[0] != "evidence_tier: " + EVIDENCE:
            raise AssertionError("Markdown evidence tier missing")
        (out / name).write_text(content, encoding="utf-8")
    write_csv(out / "phase_cells.csv", phase_rows)
    write_csv(out / "anchor_cells.csv", anchor_rows)
    write_csv(out / "stress_cells.csv", stress_rows)
    if hashlib.sha256(args.summary.read_bytes()).hexdigest() != digest:
        raise AssertionError("Read-only source summary changed during export")
    elapsed = time.perf_counter()-started
    print(f"evidence_tier: {EVIDENCE}")
    print(f"Exported {len(anchor_rows)} anchors, {rule_count} selection rows, 80 T scans, {len(phase_rows)} phase rows, {len(stress_rows)} stress rows in {elapsed:.3f}s.")
    print(f"Source summary SHA256 unchanged: {digest}")
    if elapsed >= 60:
        raise RuntimeError("Export exceeded the 60-second budget")


if __name__ == "__main__":
    main()
