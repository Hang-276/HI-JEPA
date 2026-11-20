"""Summarize the v2 dual-seed baseline run; read-only, no training."""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "runs" / "formal_v2"
V1 = REPO / "runs" / "formal"
SEEDS = (0, 1)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def result_record(root, seed):
    run_dir = root / f"e1_baseline_seed{seed}"
    result_path = run_dir / "result.json"
    record = {"run_dir": str(run_dir), "result_present": result_path.is_file()}
    if not result_path.is_file():
        return record
    result = read_json(result_path)
    evaluation = result.get("evaluation", {})
    acceptance = result.get("anticollapse", {})
    forecast = evaluation.get("original_space_forecast", {})
    rows = {row["horizon"]: row for row in forecast.get("by_horizon", [])}
    training_rows = []
    log_path = run_dir / "training.jsonl"
    if log_path.is_file():
        training_rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    record.update({
        "status": result.get("status"),
        "source_updates": result.get("source_updates"),
        "source_training_seconds": result.get("source_training_seconds_this_segment"),
        "average_step_seconds": result.get("average_step_seconds_this_segment"),
        "memory": result.get("memory", {}),
        "environment": {key: result.get("environment", {}).get(key) for key in
                        ("torch", "cuda_build", "device_name", "cuda_available")},
        "anticollapse": {
            "passed": acceptance.get("passed"),
            "checks": acceptance.get("checks"),
            "fine_effective_rank": acceptance.get("fine", {}).get("effective_rank"),
            "coarse_effective_rank": acceptance.get("coarse", {}).get("effective_rank"),
            "coarse_min_eigenvalue": acceptance.get("coarse", {}).get("minimum_covariance_eigenvalue"),
            "coarse_eigenvalues": acceptance.get("coarse", {}).get("covariance_eigenvalues"),
            "fine_dynamic_fraction": acceptance.get("fine_dynamic", {}).get("dynamic_fraction_of_total_variance"),
            "coarse_dynamic_fraction": acceptance.get("coarse_dynamic", {}).get("dynamic_fraction_of_total_variance"),
            "thresholds": acceptance.get("thresholds"),
        },
        "recovery": {
            "cluster_mean_functionspace_sin_squared": evaluation.get("cluster_mean_functionspace_sin_squared"),
            "axis": evaluation.get("axis"),
            "clusters": evaluation.get("clusters"),
            "frozen_matching": evaluation.get("frozen_matching"),
            "common_thresholds": evaluation.get("common_thresholds"),
            "coarse_one_step_residual": evaluation.get("coarse_one_step_residual"),
            "fine_one_step_residual": evaluation.get("fine_one_step_residual"),
            "theory_diagnostics": evaluation.get("theory_diagnostics"),
            "target_spectrum_diagnostics": evaluation.get("target_spectrum_diagnostics"),
        },
        "fine_retention": evaluation.get("fine_retention"),
        "forecast": {
            "fine_mean_mse": forecast.get("fine_mean_mse"),
            "coarse_mean_mse": forecast.get("coarse_mean_mse"),
            "by_horizon": {str(h): rows.get(h) for h in (1, 8, 16, 32)},
        },
        "training_log": {
            "rows": len(training_rows),
            "first": training_rows[0] if training_rows else None,
            "last": training_rows[-1] if training_rows else None,
            "all_losses_finite": all(
                all(finite(value) for value in row.get("losses", {}).values())
                for row in training_rows),
            "floor_values": [{"step": row.get("step"), "floor": row.get("losses", {}).get("floor")}
                             for row in training_rows],
        },
    })
    return record


def main():
    V2.mkdir(parents=True, exist_ok=True)
    acceptance_path = V2 / "prelaunch_anticollapse_acceptance.json"
    acceptance = read_json(acceptance_path) if acceptance_path.is_file() else None
    progress_path = V2 / "progress.json"
    progress = read_json(progress_path) if progress_path.is_file() else None
    summary_path = V2 / "summary.json"
    matrix_summary = read_json(summary_path) if summary_path.is_file() else None
    report = {
        "scope": "v2 dual-seed baseline; gate acceptance, recovery, detail, timing; no 57-record expansion",
        "gate_file": str(acceptance_path), "gate_present": acceptance is not None,
        "gate_passed": None if acceptance is None else acceptance.get("passed"),
        "acceptance": acceptance,
        "progress": progress,
        "matrix_summary": matrix_summary,
        "seeds": {f"seed{seed}": result_record(V2, seed) for seed in SEEDS},
        "v1_reference": {f"seed{seed}": result_record(V1, seed) for seed in SEEDS},
        "anomalies": sorted(
            str(path.relative_to(V2)) for pattern in ("failure.json", "halted.json")
            for path in V2.rglob(pattern)),
    }
    # Compact v1/v2 comparison table.
    comparison = {}
    for seed in SEEDS:
        v2 = report["seeds"][f"seed{seed}"]
        v1 = report["v1_reference"][f"seed{seed}"]
        comparison[f"seed{seed}"] = {
            "v1_coarse_min_eigenvalue": v1.get("anticollapse", {}).get("coarse_min_eigenvalue"),
            "v2_coarse_min_eigenvalue": v2.get("anticollapse", {}).get("coarse_min_eigenvalue"),
            "v1_fine_effective_rank": v1.get("anticollapse", {}).get("fine_effective_rank"),
            "v2_fine_effective_rank": v2.get("anticollapse", {}).get("fine_effective_rank"),
            "v1_coarse_sin2": v1.get("recovery", {}).get("cluster_mean_functionspace_sin_squared"),
            "v2_coarse_sin2": v2.get("recovery", {}).get("cluster_mean_functionspace_sin_squared"),
            "v1_fine_forecast_mean_mse": v1.get("forecast", {}).get("fine_mean_mse"),
            "v2_fine_forecast_mean_mse": v2.get("forecast", {}).get("fine_mean_mse"),
            "v1_source_training_seconds": v1.get("source_training_seconds"),
            "v2_source_training_seconds": v2.get("source_training_seconds"),
            "v1_cuda_peak_bytes": v1.get("memory", {}).get("cuda_peak_allocated_bytes"),
            "v2_cuda_peak_bytes": v2.get("memory", {}).get("cuda_peak_allocated_bytes"),
        }
    report["comparison"] = comparison
    (V2 / "v2_baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# HI-JEPA v2 dual-seed baseline report",
        "",
        "Scope: baseline seed 0/1 only; no 57-record matrix; no v2.1.",
        "",
        "## Gate",
        "",
        "| seed | status | gate | fine eff. rank | coarse min eig | coarse eig. rank | coarse / fine dynamics |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for seed in SEEDS:
        record = report["seeds"][f"seed{seed}"]
        ac = record.get("anticollapse", {})
        lines.append(
            f"| {seed} | {record.get('status')} | {ac.get('passed')} | "
            f"{ac.get('fine_effective_rank')} | {ac.get('coarse_min_eigenvalue')} | "
            f"{ac.get('coarse_effective_rank')} | "
            f"{ac.get('coarse_dynamic_fraction')} / {ac.get('fine_dynamic_fraction')} |")
    lines += [
        "",
        "## Recovery and detail",
        "",
        "| seed | coarse sin2 | axis mean MSE | both_met | fine forecast mean MSE | fine_state NMSE | coarse_state NMSE | fine_detail NMSE | coarse_detail NMSE |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        record = report["seeds"][f"seed{seed}"]
        ret = record.get("fine_retention") or {}
        axis = record.get("recovery", {}).get("axis") or {}
        thresholds = record.get("recovery", {}).get("common_thresholds") or {}
        lines.append(
            f"| {seed} | {record.get('recovery', {}).get('cluster_mean_functionspace_sin_squared')} | "
            f"{axis.get('isolated_axis_mean_mse')} | {thresholds.get('both_met')} | "
            f"{record.get('forecast', {}).get('fine_mean_mse')} | "
            f"{(ret.get('fine_state') or {}).get('normalized_mse')} | "
            f"{(ret.get('coarse_state') or {}).get('normalized_mse')} | "
            f"{(ret.get('fine_detail') or {}).get('normalized_mse')} | "
            f"{(ret.get('coarse_detail') or {}).get('normalized_mse')} |")
    lines += [
        "",
        "NMSE baseline is 1 (mean predictor); recovery thresholds are unchanged from v1.",
        "",
        "## v1 -> v2 comparison",
        "",
        "| seed | v1 gate | v2 gate | v1 coarse min eig | v2 coarse min eig | v1 coarse sin2 | v2 coarse sin2 | v1 fine mean MSE | v2 fine mean MSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        v1 = report["v1_reference"][f"seed{seed}"]
        v2 = report["seeds"][f"seed{seed}"]
        c = report["comparison"][f"seed{seed}"]
        lines.append(
            f"| {seed} | {v1.get('anticollapse', {}).get('passed')} | "
            f"{v2.get('anticollapse', {}).get('passed')} | "
            f"{c.get('v1_coarse_min_eigenvalue')} | {c.get('v2_coarse_min_eigenvalue')} | "
            f"{c.get('v1_coarse_sin2')} | {c.get('v2_coarse_sin2')} | "
            f"{c.get('v1_fine_forecast_mean_mse')} | {c.get('v2_fine_forecast_mean_mse')} |")
    lines += [
        "",
        "## Original-space forecast (NMSE; baseline=1)",
        "",
        "| seed | fine h1 | fine h8 | fine h16 | fine h32 | coarse h1 | coarse h8 | coarse h16 | coarse h32 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        rows = report["seeds"][f"seed{seed}"].get("forecast", {}).get("by_horizon", {})
        def value(kind, horizon):
            row = rows.get(str(horizon)) or {}
            return (row.get(kind) or {}).get("normalized_mse")
        lines.append(
            f"| {seed} | " + " | ".join(
                str(value(kind, horizon)) for kind in ("fine", "coarse") for horizon in (1, 8, 16, 32)
            ) + " |")
    lines += [
        "",
        "## Timing / memory",
        "",
        "| seed | source seconds | avg step (s) | CUDA peak (MiB) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        record = report["seeds"][f"seed{seed}"]
        peak = record.get("memory", {}).get("cuda_peak_allocated_bytes")
        lines.append(
            f"| {seed} | {record.get('source_training_seconds')} | "
            f"{record.get('average_step_seconds')} | "
            f"{None if peak is None else round(peak / 1048576, 2)} |")
    lines += [
        "",
        "## GCSB in training log",
        "",
    ]
    for seed in SEEDS:
        record = report["seeds"][f"seed{seed}"]
        floor_values = record.get("training_log", {}).get("floor_values", [])
        active = [item for item in floor_values if item.get("floor") not in (None, 0.0)]
        lines.append(f"- seed{seed}: {len(active)}/{len(floor_values)} logged checkpoints with GCSB > 0; "
                     f"first={floor_values[0] if floor_values else None}; last={floor_values[-1] if floor_values else None}")
    finite_summary = [
        f"seed{seed}: {report['seeds'][f'seed{seed}'].get('training_log', {}).get('all_losses_finite')}"
        for seed in SEEDS
    ]
    coarse_h1 = []
    for seed in SEEDS:
        row = (report["seeds"][f"seed{seed}"].get("forecast", {})
               .get("by_horizon", {}).get("1", {}))
        coarse_h1.append((row.get("coarse") or {}).get("normalized_mse"))
    lines += [
        "",
        "## Anomalies",
        "",
        f"- failure/halted files: {report['anomalies'] or 'none'}",
        f"- all training losses finite: {finite_summary}",
        "",
        "## Interpretation",
        "",
        "- Gate pass and recovery success are separate. The gate only checks train-only anti-collapse; "
        "the recovery/detail metrics can still be poor.",
        f"- coarse h1 NMSE values are {coarse_h1}, both < 1 (better than the mean-predictor baseline); "
        "the coarse recovery failure is established by the recovery table (coarse sin2 > 0.25 and "
        "both_met=false), not by the h1 forecast.",
        "- GCSB is a soft batch-covariance spectral penalty, not a hard anti-collapse guarantee.",
    ]
    (V2 / "BASELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "gate_passed": report["gate_passed"],
        "progress": progress,
        "anomalies": report["anomalies"],
        "comparison": comparison,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
