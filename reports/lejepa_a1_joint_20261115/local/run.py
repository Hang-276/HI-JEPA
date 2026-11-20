"""Freeze independent labeled CAL coefficients, then score the physical mechanism."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parents[1]
ROOT = REPORTS.parent
LEGACY = REPORTS / "lejepa_dynamic_boundary_20261113"
ROUNDS = {"primary": (2026111521, 2026111523), "replication": (2026111522, 2026111524)}
LAGS = (4, 16, 32)
N_CAL = 32768
N_TEST = 1024
LENGTH = 193
BOOTSTRAPS = 4096
MIN_GAIN = 0.10
ARMS = ("zero", "mean", "self_mean", "self_weighted", "degree2", "oracle")
PAIR_AXES = np.triu_indices(4, k=1)


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


B = module("quadratic_legacy_runner", LEGACY / "run.py")
S = module("quadratic_legacy_score", LEGACY / "score.py")


def rng(master, role):
    return np.random.default_rng(np.random.SeedSequence([master, role]))


def paths(master, rates):
    innovations = rng(master, 102).normal(size=(N_TEST, LENGTH, 4))
    result = innovations.copy()
    scale = np.sqrt(1 - rates**2)
    for t in range(1, LENGTH):
        result[:, t] = rates * result[:, t - 1] + scale * innovations[:, t]
    return result


def quadratic(states):
    left, right = PAIR_AXES
    return np.concatenate((states**2 - 1, states[..., left] * states[..., right]), axis=-1)


def design():
    C, identity, cells, config, sources = B.design()
    paths = [HERE / "PROTOCOL.md", HERE / "run.py", LEGACY / "DESIGN_LOCK.json",
             LEGACY / "forecast/RECEIPT.json"]
    sources.update({str(path.relative_to(ROOT)): C.digest(path) for path in paths})
    receipt = C.read(LEGACY / "forecast/RECEIPT.json")
    if C.digest(LEGACY / "DESIGN_LOCK.json") != receipt["design_sha256"]:
        raise ValueError("Legacy design receipt mismatch")
    axes = {}
    for round_name in ROUNDS:
        axes[round_name] = {}
        for cell in cells:
            key = "__".join(cell[field] for field in ("layer", "process", "name"))
            path = LEGACY / f"forecast/{round_name}/{key}.json"
            expected = receipt["files"][str(path.relative_to(LEGACY))]
            if C.digest(path) != expected:
                raise ValueError(f"Legacy normalization receipt mismatch: {path}")
            sources[str(path.relative_to(ROOT))] = expected
            archive = C.read(path)
            if archive["cell"] != cell or archive["provenance"] != cell["provenance"]:
                raise ValueError(f"Legacy model identity mismatch: {path}")
            params = {k: archive[k] for k in ("normalizer", "reference_variance")}
            if axes[round_name].setdefault(cell["name"], params) != params:
                raise ValueError(f"Legacy parameter sharing mismatch: {path}")
    return C, identity, cells, config, sources, axes


def seal_calibration(C, identity, cells, sources, axes):
    if (HERE / "DESIGN_LOCK.json").exists() or (HERE / "CAL_RECEIPT.json").exists():
        raise FileExistsError("Study already locked")
    lock = HERE / "DESIGN_LOCK.json"
    S.write(lock, {"sources": sources, "cells": cells, "rounds": ROUNDS})
    models = {cell["name"]: cell for cell in cells}
    coefficients = {}
    for round_name, (master, _) in ROUNDS.items():
        z = rng(master, 101).normal(size=(N_CAL, 4))
        basis = np.column_stack((np.ones(N_CAL), z, quadratic(z)))
        coefficients[round_name] = {}
        for name, cell in models.items():
            readout, _ = B.LOAD.load_readout(cell, identity)
            v = B.apply_normalizer(readout(z), axes[round_name][name]["normalizer"])
            beta, _, rank, _ = np.linalg.lstsq(basis, v - z, rcond=None)
            if rank != 15 or not np.isfinite(beta).all():
                raise ValueError(f"Invalid CAL fit: {name}")
            coefficients[round_name][name] = beta.tolist()
    S.write(HERE / "CAL_COEFFICIENTS.json", coefficients)
    S.write(HERE / "CAL_RECEIPT.json", {
        "design_sha256": C.digest(lock),
        "coefficients_sha256": C.digest(HERE / "CAL_COEFFICIENTS.json"),
    })
    print("Sealed both CAL fits. No TEST generation in this stage.", flush=True)


def condition_coefficients(rates):
    time = np.arange(LENGTH)
    gamma = rates[None, None, :] ** np.abs(time[:, None, None] - time[None, :, None])
    cov_time_mean = gamma.mean(axis=1)
    variance = cov_time_mean.mean(axis=0)
    q = cov_time_mean / variance
    a, b = np.zeros((2, 4))
    for h in LAGS:
        summed = q[h:] + q[:-h]
        squared = (q[h:] - q[:-h])**2
        a += (summed * (2 * (1 - rates**h) - 3 * variance * squared)).mean(axis=0)
        b += (summed * squared).mean(axis=0)
    return a / len(LAGS), b / len(LAGS)


def component_cross(z, terms):
    """Evaluate 2 E[ΔZ Δf] with the legacy finite-window lag weights."""
    result = np.zeros((len(z), 4))
    for h in LAGS:
        result += 2 * ((z[:, h:] - z[:, :-h]) * (terms[:, h:] - terms[:, :-h])).mean(axis=1)
    return result / len(LAGS)


def verify_weighted(z, truth, beta, expected):
    weighted_midpoint = np.zeros_like(truth)
    for h in LAGS:
        a, b = z[:, :-h], z[:, h:]
        weighted_midpoint += (0.5 * (a + b) * (b - a)**2).mean(axis=1)
    weighted_midpoint /= len(LAGS) * truth
    np.testing.assert_allclose(4 * beta * truth * weighted_midpoint, expected, atol=5e-13, rtol=5e-12)


def evaluate_cell(z, polynomial_features, v, variance, rates, beta):
    truth, base, _, parts, _ = S.exact_parts(z, v, variance, LAGS)
    slow, fast = S.boundary_pairs(truth)
    mean = 2 * np.diag(beta[1:5]) * (2 * (1 - rates[None, :] ** np.array(LAGS)[:, None])).mean(axis=0)
    self_beta = np.diag(beta[5:9])
    window_mean = z.mean(axis=1)
    a, b = condition_coefficients(rates)
    self_mean = 2 * self_beta * (a * window_mean + b * window_mean**3)
    self_weighted = component_cross(z, polynomial_features[..., :4] * self_beta)
    verify_weighted(z, truth, self_beta, self_weighted)
    full = component_cross(z, polynomial_features @ beta[5:])
    k = parts["cross"]
    corrections = {"zero": np.zeros_like(k), "mean": np.broadcast_to(mean / variance, k.shape),
                   "self_mean": (mean + self_mean) / variance, "self_weighted": (mean + self_weighted) / variance,
                   "degree2": (mean + full) / variance, "oracle": k}
    truth_gap = S.boundary_scores(k, slow, fast)
    wrong_base = S.boundary_wrong(base, slow, fast)
    wrong_oracle = S.boundary_wrong(base - k, slow, fast)
    subsets = {"adverse": wrong_base & ~wrong_oracle, "helpful": ~wrong_base & wrong_oracle}
    truth_top = S.top2(truth)
    mse = {}
    statistics = {}
    for arm, correction in corrections.items():
        predicted_gap = S.boundary_scores(correction, slow, fast)
        errors = S.boundary_wrong(base - correction, slow, fast)
        exact = ~errors.any(axis=1)
        np.testing.assert_array_equal(exact, np.all(S.top2(base - correction) == truth_top, axis=1))
        sign = np.sign(predicted_gap) == np.sign(truth_gap)
        mse[arm] = ((predicted_gap - truth_gap)**2).mean(axis=1)
        statistics[arm] = {
            "top2_correct": int(exact.sum()), "top2_opportunities": len(z),
            "top2_repaired": int((wrong_base.any(axis=1) & exact).sum()),
            "top2_harmed": int((~wrong_base.any(axis=1) & ~exact).sum()),
            "correct_boundary_direction": int(sign.sum()), "boundary_opportunities": int(errors.size),
            "boundary_repaired": int((wrong_base & ~errors).sum()),
            "boundary_harmed": int((~wrong_base & errors).sum()),
            "adverse_total": int(subsets["adverse"].sum()),
            "adverse_sign_correct": int((subsets["adverse"] & sign).sum()),
            "adverse_flip_reproduced": int((subsets["adverse"] & ~errors).sum()),
            "helpful_total": int(subsets["helpful"].sum()),
            "helpful_sign_correct": int((subsets["helpful"] & sign).sum()),
            "helpful_flip_reproduced": int((subsets["helpful"] & errors).sum()),
        }
    return {"mse": mse, "counts": statistics}


def aggregate(records, resample=None):
    mse_rows = {arm: np.stack([record["mse"][arm] for record in records]).mean(axis=0) for arm in ARMS}
    result = {}
    for arm in ARMS:
        counts = {key: sum(record["counts"][arm][key] for record in records)
                  for key in records[0]["counts"][arm]}
        mse = mse_rows[arm].mean()
        mean_relative_gain = 1 - mse / mse_rows["mean"].mean()
        record = {**counts, "mse": float(mse), "vs_mean_relative_mse_gain": float(mean_relative_gain),
                  "vs_zero_relative_mse_gain": float(1 - mse / mse_rows["zero"].mean()),
                  "top2_accuracy": counts["top2_correct"] / counts["top2_opportunities"]}
        if resample is not None:
            gains = 1 - mse_rows[arm][resample].mean(axis=1) / mse_rows["mean"][resample].mean(axis=1)
            lower = float(np.quantile(gains, 0.05))
            record.update({"one_sided_95pct_relative_gain_lower": lower,
                           "passes_mse_gate": bool(mean_relative_gain >= MIN_GAIN and lower > 0)})
        result[arm] = record
    return result


def score(C, identity, cells, config, sources, axes):
    out = HERE / "SUMMARY.json"
    if out.exists():
        raise FileExistsError(out)
    lock = HERE / "DESIGN_LOCK.json"
    coefficients_path = HERE / "CAL_COEFFICIENTS.json"
    receipt_path = HERE / "CAL_RECEIPT.json"
    receipt = C.read(receipt_path)
    if (receipt != {"design_sha256": C.digest(lock), "coefficients_sha256": C.digest(coefficients_path)}
            or C.read(lock)["sources"] != sources):
        raise ValueError("Pre-TEST code, archive, or coefficient lock changed")
    coefficients = C.read(coefficients_path)
    summaries = {}
    for round_name, (master, bootstrap_seed) in ROUNDS.items():
        datasets = {}
        for process, rates in config["processes"].items():
            rates = np.asarray(rates)
            z = paths(master, rates)
            datasets[process] = (rates, z, quadratic(z))
        results = {}
        for cell in cells:
            name, process = cell["name"], cell["process"]
            rates, z, poly = datasets[process]
            readout, _ = B.LOAD.load_readout(cell, identity)
            legacy = axes[round_name][name]
            v = B.apply_normalizer(readout(z), legacy["normalizer"])
            results[(process, name)] = evaluate_cell(z, poly, v, np.asarray(legacy["reference_variance"]),
                                                      rates, np.asarray(coefficients[round_name][name]))
        resample = rng(bootstrap_seed, 103).integers(0, N_TEST, size=(BOOTSTRAPS, N_TEST))
        summaries[round_name] = {
            "pooled": aggregate(list(results.values()), resample),
            "by_process": {process: aggregate([r for (p, _), r in results.items() if p == process])
                           for process in datasets},
            "by_readout": {name: aggregate([r for (_, k), r in results.items() if k == name])
                           for name in coefficients[round_name]},
        }
        print(round_name, {arm: summaries[round_name]["pooled"][arm]["vs_mean_relative_mse_gain"]
                           for arm in ("self_mean", "self_weighted", "degree2")}, flush=True)
    primary = all(s["pooled"]["degree2"]["passes_mse_gate"] for s in summaries.values())
    secondary = primary and all(s["pooled"]["self_weighted"]["passes_mse_gate"] for s in summaries.values())
    S.write(out, {"cal_receipt_sha256": C.digest(receipt_path), "primary_pass": primary,
                  "hierarchical_secondary_pass": secondary, "rounds": summaries})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "score"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    C, identity, cells, config, sources, axes = design()
    if args.stage == "freeze":
        seal_calibration(C, identity, cells, sources, axes)
    else:
        score(C, identity, cells, config, sources, axes)


if __name__ == "__main__":
    main()
