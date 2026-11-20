"""Freeze calibration and both visible ranking scores before TEST scoring."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from panel import run as panel


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def generator(master, role):
    return np.random.default_rng(np.random.SeedSequence([master, role]))


def trajectories(config, master, role, count, rates):
    innovations = generator(master, role).normal(size=(count, config["length"], 4))
    values = innovations.copy()
    rho = np.asarray(rates)
    scale = np.sqrt(1 - rho**2)
    for time in range(1, config["length"]):
        values[:, time] = rho * values[:, time - 1] + scale * innovations[:, time]
    return values


def slow_scores(paths, lags):
    return np.mean([np.mean((paths[:, h:] - paths[:, :-h])**2, axis=1) for h in lags], axis=0)


def ordered_scores(scores):
    order = np.argsort(scores, axis=1, kind="stable")
    selected = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(selected, order[:, :2], True, axis=1)
    pairs = selected[:, :, None] & ~selected[:, None, :]
    gaps = scores[:, None, :] - scores[:, :, None]
    return order, pairs, gaps


def calibrate(predicted, physical, config):
    _, pairs, model_gaps = ordered_scores(predicted)
    overstatement = model_gaps - (physical[:, None, :] - physical[:, :, None])
    allowances = np.zeros((4, 4))
    counts = pairs.sum(axis=0)
    fallback = np.zeros((4, 4), dtype=bool)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            values = overstatement[:, i, j]
            fallback[i, j] = counts[i, j] < config["minimum_pair_cal"]
            if not fallback[i, j]:
                values = values[pairs[:, i, j]]
            allowances[i, j] = max(0, np.quantile(values, config["quantile"], method="higher"))
    return allowances, counts, fallback


def population_inputs(config, master, rates):
    reference = generator(master, config["reference_role"]).normal(size=(config["n_reference"], 4))
    cal = trajectories(config, master, config["cal_role"], config["n_cal"], rates)
    return reference, cal


def model_calibration(readout, reference, cal, config):
    variance = readout(reference).var(axis=0)
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("Undefined reference variance")
    cal_prediction = slow_scores(readout(cal), config["lags"]) / variance
    cal_physical = slow_scores(cal, config["lags"])
    allowances, counts, fallback = calibrate(cal_prediction, cal_physical, config)
    return variance, allowances, counts, fallback


def old_development(config):
    archive = ROOT / "reports/lejepa_natural_direction_repair_20261113/loaders.py"
    spec = importlib.util.spec_from_file_location("a4_old_readouts", archive)
    loaders = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaders)
    _, identity, cells = loaders.load_panel()
    diagnostics = []
    for offset, (process, rates) in enumerate(config["processes"].items()):
        reference, cal = population_inputs(config, config["old_master"] + offset, rates)
        matched = [cell for cell in cells if cell["layer"] == "mlp" and cell["process"] == process]
        if len(matched) != 14:
            raise ValueError("Unexpected development panel")
        for cell in matched:
            readout, _ = loaders.load_readout(cell, identity)
            variance, allowance, counts, fallback = model_calibration(readout, reference, cal, config)
            diagnostics.append({"name": cell["name"], "process": process, "reference_variance": variance.tolist(),
                                "allowance": allowance.tolist(), "cal_pair_counts": counts.tolist(),
                                "fallback": fallback.tolist()})
    write(HERE / "OLD_DEV.json", {"config": config, "calibration_only": diagnostics})
    print("Locked rule evaluated on old calibration streams only.", flush=True)


def new_forecast(config):
    known = panel.sources_manifest()
    sources = {os.path.relpath(ROOT / path, HERE): sha for path, sha in known.items()}
    paths = ("forecast.py", "config.json", "PROTOCOL.md", "score.py", "OLD_DEV.json")
    sources.update({name: digest(HERE / name) for name in paths})
    write(HERE / "FORECAST_LOCK.json", {"config": config, "sources": sources})
    sources["FORECAST_LOCK.json"] = digest(HERE / "FORECAST_LOCK.json")
    folder = HERE / "new_forecast"
    folder.mkdir(exist_ok=False)
    cells = []
    for seed in config["seeds"]:
        for offset, (process, rates) in enumerate(config["processes"].items()):
            master = config["new_master_base"] + seed * 10 + offset
            reference, cal = population_inputs(config, master, rates)
            visible_inputs = trajectories(config, master, config["test_role"], config["n_test"], rates)
            for training_case in config["training_cases"]:
                name = f"{training_case}_{seed}"
                readout, hashes = panel.load_readout(name)
                if any(known.get(path) != sha for path, sha in hashes.items()):
                    raise ValueError(f"Model identity mismatch: {name}")
                result = json.loads((HERE / "panel" / "runs" / f"{name}.json").read_text())
                state_r2 = result["audit_r2"]
                variance, allowance, counts, fallback = model_calibration(readout, reference, cal, config)
                forecast = slow_scores(readout(visible_inputs), config["lags"]) / variance
                order, pairs, gaps = ordered_scores(forecast)
                gap = np.where(pairs, gaps, np.inf).min(axis=(1, 2))
                adjusted = np.where(pairs, gaps - allowance, np.inf).min(axis=(1, 2))
                path = folder / f"{name}__{process}.npz"
                with path.open("xb") as handle:
                    np.savez_compressed(handle, predicted_scores=forecast, gap=gap, adjusted_gap=adjusted,
                                        top2=np.sort(order[:, :2], axis=1))
                cells.append({"name": name, "seed": seed, "training_case": training_case, "process": process,
                              "eligible": min(state_r2) >= config["minimum_audit_r2"], "state_r2": state_r2,
                              "path": str(path.relative_to(HERE)), "sha256": digest(path), "test_master": master,
                              "reference_variance": variance.tolist(), "allowance": allowance.tolist(),
                              "cal_pair_counts": counts.tolist(), "fallback": fallback.tolist()})
            print(f"Sealed scores for {seed} / {process}", flush=True)
    for path, sha in sources.items():
        if digest(HERE / path) != sha:
            raise ValueError(f"Changed source: {path}")
    write(HERE / "NEW_FORECAST.json", {"config": config, "cells": cells, "sources": sources})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("old", "new"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cfg = json.loads((HERE / "config.json").read_text())
    if args.phase == "old":
        old_development(cfg)
    else:
        new_forecast(cfg)
