"""Lock common gap thresholds using archived calibration streams, then seal TEST outputs."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A4 = HERE.parent / "lejepa_a4_20261115"
sys.path.insert(0, str(A4))
import forecast as a4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def score_readout(readout, states, variance, lags):
    # Restrict intermediate memory while preserving the original within-window means.
    return np.concatenate([a4.slow_scores(readout(states[i:i+256]), lags) / variance
                           for i in range(0, len(states), 256)])


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config = read(HERE / "config.json")
    old_receipt_path = A4 / "NEW_FORECAST.json"
    old_receipt = read(old_receipt_path)
    old_config = old_receipt["config"]
    old_sources = {os.path.relpath((A4 / path).resolve(), HERE): sha
                   for path, sha in old_receipt["sources"].items()}
    sources = {**old_sources, os.path.relpath(old_receipt_path, HERE): digest(old_receipt_path)}
    for name in ("config.json", "PROTOCOL.md", "forecast.py", "score.py"):
        sources[name] = digest(HERE / name)
    for path, sha in sources.items():
        if digest(HERE / path) != sha:
            raise ValueError(f"Changed source: {path}")
    for key in ("processes", "training_cases", "seeds", "length", "lags"):
        if old_config[key] != config[key]:
            raise ValueError(f"A4 settings do not match: {key}")
    if config["test_role"] in (old_config["cal_role"], old_config["reference_role"], old_config["test_role"]):
        raise ValueError("A3 test streams must be independent of A4")
    expected_cells = {(f"{case}_{seed}", process) for case in config["training_cases"]
                      for seed in config["seeds"] for process in config["processes"]}
    old_cells = old_receipt["cells"]
    if len(old_cells) != len(expected_cells) or {(c["name"], c["process"]) for c in old_cells} != expected_cells:
        raise ValueError("Wrong fixed model panel")
    a4.write(HERE / "DESIGN_LOCK.json", {"config": config, "sources": sources})
    sources["DESIGN_LOCK.json"] = digest(HERE / "DESIGN_LOCK.json")

    readouts = {name: a4.panel.load_readout(name)[0] for name in sorted({c["name"] for c in old_cells})}
    cal_gaps = {}
    for cell in old_cells:
        name, process, master = (cell[field] for field in ("name", "process", "test_master"))
        cal = a4.trajectories(old_config, master, old_config["cal_role"], old_config["n_cal"],
                              config["processes"][process])
        scores = score_readout(readouts[name], cal, np.asarray(cell["reference_variance"]), config["lags"])
        sorted_scores = np.sort(scores, axis=1)
        cal_gaps[f"{name}__{process}"] = sorted_scores[:, 2] - sorted_scores[:, 1]
    pooled = np.concatenate(list(cal_gaps.values()))
    thresholds = {name: float(np.quantile(pooled, level, method="higher"))
                  for name, level in config["quantiles"].items()}
    if len(set(thresholds.values())) != len(thresholds) or not all(x > 0 for x in thresholds.values()):
        raise ValueError("Expected positive, distinct thresholds")
    data_path = HERE / "CAL_GAPS.npz"
    with data_path.open("xb") as handle:
        np.savez_compressed(handle, **cal_gaps)
    sources[data_path.name] = digest(data_path)
    lock = HERE / "THRESHOLDS.json"
    a4.write(lock, {"thresholds": thresholds, "n_cal_per_cell": old_config["n_cal"],
                   "coverage_cal": {name: float(np.mean(pooled >= value)) for name, value in thresholds.items()},
                   "sources": sources})
    sources[lock.name] = digest(lock)
    print(f"Locked CAL thresholds {thresholds}", flush=True)

    folder = HERE / "forecast"
    folder.mkdir(exist_ok=False)
    results = []
    for cell in old_cells:
        name, process, master = (cell[field] for field in ("name", "process", "test_master"))
        raw_inputs = a4.trajectories(config, master, config["test_role"], config["n_test"],
                                     config["processes"][process])
        scores = score_readout(readouts[name], raw_inputs, np.asarray(cell["reference_variance"]), config["lags"])
        path = folder / f"{name}__{process}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, predicted_scores=scores)
        results.append({**{key: cell[key] for key in ("name", "seed", "training_case", "process", "test_master")},
                        "path": str(path.relative_to(HERE)), "sha256": digest(path)})
        print(f"Sealed {name} / {process}", flush=True)
    for path, sha in sources.items():
        if digest(HERE / path) != sha:
            raise ValueError(f"Changed source: {path}")
    a4.write(HERE / "FORECAST.json", {"config": config, "thresholds": thresholds, "cells": results,
                                   "sources": sources})


if __name__ == "__main__":
    main()
