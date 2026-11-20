"""Seal fixed-threshold A2 scores and the independent readout quality audit."""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A4 = HERE.parent / "lejepa_a4_20261115"
A3 = HERE.parent / "lejepa_a3_gap_20261115"
sys.path.insert(0, str(A4))
import forecast as a4


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_readout(readout, states, variance, lags):
    return np.concatenate([a4.slow_scores(readout(states[i:i+256]), lags) / variance
                           for i in range(0, len(states), 256)])


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    weak = load_module(HERE / "weak/run.py", "a2_weak_readout")
    config = a4.panel.read(HERE / "config.json")
    prior_path = A4 / "NEW_FORECAST.json"
    prior = a4.panel.read(prior_path)
    prior_config = prior["config"]
    old_thresholds = a4.panel.read(A3 / "THRESHOLDS.json")["thresholds"]
    if config["threshold"] != old_thresholds["g50"]:
        raise ValueError("A2 must use exactly the passing A3 threshold")
    for key in ("processes", "training_cases", "seeds", "length", "lags"):
        if config[key] != prior_config[key]:
            raise ValueError(f"A4 settings do not match: {key}")
    old_roles = {prior_config[k] for k in ("test_role", "reference_role", "cal_role")}
    old_roles.add(a4.panel.read(A3 / "config.json")["test_role"])
    if config["test_role"] in old_roles:
        raise ValueError("Expected independent A2 TEST streams")
    sources = {os.path.relpath(ROOT / path, HERE): digest
               for path, digest in weak.sources_manifest().items()}
    sources.update({os.path.relpath((A4 / path).resolve(), HERE): digest
                    for path, digest in prior["sources"].items()})
    for path in (HERE / "config.json", HERE / "PROTOCOL.md", HERE / "forecast.py", HERE / "score.py",
                 A3 / "THRESHOLDS.json", A3 / "score.py", A3 / "config.json", prior_path):
        sources[os.path.relpath(path, HERE)] = a4.digest(path)
    for path, digest in sources.items():
        if a4.digest(HERE / path) != digest:
            raise ValueError(f"Origin digest changed: {path}")
    source_lock = HERE / "DESIGN_LOCK.json"
    a4.write(source_lock, {"config": config, "sources": sources})
    sources[source_lock.name] = a4.digest(source_lock)

    models = {f"{case}_{seed}" for seed in config["seeds"] for case in config["training_cases"]}
    weak_receipt = a4.panel.read(HERE / "weak/RECEIPT.json")["members"]
    if set(weak_receipt) != models or {(c["name"], c["process"]) for c in prior["cells"]} != {
        (name, process) for name in models for process in config["processes"]
    }:
        raise ValueError("Unexpected encoder or evaluation process list")
    audits = {}
    for name, member in weak_receipt.items():
        scores = member["audit"]
        strong = min(scores["strong_r2"])
        low = min(scores["weak_r2"])
        drop = strong - low
        audits[name] = {"strong_min_r2": strong, "weak_min_r2": low, "drop": drop,
                        "passes": drop >= config["minimum_r2_drop"] and low >= config["minimum_weak_r2"]}
    qualified = sum(member["passes"] for member in audits.values())
    audit_path = HERE / "AUDIT_GATE.json"
    a4.write(audit_path, {"qualified": qualified, "n_models": len(models), "by_model": audits,
                         "passes": qualified >= config["minimum_separated_encoders"]})
    sources[audit_path.name] = a4.digest(audit_path)
    print(f"Recovery manipulation: {qualified}/{len(models)} meet the prespecified contrast.", flush=True)

    readouts = {(name, arm): loader(name)[0] for name in sorted(models)
                for arm, loader in (("strong", a4.panel.load_readout), ("weak", weak.load_readout))}
    folder = HERE / "forecast"
    folder.mkdir(exist_ok=False)
    cells = []
    for old_cell in prior["cells"]:
        name, process, master = (old_cell[k] for k in ("name", "process", "test_master"))
        reference = a4.generator(master, prior_config["reference_role"]).normal(size=(prior_config["n_reference"], 4))
        states = a4.trajectories(config, master, config["test_role"], config["n_test"],
                                 config["processes"][process])
        for arm in ("strong", "weak"):
            readout = readouts[name, arm]
            variance = readout(reference).var(axis=0)
            if (variance <= 0).any() or not np.isfinite(variance).all():
                raise ValueError(f"Invalid reference variance: {name}, {arm}")
            if arm == "strong":
                sealed = np.asarray(old_cell["reference_variance"])
                np.testing.assert_allclose(variance, sealed, rtol=2e-11, atol=2e-11)
                variance = sealed
            predicted = score_readout(readout, states, variance, config["lags"])
            path = folder / f"{name}__{process}__{arm}.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, predicted_scores=predicted)
            cells.append({**{key: old_cell[key] for key in ("name", "seed", "process", "training_case", "test_master")},
                          "arm": arm, "path": str(path.relative_to(HERE)), "sha256": a4.digest(path),
                          "reference_variance": variance.tolist()})
        print(f"Sealed paired predictions for {name} / {process}.", flush=True)
    for path, digest in sources.items():
        if a4.digest(HERE / path) != digest:
            raise ValueError(f"Changed source: {path}")
    a4.write(HERE / "FORECAST.json", {"config": config, "cells": cells, "sources": sources})


if __name__ == "__main__":
    main()
