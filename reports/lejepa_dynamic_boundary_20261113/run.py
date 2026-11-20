"""Seal prediction of dynamic cross terms before revealing physical TEST labels."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROLES = {"state_cal": 1, "reference": 2, "fit": 3, "validation": 4, "test": 5}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


LOAD = module("boundary_frozen_loaders", HERE.parent / "lejepa_natural_direction_repair_20261113/loaders.py")
M = module("boundary_cross_calibration", HERE / "model.py")


def write(path, values):
    with path.open("x") as handle:
        json.dump(values, handle, indent=2, allow_nan=False)
        handle.write("\n")


def rng(master, role):
    return np.random.default_rng(np.random.SeedSequence([master, ROLES[role]]))


def iids(master, role, n):
    return rng(master, role).normal(size=(n, 4))


def paths(master, role, n, length, rates):
    rates = np.asarray(rates, dtype=np.float64)
    if rates.shape != (4,) or np.any(np.abs(rates) >= 1):
        raise ValueError("Invalid AR process")
    innovation = rng(master, role).normal(size=(n, length, 4))
    z = innovation.copy()
    scale = np.sqrt(1 - rates**2)
    for t in range(1, length):
        z[:, t] = rates * z[:, t-1] + scale * innovation[:, t]
    return z


def design(smoke=False):
    C, identity, panel = LOAD.load_panel()
    cells = [cell for cell in panel if cell["layer"] == "mlp"]
    config = C.read(HERE / "protocol.json")
    if len(cells) != 28 or tuple(config["arms"]) != M.ARMS:
        raise ValueError("Unexpected frozen panel or method list")
    sources = LOAD.sources_manifest()
    for filename in ("run.py", "model.py", "score.py", "protocol.json", "PROTOCOL.md", "THEORY.md"):
        path = HERE / filename
        sources[str(path.relative_to(ROOT))] = C.digest(path)
    for path in sorted((HERE / "pilot").glob("*")):
        if path.is_file():
            sources[str(path.relative_to(ROOT))] = C.digest(path)
    if smoke:
        cells = cells[:1]
        config = {**config, "rounds": {"smoke": 2026111386}, "n_state_cal": 2048,
                  "n_reference": 512, "n_fit": 64, "n_validation": 32, "n_test": 16}
    return C, identity, cells, config, sources


def apply_normalizer(u, normalizer):
    return (u - normalizer["offset"]) / normalizer["scale"]


def calibration_expectations(parts, gram, fitted, variance, config, C):
    scores = M.predict(gram, fitted, variance, config)
    base = scores["base"]
    k, e = parts["cross"] / variance, parts["energy"] / variance
    ref = -parts["truth"] * (variance - 1) / variance
    scores.update({"oracle_cross": base-k, "oracle_energy": base-e, "oracle_reference": base-ref,
                   "oracle_cross_energy": base-k-e, "oracle_cross_abs": base-np.abs(k),
                   "oracle_cross_reverse": base+k})
    truth = C.top_two(parts["truth"])
    accuracy = {arm: float(np.all(C.top_two(value) == truth, axis=1).mean()) for arm, value in scores.items()}
    return {arm: {"accuracy": value, "gain": value-accuracy["base"]} for arm, value in accuracy.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    C, identity, cells, config, sources = design(args.smoke)
    base = HERE / "smoke" if args.smoke else HERE
    if args.smoke:
        base.mkdir(exist_ok=False)
    forecast = base / "forecast"
    forecast.mkdir(exist_ok=False)
    lock = base / "DESIGN_LOCK.json"
    write(lock, {"sources": sources, "config": config, "cells": cells})
    started = time.monotonic()
    for stage, master in config["rounds"].items():
        folder = forecast / stage
        folder.mkdir()
        cal = iids(master, "state_cal", config["n_state_cal"])
        ref = iids(master, "reference", config["n_reference"])
        normalizers = {}
        for index, cell in enumerate(cells):
            key = "__".join(cell[k] for k in ("layer", "process", "name"))
            readout, provenance = LOAD.load_readout(cell, identity)
            if cell["name"] not in normalizers:
                norm = M.normalize(cal, readout(cal))
                raw_ref = readout(ref)
                normalizers[cell["name"]] = (norm, raw_ref.var(axis=0), apply_normalizer(raw_ref, norm).var(axis=0))
            normalizer, raw_variance, variance = normalizers[cell["name"]]
            rates = config["processes"][cell["process"]]
            fit_z = paths(master, "fit", config["n_fit"], config["length"], rates)
            val_z = paths(master, "validation", config["n_validation"], config["length"], rates)
            fit_v = apply_normalizer(readout(fit_z), normalizer)
            val_v = apply_normalizer(readout(val_z), normalizer)
            fit_parts = M.decompose(fit_z, fit_v, config["lags"])
            val_parts = M.decompose(val_z, val_v, config["lags"])
            fit_g, val_g = M.grams(fit_v, config["lags"]), M.grams(val_v, config["lags"])
            fitted = M.fit_predictors(fit_g, fit_parts, val_g, val_parts, variance, config)
            expectation = calibration_expectations(val_parts, val_g, fitted, variance, config, C)
            # TEST states enter only through the frozen observation and readout before sealing.
            raw = readout(paths(master, "test", config["n_test"], config["length"], rates))
            gram = M.grams(apply_normalizer(raw, normalizer), config["lags"])
            scores_dict = M.predict(gram, fitted, variance, config)
            scores = np.stack([scores_dict[arm] for arm in config["arms"]])
            correction = M.predict_corrections(gram, fitted)
            raw_scores = M.grams(raw, config["lags"]).diagonal(axis1=1, axis2=2) / raw_variance
            np.testing.assert_allclose(scores[0], raw_scores, atol=2e-12, rtol=2e-12)
            write(folder / f"{key}.json", {"cell": cell, "normalizer": normalizer, "fit": fitted,
                                           "reference_variance": variance.tolist(), "provenance": provenance,
                                           "calibration_validation": expectation})
            with (folder / f"{key}.npz").open("xb") as handle:
                np.savez_compressed(handle, scores=scores, top2=np.stack([C.top_two(s) for s in scores]),
                                    gram=gram, predicted_cross=np.stack([correction[k] for k in ("cross_mean", "cross_diag", "cross_full")]),
                                    predicted_total=correction["total_full"], raw_scores=raw_scores,
                                    raw_reference_variance=raw_variance)
            if index % 7 == 0:
                print(f"sealed {stage}: {index+1}/{len(cells)}", flush=True)
    for relative, expected in sources.items():
        if C.digest(ROOT / relative) != expected:
            raise ValueError(f"Source changed while forecasting: {relative}")
    files = {str(path.relative_to(base)): C.digest(path) for path in sorted(forecast.rglob("*")) if path.is_file()}
    write(forecast / "RECEIPT.json", {"sources": sources, "files": files, "design_sha256": C.digest(lock),
                                    "rounds": config["rounds"], "elapsed_seconds": time.monotonic()-started})
    print("Both rounds sealed before independent physical truth scoring.", flush=True)


if __name__ == "__main__":
    main()
