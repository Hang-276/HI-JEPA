"""Seal both rounds of observable state-dependent residual corrections."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


loaders = module("state_error_frozen_loaders", HERE.parent / "lejepa_natural_direction_repair_20261113/loaders.py")
M = module("state_error_calibration_model", HERE / "model.py")


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def design(smoke=False):
    C, identity, cells = loaders.load_panel()
    cells = [cell for cell in cells if cell["layer"] == "mlp"]
    if len(cells) != 28:
        raise ValueError("Expected exactly the 14 historical strong decoders in two processes")
    config = C.read(HERE / "protocol.json")
    if config["arms"] != list(M.ARMS):
        raise ValueError("Protocol arm order differs from the calibration implementation")
    sources = loaders.sources_manifest()
    for filename in ("run.py", "model.py", "score.py", "protocol.json", "PROTOCOL.md", "THEORY.md", "LEJEPA_SCOPE.md"):
        path = HERE / filename
        sources[str(path.relative_to(ROOT))] = C.digest(path)
    for path in sorted((HERE / "pilot").glob("*")):
        if path.is_file():
            sources[str(path.relative_to(ROOT))] = C.digest(path)
    if smoke:
        cells = cells[:1]
        config = {**config, "rounds": {"smoke": 2026111399}, "n_state_cal": 4096,
                  "n_reference": 512, "n_audit_state": 4096, "n_test": 16}
    return C, identity, cells, config, sources


def scores_for(visible, reference, fit, config, C):
    outputs = M.outputs(visible, fit, config)
    refs = M.outputs(reference, fit, config)
    numerators = np.stack([C.calculate_moments(outputs[name], config["lags"]).diagonal(axis1=1, axis2=2)
                           for name in config["arms"]])
    variance = np.stack([refs[name].var(axis=0) for name in config["arms"]])
    if np.any(variance <= 0):
        raise ValueError("Nonpositive reference variance")
    scores = numerators / variance[:, None, :]
    raw_variance = reference.var(axis=0)
    raw_scores = C.calculate_moments(visible, config["lags"]).diagonal(axis1=1, axis2=2) / raw_variance
    np.testing.assert_allclose(scores[0], raw_scores, atol=1e-11, rtol=1e-11)
    return scores, numerators, variance, raw_scores, raw_variance


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
    folder = base / "forecast"
    folder.mkdir(exist_ok=False)
    design_path = base / "DESIGN_LOCK.json"
    write(design_path, {"sources": sources, "config": config, "cells": cells})
    started = time.monotonic()
    for stage, master in config["rounds"].items():
        stage_dir = folder / stage
        stage_dir.mkdir()
        cal = loaders.iids(master, "state_cal", config["n_state_cal"])
        ref = loaders.iids(master, "reference", config["n_reference"])
        fitted = {}
        for index, cell in enumerate(cells):
            key = "__".join(cell[k] for k in ("layer", "process", "name"))
            readout, provenance = loaders.load_readout(cell, identity)
            if cell["name"] not in fitted:
                fitted[cell["name"]] = (M.fit_calibration(cal, readout(cal), config), readout(ref))
            fit, reference = fitted[cell["name"]]
            # Simulator truth passes only through the frozen observation/readout in this stage.
            visible = readout(loaders.paths(master, "test", config["n_test"], config["length"],
                                           config["processes"][cell["process"]]))
            scores, numerator, variance, raw_scores, raw_var = scores_for(visible, reference, fit, config, C)
            write(stage_dir / f"{key}.json", {"cell": cell, "fit": fit, "provenance": provenance,
                                               "reference_variance": variance.tolist()})
            with (stage_dir / f"{key}.npz").open("xb") as handle:
                np.savez_compressed(handle, scores=scores, numerators=numerator,
                                    top2=np.stack([C.top_two(s) for s in scores]),
                                    raw_scores=raw_scores, raw_reference_variance=raw_var)
            if index % 7 == 0:
                print(f"sealed {stage}: {index+1}/{len(cells)}", flush=True)
    for relative, expected in sources.items():
        if C.digest(ROOT / relative) != expected:
            raise ValueError(f"Source changed during forecasting: {relative}")
    files = {str(path.relative_to(base)): C.digest(path)
             for path in sorted(folder.rglob("*")) if path.is_file()}
    write(folder / "RECEIPT.json", {"sources": sources, "files": files,
                                   "design_sha256": C.digest(design_path), "rounds": config["rounds"],
                                   "elapsed_seconds": time.monotonic()-started})
    print("Both rounds sealed before physical truth scoring.", flush=True)


if __name__ == "__main__":
    main()
