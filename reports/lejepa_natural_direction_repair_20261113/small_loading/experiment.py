"""Freeze both small-loading forecasts, then reveal and score them independently."""

import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


original_path = sys.path[:]
try:
    sys.path.insert(0, str(PARENT))
    P = module("natural_repair_parent_forecast", PARENT / "run.py")
    S = module("natural_repair_parent_scoring", PARENT / "score.py")
finally:
    sys.path[:] = original_path

loaders = P.loaders
digest, read, write = S.digest, S.read, S.write


def design(smoke):
    config = read(HERE / "protocol.json")
    parent_config = read(PARENT / "protocol.json")
    allowed = {"rounds", "minimum_loading"}
    if set(config) != set(parent_config) or any(config[key] != val for key, val in parent_config.items() if key not in allowed):
        raise ValueError("unexpected change from the parent experiment")
    if config["minimum_loading"] != 0 or set(config["rounds"].values()) & set(parent_config["rounds"].values()):
        raise ValueError("wrong loading threshold or duplicate random stream")
    C, identity, cells = loaders.load_panel()
    panel = [cell for cell in cells if cell["layer"] == "mlp"]
    if len(panel) != 28:
        raise ValueError("unexpected frozen MLP panel")
    sources = loaders.sources_manifest()
    sources.update({
        str(path.relative_to(loaders.ROOT)): digest(path)
        for path in (
            Path(__file__), HERE / "protocol.json", HERE / "PROTOCOL.md",
            PARENT / "run.py", PARENT / "score.py", PARENT / "protocol.json", PARENT / "PROTOCOL.md",
        )
    })
    if smoke:
        panel = panel[:1]
        config = {
            **config, "rounds": {"smoke": 2026111395}, "n_state_cal": 512,
            "n_reference": 512, "n_audit_state": 512, "n_test": 16,
        }
    return C, identity, panel, config, sources


def forecast(base, smoke):
    C, identity, panel, config, sources = design(smoke)
    if smoke:
        base.mkdir(exist_ok=False)
    forecast_root = base / "forecast"
    forecast_root.mkdir(exist_ok=False)
    lock = base / "DESIGN_LOCK.json"
    write(lock, {"sources": sources, "config": config, "cells": panel})
    start = time.monotonic()
    for stage, master in config["rounds"].items():
        folder = forecast_root / stage
        folder.mkdir()
        cal = loaders.iids(master, "state_cal", config["n_state_cal"])
        ref = loaders.iids(master, "reference", config["n_reference"])
        test = {
            process: loaders.paths(master, "test", config["n_test"], config["length"], rates)
            for process, rates in config["processes"].items()
        }
        for cell in panel:
            key = S.cell_key(cell)
            readout, provenance = loaders.load_readout(cell, identity)
            fit = P.fit_direction(cal, readout(cal), config)
            visible, reference = readout(test[cell["process"]]), readout(ref)
            scores, gram, cov, variance, raw_var, raw_scores = P.predict(
                visible, reference, fit, config, C.calculate_moments
            )
            write(folder / f"{key}.json", {
                "cell": cell, **fit, "reference_covariance": cov.tolist(),
                "reference_variance": variance.tolist(), "provenance": provenance,
            })
            with (folder / f"{key}.npz").open("xb") as handle:
                np.savez_compressed(
                    handle, scores=scores, moments=gram, top2=np.stack([C.top_two(s) for s in scores]),
                    raw_reference_variance=raw_var, raw_scores=raw_scores,
                )
        print(f"{stage}: sealed {len(panel)} MLP predictions", flush=True)
    for path, expected in sources.items():
        if digest(loaders.ROOT / path) != expected:
            raise ValueError(f"source changed while forecasting: {path}")
    files = {str(path.relative_to(base)): digest(path) for path in sorted(forecast_root.rglob("*")) if path.is_file()}
    write(forecast_root / "RECEIPT.json", {
        "files": files, "sources": sources, "design_sha256": digest(lock),
        "rounds": config["rounds"], "elapsed_seconds": time.monotonic() - start,
    })


def verify(base, smoke):
    _, identity, cells, config, sources = design(smoke)
    lock = base / "DESIGN_LOCK.json"
    receipt_path = base / "forecast/RECEIPT.json"
    receipt = read(receipt_path)
    if digest(lock) != receipt["design_sha256"] or read(lock) != {"sources": sources, "config": config, "cells": cells}:
        raise ValueError("unrelated or changed experiment design")
    if receipt["sources"] != sources or receipt["rounds"] != config["rounds"]:
        raise ValueError("changed frozen dependency or rounds")
    expected = {
        f"forecast/{stage}/{S.cell_key(cell)}.{suffix}"
        for stage in config["rounds"] for cell in cells for suffix in ("json", "npz")
    }
    if set(receipt["files"]) != expected:
        raise ValueError("incomplete or extra prediction files")
    for path, sha in receipt["files"].items():
        if digest(base / path) != sha:
            raise ValueError(f"changed prediction: {path}")
    return identity, cells, config, receipt_path


def reveal(base, smoke):
    _, _, config, receipt = verify(base, smoke)
    label_path = base / "REVEAL.npz"
    labels = {
        S.truth_key(stage, process): loaders.paths(master, "test", config["n_test"], config["length"], rates)
        for stage, master in config["rounds"].items() for process, rates in config["processes"].items()
    }
    with label_path.open("xb") as output:
        np.savez_compressed(output, **labels)
    write(base / "REVEAL_RECEIPT.json", {
        "forecast_receipt_sha256": digest(receipt), "reveal_sha256": digest(label_path),
        "design_sha256": digest(base / "DESIGN_LOCK.json"), "label_keys": sorted(labels),
    })


def score(base, smoke):
    identity, cells, config, receipt_path = verify(base, smoke)
    label_path = base / "REVEAL.npz"
    receipt = read(base / "REVEAL_RECEIPT.json")
    if (digest(receipt_path) != receipt["forecast_receipt_sha256"]
            or digest(label_path) != receipt["reveal_sha256"]
            or digest(base / "DESIGN_LOCK.json") != receipt["design_sha256"]):
        raise ValueError("unrelated physical reveal")
    start = time.monotonic()
    count = config["n_test"]
    bootstrap = np.random.default_rng(config["bootstrap_seed"]).multinomial(
        count, np.full(count, 1 / count), size=config["bootstrap_draws"]
    ) / count
    results = {}
    with np.load(label_path, allow_pickle=False) as labels:
        if sorted(labels.files) != receipt["label_keys"]:
            raise ValueError("unexpected physical label keys")
        for stage, master in config["rounds"].items():
            ref = loaders.iids(master, "reference", config["n_reference"])
            independent = loaders.iids(master, "audit_state", config["n_audit_state"])
            truth = {process: labels[S.truth_key(stage, process)] for process in config["processes"]}
            records, correctness = {}, {}
            for cell in cells:
                key = S.cell_key(cell)
                records[key], correctness[key] = S.score_cell(
                    cell, stage, config, identity, truth[cell["process"]], ref, independent, base / "forecast"
                )
            overall = S.aggregate(records, correctness, "mlp", config, bootstrap)
            results[stage] = {
                "all": overall,
                "active": S.aggregate(records, correctness, "mlp", config, bootstrap, restrict_active=True),
                "process": {
                    process: S.aggregate(records, correctness, "mlp", config, bootstrap, process=process)
                    for process in config["processes"]
                },
                "gain_below_minimum_with_suppression": (
                    overall["active_count"] > 0
                    and overall["suppression_share"] is not None
                    and overall["suppression_share"] >= config["minimum_suppression_share"]
                    and overall["paired_target_minus"]["base"]["interval_95"][1] < config["minimum_gain"]
                ),
                "cells": records,
            }
            print(f"{stage}: scored {len(records)} fixed MLP cells", flush=True)
    write(base / "SUMMARY.json", {
        "forecast_receipt_sha256": digest(receipt_path), "reveal_sha256": digest(label_path),
        "gain_below_minimum_with_suppression_both_rounds": all(
            stage["gain_below_minimum_with_suppression"] for stage in results.values()
        ),
        "elapsed_seconds": time.monotonic() - start,
        "results": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("forecast", "reveal", "score"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = HERE / "smoke" if args.smoke else HERE
    {"forecast": forecast, "reveal": reveal, "score": score}[args.stage](base, args.smoke)


if __name__ == "__main__":
    main()
