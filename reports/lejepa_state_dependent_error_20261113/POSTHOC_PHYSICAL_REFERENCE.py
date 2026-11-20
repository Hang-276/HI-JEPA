"""Score an exact-physical-readout diagnostic after the original test reveal."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def top2(scores):
    return np.sort(np.argsort(scores, axis=-1, kind="stable")[:, :2], axis=1)


def score(values, lags):
    squared = [(values[:, lag:] - values[:, :-lag]) ** 2 for lag in lags]
    return np.mean([difference.mean(axis=1) for difference in squared], axis=0)


def iid_reference(master, count):
    # The original read-only loader's reference role is 2.
    random = np.random.default_rng(np.random.SeedSequence([master, 2]))
    return random.normal(size=(count, 4))


def estimate(matrix, bootstrap):
    window_average = np.asarray(matrix).mean(axis=0)
    return {
        "mean": float(window_average.mean()),
        "interval_95": np.quantile(bootstrap @ window_average, [0.025, 0.975]).tolist(),
    }


def diagnose():
    protocol = read(HERE / "protocol.json")
    design = read(HERE / "DESIGN_LOCK.json")
    forecast = read(HERE / "forecast/RECEIPT.json")
    summary = read(HERE / "SUMMARY.json")
    reveal_receipt = read(HERE / "REVEAL_RECEIPT.json")
    forecast_hash = digest(HERE / "forecast/RECEIPT.json")
    label_hash = digest(HERE / "REVEAL.npz")
    assert forecast_hash == summary["forecast_receipt_sha256"] == reveal_receipt["forecast_receipt_sha256"]
    assert label_hash == summary["labels_sha256"] == reveal_receipt["labels_sha256"]
    assert protocol == design["config"]
    assert digest(HERE / "DESIGN_LOCK.json") == reveal_receipt["design_sha256"] == forecast["design_sha256"]
    for name, sha in forecast["sources"].items():
        assert digest(REPOSITORY / name) == sha, name
    cells = design["cells"]
    assert len(cells) == 28 and all(cell["layer"] == "mlp" for cell in cells)
    n = protocol["n_test"]
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    bootstrap = rng.multinomial(n, np.full(n, 1 / n), size=protocol["bootstrap_draws"]) / n
    results = {}
    with np.load(HERE / "REVEAL.npz", allow_pickle=False) as labels:
        for stage, master in protocol["rounds"].items():
            physical_ref = iid_reference(master, protocol["n_reference"])
            reference_variance = physical_ref.var(axis=0)
            per_process = {}
            per_cell = {}
            archive_base, exact_correct = [], []
            for process in protocol["processes"]:
                states = labels[f"{stage}__{process}"]
                numerator = score(states, protocol["lags"])
                true_top = top2(numerator)
                exact_top = top2(numerator / reference_variance)
                exact = np.all(exact_top == true_top, axis=1)
                base_by_process = []
                names = [cell for cell in cells if cell["process"] == process]
                assert len(names) == 14
                for cell in names:
                    key = "__".join(cell[field] for field in ("layer", "process", "name"))
                    path = HERE / "forecast" / stage / f"{key}.npz"
                    assert digest(path) == forecast["files"][str(path.relative_to(HERE))]
                    with np.load(path, allow_pickle=False) as prediction:
                        base_top = top2(prediction["scores"][0])
                    base = np.all(base_top == true_top, axis=1)
                    per_cell[key] = {
                        "base_correct": int(base.sum()),
                        "physical_empirical_ref_correct": int(exact.sum()),
                        "fixes": int((exact & ~base).sum()),
                        "new_errors": int((~exact & base).sum()),
                    }
                    archive_base.append(base)
                    exact_correct.append(exact)
                    base_by_process.append(base)
                per_process[process] = {
                    "n_models": len(names),
                    "exact_readout_empirical_ref_accuracy": estimate([exact], bootstrap),
                    "base_accuracy": estimate(base_by_process, bootstrap),
                    "exact_readout_minus_base": estimate(
                        exact.astype(float)[None] - np.asarray(base_by_process), bootstrap
                    ),
                    "exact_readout_wrong_windows": int((~exact).sum()),
                }
            difference = np.asarray(exact_correct, dtype=float) - np.asarray(archive_base)
            results[stage] = {
                "reference_variances": reference_variance.tolist(),
                "exact_readout_empirical_ref_accuracy": estimate(exact_correct, bootstrap),
                "base_accuracy": estimate(archive_base, bootstrap),
                "exact_readout_minus_base": estimate(difference, bootstrap),
                "task_window_fixes": int((difference > 0).sum()),
                "task_window_new_errors": int((difference < 0).sum()),
                "processes": per_process,
                "cells": per_cell,
            }
    return {
        "status": "post-reveal explanatory diagnostic; physical truth required; no new seed or data budget",
        "sources": {
            name: digest(HERE / name)
            for name in ("protocol.json", "REVEAL.npz", "REVEAL_RECEIPT.json", "SUMMARY.json",
                         "forecast/RECEIPT.json", "DESIGN_LOCK.json")
        },
        "rounds": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    diagnostic = diagnose()
    if args.output is None:
        print(json.dumps(diagnostic, indent=2))
    else:
        with args.output.open("x") as handle:
            json.dump(diagnostic, handle, indent=2, allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
