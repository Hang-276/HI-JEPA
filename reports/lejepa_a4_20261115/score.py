"""Independent matched-coverage scoring after a sealed A4 forecast."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ("nominal", "adjusted")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def checked_path(relative, parent):
    path = Path(relative)
    full = (HERE / path).resolve()
    if path.is_absolute() or not full.is_relative_to(parent):
        raise ValueError(f"Artifact path outside {parent}: {relative}")
    return full


def top_two(values):
    return np.sort(np.argsort(values, axis=1, kind="stable")[:, :2], axis=1)


def adjusted_boundary_gap(scores, chosen, allowance):
    selected = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(selected, chosen, True, axis=1)
    boundary = selected[:, :, None] & ~selected[:, None, :]
    differences = scores[:, None, :] - scores[:, :, None]
    return np.min(np.where(boundary, differences - allowance, np.inf), axis=(1, 2))


def verify():
    receipt_path = HERE / "NEW_FORECAST.json"
    receipt = read(receipt_path)
    config = read(HERE / "config.json")
    if receipt["config"] != config:
        raise ValueError("Forecast config differs from the local locked config")
    sources = {checked_path(relative, ROOT): expected for relative, expected in receipt["sources"].items()}
    panel_path = HERE / "panel/RECEIPT.json"
    required = {HERE / "score.py", HERE / "config.json", HERE / "PROTOCOL.md", panel_path}
    if not required <= set(sources):
        raise ValueError("Forecast sources must bind the scoring code, protocol, config and new panel receipt")
    for path, expected in sources.items():
        if digest(path) != expected:
            raise ValueError(f"Forecast source digest mismatch: {path}")

    n_test = int(config["n_test"])
    expected_models = {f"{case}_{seed}": (seed, case) for seed in config["seeds"] for case in config["training_cases"]}
    panel = read(panel_path)["members"]
    if set(panel) != set(expected_models):
        raise ValueError("Training receipt does not contain the entire predeclared model panel")
    cells = {}
    models = {}
    master_by_pair = {}
    for cell in receipt["cells"]:
        name, process = cell["name"], cell["process"]
        key = (name, process)
        if (key in cells or name not in expected_models or process not in config["processes"]
                or type(cell["eligible"]) is not bool or type(cell["test_master"]) is not int):
            raise ValueError(f"Invalid or repeated forecast cell: {key}")
        model = (int(cell["seed"]), cell["training_case"], cell["eligible"])
        audit_r2 = np.asarray(panel[name]["audit_r2"])
        if (model[:2] != expected_models[name] or model[:2] != (panel[name]["seed"], panel[name]["case"])
                or audit_r2.shape != (4,) or not np.isfinite(audit_r2).all()
                or model[2] != bool(audit_r2.min() >= config["minimum_audit_r2"])):
            raise ValueError(f"Forecast identity or eligibility differs from sealed state audit: {name}")
        lodged_r2 = np.asarray(cell["state_r2"])
        expected_r2 = audit_r2.min() if lodged_r2.ndim == 0 else audit_r2
        np.testing.assert_array_equal(lodged_r2, expected_r2)
        allowance, reference = (np.asarray(cell[k]) for k in ("allowance", "reference_variance"))
        if (allowance.shape != (4, 4) or not np.isfinite(allowance).all() or np.any(allowance < 0)
                or np.any(allowance.diagonal() != 0) or reference.shape != (4,)
                or not np.isfinite(reference).all() or np.any(reference <= 0)):
            raise ValueError(f"Invalid calibration allowance or REF: {key}")
        if models.setdefault(name, model) != model:
            raise ValueError(f"Inconsistent model identity or eligibility: {name}")
        pairing = (model[0], process)
        if master_by_pair.setdefault(pairing, int(cell["test_master"])) != cell["test_master"]:
            raise ValueError(f"Forecast does not preserve intended TEST pairing: {pairing}")

        path = checked_path(cell["path"], HERE / "new_forecast")
        if digest(path) != cell["sha256"]:
            raise ValueError(f"Forecast array digest mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"predicted_scores", "gap", "adjusted_gap", "top2"}:
                raise ValueError(f"Unexpected data in sealed prediction: {path}")
            predicted, gap, adjusted, chosen = (archive[k] for k in ("predicted_scores", "gap", "adjusted_gap", "top2"))
        if (predicted.shape != (n_test, 4) or gap.shape != (n_test,) or adjusted.shape != (n_test,)
                or chosen.shape != (n_test, 2) or not np.issubdtype(chosen.dtype, np.integer)
                or not all(np.isfinite(a).all() for a in (predicted, gap, adjusted))):
            raise ValueError(f"Invalid prediction arrays: {path}")
        np.testing.assert_array_equal(chosen, top_two(predicted))
        ordered = np.sort(predicted, axis=1)
        np.testing.assert_allclose(gap, ordered[:, 2] - ordered[:, 1], rtol=2e-7, atol=2e-12)
        np.testing.assert_allclose(adjusted, adjusted_boundary_gap(predicted, chosen, allowance), rtol=2e-7, atol=2e-12)
        ranks = np.stack([np.argsort(-values, kind="stable") for values in (gap, adjusted)])
        cells[key] = {"metadata": cell, "chosen": chosen, "ranks": ranks}

    expected = {(name, process) for name in expected_models for process in config["processes"]}
    if cells.keys() != expected or len(set(master_by_pair.values())) != len(master_by_pair):
        raise ValueError("Unbalanced model × process forecast grid")
    return config, cells, receipt_path


def physical_scores(config, master, process):
    rates = np.asarray(config["processes"][process])
    rng = np.random.default_rng(np.random.SeedSequence([master, config["test_role"]]))
    innovations = rng.normal(size=(config["n_test"], config["length"], 4))
    states = innovations.copy()
    scale = np.sqrt(1 - rates**2)
    for time in range(1, config["length"]):
        states[:, time] = rates * states[:, time - 1] + scale * innovations[:, time]
    return sum(((states[:, lag:] - states[:, :-lag])**2).mean(axis=1)
               for lag in config["lags"]) / len(config["lags"])


def summarize_cell(cell, truth, keep):
    wrong = (cell["chosen"] != top_two(truth)).any(axis=1)
    accepts = np.zeros((len(ARMS), len(wrong)), dtype=bool)
    np.put_along_axis(accepts, cell["ranks"][:, :keep], True, axis=1)
    shared = accepts.all(axis=0)
    arm_only = accepts & ~shared
    counts = {name: {"accepted": keep, "wrong_accepted": int((accepts[k] & wrong).sum()),
                     "exclusive_accepted": int(arm_only[k].sum()),
                     "wrong_exclusive_accepted": int((arm_only[k] & wrong).sum())}
              for k, name in enumerate(ARMS)}
    return wrong, {"windows": len(wrong), "wrong_all": int(wrong.sum()), "shared_accepted": int(shared.sum()),
                   "wrong_shared_accepted": int((shared & wrong).sum()), "rules": counts}


def wrong_in_ranked_resample(weights, rank, wrong, keep):
    """Re-select a tail of fixed size from sampled rows with multiplicity."""
    ranked_weights = weights[:, rank]
    preceding = np.cumsum(ranked_weights, axis=1, dtype=np.int32) - ranked_weights
    accepted = np.minimum(ranked_weights, np.maximum(keep - preceding, 0))
    return accepted[:, wrong[rank]].sum(axis=1)


def paired_bootstrap(cells, config, keep):
    n_test, n_draws = config["n_test"], config["bootstrap_draws"]
    rng = np.random.default_rng(config["bootstrap_seed"])
    clusters = {}
    for key, cell in cells.items():
        clusters.setdefault(int(cell["metadata"]["test_master"]), []).append(key)
    result = {key: np.empty(n_draws, dtype=np.int32) for key in cells}
    for master in sorted(clusters):
        for start in range(0, n_draws, 128):
            stop = min(start + 128, n_draws)
            draw = rng.integers(0, n_test, size=(stop - start, n_test), dtype=np.int32)
            weights = np.stack([np.bincount(row, minlength=n_test) for row in draw]).astype(np.int32)
            for key in clusters[master]:
                cell = cells[key]
                wrong0, wrong1 = (wrong_in_ranked_resample(weights, rank, cell["wrong"], keep)
                                  for rank in cell["ranks"])
                result[key][start:stop] = wrong0 - wrong1
    return result


def group(keys, cells, effects):
    if not keys:
        return {"n_cells": 0, "n_models": 0, "n_seeds": 0, "gain": None, "gain_ci95": None}
    metadata = [cells[key]["metadata"] for key in keys]
    records = [cells[key]["counts"] for key in keys]
    sums = {k: sum(c[k] for c in records) for k in ("windows", "wrong_all", "shared_accepted", "wrong_shared_accepted")}
    arms = {}
    for name in ARMS:
        record = {k: sum(c["rules"][name][k] for c in records) for k in records[0]["rules"][name]}
        record["coverage"] = record["accepted"] / sums["windows"]
        record["selective_error"] = record["wrong_accepted"] / record["accepted"]
        arms[name] = record
    accepted = arms["nominal"]["accepted"]
    pooled = np.sum([effects[key] for key in keys], axis=0) / accepted
    wrong_difference = arms["nominal"]["wrong_accepted"] - arms["adjusted"]["wrong_accepted"]
    return {"n_cells": len(keys), "n_models": len({c["name"] for c in metadata}),
            "n_seeds": len({c["seed"] for c in metadata}), "n_window_clusters": len({c["test_master"] for c in metadata}),
            **sums, "all_window_error": sums["wrong_all"] / sums["windows"], "rules": arms,
            "wrong_accepted_difference": wrong_difference, "gain": wrong_difference / accepted,
            "gain_ci95": np.quantile(pooled, (0.025, 0.975)).tolist()}


def summarize_panel(keys, cells, effects):
    result = {"overall": group(keys, cells, effects)}
    for field in ("seed", "process", "training_case"):
        levels = sorted({cells[key]["metadata"][field] for key in keys})
        result[f"by_{field}"] = {str(level): group([k for k in keys if cells[k]["metadata"][field] == level], cells, effects)
                                 for level in levels}
    return result


def score(config, cells, receipt_path):
    destination = HERE / "SUMMARY.json"
    if destination.exists():
        raise FileExistsError(destination)
    keep = int(np.floor(config["n_test"] * config["coverage"]))
    if not 0 < keep < config["n_test"]:
        raise ValueError("Invalid tail size for matched coverage")
    truth = {}
    for cell in cells.values():
        metadata = cell["metadata"]
        master, process = metadata["test_master"], metadata["process"]
        if (master, process) not in truth:
            truth[master, process] = physical_scores(config, master, process)
        cell["wrong"], cell["counts"] = summarize_cell(cell, truth[master, process], keep)
    effects = paired_bootstrap(cells, config, keep)
    all_keys = sorted(cells)
    eligible = summarize_panel([key for key in all_keys if cells[key]["metadata"]["eligible"]], cells, effects)
    primary = eligible["overall"]
    seed_gains = {seed: result["gain"] for seed, result in eligible["by_seed"].items()}
    fraction_positive = sum(gain > 0 for gain in seed_gains.values()) / len(seed_gains) if seed_gains else 0.0
    conditions = {
        "model_count": primary["n_models"] >= config["minimum_eligible_models"],
        "seed_count": primary["n_seeds"] >= config["minimum_eligible_seeds"],
        "point_gain": primary["gain"] is not None and primary["gain"] >= config["minimum_gain"],
        "confidence_interval": primary["gain_ci95"] is not None and primary["gain_ci95"][0] > 0,
        "seed_concordance": fraction_positive >= config["minimum_positive_seed_fraction"],
    }
    by_cell = {"__".join(key): {
        **group([key], cells, effects),
        **{k: cells[key]["metadata"][k] for k in ("state_r2", "eligible", "seed", "training_case", "process", "test_master")},
    } for key in all_keys}
    write(destination, {"forecast_sha256": digest(receipt_path), "n_accepted_per_rule_per_cell": keep,
                        "eligible": eligible, "all": summarize_panel(all_keys, cells, effects), "by_cell": by_cell,
                        "primary_gate": {"passed": all(conditions.values()), "conditions": conditions,
                                         "seed_gains": seed_gains, "fraction_positive_seeds": fraction_positive}})
    print("Eligible selective error gain:", primary["gain"], primary["gain_ci95"], flush=True)
    print("Primary gate:", conditions, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "score"))
    args = parser.parse_args()
    config, cells, receipt_path = verify()
    print(f"Verified sealed forecast for {len(cells)} cells. SHA256 {digest(receipt_path)}", flush=True)
    if args.stage == "score":
        score(config, cells, receipt_path)


if __name__ == "__main__":
    main()
