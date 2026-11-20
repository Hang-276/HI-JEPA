"""Reveal frozen LeJEPA forecasts and independently audit Hermite corrections."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

import model as M


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOADER_PATH = HERE.parent / "lejepa_natural_direction_repair_20261113/loaders.py"
SPEC = importlib.util.spec_from_file_location("frozen_readout_loader", LOADER_PATH)
LOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOAD)
ORACLES = {"oracle_mixed_both": "mixed", "oracle_self_both": "self"}


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")


def cell_key(cell):
    return "__".join(cell[field] for field in ("layer", "process", "name"))


def verified_forecast(smoke):
    work = HERE / "smoke" if smoke else HERE
    archive = work / "forecast"
    if smoke and not archive.exists():
        archive = work
    receipt_path, design_path = archive / "RECEIPT.json", work / "DESIGN_LOCK.json"
    receipt, design = read(receipt_path), read(design_path)
    config, cells = design["config"], design["cells"]
    if receipt["design_sha256"] != digest(design_path) or receipt["sources"] != design["sources"]:
        raise ValueError("Forecast does not match the frozen design.")
    if receipt["rounds"] != config["rounds"] or tuple(config["arms"]) != M.ARMS:
        raise ValueError("Forecast does not contain the expected rounds and arms.")
    if smoke:
        if config["rounds"] != {"smoke": 2026111399}:
            raise ValueError("Unknown smoke namespace.")
    elif config != read(HERE / "protocol.json"):
        raise ValueError("Formal config changed.")
    for relative, value in receipt["sources"].items():
        if digest(ROOT / relative) != value:
            raise ValueError(f"Source hash mismatch: {relative}")
    expected_files = {
        str((archive / stage / f"{cell_key(cell)}.{suffix}").relative_to(work))
        for stage in config["rounds"] for cell in cells for suffix in ("json", "npz")
    }
    if not expected_files <= set(receipt["files"]):
        raise ValueError("Some locked cell artifacts are missing.")
    for relative, value in receipt["files"].items():
        if digest(work / relative) != value:
            raise ValueError(f"Prediction hash mismatch: {relative}")
    _, identity, full_panel = LOAD.load_panel()
    expected_panel = [cell for cell in full_panel if cell["layer"] == "mlp"]
    if not smoke and (len(expected_panel) != 28 or {cell_key(cell): cell for cell in cells}
                      != {cell_key(cell): cell for cell in expected_panel} or len(cells) != 28):
        raise ValueError("Unexpected frozen MLP panel.")
    return work, archive, design_path, receipt_path, config, cells, identity


def label_key(round_name, process):
    return f"{round_name}__{process}"


def replay_truth(config):
    return {
        label_key(round_name, process): LOAD.paths(master, "test", config["n_test"], config["length"], rates)
        for round_name, master in config["rounds"].items()
        for process, rates in config["processes"].items()
    }


def reveal(smoke):
    work, _, design_path, receipt_path, config, _, _ = verified_forecast(smoke)
    if any((work / name).exists() for name in ("REVEAL.npz", "REVEAL_RECEIPT.json", "SUMMARY.json")):
        raise FileExistsError("The verified archive has already been revealed or scored.")
    target = work / "REVEAL.npz"
    labels = replay_truth(config)
    with target.open("xb") as output:
        np.savez_compressed(output, **labels)
    write(work / "REVEAL_RECEIPT.json", {
        "forecast_receipt_sha256": digest(receipt_path),
        "design_sha256": digest(design_path),
        "labels_sha256": digest(target),
        "keys": sorted(labels),
    })


def top2(scores):
    return np.sort(np.argsort(scores, axis=-1, kind="stable")[..., :2], axis=-1)


def increments(states, lags):
    result = np.zeros((len(states), 4), dtype=np.float64)
    for lag in lags:
        delta = states[:, lag:] - states[:, :-lag]
        result += np.square(delta).mean(axis=1)
    return result / len(lags)


def oracle_outputs(physical, base, fit):
    features = M.basis(physical)
    result = {}
    for name, group in ORACLES.items():
        revised = base.copy()
        for row in fit["groups"][group]:
            if row["active"]:
                revised[..., row["recipient"]] -= features[..., row["indices"]] @ np.asarray(row["coefficients"])
        result[name] = revised
    return result


def arm_outputs(visible, physical, fit, config):
    outputs = M.outputs(visible, fit, config)
    base = outputs["base"]
    for arm, coordinates in {
        "mixed_axis2": (0, 2, 3), "mixed_axis3": (0, 1, 3),
        **{name: (0, 3) for name in M.ARMS if name not in ("base", "mixed_axis2", "mixed_axis3")},
    }.items():
        np.testing.assert_array_equal(outputs[arm][..., coordinates], base[..., coordinates])
    np.testing.assert_allclose(outputs["mixed_both"] + outputs["reverse_both"], 2 * base, atol=2e-12, rtol=1e-12)
    for index, recipient in enumerate(config["recipients"]):
        if not fit["groups"]["mixed"][index]["active"]:
            for arm in ("mixed_both", "reverse_both", "raw_mixed_both", f"mixed_axis{recipient + 1}"):
                np.testing.assert_array_equal(outputs[arm][..., recipient], base[..., recipient])
    return {**outputs, **oracle_outputs(physical, base, fit)}


def physical_recovery(candidate, physical):
    error = (candidate - physical).reshape(-1, 4)
    mse = np.square(error).mean(axis=0)
    variance = physical.reshape(-1, 4).var(axis=0)
    return {"mse": mse.tolist(), "r2": (1 - mse / variance).tolist()}


def grouped_statistics(output, physical, coefficients, baseline):
    groups = {}
    for j in M.RECIPIENTS:
        mixed = M.group_indices("mixed", j)
        own = M.group_indices("self", j)
        nonlinear = M.group_indices("all", j)
        groups[j] = {}
        for name, indices in {
            "affine": np.arange(1, 5), "mixed": mixed, "self": own, "all_nonlinear": nonlinear,
            "other_univariate": np.setdiff1d(nonlinear, np.union1d(mixed, own)),
        }.items():
            selected, original = coefficients[indices, j], baseline[indices, j]
            denominator = np.linalg.norm(original)
            groups[j][name] = {
                "norm": float(np.linalg.norm(selected)),
                "delta_norm": float(np.linalg.norm(selected - original)),
                "signed_projection_onto_baseline_unit": float((selected @ original) / denominator) if denominator else None,
            }
    return {
        "residual_coefficients": coefficients.tolist(),
        "groups": groups,
        "state_recovery": physical_recovery(output, physical),
    }


def audit_outputs(candidates, physical, fit, config):
    design = M.basis(physical)
    residuals = np.concatenate([output - physical for output in candidates.values()], axis=1)
    joint = np.linalg.lstsq(design, residuals, rcond=None)[0]
    by_arm = dict(zip(candidates, np.split(joint, len(candidates), axis=1)))
    base = by_arm["base"]
    stats = {name: grouped_statistics(values, physical, by_arm[name], base) for name, values in candidates.items()}
    suppression = []
    for group, arm in (("mixed", "mixed_both"), ("self", "self_both"), ("all", "all_nonlinear_both")):
        for item in fit["groups"][group]:
            j = item["recipient"]
            indices = item["indices"]
            original = base[indices, j]
            final = by_arm[arm][indices, j]
            original_norm = np.linalg.norm(original)
            fraction = float(1 - np.linalg.norm(final) / original_norm) if original_norm else None
            suppression.append({
                "group": group, "recipient": j, "active": item["active"], "cal_reason": item["reason"],
                "cal_norm": item["norm"], "cal_proxy_cosine": item["proxy_cosine"],
                "cal_split_cosine": item["split_cosine"], "cal_gain": item["gamma"],
                "audit_norm_before": float(original_norm), "audit_norm_after": float(np.linalg.norm(final)),
                "audit_norm_fraction_removed": fraction,
                "audit_norm_halved": bool(fraction is not None and fraction >= config["loading_suppression_fraction"]),
            })
            oracle = f"oracle_{group}_both"
            if oracle in by_arm:
                expected = base[:, j].copy()
                if item["active"]:
                    expected[indices] -= item["coefficients"]
                np.testing.assert_allclose(by_arm[oracle][:, j], expected, rtol=1e-9, atol=2e-12)
    return {"by_arm": stats, "target_group_audit": suppression}


def swap_pairs(old, new, mask):
    old_members = (old[:, :, None] == np.arange(4)).any(axis=1)
    new_members = (new[:, :, None] == np.arange(4)).any(axis=1)
    removed = old_members & ~new_members & mask[:, None]
    added = new_members & ~old_members & mask[:, None]
    return (removed.astype(int).T @ added.astype(int)).tolist()


def accuracy(scores, expected, base):
    selected = top2(scores)
    correct = np.all(selected == expected, axis=-1)
    base_correct = np.all(base == expected, axis=-1)
    repairs, harms = correct & ~base_correct, ~correct & base_correct
    return {
        "accuracy": float(correct.mean()), "correct": int(correct.sum()),
        "repairs": int(repairs.sum()), "harms": int(harms.sum()),
        "net": int(repairs.sum() - harms.sum()),
        "removed_added_on_repairs": swap_pairs(base, selected, repairs),
        "removed_added_on_harms": swap_pairs(base, selected, harms),
    }, correct


def score_cell(root, cell, states, ref, audit, identity, config):
    fit_record = read(root.with_suffix(".json"))
    if fit_record["cell"] != cell or fit_record["provenance"] != cell["provenance"]:
        raise ValueError(f"Wrong cell identity: {root.name}")
    readout, provenance = LOAD.load_readout(cell, identity)
    if provenance != fit_record["provenance"]:
        raise ValueError("Unrelated readout loaded.")
    fit = fit_record["fit"]
    if fit["multi_indices"] != M.MULTI_INDICES.tolist():
        raise ValueError("Hermite basis changed.")
    visible_test, visible_ref = readout(states), readout(ref)
    test_outputs = arm_outputs(visible_test, states, fit, config)
    ref_outputs = arm_outputs(visible_ref, ref, fit, config)
    audit_outputs_by_arm = arm_outputs(readout(audit), audit, fit, config)
    numerators = {name: increments(output, config["lags"]) for name, output in test_outputs.items()}
    variance = {name: output.var(axis=0) for name, output in ref_outputs.items()}
    if min(np.min(values) for values in variance.values()) <= 0:
        raise ValueError("Undefined reference coordinate.")
    predictions = {name: numerator / variance[name] for name, numerator in numerators.items()}
    with np.load(root.with_suffix(".npz"), allow_pickle=False) as sealed:
        raw_variance = visible_ref.var(axis=0)
        raw_scores = increments(visible_test, config["lags"]) / raw_variance
        np.testing.assert_allclose(raw_variance, sealed["raw_reference_variance"], rtol=1e-10, atol=2e-12)
        np.testing.assert_allclose(raw_scores, sealed["raw_scores"], rtol=1e-10, atol=2e-12)
        np.testing.assert_allclose(raw_scores, predictions["base"], rtol=1e-10, atol=2e-12)
        np.testing.assert_array_equal(top2(raw_scores), top2(predictions["base"]))
        np.testing.assert_allclose(np.stack([variance[arm] for arm in M.ARMS]), fit_record["reference_variance"], rtol=1e-10, atol=2e-12)
        np.testing.assert_allclose(np.stack([numerators[arm] for arm in M.ARMS]), sealed["numerators"], rtol=1e-10, atol=2e-12)
        np.testing.assert_allclose(np.stack([predictions[arm] for arm in M.ARMS]), sealed["scores"], rtol=1e-10, atol=2e-12)
        np.testing.assert_array_equal(top2(np.stack([predictions[arm] for arm in M.ARMS])), sealed["top2"])
    physical_scores = increments(states, config["lags"])
    physical_top = top2(physical_scores)
    base_top = top2(predictions["base"])
    metrics, correctness = {}, {}
    for arm, scores in predictions.items():
        metrics[arm], correctness[arm] = accuracy(scores, physical_top, base_top)
        delta_n, delta_v = numerators[arm] - numerators["base"], variance[arm] - variance["base"]
        dynamics = delta_n / variance[arm]
        scale = -predictions["base"] * delta_v / variance[arm]
        np.testing.assert_allclose(scores - predictions["base"], dynamics + scale, rtol=1e-10, atol=2e-12)
        metrics[arm].update({
            "test_state_recovery": physical_recovery(test_outputs[arm], states),
            "reference_variance": variance[arm].tolist(),
            "mean_numerator_change": delta_n.mean(axis=0).tolist(),
            "reference_variance_change": delta_v.tolist(),
            "mean_score_change_from_numerator": dynamics.mean(axis=0).tolist(),
            "mean_score_change_from_reference": scale.mean(axis=0).tolist(),
        })
    return {
        "cell": cell, "test_by_arm": metrics,
        "audit": audit_outputs(audit_outputs_by_arm, audit, fit, config),
    }, correctness


def paired_delta(correct, target, comparator, bootstrap):
    row_differences = np.stack([values[target].astype(float) - values[comparator] for values in correct]).mean(axis=0)
    sampled = bootstrap @ row_differences
    return {"gain": float(row_differences.mean()), "interval_95": np.quantile(sampled, [0.025, 0.975]).tolist()}


def aggregate(keys, records, correctness, config, bootstrap):
    if not keys:
        return None
    cells = [records[key] for key in keys]
    correct = [correctness[key] for key in keys]
    audit = [entry for cell in cells for entry in cell["audit"]["target_group_audit"]]
    group_stats = {}
    for group in M.GROUPS:
        rows = [entry for entry in audit if entry["group"] == group and entry["active"]]
        group_stats[group] = {
            "active_recipients": len(rows), "total_recipients": len(cells) * len(M.RECIPIENTS),
            "halved_recipients": sum(entry["audit_norm_halved"] for entry in rows),
            "halved_share": float(np.mean([entry["audit_norm_halved"] for entry in rows])) if rows else None,
        }
    scores = {
        arm: {
            metric: float(np.mean([cell["test_by_arm"][arm][metric] for cell in cells]))
            for metric in ("accuracy", "correct", "repairs", "harms", "net")
        }
        for arm in (*M.ARMS, *ORACLES)
    }
    diffs = {arm: paired_delta(correct, arm, "base", bootstrap) for arm in scores if arm != "base"}
    vs_reverse = paired_delta(correct, "mixed_both", "reverse_both", bootstrap)
    share = group_stats["mixed"]["halved_share"]
    passes = (
        share is not None and share >= config["minimum_suppression_share"]
        and all(item["gain"] >= config["minimum_gain"] and item["interval_95"][0] > 0
                for item in (diffs["mixed_both"], vs_reverse))
    )
    return {
        "cells": len(cells), "group_audit": group_stats, "by_arm": scores, "paired_arm_minus_base": diffs,
        "paired_mixed_both_minus_reverse_both": vs_reverse, "passes": passes,
    }


def score(smoke):
    work, archive, design_path, receipt_path, config, cells, identity = verified_forecast(smoke)
    target, reveal_path = work / "REVEAL.npz", work / "REVEAL_RECEIPT.json"
    label_receipt = read(reveal_path)
    if (label_receipt["forecast_receipt_sha256"] != digest(receipt_path)
            or label_receipt["labels_sha256"] != digest(target)
            or label_receipt["design_sha256"] != digest(design_path)):
        raise ValueError("Truth is unrelated to this sealed design or forecast.")
    n = config["n_test"]
    bootstrap = np.random.default_rng(config["bootstrap_seed"]).multinomial(
        n, np.full(n, 1 / n), size=config["bootstrap_draws"]
    ) / n
    results = {}
    with np.load(target, allow_pickle=False) as labels:
        if sorted(labels.files) != label_receipt["keys"]:
            raise ValueError("Missing or extra truth arrays.")
        for stage, master in config["rounds"].items():
            ref = LOAD.iids(master, "reference", config["n_reference"])
            audit = LOAD.iids(master, "audit_state", config["n_audit_state"])
            states = {process: labels[label_key(stage, process)] for process in config["processes"]}
            records, correctness = {}, {}
            for index, cell in enumerate(cells):
                key = cell_key(cell)
                records[key], correctness[key] = score_cell(
                    archive / stage / key, cell, states[cell["process"]], ref, audit, identity, config
                )
                if (index + 1) % 14 == 0:
                    print(f"{stage}: independently replayed {index + 1}/{len(cells)} frozen cells", flush=True)
            active = [key for key, record in records.items() if any(
                item["group"] == "mixed" and item["active"] for item in record["audit"]["target_group_audit"]
            )]
            results[stage] = {
                "overall": aggregate(list(records), records, correctness, config, bootstrap),
                "active_mixed_descriptive": aggregate(active, records, correctness, config, bootstrap),
                "process_descriptive": {
                    process: aggregate([key for key, record in records.items() if record["cell"]["process"] == process],
                                       records, correctness, config, bootstrap)
                    for process in config["processes"]
                },
                "records": records,
            }
    write(work / "SUMMARY.json", {
        "forecast_receipt_sha256": digest(receipt_path), "labels_sha256": label_receipt["labels_sha256"],
        "passes_both_rounds": bool(all(value["overall"]["passes"] for value in results.values())),
        "results": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reveal", "score"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "score":
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    {"reveal": reveal, "score": score}[args.stage](args.smoke)


if __name__ == "__main__":
    main()
