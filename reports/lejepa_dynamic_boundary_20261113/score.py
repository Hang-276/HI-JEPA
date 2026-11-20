"""Independently reveal physical paths and evaluate sealed dynamic corrections."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORACLES = ("oracle_cross", "oracle_energy", "oracle_reference", "oracle_cross_energy", "oracle_cross_abs", "oracle_cross_reverse")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")


def top2(scores):
    return np.sort(np.argsort(scores, axis=-1, kind="stable")[..., :2], axis=-1)


def exact_parts(physical, visible, variance, lags):
    true_score, numerator, cross, energy = np.zeros((4, len(physical), 4), dtype=float)
    for lag in lags:
        dz = physical[:, lag:] - physical[:, :-lag]
        dv = visible[:, lag:] - visible[:, :-lag]
        de = dv - dz
        true_score += np.mean(dz**2, axis=1)
        numerator += np.mean(dv**2, axis=1)
        cross += 2 * np.mean(dz * de, axis=1)
        energy += np.mean(de**2, axis=1)
    true_score, numerator, cross, energy = (part / len(lags) for part in (true_score, numerator, cross, energy))
    base = numerator / variance
    k = cross / variance
    q = energy / variance
    r = true_score * (1 - variance) / variance
    np.testing.assert_allclose(base - true_score, k + q + r, rtol=1e-10, atol=2e-13)
    scores = {
        "oracle_cross": base - k,
        "oracle_energy": base - q,
        "oracle_reference": base - r,
        "oracle_cross_energy": base - (k + q),
        "oracle_cross_abs": base - np.abs(k),
        "oracle_cross_reverse": base + k,
    }
    np.testing.assert_allclose(base - (k + q + r), true_score, rtol=1e-10, atol=2e-13)
    return true_score, base, numerator, {"cross": k, "energy": q, "reference": r}, scores


def boundary_pairs(truth):
    """The four unordered pairs joining truth's selected and unselected axes."""
    order = np.argsort(truth, axis=1, kind="stable")
    return order[:, [0, 0, 1, 1]], order[:, [2, 3, 2, 3]]


def boundary_scores(scores, slow, fast):
    return np.take_along_axis(scores, fast, axis=1) - np.take_along_axis(scores, slow, axis=1)


def boundary_wrong(scores, slow, fast):
    gap = boundary_scores(scores, slow, fast)
    return (gap < 0) | ((gap == 0) & (fast < slow))


def first_index(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def accuracy_record(scores, truth_set, base_top, slow, fast, wrong_base):
    selected = top2(scores)
    correct = np.all(selected == truth_set, axis=1)
    baseline_correct = np.all(base_top == truth_set, axis=1)
    wrong = boundary_wrong(scores, slow, fast)
    np.testing.assert_array_equal(correct, ~wrong.any(axis=1))
    repairs, harms = correct & ~baseline_correct, ~correct & baseline_correct
    wrong_pairs_repaired = wrong_base & ~wrong
    new_wrong_pairs = ~wrong_base & wrong
    margin = boundary_scores(scores, slow, fast)
    return {
        "accuracy": float(correct.mean()),
        "correct": int(correct.sum()),
        "repairs": int(repairs.sum()),
        "new_errors": int(harms.sum()),
        "net_correct": int(repairs.sum() - harms.sum()),
        "wrong_cross_boundary_pairs": int(wrong.sum()),
        "cross_boundary_pairs_repaired": int(wrong_pairs_repaired.sum()),
        "new_wrong_cross_boundary_pairs": int(new_wrong_pairs.sum()),
        "cross_boundary_ties": int((margin == 0).sum()),
        "pair_flip_with_no_set_repair_windows": int((wrong_pairs_repaired.any(1) & ~correct).sum()),
        "mean_negative_cross_boundary_margin": float(np.maximum(-margin.min(axis=1), 0).mean()),
        "example_repair_row": first_index(repairs),
        "example_new_error_row": first_index(harms),
    }, correct


def pair_component_summary(terms, truth, base, slow, fast):
    physical = boundary_scores(truth, slow, fast)
    errors = {name: boundary_scores(value, slow, fast) for name, value in terms.items()}
    visible = boundary_scores(base, slow, fast)
    np.testing.assert_allclose(visible, physical + sum(errors.values()), rtol=1e-10, atol=2e-13)
    actual = boundary_wrong(base, slow, fast)
    no_cross = boundary_wrong(base - terms["cross"], slow, fast)
    return {
        "target_pair_mean_margin": float(physical.mean()),
        "mean_component_bias": {name: float(value.mean()) for name, value in errors.items()},
        "mean_component_absolute_size": {name: float(np.abs(value).mean()) for name, value in errors.items()},
        "cross_induced_wrong": int((actual & ~no_cross).sum()),
        "cross_cancellation_dependent_correct": int((~actual & no_cross).sum()),
        "still_wrong_without_cross": int((actual & no_cross).sum()),
        "still_correct_without_cross": int((~actual & ~no_cross).sum()),
    }


def summarize_cell(physical, visible, variance, scores, cross_predictions, lags):
    truth, base, numerator, terms, oracle = exact_parts(physical, visible, variance, lags)
    np.testing.assert_allclose(scores["base"], base, rtol=1e-10, atol=2e-12)
    true_set, baseline_top = top2(truth), top2(base)
    slow, fast = boundary_pairs(truth)
    wrong_base = boundary_wrong(base, slow, fast)
    metrics, correct = {}, {}
    for arm, values in {**scores, **oracle}.items():
        metrics[arm], correct[arm] = accuracy_record(values, true_set, baseline_top, slow, fast, wrong_base)
    K = terms["cross"]
    fixed_gap = K[:, 2] - K[:, 1]
    pair_squared_error = {}
    cross_quality = {}
    for key, prediction in cross_predictions.items():
        normalized = prediction / variance
        gap = normalized[:, 2] - normalized[:, 1]
        pair_squared_error[key] = np.square(gap - fixed_gap)
        cross_quality[key] = {
            "fixed_pair_mse": float(pair_squared_error[key].mean()),
            "fixed_pair_sign_accuracy": float(((gap < 0) == (fixed_gap < 0)).mean()),
            "component_mse_by_axis": np.square(normalized - K).mean(axis=0).tolist(),
        }
    return {
        "by_arm": metrics,
        "fixed_2_3_cross_prediction": cross_quality,
        "cross_boundary_accounting": pair_component_summary(terms, truth, base, slow, fast),
        "true_top2_boundary_ties": int((boundary_scores(truth, slow, fast) == 0).sum()),
        "component_mean_by_axis": {name: values.mean(axis=0).tolist() for name, values in terms.items()},
        "component_absolute_mean_by_axis": {name: np.abs(values).mean(axis=0).tolist() for name, values in terms.items()},
    }, correct, pair_squared_error, numerator, truth, terms


def paired_gain(row, bootstrap):
    return {
        "gain": float(row.mean()),
        "interval_95": np.quantile(bootstrap @ row, [0.025, 0.975]).tolist(),
    }


def aggregate(records, correct, pair_errors, bootstrap, config):
    arm_names = tuple(config["arms"]) + ORACLES
    score_rows = {arm: np.stack([cell[arm] for cell in correct]) for arm in arm_names}
    base = score_rows["base"]
    accuracy_summary = {}
    for arm in arm_names:
        data = [record["by_arm"][arm] for record in records]
        accuracy_summary[arm] = {
            "accuracy": float(score_rows[arm].mean()),
            "repairs": sum(item["repairs"] for item in data),
            "new_errors": sum(item["new_errors"] for item in data),
            "net_correct": sum(item["net_correct"] for item in data),
            "wrong_cross_boundary_pairs": sum(item["wrong_cross_boundary_pairs"] for item in data),
            "cross_boundary_pairs_repaired": sum(item["cross_boundary_pairs_repaired"] for item in data),
            "new_wrong_cross_boundary_pairs": sum(item["new_wrong_cross_boundary_pairs"] for item in data),
            "cross_boundary_ties": sum(item["cross_boundary_ties"] for item in data),
            "versus_base": paired_gain((score_rows[arm].astype(float) - base).mean(axis=0), bootstrap),
        }
    comparisons = {
        f"{left}_minus_{right}": paired_gain(
            (score_rows[left].astype(float) - score_rows[right]).mean(axis=0), bootstrap
        )
        for left, right in (("oracle_cross", "oracle_energy"), ("cross_full", "cross_mean"),
                            ("cross_full", "cross_diag"), ("cross_full", "cross_full_reverse"),
                            ("cross_full", "total_full"))
    }
    errors = {name: np.stack([entry[name] for entry in pair_errors]).mean(axis=0) for name in pair_errors[0]}
    reference = errors["cross_mean"]
    quality = {}
    for name, rows in errors.items():
        n, d = bootstrap @ rows, bootstrap @ reference
        valid = d > 0
        quality[name] = {
            "mse": float(rows.mean()),
            "r2_relative_to_calmean": float(1 - rows.mean() / reference.mean()) if reference.mean() > 0 else None,
            "r2_interval_95": np.quantile(1 - n[valid] / d[valid], [0.025, 0.975]).tolist() if valid.any() else None,
        }
    def positive_at_least(item, minimum):
        return item["gain"] >= minimum and item["interval_95"][0] > 0

    mechanism = (
        positive_at_least(accuracy_summary["oracle_cross"]["versus_base"], config["minimum_gain"])
        and positive_at_least(comparisons["oracle_cross_minus_oracle_energy"], config["minimum_gain"])
    )
    pair_r2 = quality["cross_full"]["r2_relative_to_calmean"]
    usability = (
        positive_at_least(accuracy_summary["cross_full"]["versus_base"], config["minimum_gain"])
        and positive_at_least(comparisons["cross_full_minus_cross_mean"], config["minimum_window_specific_gain"])
        and pair_r2 is not None and pair_r2 >= config["minimum_pair_cross_r2"]
    )
    return {
        "arm_summary": accuracy_summary,
        "contrasts": comparisons,
        "fixed_2_3_cross_prediction": quality,
        "true_top2_boundary_ties": sum(record["true_top2_boundary_ties"] for record in records),
        "mechanism_gate": bool(mechanism),
        "usability_gate": bool(usability),
    }


def cell_key(cell):
    return "__".join(cell[field] for field in ("layer", "process", "name"))


def verify(smoke):
    import run as runner

    work = HERE / "smoke" if smoke else HERE
    archive = work / "forecast"
    design_path, receipt_path = work / "DESIGN_LOCK.json", archive / "RECEIPT.json"
    lock, receipt = read(design_path), read(receipt_path)
    if lock["sources"] != receipt["sources"] or receipt["design_sha256"] != digest(design_path):
        raise ValueError("Forecast is unrelated to frozen design.")
    _, identity, cells, config, sources = runner.design(smoke)
    if lock != {"sources": sources, "config": config, "cells": cells}:
        raise ValueError("Frozen design does not match current sources and declared panel.")
    if receipt["rounds"] != config["rounds"] or tuple(config["oracle_arms"]) != ORACLES:
        raise ValueError("Forecast rounds or oracle recipe differs from the protocol.")
    expected_files = {
        f"forecast/{stage}/{cell_key(cell)}.{suffix}"
        for stage in config["rounds"] for cell in cells for suffix in ("json", "npz")
    }
    if expected_files != set(receipt["files"]):
        raise ValueError("Sealed files do not match the full frozen panel.")
    for relative, sha in receipt["files"].items():
        if digest(work / relative) != sha:
            raise ValueError(f"Forecast hash mismatch: {relative}")
    return work, runner, config, identity, cells, receipt_path, design_path


def key_for_truth(stage, process):
    return f"{stage}__{process}"


def reveal(smoke):
    work, runner, config, _, _, receipt_path, design_path = verify(smoke)
    target_path, metadata = work / "REVEAL.npz", work / "REVEAL_RECEIPT.json"
    if target_path.exists() or metadata.exists() or (work / "SUMMARY.json").exists():
        raise FileExistsError("Prediction already revealed or scored.")
    truth = {
        key_for_truth(stage, process): runner.paths(master, "test", config["n_test"], config["length"], rates)
        for stage, master in config["rounds"].items() for process, rates in config["processes"].items()
    }
    with target_path.open("xb") as destination:
        np.savez_compressed(destination, **truth)
    write(metadata, {
        "forecast_receipt_sha256": digest(receipt_path),
        "design_sha256": digest(design_path),
        "labels_sha256": digest(target_path),
        "label_keys": sorted(truth),
    })


def independently_replay(visible, reference, gram, fit, config):
    """Re-evaluate frozen coefficient arithmetic without calling the predictor."""
    variance = reference.var(axis=0)
    if np.any(variance <= 0) or not np.isfinite(variance).all():
        raise ValueError("Invalid fixed reference variance.")
    upper = np.triu_indices(4)
    diagonal = gram.diagonal(axis1=1, axis2=2)
    full = gram[:, upper[0], upper[1]]
    coefficients = {}
    for method, x in (("cross_diag", diagonal), ("cross_full", full), ("total_full", full)):
        solution = fit[method]
        standardized = (x - np.asarray(solution["feature_mean"])) / np.asarray(solution["feature_std"])
        coefficients[method] = standardized @ np.asarray(solution["coefficients"]) + np.asarray(solution["target_mean"])
    coefficients["cross_mean"] = np.broadcast_to(np.asarray(fit["mean_cross"]), diagonal.shape)
    crossed = {name: coefficients[name] for name in ("cross_mean", "cross_diag", "cross_full")}
    base = diagonal / variance
    scores = {
        "base": base,
        **{name: base - values / variance for name, values in crossed.items()},
        "cross_full_reverse": base + crossed["cross_full"] / variance,
        "total_full": base - coefficients["total_full"],
    }
    if tuple(scores) != tuple(config["arms"]):
        raise ValueError("Arm order mismatch.")
    return scores, crossed, coefficients["total_full"], variance


def exact_gram(values, lags):
    result = np.zeros((len(values), 4, 4), dtype=np.float64)
    for lag in lags:
        delta = values[:, lag:] - values[:, :-lag]
        result += np.einsum("nti,ntj->nij", delta, delta) / delta.shape[1]
    return result / len(lags)


def evaluate_cell(work, runner, config, identity, cell, stage, states, physical_ref):
    folder = work / "forecast" / stage
    name = cell_key(cell)
    record = read(folder / f"{name}.json")
    if record["cell"] != cell or record["provenance"] != cell["provenance"]:
        raise ValueError(f"Frozen cell provenance mismatch: {name}")
    readout, provenance = runner.LOAD.load_readout(cell, identity)
    if provenance != record["provenance"]:
        raise ValueError(f"Loaded readout provenance mismatch: {name}")
    raw_test, raw_ref = readout(states), readout(physical_ref)
    offset, scale = (np.asarray(record["normalizer"][key]) for key in ("offset", "scale"))
    visible, reference = (raw_test - offset) / scale, (raw_ref - offset) / scale
    gram = exact_gram(visible, config["lags"])
    scores, cross_predictions, total, variance = independently_replay(visible, reference, gram, record["fit"], config)
    raw_reference_variance = raw_ref.var(axis=0)
    raw_scores = exact_gram(raw_test, config["lags"]).diagonal(axis1=1, axis2=2) / raw_reference_variance
    score_stack = np.stack(list(scores.values()))
    predicted_cross = np.stack(list(cross_predictions.values()))
    with np.load(folder / f"{name}.npz", allow_pickle=False) as sealed:
        for label, actual in (
            ("gram", gram), ("scores", score_stack), ("predicted_cross", predicted_cross),
            ("predicted_total", total), ("raw_scores", raw_scores), ("raw_reference_variance", raw_reference_variance),
        ):
            np.testing.assert_allclose(sealed[label], actual, rtol=1e-10, atol=2e-12)
        np.testing.assert_array_equal(top2(score_stack), sealed["top2"])
    np.testing.assert_allclose(variance, record["reference_variance"], rtol=1e-10, atol=2e-12)
    np.testing.assert_allclose(raw_scores, scores["base"], rtol=1e-10, atol=2e-12)
    np.testing.assert_array_equal(top2(raw_scores), top2(scores["base"]))
    result, correct, errors, numerator, physical_score, terms = summarize_cell(
        states, visible, variance, scores, cross_predictions, config["lags"]
    )
    np.testing.assert_allclose(numerator, gram.diagonal(axis1=1, axis2=2), rtol=1e-10, atol=2e-12)
    # Preserves independently replayable test truth diagnostics without feeding them to predictions.
    return {
        "cell": cell, "reference_variance": variance.tolist(),
        "calibration_validation": record["calibration_validation"], **result,
    }, correct, errors, {
        "physical_scores": physical_score,
        "cross_over_reference": terms["cross"],
        "energy_over_reference": terms["energy"],
        "reference_accounting": terms["reference"],
    }


def score(smoke):
    work, runner, config, identity, cells, receipt_path, design_path = verify(smoke)
    labels_path = work / "REVEAL.npz"
    reveal_record = read(work / "REVEAL_RECEIPT.json")
    if (reveal_record["forecast_receipt_sha256"] != digest(receipt_path)
            or reveal_record["design_sha256"] != digest(design_path)
            or reveal_record["labels_sha256"] != digest(labels_path)):
        raise ValueError("Truth archive is unrelated to the forecast.")
    if (work / "SUMMARY.json").exists() or (work / "SCORE_COMPONENTS.npz").exists():
        raise FileExistsError("Formal scores already exist.")
    count = config["n_test"]
    bootstrap = np.random.default_rng(config["bootstrap_seed"]).multinomial(
        count, np.full(count, 1 / count), size=config["bootstrap_draws"]
    ) / count
    results, components = {}, {}
    with np.load(labels_path, allow_pickle=False) as label_archive:
        if sorted(label_archive.files) != reveal_record["label_keys"]:
            raise ValueError("Wrong TEST arrays in truth archive.")
        for stage, master in config["rounds"].items():
            references = runner.iids(master, "reference", config["n_reference"])
            physical = {process: label_archive[key_for_truth(stage, process)] for process in config["processes"]}
            rows, correctness, errors = [], [], []
            for index, cell in enumerate(cells):
                record, correct, pair_errors, parts = evaluate_cell(
                    work, runner, config, identity, cell, stage, physical[cell["process"]], references
                )
                rows.append(record)
                correctness.append(correct)
                errors.append(pair_errors)
                for label, values in parts.items():
                    components[f"{stage}__{cell_key(cell)}__{label}"] = values
                if (index + 1) % 7 == 0:
                    print(f"{stage}: independently replayed {index + 1}/{len(cells)} frozen predictions", flush=True)
            by_process = {}
            for process in config["processes"]:
                indices = [index for index, cell in enumerate(cells) if cell["process"] == process]
                if indices:
                    by_process[process] = aggregate(
                        [rows[index] for index in indices], [correctness[index] for index in indices],
                        [errors[index] for index in indices], bootstrap, config,
                    )
            results[stage] = {
                "overall": aggregate(rows, correctness, errors, bootstrap, config),
                "process_descriptive": by_process, "cells": rows,
            }
    component_path = work / "SCORE_COMPONENTS.npz"
    with component_path.open("xb") as destination:
        np.savez_compressed(destination, **components)
    write(work / "SUMMARY.json", {
        "forecast_receipt_sha256": digest(receipt_path),
        "label_sha256": reveal_record["labels_sha256"],
        "components_sha256": digest(component_path),
        "mechanism_passes_both_rounds": all(entry["overall"]["mechanism_gate"] for entry in results.values()),
        "usability_passes_both_rounds": all(entry["overall"]["usability_gate"] for entry in results.values()),
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
