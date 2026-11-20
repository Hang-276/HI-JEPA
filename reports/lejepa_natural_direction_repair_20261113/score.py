"""Independent reveal, physical scoring, and mechanism audit of frozen repairs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import loaders


HERE = Path(__file__).resolve().parent
FORECAST = HERE / "forecast"
RECEIPT = FORECAST / "RECEIPT.json"
DESIGN = HERE / "DESIGN_LOCK.json"
REVEAL = HERE / "REVEAL.npz"
REVEAL_RECEIPT = HERE / "REVEAL_RECEIPT.json"
SUMMARY = HERE / "SUMMARY.json"
ARMS = ("base", "target", "reverse", "wrong_donor", "joint_linear")


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


def verify():
    config = read(HERE / "protocol.json")
    design, receipt = read(DESIGN), read(RECEIPT)
    if design["sources"] != receipt["sources"] or design["config"] != config or tuple(config["arms"]) != ARMS:
        raise ValueError("Frozen design or configuration does not match the forecast.")
    if receipt["design_sha256"] != digest(DESIGN):
        raise ValueError("Unrelated forecast receipt.")
    for relative, expected in receipt["sources"].items():
        path = loaders.ROOT / relative
        if digest(path) != expected:
            raise ValueError(f"Forecast source changed: {relative}")
    _, identity, cells = loaders.load_panel()
    if design["cells"] != cells:
        raise ValueError("Forecast panel differs from the frozen cell list.")
    if receipt["rounds"] != config["rounds"]:
        raise ValueError("Forecast does not include both locked rounds.")
    files = receipt["files"]
    mandatory = {
        f"forecast/{stage}/{cell_key(cell)}.{suffix}"
        for stage in config["rounds"] for cell in cells for suffix in ("json", "npz")
    }
    if not mandatory <= set(files):
        raise ValueError("One or more complete frozen forecast cells are missing.")
    for relative, expected in files.items():
        if digest(HERE / relative) != expected:
            raise ValueError(f"Forecast artifact changed: {relative}")
    return config, receipt, identity, cells


def truth_key(stage, process):
    return f"{stage}__{process}"


def reveal():
    if any(path.exists() for path in (REVEAL, REVEAL_RECEIPT, SUMMARY)):
        raise FileExistsError("Already revealed or scored.")
    config, _, _, _ = verify()
    labels = {
        truth_key(stage, process): loaders.paths(master, "test", config["n_test"], config["length"], rates)
        for stage, master in config["rounds"].items() for process, rates in config["processes"].items()
    }
    with REVEAL.open("xb") as output:
        np.savez_compressed(output, **labels)
    write(REVEAL_RECEIPT, {
        "forecast_receipt_sha256": digest(RECEIPT), "design_sha256": digest(DESIGN),
        "reveal_sha256": digest(REVEAL), "label_keys": sorted(labels),
    })


def top2(values):
    return np.sort(np.argsort(values, axis=-1, kind="stable")[..., :2], axis=-1)


def increments(values, lags):
    total = np.zeros((len(values), values.shape[-1]), dtype=np.float64)
    for lag in lags:
        difference = values[:, lag:] - values[:, :-lag]
        total += np.square(difference).mean(axis=1)
    return total / len(lags)


def regression(output, physical):
    centered_physical = physical - physical.mean(axis=0)
    centered_output = output - output.mean(axis=0)
    return np.linalg.lstsq(centered_physical, centered_output, rcond=None)[0].T


def recovery(output, physical):
    mse = ((output - physical)**2).reshape(-1, 4).mean(axis=0)
    return {"mse": mse.tolist(), "r2": (1 - mse / physical.reshape(-1, 4).var(axis=0)).tolist()}


def arm_values(normalized, calibration):
    arrays = {name: normalized.copy() for name in ARMS[:4]}
    j = calibration["recipient"]
    if calibration["active"]:
        d, other = calibration["donor"], calibration["wrong_donor"]
        center = np.asarray(calibration["mean_v"])
        primary = calibration["coefficient"] * (normalized[..., d] - center[d])
        other_update = calibration["wrong_coefficient"] * (normalized[..., other] - center[other])
        arrays["target"][..., j] -= primary
        arrays["reverse"][..., j] += primary
        arrays["wrong_donor"][..., j] -= other_update
    matrix = np.asarray(calibration["matrices"], dtype=float)
    intercept = np.asarray(calibration["intercepts"], dtype=float)
    arrays["joint_linear"] = normalized @ matrix[4].T + intercept[4]
    for idx, arm in enumerate(ARMS):
        np.testing.assert_allclose(arrays[arm], normalized @ matrix[idx].T + intercept[idx], atol=2e-12, rtol=1e-12)
    return arrays


def audit_state(arrays, physical, calibration, config):
    j, donor = calibration["recipient"], calibration["donor"]
    forward = {arm: regression(candidate, physical) for arm, candidate in arrays.items()}
    loadings = {
        arm: {
            "forward_loading_rows": matrix.tolist(),
            "intended_donor_loading": None if donor is None else float(matrix[j, donor]),
            "recipient_own_loading": float(matrix[j, j]),
            "other_donor_l2": float(np.linalg.norm(matrix[j, [axis for axis in range(4) if axis not in (j, donor)]])),
            "physical_recovery": recovery(arrays[arm], physical),
        }
        for arm, matrix in forward.items()
    }
    fraction = None
    sufficient = None
    if calibration["active"]:
        c = calibration["coefficient"]
        expected_oracle = forward["base"][j].copy()
        expected_oracle[donor] -= c
        np.testing.assert_allclose(forward["target"][j], forward["base"][j] - c * forward["base"][donor], atol=2e-12)
        np.testing.assert_allclose(forward["reverse"][j], forward["base"][j] + c * forward["base"][donor], atol=2e-12)
        np.testing.assert_allclose(forward["oracle_diagnostic"][j], expected_oracle, atol=2e-12)
        base = abs(forward["base"][j, donor])
        after = abs(forward["target"][j, donor])
        fraction = float(1 - after / base) if base > 0 else None
        sufficient = bool(base > 0 and after <= (1 - config["loading_suppression_fraction"]) * base)
    return {
        "audit_by_arm": loadings, "audit_loading_suppression_fraction": fraction,
        "audit_loading_reduced_by_half": sufficient,
    }


def swap_pairs(old, new, mask):
    """Count co-occurring omissions/admissions, including all pairs on multi-axis changes."""
    old_membership = (old[:, :, None] == np.arange(4)).any(axis=1)
    new_membership = (new[:, :, None] == np.arange(4)).any(axis=1)
    removed = old_membership & ~new_membership & mask[:, None]
    added = new_membership & ~old_membership & mask[:, None]
    return (removed.astype(int).T @ added.astype(int)).tolist()


def accuracy_metrics(score_values, truth, base_top):
    chosen = top2(score_values)
    correct = np.all(chosen == truth, axis=-1)
    base_correct = np.all(base_top == truth, axis=-1)
    fixes = correct & ~base_correct
    harms = ~correct & base_correct
    def first(mask):
        selected = np.flatnonzero(mask)
        return int(selected[0]) if selected.size else None

    return {
        "accuracy": float(correct.mean()), "correct": int(correct.sum()), "repairs": int(fixes.sum()),
        "new_errors": int(harms.sum()), "net_change": int(fixes.sum() - harms.sum()),
        "repair_fraction_of_base_wrong": float(fixes.sum() / (~base_correct).sum()) if (~base_correct).any() else None,
        "damage_fraction_of_base_correct": float(harms.sum() / base_correct.sum()) if base_correct.any() else None,
        "removed_added_pairs_on_repairs": swap_pairs(base_top, chosen, fixes),
        "removed_added_pairs_on_harms": swap_pairs(base_top, chosen, harms),
        "first_repair_row": first(fixes),
        "first_new_error_row": first(harms),
    }, correct


def mechanism_decomposition(base, changed, ref_base, ref_changed, j):
    baseline_score = base[:, j] / ref_base[j]
    actual_delta = changed[:, j] / ref_changed[j] - baseline_score
    dynamic_term = (changed[:, j] - base[:, j]) / ref_changed[j]
    scale_term = -baseline_score * (ref_changed[j] - ref_base[j]) / ref_changed[j]
    np.testing.assert_allclose(actual_delta, dynamic_term + scale_term, rtol=1e-10, atol=2e-12)
    return {
        "recipient_numerator_change": float((changed[:, j] - base[:, j]).mean()),
        "recipient_reference_variance_change": float(ref_changed[j] - ref_base[j]),
        "recipient_score_change": float(actual_delta.mean()),
        "score_change_from_numerator": float(dynamic_term.mean()),
        "score_change_from_reference": float(scale_term.mean()),
    }


def score_cell(cell, stage, config, identity, states, physical_ref, independent_state, forecast_root=FORECAST):
    root = forecast_root / stage / cell_key(cell)
    calibration = read(root.with_suffix(".json"))
    if calibration["cell"] != cell or calibration["provenance"] != cell["provenance"]:
        raise ValueError(f"Unrelated calibration cell: {root.name}")
    readout, provenance = loaders.load_readout(cell, identity)
    if provenance != cell["provenance"]:
        raise ValueError("Archived weight hashes no longer match.")
    offset, scale = (np.asarray(calibration[name]) for name in ("offset", "scale"))
    u_test, u_ref = readout(states), readout(physical_ref)
    v_test, v_ref = (u_test - offset) / scale, (u_ref - offset) / scale
    v_audit = (readout(independent_state) - offset) / scale
    tests, references, audits = (arm_values(data, calibration) for data in (v_test, v_ref, v_audit))
    arrays = {}
    truth = top2(increments(states, config["lags"]))
    j, donor = calibration["recipient"], calibration["donor"]
    with np.load(root.with_suffix(".npz"), allow_pickle=False) as frozen:
        frozen_scores, frozen_top2 = frozen["scores"], frozen["top2"]
        np.testing.assert_array_equal(top2(frozen_scores), frozen_top2)
        raw_var = u_ref.var(axis=0)
        raw_score = increments(u_test, config["lags"]) / raw_var
        np.testing.assert_allclose(raw_var, frozen["raw_reference_variance"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(raw_score, frozen["raw_scores"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(raw_score, frozen_scores[0], rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(top2(raw_score), frozen_top2[0])
        np.testing.assert_allclose(
            np.cov(v_ref, rowvar=False, ddof=0), calibration["reference_covariance"], rtol=1e-10, atol=1e-12
        )
        numerators = {arm: increments(values, config["lags"]) for arm, values in tests.items()}
        variance = {arm: values.var(axis=0) for arm, values in references.items()}
        np.testing.assert_allclose(
            np.stack([variance[arm] for arm in ARMS]), calibration["reference_variance"], rtol=1e-10, atol=1e-12
        )
        for idx, arm in enumerate(ARMS):
            direct = numerators[arm] / variance[arm]
            np.testing.assert_allclose(direct, frozen_scores[idx], rtol=1e-10, atol=1e-12)
            np.testing.assert_array_equal(top2(direct), frozen_top2[idx])
            arrays[arm] = direct
        oracle_test = v_test.copy()
        oracle_ref = v_ref.copy()
        oracle_audit = v_audit.copy()
        if calibration["active"]:
            c = calibration["coefficient"]
            donor_offset = calibration["mean_v"][donor]
            oracle_test[..., j] -= c * (states[..., donor] - donor_offset)
            oracle_ref[..., j] -= c * (physical_ref[..., donor] - donor_offset)
            oracle_audit[..., j] -= c * (independent_state[..., donor] - donor_offset)
        arrays["oracle_diagnostic"] = increments(oracle_test, config["lags"]) / oracle_ref.var(axis=0)
    tests["oracle_diagnostic"] = oracle_test
    audits["oracle_diagnostic"] = oracle_audit
    output, correct = {}, {}
    baseline = top2(arrays["base"])
    for arm, values in arrays.items():
        output[arm], correct[arm] = accuracy_metrics(values, truth, baseline)
        output[arm]["test_state_recovery"] = recovery(tests[arm], states)
        if arm != "oracle_diagnostic":
            output[arm]["dynamic_change"] = mechanism_decomposition(
                numerators["base"], numerators[arm], variance["base"], variance[arm], j
            )
    audit = audit_state(audits, independent_state, calibration, config)
    base_audit = audit["audit_by_arm"]["base"]
    target_audit = audit["audit_by_arm"]["target"]
    audit_base_mse = base_audit["physical_recovery"]["mse"][j]
    return {
        "cell": cell, "recipient": j, "donor": donor, "wrong_donor": calibration["wrong_donor"],
        "active": calibration["active"], "coefficient": calibration["coefficient"],
        "cal_state_mse": dict(zip(ARMS, calibration["cal_mse"])),
        "c_squared_over_audit_baseline_recipient_mse": (
            float(calibration["coefficient"] ** 2 / audit_base_mse) if audit_base_mse > 0 else None
        ),
        "audit_target_minus_base_other_donor_l2": target_audit["other_donor_l2"] - base_audit["other_donor_l2"],
        "audit_target_minus_base_own_loading": target_audit["recipient_own_loading"] - base_audit["recipient_own_loading"],
        **audit, "scores": output,
    }, correct


def bootstrap_difference(rows, bootstrap):
    draws = bootstrap @ rows
    return {"gain": float(rows.mean()), "interval_95": np.quantile(draws, [0.025, 0.975]).tolist()}


def aggregate(records, correctness, layer, config, bootstrap, restrict_active=False, process=None, names=None):
    keys = [
        key for key, value in records.items()
        if value["cell"]["layer"] == layer and (not restrict_active or value["active"])
        and (process is None or value["cell"]["process"] == process)
        and (names is None or value["cell"]["name"] in names)
    ]
    if not keys:
        return None
    eligible = [key for key in keys if records[key]["active"]]
    suppressed = sum(records[key]["audit_loading_reduced_by_half"] for key in eligible)
    share = suppressed / len(eligible) if eligible else None
    result = {
        "task_count": len(keys), "active_count": len(eligible), "suppression_count": suppressed,
        "suppression_share": share,
    }
    by_arm = {}
    for arm in (*ARMS, "oracle_diagnostic"):
        by_arm[arm] = {metric: float(np.mean([records[key]["scores"][arm][metric] for key in keys]))
                       for metric in ("accuracy", "correct", "repairs", "new_errors", "net_change")}
    result["by_arm"] = by_arm
    result["paired_target_minus"] = {}
    for comparator in ("base", "reverse", "wrong_donor", "joint_linear"):
        delta = np.stack([correctness[key]["target"].astype(float) - correctness[key][comparator] for key in keys]).mean(axis=0)
        result["paired_target_minus"][comparator] = bootstrap_difference(delta, bootstrap)
    result["passes"] = bool(
        share is not None and share >= config["minimum_suppression_share"] and all(
            result["paired_target_minus"][name]["gain"] >= config["minimum_gain"]
            and result["paired_target_minus"][name]["interval_95"][0] > 0 for name in ("base", "reverse")
        )
    )
    return result


def score():
    config, _, identity, cells = verify()
    receipt = read(REVEAL_RECEIPT)
    if receipt["forecast_receipt_sha256"] != digest(RECEIPT) or receipt["design_sha256"] != digest(DESIGN):
        raise ValueError("Labels are not bound to this forecast.")
    if receipt["reveal_sha256"] != digest(REVEAL):
        raise ValueError("Physical labels changed after reveal.")
    count = config["n_test"]
    bootstrap = np.random.default_rng(config["bootstrap_seed"]).multinomial(
        count, np.full(count, 1 / count), size=config["bootstrap_draws"]
    ) / count
    results = {}
    with np.load(REVEAL, allow_pickle=False) as labels:
        for stage, master in config["rounds"].items():
            ref = loaders.iids(master, "reference", config["n_reference"])
            independent = loaders.iids(master, "audit_state", config["n_audit_state"])
            truth = {process: labels[truth_key(stage, process)] for process in config["processes"]}
            records, correctness = {}, {}
            for cell in cells:
                key = cell_key(cell)
                records[key], correctness[key] = score_cell(
                    cell, stage, config, identity, truth[cell["process"]], ref, independent
                )
            results[stage] = {
                "all": {layer: aggregate(records, correctness, layer, config, bootstrap) for layer in ("lag", "mlp")},
                "active_descriptive": {layer: aggregate(records, correctness, layer, config, bootstrap, True) for layer in ("lag", "mlp")},
                "process_descriptive": {
                    layer: {process: aggregate(records, correctness, layer, config, bootstrap, process=process)
                            for process in config["processes"]}
                    for layer in ("lag", "mlp")
                },
                "common14_descriptive": {
                    layer: aggregate(records, correctness, layer, config, bootstrap,
                                     names={cell["name"] for cell in cells if cell["layer"] == "mlp"})
                    for layer in ("lag", "mlp")
                },
                "cells": records,
            }
            print(f"{stage}: scored {len(records)} fixed cells", flush=True)
    write(SUMMARY, {
        "forecast_receipt_sha256": receipt["forecast_receipt_sha256"], "reveal_sha256": receipt["reveal_sha256"],
        "primary_passes_both_rounds": all(value["all"]["lag"]["passes"] for value in results.values()),
        "results": results,
    })


def smoke():
    """Exercise the scorer only on the predeclared engineering namespace."""
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = HERE / "smoke"
    lock = read(root / "DESIGN_LOCK.json")
    receipt = read(root / "RECEIPT.json")
    if (lock["sources"] != receipt["sources"] or lock["config"]["rounds"] != {"smoke": 2026111394}
            or receipt["design_sha256"] != digest(root / "DESIGN_LOCK.json")):
        raise ValueError("Expected the separate smoke-only namespace.")
    _, identity, _ = loaders.load_panel()
    config = lock["config"]
    master = config["rounds"]["smoke"]
    ref = loaders.iids(master, "reference", config["n_reference"])
    audit = loaders.iids(master, "audit_state", config["n_audit_state"])
    for cell in lock["cells"]:
        states = loaders.paths(master, "test", config["n_test"], config["length"], config["processes"][cell["process"]])
        score_cell(cell, "smoke", config, identity, states, ref, audit, root)
    print("Independent direct-score replay, recovery audit and truth-only oracle passed on smoke data.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reveal", "score", "smoke"))
    args = parser.parse_args()
    if args.stage == "score":
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    {"reveal": reveal, "score": score, "smoke": smoke}[args.stage]()


if __name__ == "__main__":
    main()
