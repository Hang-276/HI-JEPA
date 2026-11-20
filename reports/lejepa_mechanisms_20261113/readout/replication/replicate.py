"""Independent two-stage replication with frozen LeJEPA diagnostic readouts."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
READOUT = HERE.parent
spec = importlib.util.spec_from_file_location("readout_run", READOUT / "run.py")
readout = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = readout
spec.loader.exec_module(readout)
RULES = readout.RULES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata():
    manifest = json.loads((READOUT / "runs/results.json").read_text())
    if digest(READOUT / "run.py") != manifest["source_sha256"]:
        raise ValueError("original readout code changed")
    config = json.loads((HERE / "protocol.json").read_text())
    for key in ("n_test", "n_reference", "trajectory_length", "lags"):
        if config[key] != manifest["config"][key]:
            raise ValueError(f"unexpected budget change: {key}")
    return manifest, config


def data_inputs(config, rates):
    reference = readout.stream(config["master"], "reference").normal(size=(config["n_reference"], 4))
    innovation = readout.stream(config["master"], "test").normal(
        size=(config["n_test"], config["trajectory_length"], 4)
    )
    test = {}
    for case, rho in rates.items():
        rho = np.asarray(rho)
        paths = innovation.copy()
        noise = np.sqrt(1 - rho**2)
        for step in range(1, config["trajectory_length"]):
            paths[:, step] = rho * paths[:, step - 1] + noise * innovation[:, step]
        test[case] = paths
    return reference, test


def predictions(record, cases, reference, weights, config, encoder):
    def encode(states):
        h = readout.inputs(record["mode"], states, encoder).reshape(-1, 4)
        return (h - weights["input_mean"]) / weights["input_std"]

    encoded_reference = encode(reference)
    encoded_test = {key: encode(test) for key, test in cases.items()}
    predictions = {}
    for rule in RULES:
        if rule == "mlp":
            decoder = readout.mlp(4, config["mlp_width"]).eval()
            decoder.load_state_dict(weights[rule])
            def predict(x):
                return readout.predict_mlp(x, decoder)
        else:
            def predict(x):
                return readout.predict_ridge(x, weights[rule], rule == "cubic")
        predictions[f"{rule}_reference_variance"] = predict(encoded_reference).var(axis=0)
        for case, test in cases.items():
            predicted = predict(encoded_test[case]).reshape(test.shape)
            predictions[f"{case}_{rule}_states"] = predicted.astype(np.float32)
    return predictions


def forecast(manifest, config):
    output = HERE / "forecast"
    output.mkdir(exist_ok=False)
    parent_config = json.loads((readout.TRAINING / "protocol.json").read_text())
    rates = parent_config["cases"]
    reference, tests = data_inputs(config, rates)
    fit_source = {}
    files = {}
    for name, record in manifest["runs"].items():
        fit_path = READOUT / "runs" / f"{name}_readouts.pt"
        fit_source[name] = digest(fit_path)
        weights = torch.load(fit_path, weights_only=False)
        encoder = None
        if record["mode"] == "frozen":
            checkpoint = readout.TRAINING / "runs" / f"{name}.pt"
            if digest(checkpoint) != record["encoder_sha256"]:
                raise ValueError(f"checkpoint changed: {name}")
            encoder = readout.make_mlp_encoder(
                4, parent_config["hidden"], parent_config["layers"], device="cpu"
            ).eval()
            encoder.load_state_dict(torch.load(checkpoint, weights_only=True))
        gap = record["case"].split("_")[-1]
        cases = {key: test for key, test in tests.items() if key.endswith("_" + gap)}
        predicted = predictions(record, cases, reference, weights, manifest["config"], encoder)
        file = output / f"{name}.npz"
        np.savez_compressed(file, **predicted)
        files[file.name] = digest(file)
        print(name, flush=True)
    readout.json_dump(HERE / "FORECAST_RECEIPT.json", {
        "source_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(HERE / "protocol.json"),
        "protocol_text_sha256": digest(HERE / "PROTOCOL.md"),
        "original_summary_sha256": digest(READOUT / "runs/results.json"),
        "fitted_readouts_sha256": fit_source,
        "forecast_files_sha256": files,
    })


def summarize_targets(results, config):
    strong_cases = {}
    optional = {}
    for name, panel in results.items():
        if panel["mode"] != "frozen":
            continue
        if panel["case"].startswith("crossed") and panel["seed"] in config["strong_recovery_seeds"]:
            strong_cases[name] = {
                case: (
                    metrics["mlp"]["fastest_r2"] >= config["strong_recovery_r2_threshold"]
                    and metrics["mlp"]["top2_delta_vs_linear"] > 0
                )
                for case, metrics in panel["evaluation"].items()
            }
        if panel["case"].startswith("crossed") and panel["seed"] == config["secondary_seed"]:
            same = panel["evaluation"][panel["case"]]
            optional[name] = (
                same["mlp"]["fastest_r2"] > same["cubic"]["fastest_r2"]
                and same["mlp"]["top2_accuracy"] < same["cubic"]["top2_accuracy"]
            )
    case_means = {}
    cases = sorted({panel["case"] for panel in results.values()})
    for train_case in cases:
        grouped = [panel for panel in results.values() if panel["case"] == train_case and panel["mode"] == "frozen"]
        case_means[train_case] = {}
        for eval_case in grouped[0]["evaluation"]:
            rows = [panel["evaluation"][eval_case] for panel in grouped]
            case_means[train_case][eval_case] = {
                rule: {
                    "fastest_r2_mean": float(np.mean([row[rule]["fastest_r2"] for row in rows])),
                    "top2_accuracy_mean": float(np.mean([row[rule]["top2_accuracy"] for row in rows])),
                    "n_seeds_top2_improved_vs_linear": sum(row[rule]["top2_delta_vs_linear"] > 0 for row in rows),
                    "n_seeds_fastest_r2_improved_vs_linear": sum(row[rule]["fastest_r2_delta_vs_linear"] > 0 for row in rows),
                }
                for rule in RULES
            }
    return {"strong_recovery_predeclared": strong_cases, "secondary_7811_reproduced": optional, "case_means": case_means}


def reveal(manifest, config):
    output = HERE / "RESULTS.json"
    if output.exists():
        raise FileExistsError("replication reveal already exists")
    receipt = json.loads((HERE / "FORECAST_RECEIPT.json").read_text())
    assert digest(Path(__file__)) == receipt["source_sha256"]
    assert digest(HERE / "protocol.json") == receipt["protocol_sha256"]
    assert digest(HERE / "PROTOCOL.md") == receipt["protocol_text_sha256"]
    assert digest(READOUT / "runs/results.json") == receipt["original_summary_sha256"]
    assert len(receipt["forecast_files_sha256"]) == len(manifest["runs"])
    for name, expected in receipt["fitted_readouts_sha256"].items():
        assert digest(READOUT / "runs" / f"{name}_readouts.pt") == expected
    for file, expected in receipt["forecast_files_sha256"].items():
        assert digest(HERE / "forecast" / file) == expected
    rates = json.loads((readout.TRAINING / "protocol.json").read_text())["cases"]
    _, physical = data_inputs(config, rates)
    true_top = {
        case: readout.top_two(readout.increment_statistics(states, config["lags"]))
        for case, states in physical.items()
    }
    results = {}
    for name, panel in manifest["runs"].items():
        rows = {}
        with np.load(HERE / "forecast" / f"{name}.npz", allow_pickle=False) as archive:
            for eval_case in panel["readouts"]["linear"]["evaluation"]:
                physical_states = physical[eval_case]
                baseline_correct = None
                baseline_r2 = None
                metrics = {}
                for rule in RULES:
                    estimates = archive[f"{eval_case}_{rule}_states"].astype(np.float64)
                    factor_r2 = readout.r2(physical_states.reshape(-1, 4), estimates.reshape(-1, 4)).tolist()
                    variance = archive[f"{rule}_reference_variance"]
                    if not np.all(variance > 0):
                        raise ValueError("undefined model readout denominator")
                    predicted_top = readout.top_two(
                        readout.increment_statistics(estimates, config["lags"]) / variance
                    )
                    correct = np.all(predicted_top == true_top[eval_case], axis=1)
                    if rule == "linear":
                        baseline_correct = correct
                        baseline_r2 = factor_r2[3]
                    fixes = int((correct & ~baseline_correct).sum())
                    new_errors = int((~correct & baseline_correct).sum())
                    metrics[rule] = {
                        "factor_r2": factor_r2,
                        "fastest_r2": factor_r2[3],
                        "fastest_r2_delta_vs_linear": factor_r2[3] - baseline_r2,
                        "top2_accuracy": float(correct.mean()),
                        "top2_delta_vs_linear": (fixes - new_errors) / len(correct),
                        "top2_fixes_vs_linear": fixes,
                        "top2_new_errors_vs_linear": new_errors,
                        "reference_variance": variance.tolist(),
                    }
                rows[eval_case] = metrics
        results[name] = {key: panel[key] for key in ("case", "mode", "seed")}
        results[name]["evaluation"] = rows
    readout.json_dump(output, {
        "receipt_sha256": digest(HERE / "FORECAST_RECEIPT.json"),
        "n_fit_panels": len(results),
        "n_fit_evaluation_pairs": sum(len(panel["evaluation"]) for panel in results.values()),
        **summarize_targets(results, config),
        "results": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("forecast", "reveal"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest, config = metadata()
    (forecast if args.stage == "forecast" else reveal)(manifest, config)


if __name__ == "__main__":
    main()
