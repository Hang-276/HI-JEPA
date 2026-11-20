"""Replay frozen readouts on independent reference and test streams."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ANALYSIS = EXP.parent / "analysis"
spec = importlib.util.spec_from_file_location("original_experiment", EXP / "run.py")
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
digest = original.digest


def dependency_hashes():
    paths = [
        Path(__file__), HERE / "protocol.json", HERE / "PROTOCOL.md",
        EXP / "run.py", EXP / "protocol.json", EXP / "FITS_RECEIPT.json",
        EXP / "VAL_GATE_RECEIPT.json", ANALYSIS / "VAL_MATCH_LOCK.json",
        ANALYSIS / "score.py", ANALYSIS / "PROTOCOL.md", ANALYSIS / "HISTORICAL_GROUPS.json",
    ]
    # The optional replication scorer wrapper is sealed before the reveal step.
    wrapper = HERE / "score_replication.py"
    if wrapper.exists():
        paths.append(wrapper)
    return {str(path.relative_to(EXP.parent)): digest(path) for path in paths}


def inputs():
    settings = json.loads((HERE / "protocol.json").read_text())
    config = json.loads((EXP / "protocol.json").read_text())
    fits_path = EXP / "FITS_RECEIPT.json"
    fits = json.loads(fits_path.read_text())
    if config != fits["config"]:
        raise ValueError("original fit protocol differs from fit receipt")
    original.check_sources(fits)
    validation = json.loads((ANALYSIS / "VAL_MATCH_LOCK.json").read_text())
    if digest(fits_path) != validation["fits_receipt_sha256"]:
        raise ValueError("unrelated VAL lock")
    if digest(ANALYSIS / "score.py") != validation["scorer_sha256"]:
        raise ValueError("original scorer changed")
    if digest(ANALYSIS / "PROTOCOL.md") != validation["analysis_protocol_sha256"]:
        raise ValueError("original analysis protocol changed")
    independent_lock = json.loads((EXP / "VAL_GATE_RECEIPT.json").read_text())
    if set(validation["groups"]["matched"]) != set(independent_lock["matched_encoders"]):
        raise ValueError("different locked validation sets")
    return {**config, "master": settings["master"]}, fits, original.case_manifest()


def expected_predictions(config, training):
    return {
        f"{original.identity(case, encoder_seed, probe_seed, arm)}.npz"
        for case in training["cases"]
        for encoder_seed in training["seeds"]
        for probe_seed in config["probe_seeds"]
        for arm in config["arms"]
    }


def standardized_panel(test, reference, fitted):
    mean, scale = np.asarray(fitted["input_mean"]), np.asarray(fitted["input_std"])
    return (
        torch.as_tensor((test - mean) / scale, dtype=torch.float32),
        torch.as_tensor((reference - mean) / scale, dtype=torch.float32),
    )


def forecast(config, fits, training):
    output = HERE / "forecast"
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    checkpoint_hashes = original.checkpoint_lock()
    reference = original.ref_states(config)
    hashes = {}
    for case, rates in training["cases"].items():
        test = original.test_states(config, np.asarray(rates))
        for encoder_seed in training["seeds"]:
            encoder, _ = original.load_encoder(training, checkpoint_hashes, case, encoder_seed)
            test_h = original.embed(encoder, test).reshape(-1, 4)
            ref_h = original.embed(encoder, reference).reshape(-1, 4)
            for probe_seed in config["probe_seeds"]:
                for arm in config["arms"]:
                    name = original.identity(case, encoder_seed, probe_seed, arm)
                    checkpoint = EXP / "fits" / f"{name}.pt"
                    if digest(checkpoint) != fits["fitted_readouts_sha256"][checkpoint.name]:
                        raise ValueError(f"frozen fit changed: {name}")
                    fitted = torch.load(checkpoint, weights_only=True)
                    model = original.decoder(config["mlp_width"]).eval()
                    model.load_state_dict(fitted["state"])
                    test_input, ref_input = standardized_panel(test_h, ref_h, fitted)
                    test_pred = original.predict(model, test_input).reshape(test.shape)
                    ref_pred = original.predict(model, ref_input).reshape(reference.shape).astype(np.float64)
                    ref_mean = ref_pred.mean(axis=1)
                    ref_second = np.mean(ref_pred**2, axis=1)
                    ref_variance = ref_pred.var(axis=1)
                    if not np.isfinite(test_pred).all() or not np.all(ref_variance > 0):
                        raise ValueError(f"invalid prediction or denominator: {name}")
                    if not np.allclose(ref_variance, ref_second - ref_mean**2, atol=1e-12, rtol=0):
                        raise ValueError("inconsistent reference moments")
                    file = output / f"{name}.npz"
                    np.savez(
                        file,
                        test_predictions=test_pred,
                        ref_variance=ref_variance,
                        ref_mean=ref_mean,
                        ref_second_moment=ref_second,
                    )
                    hashes[file.name] = digest(file)
            print(case, encoder_seed, flush=True)
    if set(hashes) != expected_predictions(config, training):
        raise ValueError("incomplete replication forecasts")
    original.dump(HERE / "FORECAST_RECEIPT.json", {
        "source_hashes": dependency_hashes(),
        "fits_receipt_sha256": digest(EXP / "FITS_RECEIPT.json"),
        "forecast_files_sha256": hashes,
        "test_predictions_shape": list(test.shape),
        "reference_moments_shape": list(ref_variance.shape),
        "replication_master": config["master"],
        "forecast_elapsed_seconds": time.monotonic() - start,
    })


def reveal(config, training):
    path = HERE / "labels.npz"
    if path.exists():
        raise FileExistsError(path)
    receipt_path = HERE / "FORECAST_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["source_hashes"] != dependency_hashes():
        raise ValueError("replication dependencies changed after forecast")
    if set(receipt["forecast_files_sha256"]) != expected_predictions(config, training):
        raise ValueError("incomplete replication forecasts")
    for file, sha in receipt["forecast_files_sha256"].items():
        if digest(HERE / "forecast" / file) != sha:
            raise ValueError(f"forecast changed after lock: {file}")
    labels = {case: original.test_states(config, np.asarray(rates)) for case, rates in training["cases"].items()}
    np.savez_compressed(path, **labels)
    original.dump(HERE / "REVEAL_RECEIPT.json", {
        "labels_sha256": digest(path),
        "forecast_receipt_sha256": digest(receipt_path),
        "test_truth_keys": list(labels),
        "test_truth_shape": list(next(iter(labels.values())).shape),
        "test_truth_dtype": str(next(iter(labels.values())).dtype),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("forecast", "reveal"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config, fits, training = inputs()
    if args.stage == "forecast":
        forecast(config, fits, training)
    else:
        reveal(config, training)


if __name__ == "__main__":
    main()
