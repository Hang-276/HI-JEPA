"""Seal both rounds of window-level intervention predictions."""

import importlib.util
import itertools
from pathlib import Path
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_PATH = HERE.parent / "lejepa_direction_a1_20261113/controlled/controlled.py"
spec = importlib.util.spec_from_file_location("a1_controlled_source", SOURCE_PATH)
SOURCE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SOURCE)
digest, read, write = SOURCE.digest, SOURCE.read, SOURCE.write
paths, moment = SOURCE.paths, SOURCE.calculate_moments
RECIPIENT = SOURCE.RECIPIENT


def fit_arms(readout, config, master):
    state = SOURCE.iids(master, "state_cal", config["n_state_cal"])
    visible = readout(state)
    residual = visible[:, RECIPIENT] - state[:, RECIPIENT]
    arms = []
    for dose, donor, sign in itertools.product(config["deltas"], SOURCE.DONORS, (1, -1)):
        donor_state = visible[:, donor]
        b = np.mean(residual * donor_state)
        c = np.mean(donor_state**2)
        coefficient = (-b + sign * np.sqrt(b**2 + c * dose)) / c
        extra = np.mean((residual + coefficient * donor_state)**2 - residual**2)
        np.testing.assert_allclose(extra, dose, rtol=0, atol=5e-14)
        arms.append((coefficient, donor, dose, sign, extra))
    return {
        key: np.asarray(column, dtype=dtype)
        for key, dtype, column in zip(
            ("coefficients", "donors", "doses", "signs", "cal_extra_mse"),
            (float, int, float, int, float), zip(*arms),
        )
    }


def fit_physical(readout, config, master, rates):
    x, y = SOURCE.with_physical_labels(readout, master, "fit", config["n_fit"], config, rates)
    val_x, val_y = SOURCE.with_physical_labels(
        readout, master, "validation", config["n_validation"], config, rates
    )
    return SOURCE.fit_ridge(SOURCE.feature_matrix(x, "diag"), y,
                            SOURCE.feature_matrix(val_x, "diag"), val_y, config)


def predict(visible, reference, config, arms, physical_fit):
    gram = moment(visible.astype(np.float64), config["lags"])
    numerator = SOURCE.feature_matrix(gram, "diag")
    covariance = np.cov(reference.astype(np.float64), rowvar=False, ddof=0)
    variance = covariance.diagonal()
    baseline = numerator / variance
    physical = SOURCE.top_two(SOURCE.apply_ridge(numerator, physical_fit))
    base_correct = np.all(SOURCE.top_two(baseline) == physical, axis=1).astype(np.int8)
    forecasts = {key: [] for key in (
        "pred_full", "pred_drop", "prob_abs", "prob_full", "proposed_full_top", "signed_opportunity"
    )}
    categories = np.eye(3, dtype=np.float32)
    for a, donor in zip(arms["coefficients"], arms["donors"]):
        denominator = variance[RECIPIENT] + 2 * a * covariance[RECIPIENT, donor] + a**2 * variance[donor]
        if not denominator > 0:
            raise ValueError("Nonpositive intervened reference variance")
        even = numerator[:, RECIPIENT] + a**2 * numerator[:, donor]
        odd = 2 * a * gram[:, RECIPIENT, donor]
        tops = []
        effects = []
        for cross in (odd, 0, np.abs(odd), -np.abs(odd)):
            scores = baseline.copy()
            scores[:, RECIPIENT] = (even + cross) / denominator
            top = SOURCE.top_two(scores)
            tops.append(top)
            effects.append(np.all(top == physical, axis=1).astype(np.int8) - base_correct)
        full, drop, plus, minus = effects
        forecasts["pred_full"].append(full)
        forecasts["pred_drop"].append(drop)
        forecasts["prob_full"].append(categories[full + 1])
        forecasts["prob_abs"].append((categories[plus + 1] + categories[minus + 1]) / 2)
        forecasts["proposed_full_top"].append(tops[0])
        forecasts["signed_opportunity"].append(np.any(tops[2] != tops[3], axis=1))
    return {
        **{key: np.stack(value) for key, value in forecasts.items()},
        "pred_physical_top": physical,
    }


def source_hashes(identity):
    _, _, names, encoder_hashes, decoder_hashes = identity
    files = [HERE / filename for filename in ("run.py", "score.py", "protocol.json", "PROTOCOL.md")]
    files += [SOURCE.REPORTS / name for name in SOURCE.sources()]
    files.append(SOURCE.READOUT / "runs/results.json")
    training = SOURCE.helper.source
    training.source_provenance()
    files.append(training.SOURCE / "LEJEPA_SOURCE_LOCK.json")
    vendor_package = training.VENDOR / "experiments/lejepa_id"
    files.extend(vendor_package / name for name in ("__init__.py", "losses.py", "models.py"))
    result = {str(path.relative_to(ROOT)): digest(path) for path in files}
    for name in names:
        encoder = SOURCE.TRAIN / "runs" / f"{name}.pt"
        decoder = SOURCE.READOUT / "runs" / f"{name}_readouts.pt"
        for path, expected in ((encoder, encoder_hashes[encoder.name]), (decoder, decoder_hashes[name])):
            if digest(path) != expected:
                raise ValueError(f"Unexpected frozen weight {path}")
            result[str(path.relative_to(ROOT))] = expected
    return result


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = read(HERE / "protocol.json")
    identity = SOURCE.identity()
    sources = source_hashes(identity)
    write(HERE / "DESIGN_LOCK.json", {"sources": sources, "config": config})
    output = HERE / "forecast"
    output.mkdir()
    started = time.monotonic()
    old, readout_config, names, encoder_hashes, decoder_hashes = identity
    files = {}
    tasks = []
    master_cal = config["rounds"]["primary"]
    processes = {key: np.asarray(rates) for key, rates in config["processes"].items()}
    for name in names:
        readout, _ = SOURCE.frozen_model(name, old, readout_config, encoder_hashes, decoder_hashes)
        arms = fit_arms(readout, config, master_cal)
        fits = {process: fit_physical(readout, config, master_cal, rates) for process, rates in processes.items()}
        fit_path = output / f"{name}_fit.json"
        write(fit_path, {"physical_fits": fits, **{key: value.tolist() for key, value in arms.items()}})
        files[fit_path.name] = digest(fit_path)
        for round_name, master in config["rounds"].items():
            reference = readout(SOURCE.iids(master, "reference", config["n_reference"])).astype(np.float32)
            for process, rates in processes.items():
                test = paths(master, "test", config["n_test"], config["length"], rates)
                visible = readout(test).astype(np.float32)
                forecast = predict(visible, reference, config, arms, fits[process])
                filename = f"{round_name}__{process}__{name}.npz"
                with (output / filename).open("xb") as handle:
                    np.savez_compressed(handle, visible=visible, reference=reference, **arms, **forecast)
                files[filename] = digest(output / filename)
                tasks.append({"round": round_name, "process": process, "model": name, "filename": filename})
        print(f"{name}  {time.monotonic() - started:.2f}s", flush=True)
    if source_hashes(identity) != sources:
        raise ValueError("Sources changed during forecast")
    write(output / "RECEIPT.json", {
        "files": files, "tasks": tasks, "sources": sources,
        "config_sha256": digest(HERE / "protocol.json"), "design_sha256": digest(HERE / "DESIGN_LOCK.json"),
        "seconds": time.monotonic() - started,
    })


if __name__ == "__main__":
    main()
