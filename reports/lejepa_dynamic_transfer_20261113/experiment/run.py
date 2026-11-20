"""Fit and prospectively score temporally supervised readouts of frozen encoders."""

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parents[1]
ORIGINAL = REPORTS / "lejepa_slow_rank_20261112"
TRAINING = ORIGINAL / "training"
sys.path.insert(0, str(TRAINING))
from run import embed, make_mlp_encoder


ROLE = {"calibration": 1, "validation": 2, "reference": 3, "test": 4, "pairs": 5}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stream(master, role, *parts):
    return np.random.default_rng(np.random.SeedSequence([master, ROLE[role], *parts]))


def trajectories(master, role, rates, count, length):
    innovations = stream(master, role).normal(size=(count, length, 4))
    physical = innovations.copy()
    noise_scale = np.sqrt(1 - rates**2)
    for step in range(1, length):
        physical[:, step] = rates * physical[:, step - 1] + noise_scale * innovations[:, step]
    return physical


def decoder(width):
    return nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 4))


def case_manifest():
    return json.loads((TRAINING / "protocol.json").read_text())


def checkpoint_lock():
    return json.loads((ORIGINAL / "prospective/pipeline_forecast_receipt.json").read_text())["checkpoint_sha256"]


def identity(case, encoder_seed, probe_seed, arm):
    return f"{case}_{encoder_seed}_{probe_seed}_{arm}"


def load_encoder(config, checkpoint_hashes, case, seed):
    path = TRAINING / "runs" / f"{case}_{seed}.pt"
    sha = digest(path)
    if checkpoint_hashes[path.name] != sha:
        raise ValueError(f"frozen encoder changed: {path.name}")
    model = make_mlp_encoder(4, config["hidden"], config["layers"], device="cpu").eval()
    model.load_state_dict(torch.load(path, weights_only=True))
    return model, sha


def make_pair_schedule(config, probe_seed, shape):
    random = stream(config["master"], "pairs", probe_seed)
    size = (config["steps"], config["batch_pairs"])
    lag = random.choice(config["lags"], size=size)
    trajectory = random.integers(shape[0], size=size)
    origin = (random.random(size=size) * (shape[1] - lag)).astype(np.int64)
    start = trajectory * shape[1] + origin
    return start, start + lag


def pair_losses(prediction, target, shuffle):
    start_pred, future_pred = prediction
    start_truth, future_truth = target
    point_errors = (prediction - target).square()
    point = point_errors.mean()
    if shuffle:
        future_pred = torch.roll(future_pred, shifts=1, dims=0)
        future_truth = torch.roll(future_truth, shifts=1, dims=0)
    increment = ((future_pred - start_pred) - (future_truth - start_truth)).square().mean() / 2
    return point, increment


def shuffle_check():
    prediction = torch.arange(64, dtype=torch.float32).reshape(2, 8, 4) / 17
    target = prediction.square()
    plain_point, plain_increment = pair_losses(prediction, target, False)
    shuffled_point, shuffled_increment = pair_losses(prediction, target, True)
    assert plain_point == shuffled_point
    assert plain_increment != shuffled_increment
    shifted_prediction = prediction.clone()
    shifted_target = target.clone()
    shifted_prediction[1] = torch.roll(prediction[1], shifts=1, dims=0)
    shifted_target[1] = torch.roll(target[1], shifts=1, dims=0)
    point_check, increment_check = pair_losses(shifted_prediction, shifted_target, False)
    assert torch.allclose(plain_point, point_check, rtol=1e-6)
    assert shuffled_increment == increment_check


def fixed_lr(config, step):
    fraction = max(0, (2 * step - config["steps"]) / config["steps"])
    return config["adamw_lr"] * (1 + math.cos(math.pi * fraction)) / 2


@torch.inference_mode()
def predict(model, encoded):
    return np.concatenate([model(batch).numpy() for batch in encoded.split(8192)])


def fit_readout(config, method, probe_seed, h_cal, z_cal, h_val, z_val, indices):
    torch.manual_seed(probe_seed)
    model = decoder(config["mlp_width"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["adamw_lr"], weight_decay=config["adamw_weight_decay"]
    )
    temporal_weight = method["lambda"]
    validation_trace = []
    val_flat = z_val.reshape(-1, 4)
    variance = val_flat.var(axis=0)
    start_schedule, future_schedule = [torch.from_numpy(x) for x in indices]
    shape = (2, config["batch_pairs"], 4)
    for step in range(config["steps"]):
        optimizer.param_groups[0]["lr"] = fixed_lr(config, step)
        pair_rows = torch.cat([start_schedule[step], future_schedule[step]])
        prediction = model(h_cal[pair_rows]).reshape(shape)
        truth = z_cal[pair_rows].reshape(shape)
        point, increment = pair_losses(prediction, truth, method["shuffle"])
        loss = (point + temporal_weight * increment) / (1 + temporal_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % config["val_interval"] == 0 or step + 1 == config["steps"]:
            model.eval()
            decoded = predict(model, h_val)
            mse = np.mean((decoded - val_flat)**2, axis=0)
            r2 = 1 - mse / variance
            record = {
                "step": step + 1,
                "r2_by_axis": r2.tolist(),
                "mse_by_axis": mse.tolist(),
                "training_state_mse": point.item(),
                "training_increment_mse": increment.item(),
            }
            validation_trace.append(record)
            model.train()
    return model.eval(), validation_trace


def experiment_sources():
    paths = [Path(__file__), HERE / "protocol.json", HERE / "PROTOCOL.md",
             TRAINING / "protocol.json", TRAINING / "run.py",
             ORIGINAL / "prospective/pipeline_forecast_receipt.json"]
    return {str(path.relative_to(REPORTS)): digest(path) for path in paths}


def check_sources(record):
    if record["source_hashes"] != experiment_sources():
        raise ValueError("experiment sources changed after lock")


def select_cases(original, only_case):
    if only_case:
        return {only_case: original["cases"][only_case]}
    return original["cases"]


def fit(config, original, output, only_case=None, only_encoder=None, only_probe=None, finalize=True):
    if finalize:
        output.mkdir(exist_ok=False)
    elif not output.is_dir():
        raise FileNotFoundError(f"create destination before parallel fit: {output}")
    shuffle_check()
    source_hashes = experiment_sources()
    checkpoint_hashes = checkpoint_lock()
    fit_records, fitted_readouts = {}, {}
    elapsed = time.monotonic()
    cases = select_cases(original, only_case)
    seeds = original["seeds"] if only_encoder is None else [only_encoder]
    probes = config["probe_seeds"] if only_probe is None else [only_probe]
    for case, rates in cases.items():
        rates = np.asarray(rates)
        length = config["calibration_length"]
        calibration = trajectories(config["master"], "calibration", rates, config["n_calibration_trajectories"], length)
        validation = trajectories(config["master"], "validation", rates, config["n_validation_trajectories"], length)
        z_cal = torch.as_tensor(calibration.reshape(-1, 4), dtype=torch.float32)
        for encoder_seed in seeds:
            encoder, encoder_sha = load_encoder(original, checkpoint_hashes, case, encoder_seed)
            cal_embedded = embed(encoder, calibration).reshape(-1, 4)
            mean = cal_embedded.mean(axis=0)
            scale = cal_embedded.std(axis=0)
            if not np.all(scale > 0):
                raise ValueError("constant calibration embedding coordinate")
            h_cal = torch.as_tensor((cal_embedded - mean) / scale, dtype=torch.float32)
            h_val = torch.as_tensor((embed(encoder, validation).reshape(-1, 4) - mean) / scale, dtype=torch.float32)
            for probe_seed in probes:
                schedule = make_pair_schedule(config, probe_seed, calibration.shape)
                schedule_hash = hashlib.sha256(schedule[0].tobytes() + schedule[1].tobytes()).hexdigest()
                for arm, method in config["arms"].items():
                    name = identity(case, encoder_seed, probe_seed, arm)
                    model_path = output / f"{name}.pt"
                    record_path = output / f"{name}.json"
                    if model_path.exists() or record_path.exists():
                        raise FileExistsError(name)
                    started = time.monotonic()
                    readout, validation_trace = fit_readout(
                        config, method, probe_seed, h_cal, z_cal, h_val, validation, schedule
                    )
                    torch.save(
                        {"state": readout.state_dict(), "input_mean": mean.tolist(), "input_std": scale.tolist()},
                        model_path,
                    )
                    record = {
                        "arm": arm,
                        "case": case,
                        "encoder_seed": encoder_seed,
                        "encoder_sha256": encoder_sha,
                        "probe_seed": probe_seed,
                        "pair_schedule_sha256": schedule_hash,
                        "validation_trace": validation_trace,
                        "val_final_r2": validation_trace[-1]["r2_by_axis"],
                        "fit_seconds": time.monotonic() - started,
                    }
                    dump(record_path, record)
                    fitted_readouts[model_path.name] = digest(model_path)
                    fit_records[record_path.name] = digest(record_path)
                    print(f"{name}  {record['fit_seconds']:.2f}s", flush=True)
    result = {
        "source_hashes": source_hashes,
        "config": config,
        "fit_records_sha256": fit_records,
        "fitted_readouts_sha256": fitted_readouts,
        "fit_seconds_elapsed": time.monotonic() - elapsed,
        "architecture": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    if finalize:
        receipt_name = "FITS_RECEIPT.json" if output.name == "fits" else "RECEIPT.json"
        destination = output.parent if output.name == "fits" else output
    else:
        receipt_name = f"FIT_SHARD_{only_case}.json"
        destination = output.parent
    dump(destination / receipt_name, result)


def join_fit_receipts(config, original):
    shards = [
        json.loads((HERE / f"FIT_SHARD_{case}.json").read_text())
        for case in original["cases"]
    ]
    records, weights = {}, {}
    for shard in shards:
        check_sources(shard)
        if shard["config"] != config:
            raise ValueError("fit shards used inconsistent config")
        records.update(shard["fit_records_sha256"])
        weights.update(shard["fitted_readouts_sha256"])
    expected = {
        identity(case, encoder_seed, probe_seed, arm)
        for case in original["cases"]
        for encoder_seed in original["seeds"]
        for probe_seed in config["probe_seeds"]
        for arm in config["arms"]
    }
    if set(records) != {f"{name}.json" for name in expected} or set(weights) != {f"{name}.pt" for name in expected}:
        raise ValueError("fit shard receipts do not cover all expected runs")
    for file, sha in {**records, **weights}.items():
        if digest(HERE / "fits" / file) != sha:
            raise ValueError(f"fitted artifact failed checksum: {file}")
    dump(HERE / "FITS_RECEIPT.json", {
        "source_hashes": experiment_sources(),
        "config": config,
        "fit_records_sha256": records,
        "fitted_readouts_sha256": weights,
        "parallel_case_workers": len(shards),
        "fit_seconds_worker_sum": sum(s["fit_seconds_elapsed"] for s in shards),
        "fit_seconds_worker_max": max(s["fit_seconds_elapsed"] for s in shards),
    })


def ref_states(config):
    return np.stack([
        stream(config["master"], "reference", bank).normal(size=(config["n_reference_states_per_bank"], 4))
        for bank in range(config["n_reference_banks"])
    ])


def test_states(config, rates):
    return trajectories(config["master"], "test", rates, config["n_test_trajectories"], max(config["test_lengths"]))


def forecast(config, original):
    output = HERE / "forecast"
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    fit_receipt = json.loads((HERE / "FITS_RECEIPT.json").read_text())
    check_sources(fit_receipt)
    checkpoint_hashes = checkpoint_lock()
    ref = ref_states(config)
    hashes = {}
    for case, rates in original["cases"].items():
        test = test_states(config, np.asarray(rates))
        for encoder_seed in original["seeds"]:
            encoder, _ = load_encoder(original, checkpoint_hashes, case, encoder_seed)
            test_h = embed(encoder, test).reshape(-1, 4)
            ref_h = embed(encoder, ref).reshape(-1, 4)
            for probe_seed in config["probe_seeds"]:
                for arm in config["arms"]:
                    name = identity(case, encoder_seed, probe_seed, arm)
                    model_path = HERE / "fits" / f"{name}.pt"
                    if digest(model_path) != fit_receipt["fitted_readouts_sha256"][model_path.name]:
                        raise ValueError(f"frozen fit changed: {name}")
                    saved = torch.load(model_path, weights_only=True)
                    model = decoder(config["mlp_width"]).eval()
                    model.load_state_dict(saved["state"])
                    mean, scale = np.asarray(saved["input_mean"]), np.asarray(saved["input_std"])
                    test_prediction = predict(model, torch.as_tensor((test_h - mean) / scale, dtype=torch.float32))
                    ref_prediction = predict(model, torch.as_tensor((ref_h - mean) / scale, dtype=torch.float32))
                    ref_prediction = ref_prediction.reshape(ref.shape).astype(np.float64)
                    ref_mean = ref_prediction.mean(axis=1)
                    ref_second = np.mean(ref_prediction**2, axis=1)
                    ref_variance = ref_prediction.var(axis=1)
                    if not all(np.isfinite(a).all() for a in (test_prediction, ref_variance, ref_mean, ref_second)):
                        raise FloatingPointError(f"nonfinite forecast: {name}")
                    if not np.allclose(ref_variance, ref_second - ref_mean**2, atol=1e-12, rtol=0):
                        raise ValueError("reference variance disagrees with archived sufficient statistics")
                    path = output / f"{name}.npz"
                    np.savez(
                        path,
                        test_predictions=test_prediction.reshape(test.shape),
                        ref_variance=ref_variance,
                        ref_mean=ref_mean,
                        ref_second_moment=ref_second,
                    )
                    hashes[path.name] = digest(path)
            print(f"Forecast ready: {case} encoder {encoder_seed}", flush=True)
    dump(HERE / "FORECAST_RECEIPT.json", {
        "source_hashes": experiment_sources(),
        "fits_receipt_sha256": digest(HERE / "FITS_RECEIPT.json"),
        "forecast_files_sha256": hashes,
        "test_predictions_shape": list(test.shape),
        "reference_moments_shape": list(ref_variance.shape),
        "forecast_elapsed_seconds": time.monotonic() - start,
    })


def reveal(config, original):
    path = HERE / "labels.npz"
    if path.exists():
        raise FileExistsError(path)
    receipt_path = HERE / "FORECAST_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    check_sources(receipt)
    if digest(HERE / "FITS_RECEIPT.json") != receipt["fits_receipt_sha256"]:
        raise ValueError("fit receipt changed after forecast")
    for file, sha in receipt["forecast_files_sha256"].items():
        if digest(HERE / "forecast" / file) != sha:
            raise ValueError(f"forecast changed after lock: {file}")
    labels = {case: test_states(config, np.asarray(rates)) for case, rates in original["cases"].items()}
    np.savez_compressed(path, **labels)
    dump(HERE / "REVEAL_RECEIPT.json", {
        "labels_sha256": digest(path),
        "forecast_receipt_sha256": digest(receipt_path),
        "test_truth_keys": list(labels),
        "test_truth_shape": list(next(iter(labels.values())).shape),
        "test_truth_dtype": str(next(iter(labels.values())).dtype),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "fit", "fit-shard", "join-fits", "forecast", "reveal"))
    parser.add_argument("--case")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((HERE / "protocol.json").read_text())
    original = case_manifest()
    if args.action == "smoke":
        isolated = {**config, "master": config["master"] + 10000, "steps": 500}
        fit(isolated, original, HERE / "smoke", next(iter(original["cases"])), original["seeds"][0], config["probe_seeds"][0])
    elif args.action == "fit":
        fit(config, original, HERE / "fits")
    elif args.action == "fit-shard":
        if args.case not in original["cases"]:
            parser.error("--case must name a locked case for fit-shard")
        fit(config, original, HERE / "fits", args.case, finalize=False)
    elif args.action == "join-fits":
        join_fit_receipts(config, original)
    elif args.action == "forecast":
        forecast(config, original)
    else:
        reveal(config, original)


if __name__ == "__main__":
    main()
