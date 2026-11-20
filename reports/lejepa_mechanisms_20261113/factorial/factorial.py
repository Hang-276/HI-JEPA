"""Forecast the eight LeJEPA readout arms and separately reveal the physical targets."""

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parents[1] / "lejepa_slow_rank_20261112"
TRAINING = PRIOR / "training"
PIPELINE = PRIOR / "prospective"
sys.path.insert(0, str(PIPELINE))
import pipeline as prior
from score import sha256, top2, write_json


ARMS = np.array(list(itertools.product(range(2), repeat=3)))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def seed(config, role, *parts):
    return np.random.default_rng(np.random.SeedSequence([config["master_rng"], config["roles"][role], *parts]))


def ar(config, role, bank, count, rates):
    innovations = seed(config, role, bank).normal(size=(count, config["trajectory_length"], 4))
    return prior.innovations_to_paths(rates, innovations)


def write_npz(path, **values):
    with path.open("xb") as destination:
        np.savez_compressed(destination, **values)


def current_provenance(train):
    checkpoints = {
        f"{condition}_{model_seed}": sha256(TRAINING / "runs" / f"{condition}_{model_seed}.pt")
        for condition in train["cases"]
        for model_seed in train["seeds"]
    }
    return {
        "protocol_sha256": sha256(HERE / "protocol.json"),
        "protocol_notes_sha256": sha256(HERE / "PROTOCOL.md"),
        "runner_sha256": sha256(Path(__file__)),
        "score_sha256": sha256(HERE / "score_factorial.py"),
        "training_runner_sha256": sha256(TRAINING / "run.py"),
        "training_protocol_sha256": sha256(TRAINING / "protocol.json"),
        "prior_pipeline_sha256": sha256(Path(prior.__file__)),
        "checkpoint_sha256": checkpoints,
        "author_source": prior.source_provenance(),
    }


def environment():
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "machine": platform.machine(),
        "os": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mps_available": torch.backends.mps.is_available(),
        "device": "cpu",
    }


def banks(config, train):
    return [
        (name, model_seed, repeat, np.asarray(rates))
        for name, rates in train["cases"].items()
        for model_seed in train["seeds"]
        for repeat in range(config["banks_per_checkpoint"])
    ]


def model_for(train, name, model_seed):
    path = TRAINING / "runs" / f"{name}_{model_seed}.pt"
    model = prior.make_mlp_encoder(4, train["hidden"], train["layers"], device="cpu").eval()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    return model


def predict_bank(model, config, bank, rates):
    calibration = prior.embed(model, ar(config, "calibration", bank, config["calibration_trajectories"][1], rates))
    bases = seed(config, "anchors", bank).normal(size=(config["named_anchor_bases"][1], 4))
    dose = config["anchor_dose"]
    delta = np.eye(4)[:, None, :] * dose
    responses = (prior.embed(model, bases[None] + delta) - prior.embed(model, bases[None] - delta)) / (2 * dose)
    reference_trajectories = ar(config, "reference_trajectory", bank, config["reference_trajectories"], rates)
    reference_iid = seed(config, "reference_iid", bank).normal(size=(config["reference_iid_states"], 4))
    reference = [prior.embed(model, states).reshape(-1, 4) for states in (reference_trajectories, reference_iid)]
    test = ar(config, "test", bank, config["test_windows_per_bank"], rates)
    features = prior.embed(model, test)
    feature_moments = prior.dynamic_moments(features, config["lags"])

    fitted = [prior.fit_lag_axes(calibration[:count]) for count in config["calibration_trajectories"]]
    response_prefixes = [responses[:, :count].mean(axis=1) for count in config["named_anchor_bases"]]
    unpermuted = np.stack([row[0] for row in fitted])
    ordered = np.empty((8, 4, 4))
    permutations = np.empty((8, 4), dtype=np.int64)
    refvar = np.empty((8, 4))
    naming_gap = np.empty(8)
    numerator = np.empty((8, len(test), 4))
    for index, (cal, naming, ref) in enumerate(ARMS):
        names, normalized = prior.assign_names(response_prefixes[naming] @ unpermuted[cal])
        assignment_scores = np.sort(normalized[np.arange(4)[None, :], prior.PERMUTATIONS].sum(axis=1))
        naming_gap[index] = assignment_scores[-1] - assignment_scores[-2]
        permutations[index] = names
        ordered[index] = unpermuted[cal][:, names]
        refvar[index] = (reference[ref] @ ordered[index]).var(axis=0)
        numerator[index] = np.einsum("dj,nde,ej->nj", ordered[index], feature_moments, ordered[index])
    if np.min(refvar) <= 1e-12:
        raise ValueError("Undefined reference variance.")
    scores = numerator / refvar[:, None, :]
    for cal, ref in itertools.product(range(2), repeat=2):
        low, high = cal * 4 + ref, cal * 4 + 2 + ref
        np.testing.assert_allclose(np.sort(scores[low], axis=1), np.sort(scores[high], axis=1), atol=2e-12)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite forecast scores.")
    return {
        "numerators": numerator,
        "scores": scores,
        "top2": np.stack([top2(arm_scores) for arm_scores in scores]),
        "basis": unpermuted,
        "named_basis": ordered,
        "permutation": permutations,
        "reference_variance": refvar,
        "assignment_gap": naming_gap,
        "lag_eigenvalues": np.stack([row[1] for row in fitted]),
        "calibration_eigenvalues": np.stack([row[2] for row in fitted]),
    }


def forecast(config, train):
    provenance = current_provenance(train)
    started = time.monotonic()
    records = []
    current = None
    for bank, (name, model_seed, repeat, rates) in enumerate(banks(config, train)):
        if current != (name, model_seed):
            model = model_for(train, name, model_seed)
            current = (name, model_seed)
        records.append(predict_bank(model, config, bank, rates))
        if (bank + 1) % 4 == 0:
            print(f"{bank + 1}/64 complete, {time.monotonic() - started:.1f}s", flush=True)
    output = HERE / "FORECAST.npz"
    write_npz(output, arms=ARMS, **{key: np.stack([record[key] for record in records]) for key in records[0]})
    write_json(
        HERE / "FORECAST_LOCK.json",
        {**provenance, "forecast_sha256": sha256(output), "environment": environment(), "seconds": time.monotonic() - started},
    )


def verify_forecast(train):
    expected = load(HERE / "FORECAST_LOCK.json")
    if any(expected[key] != value for key, value in current_provenance(train).items()):
        raise ValueError("Forecast source integrity mismatch.")
    if expected["forecast_sha256"] != sha256(HERE / "FORECAST.npz"):
        raise ValueError("Forecast artifact integrity mismatch.")
    return expected


def covariance(values):
    values = values.reshape(-1, values.shape[-1])
    centered = values - values.mean(axis=0)
    return centered.T @ centered / len(centered)


def reveal(config, train):
    locked = verify_forecast(train)
    started = time.monotonic()
    scores = []
    stats = []
    current = None
    with np.load(HERE / "FORECAST.npz", allow_pickle=False) as forecast_file:
        for bank, (name, model_seed, repeat, rates) in enumerate(banks(config, train)):
            if current != (name, model_seed):
                model = model_for(train, name, model_seed)
                current = (name, model_seed)
            physical = ar(config, "test", bank, config["test_windows_per_bank"], rates)
            scores.append(np.diagonal(prior.dynamic_moments(physical, config["lags"]), axis1=1, axis2=2))
            features = prior.embed(model, physical)
            covariance_all = covariance(np.concatenate((physical, features), axis=-1))
            covariance_z = covariance_all[:4, :4]
            covariance_x = covariance_all[4:, 4:]
            cross = covariance_all[:4, 4:]
            beta = np.linalg.solve(covariance_z, cross)
            all_axes = forecast_file["named_basis"][bank]
            slopes = beta @ all_axes
            direction_var = np.einsum("bdj,de,bej->bj", all_axes, covariance_x, all_axes)
            total_linear_var = np.einsum("bdj,de,bej->bj", slopes, covariance_z, slopes)
            named_slope = np.diagonal(slopes, axis1=1, axis2=2)
            other_slopes = slopes.copy()
            indices = np.arange(4)
            other_slopes[:, indices, indices] = 0
            other_var = np.einsum("bdj,de,bej->bj", other_slopes, covariance_z, other_slopes)
            stats.append(
                {
                    "physical_slopes": slopes,
                    "named_slopes": named_slope,
                    "linear_cross_axis_fraction": other_var / direction_var,
                    "residual_fraction": (direction_var - total_linear_var) / direction_var,
                    "reference_physical_scale": forecast_file["reference_variance"][bank] / named_slope**2,
                }
            )
    path = HERE / "REVEAL.npz"
    write_npz(
        path,
        physical_scores=np.stack(scores),
        forecasts_sha256=locked["forecast_sha256"],
        **{key: np.stack([record[key] for record in stats]) for key in stats[0]},
    )
    write_json(
        HERE / "REVEAL_LOCK.json",
        {"reveal_sha256": sha256(path), "forecast_lock_sha256": sha256(HERE / "FORECAST_LOCK.json"), "seconds": time.monotonic() - started},
    )


def smoke(config, train):
    config = {**config, "master_rng": config["smoke_rng"], "test_windows_per_bank": 12}
    name, model_seed, repeat, rates = banks(config, train)[0]
    result = predict_bank(model_for(train, name, model_seed), config, 0, rates)
    assert result["scores"].shape == (8, 12, 4)
    print("CPU checkpoint compatibility, forecast shapes, and pure-permutation invariance passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "forecast", "reveal"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = load(HERE / "protocol.json")
    train = load(TRAINING / "protocol.json")
    {"smoke": smoke, "forecast": forecast, "reveal": reveal}[args.stage](config, train)


if __name__ == "__main__":
    main()
