"""Train the locked author objective on a prespecified controlled AR family."""

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO / "work/hcf_stable_id_20261027/v1"
VENDOR = SOURCE / "vendor/lejepa_identifiability"
sys.path.insert(0, str(VENDOR / "experiments"))
from lejepa_id.losses import SIGReg, alignment_loss
from lejepa_id.models import make_mlp_encoder


HADAMARD = torch.tensor(
    [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
    dtype=torch.float32,
) / 2


def source_provenance():
    lock = json.loads((SOURCE / "LEJEPA_SOURCE_LOCK.json").read_text())
    hashes = {}
    for name in ("losses.py", "models.py"):
        relative = f"experiments/lejepa_id/{name}"
        digest = hashlib.sha256((VENDOR / relative).read_bytes()).hexdigest()
        if digest != lock["author_files_sha256"][relative]:
            raise ValueError(f"Unexpected author source content: {relative}")
        hashes[relative] = digest
    return {"repository": lock["repository"], "commit": lock["commit"], "sha256": hashes}


def observe(z):
    return torch.stack(
        (
            z[..., 0] + 0.55 * torch.sin(1.5 * z[..., 1]),
            z[..., 1] + 0.4 * (z[..., 2].square() - 1),
            z[..., 2] + 0.55 * torch.sin(1.5 * z[..., 3]),
            z[..., 3],
        ),
        dim=-1,
    ) @ HADAMARD


def train(config, seed, rates, steps):
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000_000)
    model = make_mlp_encoder(4, config["hidden"], config["layers"], device="cpu")
    sigreg = SIGReg(n_slices=config["sigreg_slices"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    rho = torch.as_tensor(rates, dtype=torch.float32)
    noise_scale = torch.sqrt(1 - rho.square())
    weight = config["sigreg_weight"]
    pair_moments = np.zeros((5, 4))
    logs = []
    for step in range(steps):
        decay_fraction = max(0, 2 * step / steps - 1)
        optimizer.param_groups[0]["lr"] = config["lr"] * (1 + math.cos(math.pi * decay_fraction)) / 2
        current = torch.randn((config["batch"], 4), generator=generator)
        innovation = torch.randn(current.shape, generator=generator)
        future = rho * current + noise_scale * innovation
        pair_moments += torch.stack(
            [current, future, current.square(), future.square(), current * future]
        ).sum(dim=1).numpy()
        states = torch.stack((current, future))
        embedded = model(observe(states))
        alignment = alignment_loss(embedded)
        penalty = sigreg(embedded)
        loss = (1 - weight) * alignment + weight * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 800 == 0 or step == steps - 1:
            values = [alignment.item(), penalty.item(), loss.item()]
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError(f"nonfinite loss at step {step + 1}")
            logs.append({"step": step + 1, "alignment": values[0], "sigreg": values[1], "loss": values[2]})
    mean_current, mean_future, second_current, second_future, cross = pair_moments / (steps * config["batch"])
    denominator = np.sqrt((second_current - mean_current**2) * (second_future - mean_future**2))
    pair_correlation = (cross - mean_current * mean_future) / denominator
    return model.eval(), logs, pair_correlation


def trajectory(rates, seed, split, count, length):
    random = np.random.default_rng(np.random.SeedSequence([seed, split]))
    innovations = random.normal(size=(count, length, 4))
    sequence = innovations.copy()
    scale = np.sqrt(1 - rates**2)
    for t in range(1, length):
        sequence[:, t, :] = rates * sequence[:, t - 1, :] + scale * innovations[:, t, :]
    return sequence


@torch.inference_mode()
def embed(model, states):
    flat = torch.as_tensor(states.reshape(-1, 4), dtype=torch.float32)
    batches = [model(observe(batch)).numpy() for batch in flat.split(8192)]
    return np.concatenate(batches).astype(np.float64).reshape(states.shape)


def fit_lag_axes(latent):
    flat = latent.reshape(-1, 4)
    centered = flat - flat.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    values, vectors = np.linalg.eigh(covariance)
    if values[0] <= 1e-8:
        raise ValueError(f"singular calibration covariance: {values}")
    whitening = (vectors / np.sqrt(values)) @ vectors.T
    h = (centered @ whitening).reshape(latent.shape)
    left = h[:, :-1, :].reshape(-1, 4)
    right = h[:, 1:, :].reshape(-1, 4)
    covariance_lag = (left.T @ right + right.T @ left) / (2 * len(left))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_lag)
    return whitening @ eigenvectors[:, ::-1], eigenvalues[::-1], values


def assign_names(responses):
    absolute = np.abs(responses)
    norm = np.linalg.norm(absolute, axis=0)
    normalized = absolute / np.maximum(norm, 1e-12)
    axes = range(4)
    names = max(itertools.permutations(axes), key=lambda order: normalized[axes, order].sum())
    return np.array(names), normalized


def named_responses(model, seed, config):
    random = np.random.default_rng(np.random.SeedSequence([seed, 103]))
    anchors = random.normal(size=(config["named_anchor_bases"], 4))
    perturbation = np.eye(4)[:, None, :] * config["named_anchor_dose"]
    plus = embed(model, anchors[None] + perturbation)
    minus = embed(model, anchors[None] - perturbation)
    return ((plus - minus) / (2 * config["named_anchor_dose"])).mean(axis=1)


def add_intercept(x):
    return np.column_stack((x.reshape(-1, x.shape[-1]), np.ones(x.size // x.shape[-1])))


def fit(x, y):
    return np.linalg.lstsq(add_intercept(x), y.reshape(-1, y.shape[-1]), rcond=None)[0]


def predict(x, fitted):
    return x @ fitted[:-1, :] + fitted[-1, :]


def r2(actual, predicted):
    actual = actual.reshape(-1, actual.shape[-1])
    predicted = predicted.reshape(actual.shape)
    error = (predicted - actual)**2
    return 1 - error.mean(axis=0) / actual.var(axis=0)


def scores(y, reference, lags):
    variance = reference.var(axis=(0, 1))
    if np.min(variance) <= 1e-12:
        raise ValueError("undefined slow score for constant coordinate")
    lagged = [np.mean((y[:, lag:] - y[:, :-lag])**2, axis=1) for lag in lags]
    return np.mean(lagged, axis=0) / variance


def top_two(y):
    return np.sort(np.argsort(y, axis=1, kind="stable")[:, :2], axis=1)


def gap(y):
    ordered = np.sort(y, axis=1)
    return ordered[:, 2] - ordered[:, 1]


def ranking_metrics(expected, estimated, edges):
    expected_top = top_two(expected)
    correct = np.all(top_two(estimated) == expected_top, axis=1)
    test_gaps = gap(expected)
    bins = np.digitize(test_gaps, edges, right=True)
    return {
        "accuracy": float(correct.mean()),
        "correct": int(correct.sum()),
        "windows": int(len(correct)),
        "margin_bins": [
            {"correct": int(correct[bins == j].sum()), "windows": int((bins == j).sum())}
            for j in range(len(edges) + 1)
        ],
    }


def hermite2(z):
    return (z**2 - 1) / np.sqrt(2)


def hermite_diagnostic(cal_z, cal_h, test_z, test_h):
    cal_h2 = hermite2(cal_z)
    test_h2 = hermite2(test_z)
    base_fit = fit(cal_z, cal_h)
    base_r2 = r2(test_h, predict(test_z, base_fit))
    gains = []
    for axis in range(4):
        cal_extended = np.concatenate([cal_z, cal_h2[..., [axis]]], axis=-1)
        test_extended = np.concatenate([test_z, test_h2[..., [axis]]], axis=-1)
        extended_fit = fit(cal_extended, cal_h)
        extended_r2 = r2(test_h, predict(test_extended, extended_fit))
        gains.append(float(np.mean(extended_r2 - base_r2)))
    return {
        "test_h2_linear_readout_r2": r2(test_h2, predict(test_h, fit(cal_h, cal_h2))).tolist(),
        "test_embedding_r2_gain_from_h2_after_linear_factors": gains,
        "test_embedding_linear_r2": base_r2.tolist(),
    }


def evaluate(model, config, seed, rates):
    length = config["trajectory_length"]
    cal_z = trajectory(rates, seed, 101, config["cal_trajectories"], length)
    ref_z = trajectory(rates, seed, 102, config["ref_trajectories"], length)
    test_z = trajectory(rates, seed, 104, config["test_trajectories"], length)
    cal_h, ref_h, test_h = [embed(model, z) for z in (cal_z, ref_z, test_z)]
    spectral, spectral_eigs, cal_cov_eigs = fit_lag_axes(cal_h)
    responses = named_responses(model, seed, config)
    native_order, native_responses = assign_names(responses)
    spectral_order, spectral_responses = assign_names(responses @ spectral)
    oracle = fit(cal_h, cal_z)
    physical = scores(test_z, ref_z, config["lags"])
    references = {
        "native_named": ref_h[:, :, native_order],
        "spectral_named": (ref_h @ spectral)[:, :, spectral_order],
        "spectral_population_order": ref_h @ spectral,
        "oracle_linear": predict(ref_h, oracle),
    }
    readouts = {
        "native_named": test_h[:, :, native_order],
        "spectral_named": (test_h @ spectral)[:, :, spectral_order],
        "spectral_population_order": test_h @ spectral,
        "oracle_linear": predict(test_h, oracle),
    }
    case_scores = {
        name: scores(values, references[name], config["lags"])
        for name, values in readouts.items()
    }
    calibration_window_correlation = [
        float(np.corrcoef(cal_z[:, :-1, i].ravel(), cal_z[:, 1:, i].ravel())[0, 1])
        for i in range(4)
    ]
    return {
        "calibration_physical_lag1_correlation": calibration_window_correlation,
        "calibration_covariance_eigenvalues": cal_cov_eigs.tolist(),
        "calibration_latent_lag1_eigenvalues": spectral_eigs.tolist(),
        "native_named_permutation": native_order.tolist(),
        "spectral_named_permutation": spectral_order.tolist(),
        "native_named_normalized_responses": native_responses.tolist(),
        "spectral_named_normalized_responses": spectral_responses.tolist(),
        "oracle_linear_factor_r2": r2(test_z, readouts["oracle_linear"]).tolist(),
        "top2": {
            name: ranking_metrics(physical, candidate, config["margin_bins"])
            for name, candidate in case_scores.items()
        },
        "hermite": hermite_diagnostic(cal_z, cal_h, test_z, test_h),
        "physical_gap_quantiles": np.quantile(gap(physical), [0.1, 0.5, 0.9]).tolist(),
        "physical_population_set_window_agreement": float(np.all(top_two(physical) == [0, 1], axis=1).mean()),
    }


def analytic_case(rates, lags):
    rates = np.asarray(rates)
    population_scores = 2 - 2 * np.mean(rates[None, :] ** np.array(lags)[:, None], axis=0)
    linear = [(f"z{i}", rate) for i, rate in enumerate(rates)]
    quadratic = [
        ((f"H2(z{i})" if i == j else f"z{i} z{j}"), rates[i] * rates[j])
        for i, j in itertools.combinations_with_replacement(range(len(rates)), 2)
    ]
    selected = sorted(linear + quadratic, key=lambda pair: pair[1], reverse=True)[:len(rates)]
    if not selected[-1][1] > np.max(rates)**3:
        raise ValueError("higher Hermite degrees must be searched for these rates")
    return {
        "rates": rates.tolist(),
        "linear_nonlinear_spectral_gap": float(np.min(rates) - np.max(rates)**2),
        "physical_population_scores": population_scores.tolist(),
        "physical_population_gap_2_3": float(population_scores[2] - population_scores[1]),
        "whitening_relaxation_top_modes": selected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((HERE / "protocol.json").read_text())
    origin = source_provenance()
    output = HERE / ("smoke" if args.smoke else "runs")
    output.mkdir(exist_ok=False)
    results = {
        "protocol_sha256": hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": origin,
        "environment": {"cpu_arch": platform.machine(), "torch": torch.__version__, "numpy": np.__version__},
        "cases": {},
    }
    cases = list(config["cases"].items())[:1] if args.smoke else config["cases"].items()
    seeds = config["seeds"][:1] if args.smoke else config["seeds"]
    steps = 30 if args.smoke else config["steps"]
    for case, rates in cases:
        results["cases"][case] = analytic_case(rates, config["lags"])
        results["cases"][case]["runs"] = {}
        for seed in seeds:
            start = time.monotonic()
            model, log, pair_correlation = train(config, seed, rates, steps)
            result = evaluate(model, config, seed, np.asarray(rates))
            result["training_log"] = log
            result["training_pair_correlations"] = pair_correlation.tolist()
            result["elapsed_seconds"] = time.monotonic() - start
            torch.save(model.state_dict(), output / f"{case}_{seed}.pt")
            (output / f"{case}_{seed}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            results["cases"][case]["runs"][seed] = result
            print(f"{case} {seed}: {result['elapsed_seconds']:.2f}s", flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
