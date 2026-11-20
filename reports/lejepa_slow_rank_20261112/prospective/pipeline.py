"""Build frozen LeJEPA score forecasts and separately reveal population targets."""

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent / "training"
sys.path.insert(0, str(TRAINING))
from run import assign_names, embed, fit_lag_axes, make_mlp_encoder, scores, source_provenance


MASTER = 2026111201
BANKS_PER_CHECKPOINT = 4
DEV_PER_BANK = 128
TEST_SIZE = 8192
DRAWS = 64
LOW_QUANTILE = 0.10

ROLES = {
    "calibration": 1,
    "reference": 2,
    "anchors": 3,
    "development": 4,
    "test_bank_allocation": 5,
    "test_trajectory": 6,
    "bootstrap": 7,
}
PERMUTATIONS = np.array(list(itertools.permutations(range(4))))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream(role, *parts):
    return np.random.default_rng(np.random.SeedSequence([MASTER, ROLES[role], *parts]))


def innovations_to_paths(rates, innovations):
    paths = innovations.copy()
    scale = np.sqrt(1 - rates**2)
    for t in range(1, paths.shape[1]):
        paths[:, t, :] = rates * paths[:, t - 1, :] + scale * innovations[:, t, :]
    return paths


def panel_paths(rates, role, bank, count, length):
    return innovations_to_paths(rates, stream(role, bank).normal(size=(count, length, 4)))


def test_paths(rates, indices, length):
    if len(indices) == 0:
        return np.empty((0, length, 4))
    innovations = np.stack([stream("test_trajectory", int(i)).normal(size=(length, 4)) for i in indices])
    return innovations_to_paths(rates, innovations)


def bank_definitions(config):
    return [
        (case_index, case, seed, replicate, np.asarray(rates))
        for case_index, (case, rates) in enumerate(config["cases"].items())
        for seed in config["seeds"]
        for replicate in range(BANKS_PER_CHECKPOINT)
    ]


def bootstrap_weights(random, count):
    probability = np.full(count, 1.0 / count)
    draws = random.multinomial(count, probability, size=DRAWS) / count
    return np.vstack([probability, draws])


def anchor_response_samples(model, bank, config):
    dose = config["named_anchor_dose"]
    bases = stream("anchors", bank).normal(size=(config["named_anchor_bases"], 4))
    perturbations = np.eye(4)[:, None, :] * dose
    return ((embed(model, bases[None] + perturbations) - embed(model, bases[None] - perturbations)) / (2 * dose)).transpose(1, 0, 2)


def outer(x, y):
    return x[..., :, None] * y[..., None, :]


def second_moment(y):
    return np.einsum("ntd,nte->nde", y, y) / y.shape[1]


def mean_and_cov(y, weights):
    mean = weights @ y.mean(axis=1)
    second = np.einsum("bn,nde->bde", weights, second_moment(y))
    return mean, second - outer(mean, mean)


def lag_transforms(calibration, weights):
    mean, covariance = mean_and_cov(calibration, weights)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    if np.min(eigvals) <= 1e-8:
        raise ValueError(f"singular calibration covariance: {np.min(eigvals)}")
    whitening = (eigvecs / np.sqrt(eigvals[:, None, :])) @ eigvecs.transpose(0, 2, 1)
    left = calibration[:, :-1, :]
    right = calibration[:, 1:, :]
    raw = np.einsum("ntd,nte->nde", left, right) / left.shape[1]
    centered = np.einsum("bn,nde->bde", weights, (raw + raw.transpose(0, 2, 1)) / 2)
    mean_adjacent = weights @ ((left.mean(axis=1) + right.mean(axis=1)) / 2)
    centered -= outer(mean_adjacent, mean) + outer(mean, mean_adjacent) - outer(mean, mean)
    lag_values, lag_vectors = np.linalg.eigh(whitening @ centered @ whitening)
    return whitening @ lag_vectors[:, :, ::-1], lag_values[:, ::-1], eigvals


def named_permutations(responses):
    absolute = np.abs(responses)
    norms = np.linalg.norm(absolute, axis=1, keepdims=True)
    normalized = absolute / np.maximum(norms, 1e-12)
    matches = normalized[:, np.arange(4)[None, :], PERMUTATIONS].sum(axis=-1)
    return PERMUTATIONS[np.argmax(matches, axis=1)], normalized


def dynamic_moments(embedded, lags):
    result = np.zeros((len(embedded), 4, 4))
    for lag in lags:
        differences = embedded[:, lag:] - embedded[:, :-lag]
        result += second_moment(differences)
    return result / len(lags)


def candidate_scores(embedded, transforms, reference_variances, lags):
    moments = dynamic_moments(embedded, lags)
    numerators = np.einsum("bdj,nde,bej->bnj", transforms, moments, transforms, optimize=True)
    return numerators / reference_variances[:, None, :]


def nominal_and_uncertainty(candidates):
    nominal = candidates[0]
    membership = np.argsort(nominal, axis=1, kind="stable")[:, :2]
    inside = np.zeros(nominal.shape, dtype=bool)
    np.put_along_axis(inside, membership, True, axis=1)
    outside_min = np.where(~inside, candidates[1:], np.inf).min(axis=2)
    inside_max = np.where(inside, candidates[1:], -np.inf).max(axis=2)
    return nominal, np.quantile(outside_min - inside_max, LOW_QUANTILE, axis=0)


def frozen_bank(model, config, rates, bank):
    length = config["trajectory_length"]
    calibration = embed(model, panel_paths(rates, "calibration", bank, config["cal_trajectories"], length))
    reference = embed(model, panel_paths(rates, "reference", bank, config["ref_trajectories"], length))
    anchors = anchor_response_samples(model, bank, config)
    random = stream("bootstrap", bank)
    transforms, lag_eigenvalues, cov_eigenvalues = lag_transforms(
        calibration, bootstrap_weights(random, len(calibration))
    )
    responses = np.einsum("bn,nij->bij", bootstrap_weights(random, len(anchors)), anchors)
    names, normalized = named_permutations(responses @ transforms)
    transforms = np.take_along_axis(transforms, names[:, None, :], axis=2)
    _, reference_cov = mean_and_cov(reference, bootstrap_weights(random, len(reference)))
    ref_var = np.einsum("bdj,bde,bej->bj", transforms, reference_cov, transforms)
    if np.min(ref_var) <= 1e-12:
        raise ValueError("zero reference variance in candidate")
    direct_transform, direct_eigenvalues, _ = fit_lag_axes(calibration)
    direct_names, _ = assign_names(anchors.mean(axis=0) @ direct_transform)
    if not (
        np.allclose(lag_eigenvalues[0], direct_eigenvalues, rtol=0, atol=1e-12)
        and np.array_equal(names[0], direct_names)
        and np.allclose(np.abs(transforms[0]), np.abs(direct_transform[:, direct_names]), rtol=0, atol=1e-9)
        and np.allclose(ref_var[0], (reference @ transforms[0]).var(axis=(0, 1)), rtol=0, atol=1e-12)
    ):
        raise ValueError("nominal bank differs from prior locked readout")
    diagnostics = {
        "lag_eigenvalues": lag_eigenvalues[0].tolist(),
        "minimum_lag_eigengap": float(np.min(-np.diff(lag_eigenvalues[0]))),
        "minimum_calibration_cov_eigenvalue": float(np.min(cov_eigenvalues[0])),
        "named_permutation": names[0].tolist(),
        "normalized_anchor_responses": normalized[0].tolist(),
        "reference_variances": ref_var[0].tolist(),
        "bootstrap_permutation_changes": int(np.any(names[1:] != names[0], axis=1).sum()),
    }
    return transforms, ref_var, diagnostics


def allocation(bank_count):
    return stream("test_bank_allocation").integers(0, bank_count, size=TEST_SIZE)


def forecast(config):
    output = HERE / "forecasts.npz"
    receipt = HERE / "pipeline_forecast_receipt.json"
    if output.exists() or receipt.exists():
        raise FileExistsError("forecast or receipt already exists")
    start = time.monotonic()
    banks = bank_definitions(config)
    assigned = allocation(len(banks))
    dev_bank = np.repeat(np.arange(len(banks)), DEV_PER_BANK)
    dev_scores = np.zeros((len(dev_bank), 4))
    dev_uncertainty = np.zeros(len(dev_bank))
    test_scores = np.zeros((TEST_SIZE, 4))
    test_uncertainty = np.zeros(TEST_SIZE)
    diagnostics = []
    checkpoint_sha256 = {}
    model = None
    previous_checkpoint = None
    for bank, (case_index, case, seed, replicate, rates) in enumerate(banks):
        checkpoint = TRAINING / "runs" / f"{case}_{seed}.pt"
        if previous_checkpoint != checkpoint:
            model = make_mlp_encoder(4, config["hidden"], config["layers"], device="cpu").eval()
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            checkpoint_sha256[checkpoint.name] = digest(checkpoint)
            previous_checkpoint = checkpoint
        transforms, variances, diagnostic = frozen_bank(model, config, rates, bank)
        indices = np.flatnonzero(assigned == bank)
        development = panel_paths(rates, "development", bank, DEV_PER_BANK, config["trajectory_length"])
        heldout = test_paths(rates, indices, config["trajectory_length"])
        candidates = candidate_scores(embed(model, development), transforms, variances, config["lags"])
        dev_rows = slice(bank * DEV_PER_BANK, (bank + 1) * DEV_PER_BANK)
        dev_scores[dev_rows], dev_uncertainty[dev_rows] = nominal_and_uncertainty(candidates)
        if len(indices):
            candidates = candidate_scores(embed(model, heldout), transforms, variances, config["lags"])
            test_scores[indices], test_uncertainty[indices] = nominal_and_uncertainty(candidates)
        diagnostics.append({"bank": bank, "case": case, "seed": seed, "replicate": replicate, **diagnostic})
        if (bank + 1) % BANKS_PER_CHECKPOINT == 0:
            print(f"{bank + 1}/{len(banks)} frozen banks ready", flush=True)
    for values in (dev_scores, test_scores, dev_uncertainty, test_uncertainty):
        if not np.isfinite(values).all():
            raise ValueError("nonfinite scores")
    np.savez_compressed(
        output,
        dev_scores=dev_scores,
        test_scores=test_scores,
        dev_uncertainty=dev_uncertainty,
        test_uncertainty=test_uncertainty,
        dev_bank=dev_bank,
        test_bank=assigned,
        bank_case=np.array([row[0] for row in banks]),
        bank_model_seed=np.array([row[2] for row in banks]),
        bank_index=np.array([row[3] for row in banks]),
        case_names=np.array(list(config["cases"])),
        model_seeds=np.array(config["seeds"]),
    )
    state = {
        "forecasts_sha256": digest(output),
        "pipeline_sha256": digest(Path(__file__)),
        "checkpoint_sha256": checkpoint_sha256,
        "training_protocol_sha256": digest(TRAINING / "protocol.json"),
        "author_source": source_provenance(),
        "master_rng": MASTER,
        "rng_roles": ROLES,
        "draws": DRAWS,
        "uncertainty_quantile": LOW_QUANTILE,
        "dev_per_bank": DEV_PER_BANK,
        "test_size": TEST_SIZE,
        "test_bank_counts": np.bincount(assigned, minlength=len(banks)).tolist(),
        "bank_diagnostics": diagnostics,
        "seconds": time.monotonic() - start,
    }
    receipt.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")


def reveal(config):
    forecast_path = HERE / "forecasts.npz"
    receipt = json.loads((HERE / "pipeline_forecast_receipt.json").read_text())
    if digest(forecast_path) != receipt["forecasts_sha256"] or digest(Path(__file__)) != receipt["pipeline_sha256"]:
        raise ValueError("forecast or pipeline no longer matches pre-reveal receipt")
    if digest(TRAINING / "protocol.json") != receipt["training_protocol_sha256"]:
        raise ValueError("training protocol changed")
    output = HERE / "reveal.npz"
    if output.exists():
        raise FileExistsError("reveal artifact already exists")
    banks = bank_definitions(config)
    assigned = allocation(len(banks))
    with np.load(forecast_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["test_bank"], assigned):
            raise ValueError("fresh replay does not match locked bank allocation")
    physical_scores = np.zeros((TEST_SIZE, 4))
    known_reference = np.array([[-1.0] * 4, [1.0] * 4])[None, :, :]
    for bank, (_, _, _, _, rates) in enumerate(banks):
        indices = np.flatnonzero(assigned == bank)
        if len(indices):
            states = test_paths(rates, indices, config["trajectory_length"])
            physical_scores[indices] = scores(states, known_reference, config["lags"])
    np.savez_compressed(
        output,
        test_physical_scores=physical_scores,
        test_bank=assigned,
        forecasts_sha256=receipt["forecasts_sha256"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("forecast", "reveal"))
    args = parser.parse_args()
    config = json.loads((TRAINING / "protocol.json").read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.action == "forecast":
        forecast(config)
    else:
        reveal(config)


if __name__ == "__main__":
    main()
