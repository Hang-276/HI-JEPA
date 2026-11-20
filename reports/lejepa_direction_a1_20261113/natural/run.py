"""Frozen natural LeJEPA direction experiment with separate forecast and reveal."""

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parents[1]
ORIGINAL = REPORTS / "lejepa_slow_rank_20261112"
TRAINING = ORIGINAL / "training"
spec = importlib.util.spec_from_file_location("direction_source", TRAINING / "run.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)

ROLES = {"axes": 1, "naming": 2, "reference": 3, "fit": 4, "validation": 5, "conformal": 6, "test": 7}
METHODS = ("diag", "full")
PAIRS = np.array(list(itertools.combinations(range(4), 2)))
UPPER = np.triu_indices(4)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, result):
    with path.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


def sources():
    paths = [
        Path(__file__), HERE / "PROTOCOL.md", HERE / "protocol.json", HERE / "score.py",
        TRAINING / "run.py", TRAINING / "protocol.json",
        ORIGINAL / "prospective/pipeline_forecast_receipt.json",
    ]
    return {str(path.relative_to(REPORTS)): digest(path) for path in paths}


def rng(master, role):
    return np.random.default_rng(np.random.SeedSequence([master, ROLES[role]]))


def trajectories(master, role, count, length, rates):
    innovations = rng(master, role).normal(size=(count, length, 4))
    path = innovations.copy()
    scale = np.sqrt(1 - rates**2)
    for t in range(1, length):
        path[:, t] = rates * path[:, t - 1] + scale * innovations[:, t]
    return path


def calculate_moments(values, lags):
    moments = np.zeros((len(values), 4, 4))
    for lag in lags:
        delta = values[:, lag:] - values[:, :-lag]
        moments += np.einsum("nti,ntj->nij", delta, delta) / delta.shape[1]
    return moments / len(lags)


def physical_scores(states, lags):
    return calculate_moments(states, lags).diagonal(axis1=1, axis2=2)


def top_two(scores):
    return np.sort(np.argsort(scores, axis=1, kind="stable")[:, :2], axis=1)


def feature_matrix(moments, method):
    if method == "diag":
        return moments.diagonal(axis1=1, axis2=2)
    return moments[:, UPPER[0], UPPER[1]]


def pair_gaps(scores):
    return scores[:, PAIRS[:, 1]] - scores[:, PAIRS[:, 0]]


def load_encoder(case, seed, old, hashes):
    checkpoint = TRAINING / "runs" / f"{case}_{seed}.pt"
    sha = digest(checkpoint)
    if hashes[checkpoint.name] != sha:
        raise ValueError(f"encoder changed: {checkpoint}")
    model = source.make_mlp_encoder(4, old["hidden"], old["layers"], device="cpu").eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    return model, sha


def calibrate_axes(model, latent_axes, anchors, dose):
    transform, lag_eigenvalues, cov_eigenvalues = source.fit_lag_axes(latent_axes)
    intervention = dose * np.eye(4)[:, None, :]
    plus = source.embed(model, anchors[None] + intervention)
    minus = source.embed(model, anchors[None] - intervention)
    responses = ((plus - minus) / (2 * dose)).mean(axis=1) @ transform
    names, normalized = source.assign_names(responses)
    orientation = np.sign(responses[np.arange(4), names])
    if np.any(orientation == 0):
        raise ValueError("zero named response leaves an undefined coordinate orientation")
    named_transform = transform[:, names] * orientation
    return named_transform, {
        "lag_eigenvalues": lag_eigenvalues.tolist(),
        "calibration_covariance_eigenvalues": cov_eigenvalues.tolist(),
        "permutation": names.tolist(),
        "orientations": orientation.tolist(),
        "anchor_response": responses.tolist(),
        "normalized_anchor_response": normalized.tolist(),
    }


def apply_ridge(features, fit):
    x = (features - np.asarray(fit["feature_mean"])) / np.asarray(fit["feature_std"])
    return x @ np.asarray(fit["coefficients"]) + np.asarray(fit["target_mean"])


def fit_ridge(x, y, validation_x, validation_y, config):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    if not np.all(scale > 0):
        raise ValueError("constant feature in calibration")
    target_mean = y.mean(axis=0)
    scaled = (x - mean) / scale
    validation_scaled = (validation_x - mean) / scale
    centered = y - target_mean
    candidates = [
        np.linalg.lstsq(scaled, centered, rcond=None)[0] if alpha == 0 else
        np.linalg.solve(scaled.T @ scaled / len(x) + alpha * np.eye(x.shape[1]), scaled.T @ centered / len(x))
        for alpha in config["ridge_alphas"]
    ]
    errors = [float(np.mean((validation_scaled @ coef + target_mean - validation_y)**2)) for coef in candidates]
    choice = int(np.argmin(errors))
    return {
        "feature_mean": mean.tolist(),
        "feature_std": scale.tolist(),
        "target_mean": target_mean.tolist(),
        "coefficients": candidates[choice].tolist(),
        "alpha": config["ridge_alphas"][choice],
        "validation_mse_by_alpha": errors,
    }


def conformal_calibration(fit, validation_x, validation_y, conf_x, conf_y, alpha):
    validation_residual = pair_gaps(validation_y - apply_ridge(validation_x, fit))
    gap_scales = np.sqrt(np.mean(validation_residual**2, axis=0))
    if not np.all(gap_scales > 0):
        raise ValueError("constant zero validation gap residual")
    residuals = np.abs(pair_gaps(conf_y - apply_ridge(conf_x, fit))) / gap_scales
    maxima = residuals.max(axis=1)
    order = int(np.ceil((len(maxima) + 1) * (1 - alpha)))
    if not 1 <= order <= len(maxima):
        raise ValueError("undefined finite conformal quantile")
    quantile = float(np.sort(maxima)[order - 1])
    return {
        "gap_scales": gap_scales.tolist(),
        "conformal_order": order,
        "conformal_quantile": quantile,
        "gap_widths": (gap_scales * quantile).tolist(),
    }


def risks_and_safety(scores, nominal_set, widths):
    membership = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(membership, nominal_set, True, axis=1)
    gaps = scores[:, None, :] - scores[:, :, None]
    crossed = membership[:, :, None] & ~membership[:, None, :]
    risk_score = np.where(crossed, gaps, np.inf).min(axis=(1, 2))
    width_matrix = np.zeros((4, 4))
    width_matrix[PAIRS[:, 0], PAIRS[:, 1]] = widths
    width_matrix = width_matrix + width_matrix.T
    lower = np.where(crossed, gaps - width_matrix, np.inf).min(axis=(1, 2))
    return risk_score, lower > 0


def common_inputs(config, old, round_name, include_calibration, smoke):
    master = config[f"master_{round_name}"] + (10000 if smoke else 0)
    reference = rng(master, "reference").normal(size=(config["n_reference"], 4))
    anchors = rng(master, "naming").normal(size=(old["named_anchor_bases"], 4))
    for name, rates in config["processes"].items():
        rates = np.asarray(rates)
        roles = ("axes", "fit", "validation", "conformal", "test") if include_calibration else ("test",)
        data = {role: trajectories(master, role, config[f"n_{role}"], config["length"], rates) for role in roles}
        yield name, reference, anchors, data


def fit_parameters(config, old, encoder, reference_states, anchors, data, encoder_sha):
    h_axes = source.embed(encoder, data["axes"])
    transform, diagnostic = calibrate_axes(encoder, h_axes, anchors, old["named_anchor_dose"])
    variance = (source.embed(encoder, reference_states) @ transform).var(axis=0)
    if not np.all(variance > 0):
        raise ValueError("nonpositive reference variance")
    moments = {}
    physical = {}
    normalized_transform = transform / np.sqrt(variance)
    for role in ("fit", "validation", "conformal"):
        encoded = source.embed(encoder, data[role]) @ normalized_transform
        moments[role] = calculate_moments(encoded, config["lags"])
        physical[role] = physical_scores(data[role], config["lags"])
    methods = {}
    for method in METHODS:
        f = {key: feature_matrix(val, method) for key, val in moments.items()}
        fit = fit_ridge(f["fit"], physical["fit"], f["validation"], physical["validation"], config)
        conf = conformal_calibration(
            fit, f["validation"], physical["validation"], f["conformal"], physical["conformal"], config["conformal_alpha"]
        )
        methods[method] = {**fit, **conf}
    return {
        "encoder_sha256": encoder_sha,
        "transform": transform.tolist(),
        "calibration_ref_variance": variance.tolist(),
        "axis_diagnostic": diagnostic,
        "methods": methods,
    }


def predict(config, encoder, reference_states, test, parameters):
    transform = np.asarray(parameters["transform"])
    initial_variance = np.asarray(parameters["calibration_ref_variance"])
    variance = (source.embed(encoder, reference_states) @ transform).var(axis=0)
    if not np.all(variance > 0):
        raise ValueError("nonpositive test reference variance")
    observed_coordinates = source.embed(encoder, test) @ transform
    test_moments = calculate_moments(observed_coordinates / np.sqrt(variance), config["lags"])
    nominal = feature_matrix(test_moments, "diag")
    nominal_set = top_two(nominal)
    nominal_risk, _ = risks_and_safety(nominal, nominal_set, np.zeros(len(PAIRS)))
    # Frozen predictors are defined on the initial reference coordinate convention.
    scale = np.sqrt(variance / initial_variance)
    fixed_convention_moments = test_moments * scale[None, :, None] * scale[None, None, :]
    result = {"nominal_scores": nominal, "nominal_risk": nominal_risk, "ref_variance": variance}
    for method in METHODS:
        fit = parameters["methods"][method]
        prediction = apply_ridge(feature_matrix(fixed_convention_moments, method), fit)
        risk_score, safe = risks_and_safety(prediction, nominal_set, np.asarray(fit["gap_widths"]))
        result.update({f"{method}_scores": prediction, f"{method}_risk": risk_score, f"{method}_safe": safe})
    return result


def smoke_config(config):
    return {
        **config, "n_axes": 16, "n_fit": 48, "n_validation": 32,
        "n_conformal": 32, "n_reference": 2048, "n_test": 64,
        "processes": dict(list(config["processes"].items())[:1]),
    }


def forecast(config, round_name, base, smoke):
    start = time.monotonic()
    output = base / round_name
    output.mkdir(exist_ok=False)
    old = read(TRAINING / "protocol.json")
    frozen_hashes = read(ORIGINAL / "prospective/pipeline_forecast_receipt.json")["checkpoint_sha256"]
    initial = base / "primary"
    initial_receipt = None
    if round_name == "replication":
        initial_receipt = read(initial / "FORECAST_RECEIPT.json")
        if initial_receipt["source_hashes"] != sources() or initial_receipt["config"] != config:
            raise ValueError("initial predictor sources changed")
    parameter_hashes, prediction_hashes = {}, {}
    for process, ref, anchors, data in common_inputs(config, old, round_name, initial_receipt is None, smoke):
        for case, seed in itertools.product(old["cases"], old["seeds"]):
            if smoke and seed != old["seeds"][0]:
                continue
            name = f"{process}__{case}_{seed}"
            encoder, encoder_sha = load_encoder(case, seed, old, frozen_hashes)
            if initial_receipt is None:
                parameter_path = output / f"{name}.json"
                parameter = fit_parameters(config, old, encoder, ref, anchors, data, encoder_sha)
                write(parameter_path, parameter)
                parameter_hashes[parameter_path.name] = digest(parameter_path)
            else:
                parameter_path = initial / f"{name}.json"
                if digest(parameter_path) != initial_receipt["parameter_sha256"][parameter_path.name]:
                    raise ValueError("frozen calibration parameter changed")
                parameter = read(parameter_path)
            if parameter["encoder_sha256"] != encoder_sha:
                raise ValueError("parameter bound to a different frozen encoder")
            forecast_path = output / f"{name}.npz"
            results = predict(config, encoder, ref, data["test"], parameter)
            if not all(np.isfinite(array).all() for array in results.values()):
                raise ValueError("nonfinite frozen forecast")
            np.savez_compressed(forecast_path, **results)
            prediction_hashes[forecast_path.name] = digest(forecast_path)
            print(f"{round_name} {name}", flush=True)
    write(output / "FORECAST_RECEIPT.json", {
        "source_hashes": sources(), "config": config, "round": round_name, "smoke": smoke,
        "parameter_sha256": parameter_hashes,
        "initial_receipt_sha256": digest(initial / "FORECAST_RECEIPT.json") if initial_receipt else None,
        "prediction_sha256": prediction_hashes,
        "elapsed_seconds": time.monotonic() - start,
    })


def reveal(config, round_name, base, smoke):
    output = base / round_name
    forecast_receipt = output / "FORECAST_RECEIPT.json"
    receipt = read(forecast_receipt)
    if receipt["source_hashes"] != sources() or receipt["config"] != config or receipt["smoke"] != smoke:
        raise ValueError("forecast setup changed")
    for name, sha in receipt["prediction_sha256"].items():
        if digest(output / name) != sha:
            raise ValueError(f"prediction changed: {name}")
    initial = base / "primary"
    if round_name == "replication" and receipt["initial_receipt_sha256"] != digest(initial / "FORECAST_RECEIPT.json"):
        raise ValueError("initial receipt changed")
    for name, sha in read(initial / "FORECAST_RECEIPT.json")["parameter_sha256"].items():
        if digest(initial / name) != sha:
            raise ValueError(f"calibration parameters changed: {name}")
    master = config[f"master_{round_name}"] + (10000 if smoke else 0)
    labels = {}
    for process, rates in config["processes"].items():
        test = trajectories(master, "test", config["n_test"], config["length"], np.asarray(rates))
        labels[process] = physical_scores(test, config["lags"])
    label_path = output / "labels.npz"
    if label_path.exists():
        raise FileExistsError(label_path)
    np.savez_compressed(label_path, **labels)
    write(output / "REVEAL_RECEIPT.json", {
        "forecast_receipt_sha256": digest(forecast_receipt), "labels_sha256": digest(label_path),
        "process_names": list(labels), "label_shape": list(next(iter(labels.values())).shape),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("forecast", "reveal"))
    parser.add_argument("round", choices=("primary", "replication"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = read(HERE / "protocol.json")
    base = HERE / "smoke" if args.smoke else HERE
    if args.smoke:
        config = smoke_config(config)
        base.mkdir(exist_ok=True)
    (forecast if args.stage == "forecast" else reveal)(config, args.round, base, args.smoke)


if __name__ == "__main__":
    main()
