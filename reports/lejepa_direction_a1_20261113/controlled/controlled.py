"""Run the prespecified state-matched intervention without reading TEST truth."""

import argparse
import importlib.util
import itertools
from pathlib import Path
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parents[1]
TRAIN = REPORTS / "lejepa_slow_rank_20261112/training"
READOUT = REPORTS / "lejepa_mechanisms_20261113/readout"
HISTORICAL = REPORTS / "lejepa_dynamic_transfer_20261113/analysis/HISTORICAL_GROUPS.json"
NATURAL = HERE.parent / "natural/run.py"


def module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(result)
    return result


helper = module("a1_natural_helpers", NATURAL)
b_probe = module("a1_frozen_b_readout", READOUT / "run.py")
digest, read, write = helper.digest, helper.read, helper.write
calculate_moments, top_two = helper.calculate_moments, helper.top_two
feature_matrix, apply_ridge, fit_ridge = helper.feature_matrix, helper.apply_ridge, helper.fit_ridge
ROLES = {"state_cal": 1, "fit": 2, "validation": 3, "reference": 4, "test": 5, "cal_reference": 6}
RECIPIENT = 2
DONORS = (0, 3)
METHODS = ("diag", "full")
PAIR = (1, 2)
ROUND_FIELDS = {"primary": "master_primary", "replication": "master_replication"}


def rng(master, role):
    return np.random.default_rng(np.random.SeedSequence([master, ROLES[role]]))


def paths(master, role, count, length, rates):
    innovations = rng(master, role).normal(size=(count, length, 4))
    states = innovations.copy()
    scale = np.sqrt(1 - rates**2)
    for t in range(1, length):
        states[:, t] = rates * states[:, t - 1] + scale * innovations[:, t]
    return states


def iids(master, role, count):
    return rng(master, role).normal(size=(count, 4))


def sources():
    paths = (
        HERE / "controlled.py", HERE / "score.py", HERE / "PROTOCOL.md", HERE / "protocol.json",
        NATURAL, TRAIN / "run.py", TRAIN / "protocol.json", READOUT / "run.py", READOUT / "protocol.json",
        READOUT / "replication/FORECAST_RECEIPT.json", READOUT / "replication/RESULTS.json", HISTORICAL,
        TRAIN.parent / "prospective/pipeline_forecast_receipt.json",
    )
    return {str(path.relative_to(REPORTS)): digest(path) for path in paths}


def identity():
    old = read(TRAIN / "protocol.json")
    b_config = read(READOUT / "protocol.json")
    b_receipt = read(READOUT / "replication/FORECAST_RECEIPT.json")
    historical = read(HISTORICAL)
    root = REPORTS.parent
    if digest(root / historical["source_relative_to_repository"]) != historical["source_sha256"]:
        raise ValueError("Historical recovery group provenance changed.")
    encoder_hashes = read(TRAIN.parent / "prospective/pipeline_forecast_receipt.json")["checkpoint_sha256"]
    if digest(READOUT / "runs/results.json") != b_receipt["original_summary_sha256"]:
        raise ValueError("Historical readout manifest changed.")
    return old, b_config, historical["strong"], encoder_hashes, b_receipt["fitted_readouts_sha256"]


def frozen_model(name, old, b_config, encoder_hashes, decoder_hashes):
    encoder_path = TRAIN / "runs" / f"{name}.pt"
    decoder_path = READOUT / "runs" / f"{name}_readouts.pt"
    hash_encoder, hash_decoder = digest(encoder_path), digest(decoder_path)
    if hash_encoder != encoder_hashes[encoder_path.name] or hash_decoder != decoder_hashes[name]:
        raise ValueError(f"Unrelated frozen weights: {name}")
    encoder = b_probe.make_mlp_encoder(4, old["hidden"], old["layers"], device="cpu").eval()
    encoder.load_state_dict(torch.load(encoder_path, weights_only=True, map_location="cpu"))
    decoder = b_probe.mlp(4, b_config["mlp_width"]).eval()
    fitted = torch.load(decoder_path, weights_only=False, map_location="cpu")
    decoder.load_state_dict(fitted["mlp"])
    mean, scale = fitted["input_mean"], fitted["input_std"]
    if np.any(scale <= 0):
        raise ValueError("Invalid historical input scaler.")

    def readout(states):
        embedding = b_probe.embed(encoder, states).reshape(-1, 4)
        return b_probe.predict_mlp((embedding - mean) / scale, decoder).reshape(states.shape)

    return readout, {"encoder_sha256": hash_encoder, "probe_sha256": hash_decoder}


def arm_design(config):
    return [
        {"name": "base", "delta": 0, "donor": None},
        *({"name": f"{'slow' if donor == 0 else 'fast'}_{delta:.2f}", "delta": delta, "donor": donor}
          for delta, donor in itertools.product(config["deltas"], DONORS)),
    ]


def mix_coefficients(state, readout, arms):
    error = readout[:, RECIPIENT] - state[:, RECIPIENT]
    baseline = float(np.mean(error**2))
    candidates = []
    for arm in arms:
        donor = arm["donor"]
        if donor is None:
            candidates.append({**arm, "a": 0.0, "cal_mse": baseline})
            continue
        values = readout[:, donor]
        b = float(np.mean(error * values))
        c = float(np.mean(values**2))
        a = (-b + np.sqrt(b**2 + c * arm["delta"])) / c
        mse = float(np.mean((error + a * values)**2))
        np.testing.assert_allclose(mse - baseline, arm["delta"], atol=5e-14)
        candidates.append({**arm, "a": float(a), "cal_mse": mse, "cal_b": b, "cal_c": c})
    return candidates


def score_candidates(moment, ref, arms):
    n = moment.diagonal(axis1=1, axis2=2)
    variance = ref.var(axis=0)
    centered = ref - ref.mean(axis=0)
    scores = []
    denominators = []
    for arm in arms:
        numerator = n.copy()
        denominator = variance.copy()
        donor, a = arm["donor"], arm["a"]
        if donor is not None:
            numerator[:, RECIPIENT] += 2 * a * moment[:, RECIPIENT, donor] + a**2 * n[:, donor]
            denominator[RECIPIENT] = np.mean((centered[:, RECIPIENT] + a * centered[:, donor])**2)
        if np.any(denominator <= 0):
            raise ValueError("A reference coordinate has undefined normalization.")
        scores.append(numerator / denominator)
        denominators.append(denominator)
    return np.stack(scores), np.stack(denominators)


def with_physical_labels(readout, master, role, count, config, rates):
    physical = paths(master, role, count, config["length"], rates)
    inputs = calculate_moments(readout(physical), config["lags"])
    labels = calculate_moments(physical, config["lags"]).diagonal(axis1=1, axis2=2)
    return inputs, labels


def fit_predictors(readout, config, master, rates, coefficients, cal_ref):
    fit_x, fit_y = with_physical_labels(readout, master, "fit", config["n_fit"], config, rates)
    val_x, val_y = with_physical_labels(readout, master, "validation", config["n_validation"], config, rates)
    fits = {
        name: fit_ridge(feature_matrix(fit_x, name), fit_y, feature_matrix(val_x, name), val_y, config)
        for name in METHODS
    }
    candidate, variances = score_candidates(fit_x, cal_ref, coefficients)
    wrong_pair = (candidate[..., PAIR[1]] < candidate[..., PAIR[0]]) != (fit_y[:, PAIR[1]] < fit_y[:, PAIR[0]])
    wrong_set = np.any(np.stack([top_two(arm) for arm in candidate]) != top_two(fit_y), axis=-1)
    risk = []
    for index, delta in enumerate(config["deltas"]):
        low, high = 1 + 2 * index, 2 + 2 * index
        difference = float(wrong_pair[low].mean() - wrong_pair[high].mean())
        set_difference = float(wrong_set[low].mean() - wrong_set[high].mean())
        threshold = config["cal_direction_identifiable_min_difference"]
        risk.append({
            "delta": delta,
            "cal_pair_error_by_arm": [float(wrong_pair[a].mean()) for a in (low, high)],
            "cal_top2_error_by_arm": [float(wrong_set[a].mean()) for a in (low, high)],
            "cal_pair_slow_minus_fast": difference, "cal_pair_direction": int(np.sign(difference)),
            "cal_pair_direction_identifiable": abs(difference) >= threshold,
            "cal_top2_slow_minus_fast": set_difference, "cal_top2_direction": int(np.sign(set_difference)),
            "cal_top2_direction_identifiable": abs(set_difference) >= threshold,
        })
    return fits, risk, variances.tolist()


def forecast_one(readout, config, master, rates, ref_readout, coefficients, fitted):
    # Latent simulator states flow into the observation encoder only.
    visible = readout(paths(master, "test", config["n_test"], config["length"], rates))
    moment = calculate_moments(visible, config["lags"])
    candidate, reference = score_candidates(moment, ref_readout, coefficients)
    physical = np.stack([apply_ridge(feature_matrix(moment, name), fitted[name]) for name in METHODS])
    candidate_top = np.stack([top_two(arm) for arm in candidate])
    physical_top = np.stack([top_two(method) for method in physical])
    pair_gap = candidate[..., PAIR[1]] - candidate[..., PAIR[0]]
    physical_gap = physical[..., PAIR[1]] - physical[..., PAIR[0]]
    return {
        "visible_named_readout": visible.astype(np.float32),
        "candidate_scores": candidate,
        "candidate_top2": candidate_top,
        "candidate_pair_gap": pair_gap,
        "physical_score_prediction": physical,
        "physical_pair_gap_prediction": physical_gap,
        "pair_reversal_prediction": (physical_gap[:, None] < 0) != (pair_gap[None] < 0),
        "correct_set_prediction": np.all(candidate_top[None] == physical_top[:, None], axis=-1),
        "reference_variance": reference,
    }


def write_npz(path, **values):
    with path.open("xb") as output:
        np.savez_compressed(output, **values)


def forecast(config, smoke):
    output = HERE / ("smoke" if smoke else "forecast")
    output.mkdir(exist_ok=False)
    old, b_config, models, encoder_hashes, probe_hashes = identity()
    if smoke:
        models = models[:1]
        config = {**config, "n_state_cal": 256, "n_fit": 32, "n_validation": 24, "n_reference": 256, "n_test": 16}
    master = config["smoke_master"] if smoke else config["master_primary"]
    rounds = {"smoke": master} if smoke else {stage: config[key] for stage, key in ROUND_FIELDS.items()}
    initial = time.monotonic()
    state_cal = iids(master, "state_cal", config["n_state_cal"])
    arms = arm_design(config)
    rates = {name: np.asarray(value) for name, value in config["processes"].items()}
    weights = {}
    artifacts = {}
    for name in models:
        readout, provenance = frozen_model(name, old, b_config, encoder_hashes, probe_hashes)
        coefficients = mix_coefficients(state_cal, readout(state_cal), arms)
        cal_ref = readout(iids(master, "cal_reference", config["n_reference"]))
        fitted = {
            process: fit_predictors(readout, config, master, rho, coefficients, cal_ref)
            for process, rho in rates.items()
        }
        fit_file = output / f"{name}_fit.json"
        write(fit_file, {
            **provenance, "arms": coefficients,
            "fitted_physical_scores": {process: rows[0] for process, rows in fitted.items()},
            "calibrated_risks": {process: rows[1] for process, rows in fitted.items()},
            "calibration_reference_variance": next(iter(fitted.values()))[2],
        })
        weights[fit_file.name] = digest(fit_file)
        for stage, draw_master in rounds.items():
            ref = readout(iids(draw_master, "reference", config["n_reference"]))
            for process, rho in rates.items():
                path = output / f"{stage}_{name}_{process}.npz"
                values = forecast_one(readout, config, draw_master, rho, ref, coefficients, fitted[process][0])
                write_npz(path, **values)
                artifacts[path.name] = digest(path)
        print(name, f"{time.monotonic() - initial:.2f}s", flush=True)
    write(output / "RECEIPT.json", {
        "sources": sources(), "fit_files": weights, "forecast_files": artifacts,
        "model_names": models, "process_names": list(rates), "methods": list(METHODS), "arm_design": arms,
        "round_rng": rounds, "config": config, "roles": ROLES, "seconds": time.monotonic() - initial,
    })


def reveal():
    output = HERE / "forecast"
    receipt_path = output / "RECEIPT.json"
    receipt = read(receipt_path)
    if receipt["sources"] != sources():
        raise ValueError("Prediction dependencies changed after lock.")
    for entries in ("fit_files", "forecast_files"):
        for name, expected in receipt[entries].items():
            if digest(output / name) != expected:
                raise ValueError(f"Forecast file changed: {name}")
    config = receipt["config"]
    truth = {}
    for stage, master in receipt["round_rng"].items():
        for process, rates in config["processes"].items():
            truth[f"{stage}_{process}"] = paths(
                master, "test", config["n_test"], config["length"], np.asarray(rates)
            )
    path = HERE / "REVEAL.npz"
    write_npz(path, **truth)
    write(HERE / "REVEAL_RECEIPT.json", {"forecast_receipt_sha256": digest(receipt_path), "physical_sha256": digest(path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "forecast", "reveal"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = read(HERE / "protocol.json")
    if args.stage == "reveal":
        reveal()
    else:
        forecast(config, smoke=args.stage == "smoke")


if __name__ == "__main__":
    main()
