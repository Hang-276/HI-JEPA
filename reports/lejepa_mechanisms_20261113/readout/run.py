"""Frozen LeJEPA nonlinear decoder diagnosis with independent labelled splits."""

import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1] / "lejepa_slow_rank_20261112/training"
sys.path.insert(0, str(TRAINING))
from run import embed, make_mlp_encoder, observe, top_two


ROLE = {"calibration": 1, "validation": 2, "reference": 3, "test": 4, "probe_fit": 5}
POLYNOMIAL_POWERS = tuple(
    indices
    for order in range(1, 4)
    for indices in itertools.combinations_with_replacement(range(4), order)
)
RULES = ("linear", "cubic", "mlp")
MODES = ("frozen", "cubic_shear", "invertible_observation", "missing_fastest")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stream(master, role):
    return np.random.default_rng(np.random.SeedSequence([master, ROLE[role]]))


def datasets(config, rates):
    calibration = stream(config["master"], "calibration").normal(size=(config["n_calibration"], 4))
    validation = stream(config["master"], "validation").normal(size=(config["n_validation"], 4))
    reference = stream(config["master"], "reference").normal(size=(config["n_reference"], 4))
    innovation = stream(config["master"], "test").normal(
        size=(config["n_test"], config["trajectory_length"], 4)
    )
    test = innovation.copy()
    step_noise = np.sqrt(1 - rates**2)
    for step in range(1, config["trajectory_length"]):
        test[:, step] = rates * test[:, step - 1] + step_noise * innovation[:, step]
    return {"calibration": calibration, "validation": validation, "reference": reference, "test": test}


def normalized_h2(z):
    return (z**2 - 1) / np.sqrt(2)


def r2(y, predicted):
    return 1 - np.mean((y - predicted)**2, axis=0) / np.var(y, axis=0)


def standardizer(values):
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1
    return mean, scale


def polynomial(x):
    return np.column_stack([np.prod(x[:, term], axis=1) for term in POLYNOMIAL_POWERS])


def ridge_features(x, use_cubic):
    return polynomial(x) if use_cubic else x


def fit_ridge(x_cal, y_cal, x_val, y_val, config, use_cubic):
    cal_features = ridge_features(x_cal, use_cubic)
    val_features = ridge_features(x_val, use_cubic)
    center, scale = standardizer(cal_features)
    cal_features = (cal_features - center) / scale
    val_features = (val_features - center) / scale
    center_y = y_cal.mean(axis=0)
    dimension = cal_features.shape[1]
    gram = cal_features.T @ cal_features / len(x_cal)
    cross = cal_features.T @ (y_cal - center_y) / len(x_cal)
    coefficients = [
        np.linalg.solve(gram + alpha * np.eye(dimension), cross) for alpha in config["ridge_alphas"]
    ]
    losses = np.stack([np.mean((val_features @ beta + center_y - y_val)**2, axis=0) for beta in coefficients])
    choices = losses.argmin(axis=0)
    fitted = np.column_stack([coefficients[choice][:, j] for j, choice in enumerate(choices)])
    return {
        "coefficient": fitted,
        "feature_mean": center,
        "feature_std": scale,
        "target_mean": center_y,
        "ridge_alpha_by_target": [config["ridge_alphas"][int(choice)] for choice in choices],
        "validation_mse_by_target": losses[choices, np.arange(y_cal.shape[1])].tolist(),
    }


def predict_ridge(x, fitted, use_cubic):
    raw = ridge_features(x, use_cubic)
    features = (raw - fitted["feature_mean"]) / fitted["feature_std"]
    return features @ fitted["coefficient"] + fitted["target_mean"]


def mlp(outputs, width):
    return nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, outputs))


def fit_mlp(x_cal, y_cal, x_val, y_val, config):
    torch.manual_seed(config["master"] + ROLE["probe_fit"])
    decoder = mlp(4, config["mlp_width"])
    generator = torch.Generator(device="cpu").manual_seed(config["master"] + ROLE["probe_fit"] + 50000)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config["mlp_lr"], weight_decay=1e-4)
    x, y, vx, vy = [torch.as_tensor(value, dtype=torch.float32) for value in (x_cal, y_cal, x_val, y_val)]
    best_mse = float("inf")
    best_step = None
    best_state = None
    validation_trace = []
    for step in range(1, config["mlp_steps"] + 1):
        selected = torch.randint(len(x), (config["mlp_batch_size"],), generator=generator)
        batch_x, batch_y = x[selected], y[selected]
        physical_loss = (decoder(batch_x) - batch_y).square().mean()
        optimizer.zero_grad(set_to_none=True)
        physical_loss.backward()
        optimizer.step()
        if step % config["mlp_eval_interval"] == 0 or step == config["mlp_steps"]:
            with torch.inference_mode():
                error = float((decoder(vx) - vy).square().mean())
            if not np.isfinite(error):
                raise FloatingPointError(f"nonfinite VAL loss at step {step}")
            validation_trace.append({"step": step, "physical_mse": error})
            if error < best_mse:
                best_state = {key: value.clone() for key, value in decoder.state_dict().items()}
                best_mse = error
                best_step = step
    decoder.load_state_dict(best_state)
    decoder.eval()
    return decoder, {"selected_step": best_step, "validation_trace": validation_trace}


@torch.inference_mode()
def predict_mlp(x, decoder):
    x = torch.as_tensor(x, dtype=torch.float32)
    output = [decoder(chunk).numpy() for chunk in x.split(8192)]
    return np.concatenate(output).astype(np.float64)


def increment_statistics(states, lags):
    differences = [states[:, lag:] - states[:, :-lag] for lag in lags]
    return np.mean([np.mean(delta**2, axis=1) for delta in differences], axis=0)


def task_metrics(predicted, reference, truth, lags):
    coordinate_variances = reference.var(axis=0)
    result = {"reference_variance": coordinate_variances.tolist()}
    if np.any(coordinate_variances <= 0):
        result.update({
            "status": "undefined_constant_reference_coordinate",
            "zero_variance_axes": np.flatnonzero(coordinate_variances <= 0).tolist(),
            "windows": len(predicted),
            "top2_accuracy": None,
        })
        return result, None
    numerators = increment_statistics(predicted, lags)
    physical_scores = increment_statistics(truth, lags)
    predicted_scores = numerators / coordinate_variances
    estimate = top_two(predicted_scores)
    expected = top_two(physical_scores)
    correct = np.all(estimate == expected, axis=1)
    inside = np.zeros(expected.shape[:-1] + (4,), dtype=bool)
    np.put_along_axis(inside, expected, True, axis=1)
    physical_margin = np.where(~inside, physical_scores, np.inf).min(axis=1) - np.where(inside, physical_scores, -np.inf).max(axis=1)
    predicted_margin = np.where(~inside, predicted_scores, np.inf).min(axis=1) - np.where(inside, predicted_scores, -np.inf).max(axis=1)
    increment_error = np.mean([
        np.mean(((predicted[:, lag:] - predicted[:, :-lag]) - (truth[:, lag:] - truth[:, :-lag]))**2, axis=(0, 1))
        for lag in lags
    ], axis=0)
    result.update({
        "status": "scored",
        "windows": len(correct),
        "top2_correct": int(correct.sum()),
        "top2_accuracy": float(correct.mean()),
        "state_increment_mse_by_axis": increment_error.tolist(),
        "increment_score_mae_by_axis": np.abs(numerators - physical_scores).mean(axis=0).tolist(),
        "normalized_score_mae_by_axis": np.abs(predicted_scores - physical_scores).mean(axis=0).tolist(),
        "physical_cross_boundary_margin_mae": float(np.abs(physical_margin - predicted_margin).mean()),
        "predicted_cross_boundary_margin_negative_fraction": float(np.mean(predicted_margin <= 0)),
        "fastest_inside_true_top2_windows": int(inside[:, 3].sum()),
        "fastest_inside_true_top2_accuracy": float(correct[inside[:, 3]].mean()) if np.any(inside[:, 3]) else None,
    })
    return result, {
        "scores": predicted_scores,
        "top2_correct": correct,
        "physical_margin": physical_margin,
        "predicted_margin": predicted_margin,
    }


def inputs(mode, values, model):
    if mode == "frozen":
        return embed(model, values)
    if mode == "cubic_shear":
        transformed = values.copy()
        transformed[..., 0] += 0.2 * values[..., 1] ** 3
        return transformed
    if mode == "invertible_observation":
        return observe(torch.as_tensor(values, dtype=torch.float32)).numpy().astype(np.float64)
    return np.concatenate([values[..., :3], normalized_h2(values[..., [0]])], axis=-1)


def run_probe_panel(mode, config, values, tests, model):
    encoded = {role: inputs(mode, z, model) for role, z in values.items()}
    center, scale = standardizer(encoded["calibration"])
    standardized = {role: (x.reshape(-1, 4) - center) / scale for role, x in encoded.items()}
    x_cal, x_val = standardized["calibration"], standardized["validation"]
    y_cal, y_val = values["calibration"], values["validation"]
    encoded_tests = {
        case: (inputs(mode, z, model).reshape(-1, 4) - center) / scale
        for case, z in tests.items()
    }
    h2_probe = fit_ridge(
        x_cal, normalized_h2(y_cal[:, [0]]), x_val, normalized_h2(y_val[:, [0]]), config, False
    )
    h2_results = {
        case: float(r2(
            normalized_h2(z[..., [0]]).reshape(-1, 1),
            predict_ridge(encoded_tests[case], h2_probe, False),
        )[0])
        for case, z in tests.items()
    }
    summary, arrays, weights = {}, {}, {}
    for rule in RULES:
        started = time.monotonic()
        if rule == "mlp":
            decoder, selection = fit_mlp(x_cal, y_cal, x_val, y_val, config)
            state = decoder.state_dict()
            decoded = {case: predict_mlp(x, decoder) for case, x in encoded_tests.items()}
            decoded_ref = predict_mlp(standardized["reference"], decoder)
        else:
            use_cubic = rule == "cubic"
            state = fit_ridge(x_cal, y_cal, x_val, y_val, config, use_cubic)
            selection = {key: value for key, value in state.items() if key.startswith(("validation_", "ridge_"))}
            decoded = {case: predict_ridge(x, state, use_cubic) for case, x in encoded_tests.items()}
            decoded_ref = predict_ridge(standardized["reference"], state, use_cubic)
        summary[rule] = {
            "selection": selection,
            "evaluation": {},
        }
        for case, true_test in tests.items():
            decoded_test = decoded[case]
            task, window_arrays = task_metrics(
                decoded_test.reshape(true_test.shape), decoded_ref, true_test, config["lags"]
            )
            factor_r2 = r2(true_test.reshape(-1, 4), decoded_test)
            summary[rule]["evaluation"][case] = {
                "heldout_physical_r2": factor_r2.tolist(),
                "task": task,
            }
            if window_arrays is not None:
                arrays[rule, case] = window_arrays
        summary[rule]["seconds"] = time.monotonic() - started
        weights[rule] = state
    for rule in ("cubic", "mlp"):
        for case in tests:
            if ("linear", case) in arrays and (rule, case) in arrays:
                baseline = arrays["linear", case]["top2_correct"]
                candidate = arrays[rule, case]["top2_correct"]
                summary[rule]["evaluation"][case]["paired_top2_vs_linear"] = {
                    "fixes": int((~baseline & candidate).sum()),
                    "new_errors": int((baseline & ~candidate).sum()),
                }
    return summary, arrays, {"input_mean": center, "input_std": scale, "h2_linear": h2_probe, **weights}, h2_results


def replay_source():
    training = json.loads((TRAINING / "runs/results.json").read_text())
    source_lock = json.loads((TRAINING / "../prospective/pipeline_forecast_receipt.json").read_text())
    return training, source_lock


def run(config, smoke):
    prior, checkpoint_lock = replay_source()
    output = HERE / ("smoke" if smoke else "runs")
    output.mkdir(exist_ok=False)
    records = {}
    cases = list(prior["cases"].items())
    if smoke:
        cases = cases[:1]
    for case, profile in cases:
        values = datasets(config, np.asarray(profile["rates"]))
        test = values.pop("test")
        gap = case.rsplit("_", maxsplit=1)[-1]
        tests = {
            evaluation_case: (
                test if evaluation_case == case else datasets(config, np.asarray(evaluation_profile["rates"]))["test"]
            )
            for evaluation_case, evaluation_profile in prior["cases"].items()
            if evaluation_case.endswith("_" + gap)
        }
        if case == cases[0][0]:
            assert not any(
                np.array_equal(values[a][0], values[b][0])
                for a, b in itertools.combinations(("calibration", "validation", "reference"), 2)
            )
        probes = [(f"{case}_{seed}", "frozen", int(seed)) for seed in profile["runs"]]
        if case.startswith("separated"):
            probes += [(f"{case}_{mode}", mode, None) for mode in MODES[1:]]
        if smoke:
            probes = probes[:1] + probes[-3:]
        physical = {name: increment_statistics(z, config["lags"]) for name, z in tests.items()}
        exact_constant = test.copy()
        exact_constant[..., 3] = 0
        exact_const_ref = values["reference"].copy()
        exact_const_ref[..., 3] = 0
        zero_variance, _ = task_metrics(exact_constant, exact_const_ref, test, config["lags"])
        assert zero_variance["status"] == "undefined_constant_reference_coordinate"
        for name, mode, seed in probes:
            encoder = None
            sha = None
            if seed is not None:
                locked_protocol = json.loads((TRAINING / "protocol.json").read_text())
                checkpoint = TRAINING / "runs" / f"{case}_{seed}.pt"
                sha = digest(checkpoint)
                if sha != checkpoint_lock["checkpoint_sha256"][checkpoint.name]:
                    raise ValueError(f"checkpoint changed: {name}")
                encoder = make_mlp_encoder(4, locked_protocol["hidden"], locked_protocol["layers"], device="cpu").eval()
                encoder.load_state_dict(torch.load(checkpoint, weights_only=True))
            result, arrays, weights, h2 = run_probe_panel(mode, config, values, tests, encoder)
            np.savez_compressed(
                output / f"{name}.npz",
                **{f"{eval_case}_physical_scores": score for eval_case, score in physical.items()},
                **{
                    f"{eval_case}_{rule}_{key}": val
                    for (rule, eval_case), rule_arrays in arrays.items()
                    for key, val in rule_arrays.items()
                },
            )
            torch.save(weights, output / f"{name}_readouts.pt")
            record = {
                "case": case,
                "rates": profile["rates"],
                "mode": mode,
                "seed": seed,
                "encoder_sha256": sha,
                "zero_variance_control": zero_variance,
                "heldout_h2_slowest_linear_r2": h2,
                "readouts": result,
            }
            json_dump(output / f"{name}.json", record)
            records[name] = record
            values_by_rule = [round(result[rule]["evaluation"][case]["heldout_physical_r2"][3], 4) for rule in RULES]
            print(f"{name}: fastest R²={values_by_rule}", flush=True)
    json_dump(
        output / "results.json",
        {
            "protocol_sha256": digest(HERE / "protocol.json"),
            "source_sha256": digest(Path(__file__)),
            "config": config,
            "status": "isolated engineering smoke" if smoke else "locked decoder diagnosis",
            "environment": {"architecture": platform.machine(), "torch": torch.__version__, "numpy": np.__version__},
            "runs": records,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((HERE / "protocol.json").read_text())
    if args.smoke:
        config = {
            **config,
            "master": config["master"] + 10001,
            "n_calibration": 512,
            "n_validation": 256,
            "n_reference": 1024,
            "n_test": 16,
            "mlp_steps": 500,
            "mlp_eval_interval": 250,
        }
    run(config, args.smoke)


if __name__ == "__main__":
    main()
