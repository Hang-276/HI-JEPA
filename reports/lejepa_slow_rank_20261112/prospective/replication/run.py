"""Replicate only the original two DEV 2.5% thresholds on fresh banks and windows."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import pipeline as primary
from score import clopper_pearson_upper, confidence, risk, sha256, top2, verify_lock, write_json


REPLICATION_MASTER = 2026111202
TEST_SIZE = 32768
TARGET_DEV_COVERAGE = 0.025
RISK_LIMIT = 0.05
ALPHA = 0.05 / 2
TRAINING = primary.TRAINING


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_npz(path, **values):
    with path.open("xb") as destination:
        np.savez_compressed(destination, **values)


def bind_original():
    original_lock = PARENT / "THRESHOLD_LOCK.json"
    original_protocol = PARENT / "PROTOCOL.md"
    original_forecast = PARENT / "forecasts.npz"
    lock = verify_lock(original_forecast, original_protocol, original_lock)
    state = load(PARENT / "pipeline_forecast_receipt.json")
    if state["master_rng"] == REPLICATION_MASTER or state["master_rng"] != primary.MASTER:
        raise ValueError("Primary and replication RNG streams are not correctly separated.")
    pipeline_hash = sha256(Path(primary.__file__))
    training_protocol = TRAINING / "protocol.json"
    if pipeline_hash != state["pipeline_sha256"] or sha256(training_protocol) != state["training_protocol_sha256"]:
        raise ValueError("Primary prediction code or training protocol differs from the locked run.")
    for filename, checkpoint_hash in state["checkpoint_sha256"].items():
        if sha256(TRAINING / "runs" / filename) != checkpoint_hash:
            raise ValueError(f"Frozen checkpoint changed: {filename}")
    rules = {}
    for method, values in lock["rules"].items():
        matched = [rule for rule in values if rule["dev_target_coverage"] == TARGET_DEV_COVERAGE]
        if len(matched) != 1:
            raise ValueError("Original rule at DEV 2.5% is missing or ambiguous.")
        rules[method] = matched[0]
    provenance = {
        "primary_threshold_lock_sha256": sha256(original_lock),
        "primary_forecast_sha256": sha256(original_forecast),
        "primary_pipeline_sha256": pipeline_hash,
        "primary_scorer_sha256": lock["scorer_sha256"],
        "replication_protocol_sha256": sha256(HERE / "PROTOCOL.md"),
        "replication_runner_sha256": sha256(Path(__file__)),
        "training_protocol_sha256": sha256(training_protocol),
        "checkpoint_sha256": state["checkpoint_sha256"],
        "replication_rng": REPLICATION_MASTER,
        "original_rng": primary.MASTER,
        "test_count": TEST_SIZE,
        "rules": rules,
    }
    return load(training_protocol), provenance


def acceptance(ranks, rules):
    return {method: ranks[method] >= rule["threshold"] for method, rule in rules.items()}


def build_predictions(config, rules, show_progress):
    primary.MASTER = REPLICATION_MASTER
    primary.TEST_SIZE = TEST_SIZE
    banks = primary.bank_definitions(config)
    if len(banks) != 64:
        raise ValueError("Expected exactly 64 replication banks.")
    assigned = primary.allocation(len(banks))
    nominal = np.empty((TEST_SIZE, 4))
    uncertainty = np.empty(TEST_SIZE)
    model = None
    previous_checkpoint = None
    for bank, (_, case, seed, _, rates) in enumerate(banks):
        checkpoint = TRAINING / "runs" / f"{case}_{seed}.pt"
        if previous_checkpoint != checkpoint:
            model = primary.make_mlp_encoder(4, config["hidden"], config["layers"], device="cpu").eval()
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
            previous_checkpoint = checkpoint
        transforms, variances, _ = primary.frozen_bank(model, config, rates, bank)
        indices = np.flatnonzero(assigned == bank)
        if len(indices):
            observations = primary.test_paths(rates, indices, config["trajectory_length"])
            candidates = primary.candidate_scores(
                primary.embed(model, observations), transforms, variances, config["lags"]
            )
            nominal[indices], uncertainty[indices] = primary.nominal_and_uncertainty(candidates)
        if show_progress and (bank + 1) % 8 == 0:
            print(f"{bank + 1}/64 new calibration banks forecast", flush=True)
    ranks = confidence(nominal, uncertainty, TEST_SIZE)
    masks = acceptance(ranks, rules)
    return {
        "test_scores": nominal,
        "test_uncertainty": uncertainty,
        "test_bank": assigned,
        "bank_case": np.array([row[0] for row in banks]),
        "bank_model_seed": np.array([row[2] for row in banks]),
        "case_names": np.array(list(config["cases"])),
        **{f"accept_{name}": mask for name, mask in masks.items()},
    }


def forecast():
    config, provenance = bind_original()
    values = build_predictions(config, provenance["rules"], show_progress=True)
    path = HERE / "FORECAST.npz"
    write_npz(path, **values)
    write_json(
        HERE / "FORECAST_LOCK.json",
        {
            **provenance,
            "forecast_sha256": sha256(path),
            "test_bank_counts": np.bincount(values["test_bank"], minlength=64).tolist(),
            "label_free_accepts": {
                method: int(values[f"accept_{method}"].sum()) for method in provenance["rules"]
            },
        },
    )


def verify_replication_lock():
    config, current = bind_original()
    lock = load(HERE / "FORECAST_LOCK.json")
    if any(lock[key] != value for key, value in current.items()):
        raise ValueError("Locked replication provenance changed.")
    if sha256(HERE / "FORECAST.npz") != lock["forecast_sha256"]:
        raise ValueError("Replication forecast differs from the locked forecast.")
    return config, lock


def verify_forecast():
    config, lock = verify_replication_lock()
    replayed = build_predictions(config, lock["rules"], show_progress=False)
    with np.load(HERE / "FORECAST.npz", allow_pickle=False) as locked:
        for key, values in replayed.items():
            np.testing.assert_array_equal(locked[key], values)
    print("Exact reproduction of the 32,768 frozen forecasts and acceptance masks.")


def reveal():
    config, lock = verify_replication_lock()
    primary.MASTER = REPLICATION_MASTER
    primary.TEST_SIZE = TEST_SIZE
    banks = primary.bank_definitions(config)
    assigned = primary.allocation(len(banks))
    with np.load(HERE / "FORECAST.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["test_bank"], assigned)
    target = np.empty((TEST_SIZE, 4))
    unit_reference = np.array([[-1.0] * 4, [1.0] * 4])[None, :, :]
    for bank, (_, _, _, _, rates) in enumerate(banks):
        indices = np.flatnonzero(assigned == bank)
        if len(indices):
            states = primary.test_paths(rates, indices, config["trajectory_length"])
            target[indices] = primary.scores(states, unit_reference, config["lags"])
    write_npz(
        HERE / "REVEAL.npz",
        test_physical_scores=target,
        test_bank=assigned,
        forecasts_sha256=lock["forecast_sha256"],
        forecast_lock_sha256=sha256(HERE / "FORECAST_LOCK.json"),
    )


def score():
    _, lock = verify_replication_lock()
    reveal_path = HERE / "REVEAL.npz"
    lock_hash = sha256(HERE / "FORECAST_LOCK.json")
    with np.load(HERE / "FORECAST.npz", allow_pickle=False) as forecast_file, np.load(reveal_path, allow_pickle=False) as truth:
        if (
            str(truth["forecasts_sha256"].item()) != lock["forecast_sha256"]
            or str(truth["forecast_lock_sha256"].item()) != lock_hash
        ):
            raise ValueError("Replication truth is not bound to the forecast lock.")
        np.testing.assert_array_equal(forecast_file["test_bank"], truth["test_bank"])
        nominal = forecast_file["test_scores"]
        target = truth["test_physical_scores"]
        if target.shape != nominal.shape or not np.isfinite(target).all():
            raise ValueError("Replication truth does not match the frozen 4-axis score rows.")
        wrong = np.any(top2(nominal) != top2(target), axis=1)
        masks = acceptance(confidence(nominal, forecast_file["test_uncertainty"], TEST_SIZE), lock["rules"])
        cases = forecast_file["bank_case"][forecast_file["test_bank"]]
        case_names = forecast_file["case_names"].tolist()
        outcomes = {}
        for method, mask in masks.items():
            np.testing.assert_array_equal(mask, forecast_file[f"accept_{method}"])
            accepted = int(mask.sum())
            errors = int((mask & wrong).sum())
            upper = clopper_pearson_upper(errors, accepted, ALPHA)
            by_case = {}
            for case, case_name in enumerate(case_names):
                in_case = cases == case
                by_case[case_name] = risk(
                    int((mask & in_case).sum()), int((mask & wrong & in_case).sum()), int(in_case.sum())
                )
            outcomes[method] = {
                **risk(accepted, errors, TEST_SIZE),
                "simultaneous_selective_error_upper": upper,
                "passes": upper is not None and upper <= RISK_LIMIT,
                "case_composition_descriptive": by_case,
            }
    write_json(
        HERE / "SCORE_SUMMARY.json",
        {
            "forecast_lock_sha256": lock_hash,
            "reveal_sha256": sha256(reveal_path),
            "selective_error_limit": RISK_LIMIT,
            "per_rule_one_sided_alpha": ALPHA,
            "test_count": TEST_SIZE,
            "nominal_baseline": risk(TEST_SIZE, int(wrong.sum()), TEST_SIZE),
            "fixed_rules": outcomes,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = {"forecast": forecast, "verify": verify_forecast, "reveal": reveal, "score": score}
    parser.add_argument("stage", choices=actions)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    actions[args.stage]()


if __name__ == "__main__":
    main()
