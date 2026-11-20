"""Audit the natural-direction artifacts using their locked data and encoders."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


A1 = Path(__file__).resolve().parents[1]
NATURAL = A1 / "natural"
REPORTS = A1.parent
PAIRS = np.triu_indices(4, 1)
FIXED_PAIR = (1, 2)
RULES = ("nominal", "diag", "full")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def choose(scores):
    return np.sort(np.argsort(scores, axis=-1, kind="stable")[..., :2], axis=-1)


def moments(states, lags):
    return np.mean(
        [
            (change.transpose(0, 2, 1) @ change) / change.shape[1]
            for lag in lags
            for change in [states[:, lag:] - states[:, :-lag]]
        ],
        axis=0,
    )


def track(scores, physical, key):
    correct = np.all(choose(scores) == choose(physical), axis=1)
    left, right = PAIRS
    pair_correct = (
        (scores[:, right] - scores[:, left] >= 0)
        == (physical[:, right] - physical[:, left] >= 0)
    )
    assert (left[3], right[3]) == FIXED_PAIR
    np.testing.assert_equal(correct.mean(), key["top2_accuracy"])
    np.testing.assert_array_equal(pair_correct.mean(axis=0), key["six_pair_accuracy"])
    return correct, pair_correct[:, 3]


def safety(scores, widths, nominal, physical):
    n = len(scores)
    membership = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(membership, choose(nominal), True, axis=1)
    widths_matrix = np.zeros((4, 4))
    widths_matrix[PAIRS] = widths
    widths_matrix += widths_matrix.T.copy()
    crossed = membership[:, :, None] & ~membership[:, None, :]
    gaps = scores[:, None] - scores[:, :, None]
    safe = np.all(~crossed | (gaps > widths_matrix), axis=(1, 2))
    errors = np.abs((scores - physical)[:, PAIRS[1]] - (scores - physical)[:, PAIRS[0]])
    covered = np.all(errors <= widths, axis=1)
    nominal_wrong = np.any(choose(nominal) != choose(physical), axis=1)
    wrong_safe = nominal_wrong & safe
    assert not np.any(wrong_safe & covered)
    return safe, (int(covered.sum()), int(safe.sum()), int(wrong_safe.sum()), n)


def audit_saved(round_name, config, original):
    base = NATURAL / round_name
    forecast = read(base / "FORECAST_RECEIPT.json")
    reveal = read(base / "REVEAL_RECEIPT.json")
    result = read(base / "RESULTS.json")
    assert not forecast["smoke"]
    for path, expected in forecast["source_hashes"].items():
        assert sha(REPORTS / path) == expected, path
    assert sha(base / "FORECAST_RECEIPT.json") == reveal["forecast_receipt_sha256"] == result["forecast_receipt_sha256"]
    assert sha(base / "labels.npz") == reveal["labels_sha256"] == result["labels_sha256"]
    if round_name == "replication":
        assert not forecast["parameter_sha256"]
        assert forecast["initial_receipt_sha256"] == sha(NATURAL / "primary/FORECAST_RECEIPT.json")
    assert len(forecast["prediction_sha256"]) == len(original["parameter_sha256"]) == 32
    deltas = []
    counts = {rule: np.zeros(4, dtype=int) for rule in ("diag", "full")}
    with np.load(base / "labels.npz", allow_pickle=False) as targets:
        for name, expected in forecast["prediction_sha256"].items():
            prediction_path = base / name
            assert sha(prediction_path) == expected
            task = Path(name).stem
            process = task.split("__")[0]
            parameter_path = NATURAL / "primary" / f"{task}.json"
            assert sha(parameter_path) == original["parameter_sha256"][parameter_path.name]
            parameter = read(parameter_path)
            physical = targets[process]
            official = result["tasks"][task]["metrics"]
            with np.load(prediction_path, allow_pickle=False) as prediction:
                assert not any("physical" in key or "truth" in key for key in prediction.files)
                accuracy = {}
                for rule in RULES:
                    scores = prediction[f"{rule}_scores"]
                    accuracy[rule] = track(scores, physical, official[rule])
                    if rule == "nominal":
                        continue
                    method = parameter["methods"][rule]
                    order = int(np.ceil((1 + config["n_conformal"]) * (1 - config["conformal_alpha"])))
                    assert method["conformal_order"] == order == 245
                    widths = np.asarray(method["gap_widths"])
                    safe, totals = safety(scores, widths, prediction["nominal_scores"], physical)
                    np.testing.assert_array_equal(safe, prediction[f"{rule}_safe"])
                    covered, accepted, wrong, n = totals
                    report = official[rule]["conformal"]
                    assert report["all_six_marginal_coverage"] == covered / n
                    assert report["acceptance_coverage"] == accepted / n
                    assert report["accepted_nominal_errors"] == wrong
                    assert report["unconditional_accepted_error"] == wrong / n
                    counts[rule] += totals
                deltas.append([f.astype(float) - d for f, d in zip(accuracy["full"], accuracy["diag"], strict=True)])
    common_rows = np.mean(deltas, axis=0).T
    n = config["n_test"]
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = rng.multinomial(n, np.full(n, 1 / n), size=config["bootstrap_repetitions"])
    intervals = np.quantile(draws @ common_rows / n, [0.025, 0.975], axis=0)
    official = result["groups"]["all"]
    for i, key in enumerate(("top2", "fixed_pair")):
        np.testing.assert_allclose(intervals[:, i], official[f"{key}_paired_ci95"], atol=1e-15, rtol=0)
        np.testing.assert_allclose(common_rows[:, i].mean(), official[f"{key}_delta_full_minus_diag"], atol=1e-15, rtol=0)
    passed = common_rows[:, 0].mean() >= config["primary_minimum_gain"] and intervals[0, 0] > 0
    assert passed == result["primary_gate_passes"]
    print(round_name, "top-2/fixed-pair gains and intervals", common_rows.mean(axis=0), intervals.T, "pass", passed)
    print(round_name, "covered, accepted, wrong-and-accepted, total:", {k: v.tolist() for k, v in counts.items()})


def replay_reference(config, original):
    spec = importlib.util.spec_from_file_location("direction_reference_replay", NATURAL / "run.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    source.torch.set_num_threads(1)
    old = read(source.TRAINING / "protocol.json")
    encoder_hashes = read(source.ORIGINAL / "prospective/pipeline_forecast_receipt.json")["checkpoint_sha256"]
    master = config["master_replication"]
    ref_states = np.random.default_rng(np.random.SeedSequence([master, 3])).normal(size=(config["n_reference"], 4))
    tests = {}
    for process, rates in config["processes"].items():
        rho = np.asarray(rates)
        noise = np.random.default_rng(np.random.SeedSequence([master, 7])).normal(
            size=(config["n_test"], config["length"], 4)
        )
        physical = np.empty_like(noise)
        physical[:, 0] = noise[:, 0]
        for t in range(1, config["length"]):
            physical[:, t] = physical[:, t - 1] * rho + noise[:, t] * np.sqrt(1 - rho * rho)
        tests[process] = physical
    max_diff = 0.0
    count = 0
    for name, expected in original["parameter_sha256"].items():
        path = NATURAL / "primary" / name
        assert sha(path) == expected
        process, model_name = path.stem.split("__")
        case, seed = model_name.rsplit("_", maxsplit=1)
        model, encoder_hash = source.load_encoder(case, int(seed), old, encoder_hashes)
        params = read(path)
        assert encoder_hash == params["encoder_sha256"]
        transform = np.asarray(params["transform"])
        old_var = np.asarray(params["calibration_ref_variance"])
        y = source.source.embed(model, tests[process]) @ transform
        new_var = (source.source.embed(model, ref_states) @ transform).var(axis=0)
        # Directly normalize by the original reference. The production path
        # first uses new_var, then changes the moment coordinates back.
        original_moment = moments(y / np.sqrt(old_var), config["lags"])
        new_moment = moments(y / np.sqrt(new_var), config["lags"])
        scale = np.sqrt(new_var / old_var)
        np.testing.assert_allclose(original_moment, new_moment * scale[:, None] * scale[None, :], atol=3e-14)
        with np.load(NATURAL / "replication" / f"{path.stem}.npz", allow_pickle=False) as saved:
            np.testing.assert_allclose(new_var, saved["ref_variance"], atol=3e-14)
            np.testing.assert_allclose(new_moment.diagonal(axis1=1, axis2=2), saved["nominal_scores"], atol=3e-14)
            for rule in ("diag", "full"):
                if rule == "diag":
                    features = original_moment.diagonal(axis1=1, axis2=2)
                else:
                    features = original_moment[:, *np.triu_indices(4)]
                fit = params["methods"][rule]
                centered = (features - fit["feature_mean"]) / fit["feature_std"]
                pred = centered @ fit["coefficients"] + fit["target_mean"]
                max_diff = max(max_diff, float(np.max(np.abs(pred - saved[f"{rule}_scores"]))))
                np.testing.assert_allclose(pred, saved[f"{rule}_scores"], atol=3e-12)
        count += 1
    print("Direct old-reference forward replay:", count, "tasks, max prediction deviation", max_diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-reference", action="store_true")
    arguments = parser.parse_args()
    config = read(NATURAL / "protocol.json")
    primary = read(NATURAL / "primary/FORECAST_RECEIPT.json")
    for run in ("primary", "replication"):
        audit_saved(run, config, primary)
    if arguments.replay_reference:
        replay_reference(config, primary)
