"""Verify the locked experiment receipts and replay held-out scores from saved weights."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from run import HERE, evaluate, make_mlp_encoder, scores, top_two, trajectory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config_path = HERE / "protocol.json"
    config = json.loads(config_path.read_text())
    runs_path = HERE / "runs"
    receipt = json.loads((runs_path / "results.json").read_text())
    assert digest(config_path) == receipt["protocol_sha256"]
    assert digest(HERE / "run.py") == receipt["code_sha256"]
    assert list(receipt["cases"]) == list(config["cases"])
    summary = {
        "config_sha256": digest(config_path),
        "code_sha256": digest(HERE / "run.py"),
        "verification": "all held-out scores replayed identically from saved state dictionaries",
        "total_runs": 0,
        "total_windows": 0,
        "total_elapsed_seconds": 0.0,
        "max_abs_training_pair_correlation_error": 0.0,
        "cases": {},
    }
    for case, rates in config["cases"].items():
        runs = []
        physical_top_fastest = []
        physical_population_baseline = []
        minimum_factor_r2 = []
        rank_bins = {}
        case_result = receipt["cases"][case]
        assert list(case_result["runs"]) == list(map(str, config["seeds"]))
        for seed in config["seeds"]:
            result = case_result["runs"][str(seed)]
            assert json.loads((runs_path / f"{case}_{seed}.json").read_text()) == result
            assert result["training_log"][-1]["step"] == config["steps"]
            difference = np.abs(np.asarray(result["training_pair_correlations"]) - rates)
            assert np.max(difference) < 0.005
            summary["max_abs_training_pair_correlation_error"] = max(
                summary["max_abs_training_pair_correlation_error"], float(np.max(difference))
            )
            model = make_mlp_encoder(4, config["hidden"], config["layers"], device="cpu").eval()
            model.load_state_dict(torch.load(runs_path / f"{case}_{seed}.pt", weights_only=True))
            replay = evaluate(model, config, seed, np.asarray(rates))
            assert all(value == result[key] for key, value in replay.items()), (case, seed)
            assert sorted(replay["native_named_permutation"]) == [0, 1, 2, 3]
            assert sorted(replay["spectral_named_permutation"]) == [0, 1, 2, 3]
            physical_reference = trajectory(
                np.asarray(rates), seed, 102, config["ref_trajectories"], config["trajectory_length"]
            )
            physical_test = trajectory(
                np.asarray(rates), seed, 104, config["test_trajectories"], config["trajectory_length"]
            )
            expected = top_two(scores(physical_test, physical_reference, config["lags"]))
            physical_top_fastest.append(float(np.any(expected == 3, axis=1).mean()))
            physical_population_baseline.append(replay["physical_population_set_window_agreement"])
            minimum_factor_r2.append(min(replay["oracle_linear_factor_r2"]))
            for rule, score in replay["top2"].items():
                assert score["windows"] == config["test_trajectories"]
                assert score["correct"] == sum(bin["correct"] for bin in score["margin_bins"])
                assert score["windows"] == sum(bin["windows"] for bin in score["margin_bins"])
                if rule not in rank_bins:
                    rank_bins[rule] = np.zeros((len(config["margin_bins"]) + 1, 2), dtype=int)
                rank_bins[rule] += np.array([[b["correct"], b["windows"]] for b in score["margin_bins"]])
            summary["total_runs"] += 1
            summary["total_windows"] += config["test_trajectories"]
            summary["total_elapsed_seconds"] += result["elapsed_seconds"]
            runs.append(replay)
        gains = [
            q["top2"]["spectral_named"]["accuracy"] - q["top2"]["native_named"]["accuracy"] for q in runs
        ]
        summary["cases"][case] = {
            "linear_nonlinear_spectral_gap": case_result["linear_nonlinear_spectral_gap"],
            "physical_population_gap_2_3": case_result["physical_population_gap_2_3"],
            "whitening_relaxation_top_modes": case_result["whitening_relaxation_top_modes"],
            "factor_r2_mean": np.mean([q["oracle_linear_factor_r2"] for q in runs], axis=0).tolist(),
            "factor_r2_min_of_all_axes_seed_range": summarize(minimum_factor_r2),
            "fastest_factor_r2": [q["oracle_linear_factor_r2"][3] for q in runs],
            "slowest_h2_r2": [q["hermite"]["test_h2_linear_readout_r2"][0] for q in runs],
            "slowest_h2_forward_r2_gain": [
                q["hermite"]["test_embedding_r2_gain_from_h2_after_linear_factors"][0] for q in runs
            ],
            "top2_accuracy": {
                rule: summarize([q["top2"][rule]["accuracy"] for q in runs]) for rule in runs[0]["top2"]
            },
            "spectral_minus_native_accuracy": summarize(gains),
            "paired_spectral_improvements": sum(gain > 0 for gain in gains),
            "top2_margin_bins_counts": {rule: counts.tolist() for rule, counts in rank_bins.items()},
            "fastest_factor_in_window_slow_set": summarize(physical_top_fastest),
            "population_top2_baseline": summarize(physical_population_baseline),
        }
    assert math.isfinite(summary["total_elapsed_seconds"])
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"replayed {summary['total_runs']} runs and {summary['total_windows']} window scores")


if __name__ == "__main__":
    main()
