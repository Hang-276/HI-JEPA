"""Score the same predicted top-2 on physical targets using known V=1."""

import json

import numpy as np
import torch

from run import (
    HERE,
    assign_names,
    embed,
    fit,
    fit_lag_axes,
    make_mlp_encoder,
    named_responses,
    predict,
    ranking_metrics,
    scores,
    top_two,
    trajectory,
)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((HERE / "protocol.json").read_text())
    primary = json.loads((HERE / "runs/results.json").read_text())
    population_reference = np.array([[-1.0] * 4, [1.0] * 4])[None, :, :]
    cases = {}
    for case, rates in config["cases"].items():
        rates = np.asarray(rates)
        by_seed = {}
        for seed in config["seeds"]:
            length = config["trajectory_length"]
            cal = trajectory(rates, seed, 101, config["cal_trajectories"], length)
            ref = trajectory(rates, seed, 102, config["ref_trajectories"], length)
            test = trajectory(rates, seed, 104, config["test_trajectories"], length)
            model = make_mlp_encoder(4, config["hidden"], config["layers"], device="cpu").eval()
            weights = torch.load(HERE / f"runs/{case}_{seed}.pt", weights_only=True)
            model.load_state_dict(weights)
            cal_h, ref_h, test_h = [embed(model, z) for z in (cal, ref, test)]
            spectral, _, _ = fit_lag_axes(cal_h)
            responses = named_responses(model, seed, config)
            native, _ = assign_names(responses)
            spectral_names, _ = assign_names(responses @ spectral)
            oracle = fit(cal_h, cal)
            spectral_test = test_h @ spectral
            spectral_reference = ref_h @ spectral
            candidates = {
                "native_named": (test_h[:, :, native], ref_h[:, :, native]),
                "spectral_named": (spectral_test[:, :, spectral_names], spectral_reference[:, :, spectral_names]),
                "spectral_population_order": (spectral_test, spectral_reference),
                "oracle_linear": (predict(test_h, oracle), predict(ref_h, oracle)),
            }
            target_main = scores(test, ref, config["lags"])
            target_population = scores(test, population_reference, config["lags"])
            estimated = {key: scores(*xy, config["lags"]) for key, xy in candidates.items()}
            main = primary["cases"][case]["runs"][str(seed)]["top2"]
            for key, candidate in estimated.items():
                assert ranking_metrics(target_main, candidate, config["margin_bins"]) == main[key]
            by_seed[str(seed)] = {
                "physical_target_changes": int(
                    np.any(top_two(target_main) != top_two(target_population), axis=1).sum()
                ),
                "top2": {
                    key: ranking_metrics(target_population, candidate, config["margin_bins"])
                    for key, candidate in estimated.items()
                },
            }
        rows = list(by_seed.values())
        cases[case] = {
            "physical_target_changes": sum(row["physical_target_changes"] for row in rows),
            "windows": config["test_trajectories"] * len(rows),
            "mean_accuracy": {
                rule: float(np.mean([row["top2"][rule]["accuracy"] for row in rows]))
                for rule in rows[0]["top2"]
            },
            "by_seed": by_seed,
        }
    results = {
        "status": "supplemental sensitivity; frozen predictions verified against primary receipts",
        "target": "known population physical reference variance one",
        "prediction_change": "none; each predicted coordinate retains its original reference-panel variance",
        "cases": cases,
    }
    (HERE / "population_reference_sensitivity.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
