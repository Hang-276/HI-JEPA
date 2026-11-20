"""Post-reveal decomposition of the locked probes' 193-state residual moments."""

import numpy as np

from score import BASE, CONTROLS, EXP, HERE, digest, inputs, read, write


def moments(predicted, physical, lags):
    residual = predicted - physical
    endpoint = []
    autocross = []
    energy = []
    physical_error_cross = []
    for lag in lags:
        left = residual[:, :-lag]
        right = residual[:, lag:]
        delta = right - left
        delta_z = physical[:, lag:] - physical[:, :-lag]
        endpoint.append(((left**2 + right**2) / 2).mean(axis=(0, 1)))
        autocross.append((left * right).mean(axis=(0, 1)))
        energy.append((delta**2).mean(axis=(0, 1)))
        physical_error_cross.append((2 * delta_z * delta).mean(axis=(0, 1)))
    endpoint, autocross, energy, physical_error_cross = map(np.asarray, (endpoint, autocross, energy, physical_error_cross))
    np.testing.assert_allclose(endpoint - autocross, energy / 2, rtol=1e-12, atol=1e-14)
    return {
        "endpoint_state_mse": endpoint,
        "residual_cross_noncentered": autocross,
        "increment_residual_mse": energy,
        "physical_increment_times_residual_increment": physical_error_cross,
        "numerator_signed_bias": energy + physical_error_cross,
    }


def run():
    config, training, _ = inputs()
    lock = read(HERE / "VAL_MATCH_LOCK.json")
    forecast = read(EXP / "FORECAST_RECEIPT.json")
    reveal = read(EXP / "REVEAL_RECEIPT.json")
    assert digest(EXP / "labels.npz") == reveal["labels_sha256"]
    per_probe = {}
    with np.load(EXP / "labels.npz", allow_pickle=False) as labels:
        for case in training["cases"]:
            truth = labels[case][:, :193]
            for seed in training["seeds"]:
                for probe in config["probe_seeds"]:
                    name = f"{case}_{seed}_{probe}"
                    per_probe[name] = {}
                    for arm in config["arms"]:
                        filename = f"{name}_{arm}.npz"
                        path = EXP / "forecast" / filename
                        assert digest(path) == forecast["forecast_files_sha256"][filename]
                        with np.load(path, allow_pickle=False) as source:
                            predictions = source["test_predictions"][:, :193].astype(np.float64)
                        per_probe[name][arm] = moments(predictions, truth, config["lags"])
    groups = {}
    for group, names in lock["groups"].items():
        selected = [panel for key, panel in per_probe.items() if key.rsplit("_", 1)[0] in names]
        by_arm = {
            arm: {key: np.mean([probe[arm][key] for probe in selected], axis=0) for key in selected[0][arm]}
            for arm in config["arms"]
        }
        groups[group] = {
            "by_arm": {arm: {key: value.tolist() for key, value in metrics.items()} for arm, metrics in by_arm.items()},
            "temporal_1_minus": {
                arm: {key: (by_arm[BASE][key] - by_arm[arm][key]).tolist() for key in by_arm[BASE]} for arm in CONTROLS
            },
        }
    write(HERE / "RESIDUAL_DIAGNOSTICS.json", {
        "scope": "post-reveal arithmetic mechanism diagnostics, no new predictions or rule selection",
        "lags": config["lags"],
        "window_length": 193,
        "labels_sha256": reveal["labels_sha256"],
        "forecast_receipt_sha256": digest(EXP / "FORECAST_RECEIPT.json"),
        "groups": groups,
        "per_probe": {
            probe: {arm: {key: value.tolist() for key, value in stats.items()} for arm, stats in values.items()}
            for probe, values in per_probe.items()
        },
    })


if __name__ == "__main__":
    run()
