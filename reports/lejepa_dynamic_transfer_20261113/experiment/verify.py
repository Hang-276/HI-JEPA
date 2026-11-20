"""Validate the sealed receipts and cross-arm fit/forecast invariants."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parents[1]
TRAINING = REPORTS / "lejepa_slow_rank_20261112/training"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(name):
    return json.loads((HERE / name).read_text())


def main():
    torch.set_num_threads(1)
    config = load("protocol.json")
    original = json.loads((TRAINING / "protocol.json").read_text())
    fits, forecast, reveal = map(load, ("FITS_RECEIPT.json", "FORECAST_RECEIPT.json", "REVEAL_RECEIPT.json"))
    assert fits["config"] == config
    assert digest(HERE / "FITS_RECEIPT.json") == forecast["fits_receipt_sha256"]
    assert digest(HERE / "FORECAST_RECEIPT.json") == reveal["forecast_receipt_sha256"]
    assert digest(HERE / "labels.npz") == reveal["labels_sha256"]
    assert fits["source_hashes"] == forecast["source_hashes"]
    for relative, sha in fits["source_hashes"].items():
        assert digest(REPORTS / relative) == sha
    pairs = itertools.product(original["cases"], original["seeds"], config["probe_seeds"])
    shape = (config["n_test_trajectories"], max(config["test_lengths"]), 4)
    reference_shape = (config["n_reference_banks"], 4)
    verified = 0
    max_variance_identity_error = 0.0
    n_matched = 0
    for case, encoder_seed, probe_seed in pairs:
        schedules, input_normalizers, encoder_hashes, val_r2 = set(), set(), set(), {}
        for arm in config["arms"]:
            name = f"{case}_{encoder_seed}_{probe_seed}_{arm}"
            record_path = HERE / "fits" / f"{name}.json"
            weights_path = HERE / "fits" / f"{name}.pt"
            predict_path = HERE / "forecast" / f"{name}.npz"
            assert digest(record_path) == fits["fit_records_sha256"][record_path.name]
            assert digest(weights_path) == fits["fitted_readouts_sha256"][weights_path.name]
            assert digest(predict_path) == forecast["forecast_files_sha256"][predict_path.name]
            record = json.loads(record_path.read_text())
            schedules.add(record["pair_schedule_sha256"])
            encoder_hashes.add(record["encoder_sha256"])
            assert record["validation_trace"][-1]["step"] == config["steps"]
            assert record["val_final_r2"] == record["validation_trace"][-1]["r2_by_axis"]
            val_r2[arm] = np.asarray(record["val_final_r2"])
            weights = torch.load(weights_path, weights_only=True)
            input_normalizers.add((tuple(weights["input_mean"]), tuple(weights["input_std"])))
            with np.load(predict_path, allow_pickle=False) as saved:
                assert saved["test_predictions"].shape == shape
                assert saved["test_predictions"].dtype == np.float32
                assert np.isfinite(saved["test_predictions"]).all()
                assert all(saved[key].shape == reference_shape for key in ("ref_mean", "ref_second_moment", "ref_variance"))
                assert all(saved[key].dtype == np.float64 for key in ("ref_mean", "ref_second_moment", "ref_variance"))
                assert np.all(saved["ref_variance"] > 0)
                identity_error = np.max(np.abs(saved["ref_second_moment"] - saved["ref_mean"]**2 - saved["ref_variance"]))
                max_variance_identity_error = max(max_variance_identity_error, float(identity_error))
            verified += 1
        assert len(schedules) == len(input_normalizers) == len(encoder_hashes) == 1
        n_matched += np.max(np.abs(val_r2["state"] - val_r2["temporal_1"])) <= config["validation_match_max_abs_r2_gap"]
    assert verified == len(fits["fit_records_sha256"]) == len(forecast["forecast_files_sha256"])
    assert max_variance_identity_error <= 1e-12
    with np.load(HERE / "labels.npz", allow_pickle=False) as labels:
        assert labels.files == list(original["cases"])
        assert all(labels[key].shape == shape and labels[key].dtype == np.float64 for key in labels.files)
        for gap in ("wide", "narrow"):
            assert np.array_equal(labels[f"separated_{gap}"][..., :3], labels[f"crossed_{gap}"][..., :3])
        assert np.array_equal(labels["separated_wide"][:, 0], labels["crossed_narrow"][:, 0])
    print(f"Verified {verified} sealed fits and forecasts; VAL-matched contrasts {n_matched}/32")
    print(f"Max reference moment identity error {max_variance_identity_error:.3e}")


if __name__ == "__main__":
    main()
