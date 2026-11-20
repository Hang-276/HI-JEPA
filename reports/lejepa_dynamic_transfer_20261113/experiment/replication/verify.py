"""Verify paired replication data and forecasts against archived receipts."""

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    forecast_path = HERE / "FORECAST_RECEIPT.json"
    forecast = load(forecast_path)
    reveal = load(HERE / "REVEAL_RECEIPT.json")
    original = load(EXP / "FORECAST_RECEIPT.json")
    assert forecast["replication_master"] == 2026111305
    assert forecast["replication_master"] != load(EXP / "protocol.json")["master"]
    assert forecast["fits_receipt_sha256"] == original["fits_receipt_sha256"]
    assert forecast["fits_receipt_sha256"] == digest(EXP / "FITS_RECEIPT.json")
    assert digest(forecast_path) == reveal["forecast_receipt_sha256"]
    assert digest(HERE / "labels.npz") == reveal["labels_sha256"]
    assert set(forecast["forecast_files_sha256"]) == set(original["forecast_files_sha256"])
    for source, sha in forecast["source_hashes"].items():
        assert digest(ROOT / source) == sha
    assert len(forecast["forecast_files_sha256"]) == 128
    old_reveal = load(EXP / "REVEAL_RECEIPT.json")
    assert reveal["labels_sha256"] != old_reveal["labels_sha256"]
    with np.load(EXP / "labels.npz", allow_pickle=False) as old, np.load(HERE / "labels.npz", allow_pickle=False) as new:
        assert new.files == old.files == reveal["test_truth_keys"]
        for case in new.files:
            assert new[case].shape == tuple(reveal["test_truth_shape"])
            assert not np.array_equal(new[case][:, 0], old[case][:, 0])
        for gap in ("wide", "narrow"):
            assert np.array_equal(new[f"separated_{gap}"][..., :3], new[f"crossed_{gap}"][..., :3])
        assert np.array_equal(new["separated_wide"][:, 0], new["crossed_narrow"][:, 0])
    for file, sha in forecast["forecast_files_sha256"].items():
        path = HERE / "forecast" / file
        assert sha != original["forecast_files_sha256"][file]
        assert digest(path) == sha
        with np.load(path, allow_pickle=False) as data:
            assert data["test_predictions"].dtype == np.float32
            assert data["test_predictions"].shape == (1024, 513, 4)
            for key in ("ref_mean", "ref_second_moment", "ref_variance"):
                assert data[key].dtype == np.float64
                assert data[key].shape == (8, 4)
            assert np.all(data["ref_variance"] > 0)
            np.testing.assert_allclose(data["ref_variance"], data["ref_second_moment"] - data["ref_mean"]**2, atol=1e-12, rtol=0)
    print("Verified all 128 sealed replication predictions, untouched fits, matched REF moments, and independently sampled paired TEST labels.")


if __name__ == "__main__":
    main()
