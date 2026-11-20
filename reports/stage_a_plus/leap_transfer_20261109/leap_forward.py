"""Restore frozen LEAP V2 AR2 encoders and check public TRAIN parity on CPU."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2"
SLOT = "ar_rich_fresh2"
VALID_START = 11
SMOKE_ATOL = 2e-5
BRIDGE = ROOT / "work/hcf_stable_id_20261027/v1"
CORE = ROOT / "work/hcf_method_freeze_20261025/flowdirect_branch_v1/leap_controlled.py"
MAP_LOCK = ARCHIVE / "evaluation_v2/CAL_MAPS_LOCK.json"
MAP_LOCK_SHA256 = "9c14eb2db2a8d0d9acd975d4bdc6bcf1bdeece77cb3ac413b1c63aa2c57dd0eb"
TRAIN = ARCHIVE / "data/TRAIN_CAL" / f"{SLOT}.npz"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def reference(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def array_identity(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    digest.update(value.tobytes())
    return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": digest.hexdigest()}


def branch(seed):
    if seed not in (0, 1, 2):
        raise ValueError("Expected frozen LEAP V2 logical replica 0, 1, or 2")
    return Path("representation/LEAP_V2") / SLOT / f"seed{seed}"


@lru_cache(maxsize=1)
def core_module():
    spec = importlib.util.spec_from_file_location("_leap_transfer_controlled", CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_sources(receipt):
    vendor = CORE.parent / "vendor/leap_official/leap/modules/components"
    paths = (CORE, BRIDGE / "leap_adapter.py", BRIDGE / "leap_standardized_adapter.py",
             *(vendor / name for name in ("mlp.py", "splines.py", "transforms.py", "tc.py")))
    sources = [reference(path) for path in paths]
    for source in sources:
        require(source["sha256"] == receipt["source_sha256"].get(source["path"]),
                f"Historical LEAP source digest differs: {source['path']}")
    return sources


@lru_cache(maxsize=1)
def _public_train():
    job = read(ARCHIVE / "execution/registration" / branch(0) / "JOB.json")
    require(job["data_manifest"]["source"] == reference(TRAIN), "Public TRAIN archive differs")
    with np.load(TRAIN, allow_pickle=False) as archive:
        result = {key: archive[f"train__{key}"] for key in ("x", "actions", "regimes")}
    for key, value in result.items():
        require(array_identity(value) == job["data_identity"]["train"][key],
                f"Public TRAIN {key} digest differs")
        value.setflags(write=False)
    return result


def public_train():
    """Return verified public TRAIN x, actions and regimes as read-only arrays."""
    return _public_train().copy()


@dataclass(frozen=True)
class FrozenLEAP:
    model: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray
    permutation: tuple
    axis_sign: tuple
    batch_size: int
    provenance: dict
    valid_start: int = VALID_START

    def encode(self, x):
        """Return raw learned-axis coordinates [N,T-11,4] from raw FP32 observations."""
        x = np.asarray(x)
        if (x.dtype != np.float32 or x.ndim != 3 or x.shape[1] <= self.valid_start
                or x.shape[2] != 4 or not x.size or not np.isfinite(x).all()):
            raise ValueError("Expected nonempty finite FP32 [N,T>=12,4] observations")
        normalized = np.ascontiguousarray((x - self.mean) / self.std)
        n, t, _ = x.shape
        length = t - self.valid_start
        out = np.empty((n * length, 4), dtype=np.float32)
        offsets = np.arange(self.valid_start + 1)
        with torch.inference_mode():
            for start in range(0, len(out), self.batch_size):
                origin = np.arange(start, min(start + self.batch_size, len(out)))
                rows, left = np.divmod(origin, length)
                prefix = torch.from_numpy(normalized[rows[:, None], left[:, None] + offsets])
                out[origin] = self.model.encode_deterministic(prefix).numpy()
        require(np.isfinite(out).all(), "Nonfinite restored LEAP encoder output")
        return out.reshape(n, length, 4)


def load(seed):
    """Load one CAL-selected model and the separately locked CAL axis permutation."""
    torch.set_num_threads(1)
    suffix = branch(seed)
    folder = ARCHIVE / "execution/jobs" / suffix
    completion = reference(folder / "COMPLETED.json")
    receipt = read(Path(completion["path"]))
    profile = {"operation": "representation", "method": "LEAP_V2",
               "slot_id": SLOT, "logical_rep": seed}
    require(all(receipt[key] == profile[key] for key in ("method", "slot_id", "logical_rep"))
            and receipt["status"] == "complete" and receipt["valid_t_start"] == VALID_START
            and receipt["physical_or_FINAL_read"] is False, "Historical LEAP completion identity differs")
    job = reference(ARCHIVE / "execution/registration" / suffix / "JOB.json")
    checkpoint = reference(folder / "best.pt")
    features = reference(folder / "FEATURES_RECEIPT.json")
    require((receipt["job"], receipt["selected_artifact"], receipt["features"]) ==
            (job, checkpoint, features), "Historical LEAP checkpoint or feature receipt differs")
    job_data = read(Path(job["path"]))
    feature_data = read(Path(features["path"]))
    require(all(job_data[key] == value for key, value in profile.items())
            and job_data["data_manifest"]["source"] == reference(TRAIN),
            "LEAP registered job differs")
    require(feature_data["job"] == job and feature_data["selected_artifact"] == checkpoint
            and feature_data["valid_t_start"] == VALID_START and feature_data["no_physical_FINAL_read"],
            "LEAP TRAIN feature lineage differs")
    map_ref = reference(ARCHIVE / "evaluation_v2/CAL_maps" / suffix / "MAPS.json")
    map_lock = reference(MAP_LOCK)
    require(map_lock["sha256"] == MAP_LOCK_SHA256 and
            {"profile": profile, "maps": map_ref} in read(MAP_LOCK)["records"],
            "Historical CAL mapping lock differs")
    map_data = read(Path(map_ref["path"]))
    axis = map_data["maps"]["axis_affine"]
    permutation = tuple(axis["matched_indices"])
    signs = tuple(int(value) for value in np.sign(axis["slopes"]))
    require(map_data["profile"] == profile and map_data["parent_completion"] == completion
            and map_data["features"] == features and map_data["valid_t_start"] == VALID_START
            and not map_data["FINAL_values_used_for_fit"] and axis["fit_split"] == "CAL only"
            and axis["assignment_unique"] and not axis["heldout_used_for_fit"]
            and sorted(permutation) == list(range(4)) and len(signs) == 4 and all(signs),
            "Invalid historical CAL-only LEAP axis permutation")
    sources = verified_sources(receipt)
    require(all(feature_data["source_sha256"].get(item["path"]) == item["sha256"] for item in sources),
            "LEAP encoding sources differ from TRAIN feature receipt")
    numpy_types = [np.ndarray, np.dtype, np._core.multiarray._reconstruct,
                   type(np.dtype(np.float32)), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(numpy_types):
        payload = torch.load(Path(checkpoint["path"]), map_location="cpu", weights_only=True)
    outer = payload["adapter"]
    inner, norm = outer["inner"], outer["normalization"]
    cfg = inner["config"]
    require(payload["job"] == job and payload["step"] == receipt["selected_step"] == feature_data["selected_step"]
            and outer["format"] == norm["format"] == "LEAP_TRAIN_standardized_composition_v2"
            and inner["format"] == cfg["format"] == "stable_id_controlled_leap_adapter_v1"
            and cfg == job_data["config"] and cfg["prefix_frames"] == VALID_START + 1
            and norm["TRAIN_identity"] == job_data["data_identity"]["train"]["x"],
            "Historical LEAP state or TRAIN normalization identity differs")
    train = public_train()["x"].astype(np.float64).reshape(-1, 4)
    mean, std = norm["mean_FP32"], norm["std_FP32"]
    require(np.array_equal(train.mean(0).astype(np.float32), mean)
            and np.array_equal(train.std(0).astype(np.float32), std) and (std > 0).all(),
            "Historical LEAP TRAIN normalization differs")
    mean.setflags(write=False)
    std.setflags(write=False)
    with torch.device("cpu"):
        model = core_module().LEAPNSControlled(inner["seed"], cfg)
    model.load_state_dict(inner["model"], strict=True)
    model.eval().requires_grad_(False)
    provenance = {
        "completion": completion, "job": job, "checkpoint": checkpoint, "features": features,
        "cal_map": map_ref, "cal_map_lock": map_lock, "public_train": job_data["data_manifest"]["source"],
        "sources": sources, "selected_step": payload["step"], "logical_rep": seed,
        "torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu",
        "intraop_threads": torch.get_num_threads(),
    }
    return FrozenLEAP(model, mean, std, permutation, signs, cfg["encoding_batch_size"], provenance)


def smoke(seed, encoder=None):
    """Verify two complete public TRAIN batches using historical batch geometry."""
    frozen = load(seed) if encoder is None else encoder
    require(frozen.provenance["logical_rep"] == seed, "Smoke replica differs")
    feature_data = read(Path(frozen.provenance["features"]["path"]))
    path = ARCHIVE / "execution/jobs" / branch(seed) / "FEATURES.npz"
    require(feature_data["artifact"] == reference(path), "Historical TRAIN feature archive differs")
    with np.load(path, allow_pickle=False) as archive:
        expected = archive["train_z"]
    x = public_train()["x"]
    require(array_identity(expected) == feature_data["arrays"]["train_z"]
            and expected.shape == (len(x), x.shape[1] - VALID_START, 4),
            "Historical TRAIN feature identity differs")
    # Complete the second 256-origin batch so both archived and fresh batches match.
    episodes = (2 * frozen.batch_size + expected.shape[1] - 1) // expected.shape[1]
    checked_origins = 2 * frozen.batch_size
    actual = frozen.encode(x[:episodes]).reshape(-1, 4)[:checked_origins]
    truth = expected[:episodes].reshape(-1, 4)[:checked_origins]
    absolute = np.abs(actual.astype(np.float64) - truth).max(0)
    feature_std = expected.astype(np.float64).reshape(-1, 4).std(0)
    require((feature_std > 0).all(), "Constant archived LEAP TRAIN feature")
    standardized = absolute / feature_std
    require((standardized <= SMOKE_ATOL).all(),
            f"TRAIN CPU parity exceeds {SMOKE_ATOL}: {standardized.tolist()}")
    return {"provenance": frozen.provenance, "valid_t_start": VALID_START,
            "permutation": list(frozen.permutation), "axis_sign": list(frozen.axis_sign),
            "checked_train_origins": checked_origins, "train_features_sha256": array_identity(expected)["sha256"],
            "max_abs_by_axis": absolute.tolist(), "max_abs_in_train_std_by_axis": standardized.tolist(),
            "smoke_atol_train_std": SMOKE_ATOL, "exact": bool(np.array_equal(actual, truth))}


if __name__ == "__main__":
    print(json.dumps([smoke(seed) for seed in range(3)], indent=2, allow_nan=False))
