"""Run CAL-selected historical Fine encoders on CPU and audit TRAIN cache parity."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2"
SLOTS = {"ar_rich_fresh1", "fhn_fresh_eps008"}
SMOKE_ATOL = 2e-5  # In the original Fine coordinates standardized by TRAIN G0.
CORE_SUFFIX = "/hcf_8h_extension_20261025/fhn/core.py"
FEATURE_SOURCES = (
    CORE_SUFFIX,
    "/hcf_timescale_20261023/code/src/hcf_jepa/models.py",
    "/hcf_published_baselines_20261024/neural_baselines.py",
    "/hcf_identified_g_20261024/evaluation.py",
    "/hcf_8h_20261024/code/local_validation/eight_hour_structure.py",
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def reference(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def archived_core(path):
    spec = importlib.util.spec_from_file_location("_fine_validation_archived_core", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class FrozenFine:
    model: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray
    provenance: dict
    _feature_function: object

    def encode(self, x):
        """Return raw float32 Fine coordinates, preserving the input shape."""
        raw = np.asarray(x, dtype=np.float32)
        if raw.ndim < 2 or raw.shape[-1] != 4 or not raw.size or not np.isfinite(raw).all():
            raise ValueError("Expected a nonempty finite array of four-dimensional observations")
        with torch.inference_mode():
            value = self._feature_function(self.model, raw, batch=2048)
        require(value.shape == raw.shape and value.dtype == np.float32 and np.isfinite(value).all(),
                "Unexpected archived Fine output")
        return value


def load_fine(slot, rep):
    """Restore and hash-check one of the six permitted CAL-selected states."""
    if slot not in SLOTS or rep not in (0, 1, 2):
        raise ValueError("Expected one of the two frozen slots and replica 0, 1, or 2")
    torch.set_num_threads(1)
    branch = Path("representation/OursFine4") / slot / f"seed{rep}"
    folder = ARCHIVE / "execution/jobs" / branch
    completion = folder / "COMPLETED.json"
    record = read(completion)
    require((record["status"], record["method"], record["slot_id"], record["logical_rep"],
             record["valid_t_start"], record["optimizer_updates"]) ==
            ("complete", "OursFine4", slot, rep, 0, 24000), "Archived Fine identity differs")
    job_path = ARCHIVE / "execution/registration" / branch / "JOB.json"
    require(record["job"] == reference(job_path), "Archived Fine job receipt differs")
    checkpoint = folder / "model/Fine_latest.pt"
    checkpoint_ref = reference(checkpoint)
    require(record["selected_artifact"] == checkpoint_ref, "Archived checkpoint digest differs")
    metadata = read(folder / "model/Fine.json")
    require(metadata["checkpoint_sha256"] == checkpoint_ref["sha256"], "Checkpoint metadata differs")
    g0_path = folder / "G0.json"
    require(record["G0"] == reference(g0_path), "Archived Fine G0 differs")
    g0 = read(g0_path)
    mean = np.array(g0["mean_FP64"], dtype=np.float64)
    std = np.array(g0["std_FP64_ddof0"], dtype=np.float64)
    require(mean.shape == std.shape == (4,) and np.isfinite(mean).all()
            and np.isfinite(std).all() and (std > 0).all(), "Invalid archived G0 normalization")
    mean.setflags(write=False)
    std.setflags(write=False)

    sources = {}
    for suffix in FEATURE_SOURCES:
        matches = [Path(path) for path in record["source_sha256"] if path.endswith(suffix)]
        require(len(matches) == 1, f"Expected one archived source for {suffix}")
        path = matches[0]
        source_ref = reference(path)
        require(source_ref["sha256"] == record["source_sha256"][str(path)], f"Source digest differs: {path}")
        sources[suffix] = source_ref
    core = archived_core(sources[CORE_SUFFIX]["path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    selected = payload["selected_model"]
    require(payload["status"] == "completed" and payload["step"] == record["optimizer_updates"]
            and payload["best"]["step"] == metadata["best"]["step"] == record["selected_step"]
            and payload["identity"] == metadata["identity"], "Selected checkpoint record differs")
    tensor_hashes = {key: array_hash(value.numpy()) for key, value in selected.items()}
    require(tensor_hashes == record["selected_tensor_hashes"] == payload["selected_model_hash"]
            == metadata["selected_model_hash"], "Selected tensor digest differs")
    cfg = payload["identity"]["Fine_config"]
    # Initialization only supplies shapes; strict restoration replaces every tensor.
    with torch.device("cpu"):
        model = core.FineDensity(selected["mean"].numpy(), selected["std"].numpy(), 2, 9, 0, cfg)
    model.load_state_dict(selected, strict=True)
    model.eval().requires_grad_(False)
    require(core.state_hash(model) == tensor_hashes, "Restored Fine tensor digest differs")
    provenance = {
        "completion": reference(completion), "job": record["job"], "checkpoint": checkpoint_ref,
        "selected_step": record["selected_step"], "g0": record["G0"], "sources": list(sources.values()),
        "torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu",
        "intraop_threads": torch.get_num_threads(),
    }
    return FrozenFine(model, mean, std, provenance, core.features)


def smoke(slot, rep, fine=None):
    """Encode one existing TRAIN trajectory and compare to its archived raw Fine features."""
    fine = load_fine(slot, rep) if fine is None else fine
    completion = read(Path(fine.provenance["completion"]["path"]))
    job = read(Path(fine.provenance["job"]["path"]))
    require((completion["slot_id"], completion["logical_rep"]) == (slot, rep), "Smoke parent differs")
    public = ARCHIVE / "data/TRAIN_CAL" / f"{slot}.npz"
    require(job["data_manifest"]["source"]["path"] == str(public), "TRAIN observation source differs")
    with np.load(public, allow_pickle=False) as archive:
        x = archive["train__x"]
    require(array_hash(x) == job["data_identity"]["train"]["x"]["sha256"], "TRAIN observations differ")
    with np.load(Path(completion["features"]["path"]), allow_pickle=False) as archive:
        expected = archive["train"]
    g0 = read(Path(fine.provenance["g0"]["path"]))
    require(array_hash(expected) == g0["TRAIN_raw_Fine_sha256"] and expected.shape == x.shape,
            "TRAIN Fine features differ")
    actual = fine.encode(x[:1])
    error = np.abs(actual.astype(np.float64) - expected[:1])
    by_axis = error.max(axis=(0, 1))
    standardized = by_axis / fine.std
    require((standardized <= SMOKE_ATOL).all(),
            f"TRAIN CPU parity exceeds {SMOKE_ATOL}: {standardized.tolist()}")
    return {
        **fine.provenance,
        "slot": slot, "rep": rep, "train_episode": 0, "shape": list(actual.shape),
        "train_x_sha256": job["data_identity"]["train"]["x"]["sha256"],
        "train_features_sha256": g0["TRAIN_raw_Fine_sha256"],
        "exact": bool(np.array_equal(actual, expected[:1])),
        "max_abs_by_axis": by_axis.tolist(),
        "max_abs_in_g0_units_by_axis": standardized.tolist(),
        "smoke_atol_g0_units": SMOKE_ATOL,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", choices=sorted(SLOTS), default="ar_rich_fresh1")
    parser.add_argument("--rep", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    print(json.dumps(smoke(args.slot, args.rep), indent=2, allow_nan=False))
