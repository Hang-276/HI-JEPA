"""Stage-1 self-audit: run the audited official probe on the OFFICIAL probing data.

Downloads `fourroom_tp35_d1_probing{,_val}.h5` from the pinned HF dataset repo,
verifies them against the published SHA256SUMS, encodes them with the same
fourroom_hjepa_l2 seed-43 checkpoint, and trains the audited 60-epoch
two-head probe on levels 1 and 2.

This is a diagnostic only. It uses no fourroom_v2 data and no TEST role.

CUDA_VISIBLE_DEVICES=6 .venv/bin/python -B diagnose_official_data.py
"""

import os

_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if _vis not in {"6", "7", "6,7", "7,6"}:
    raise RuntimeError(
        f"FATAL GPU ISOLATION GUARD: CUDA_VISIBLE_DEVICES={_vis!r}; only GPUs 6 and 7 are allowed!"
    )

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from huggingface_hub import hf_hub_download

import run_v2

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
DATA_DIR = ASSETS / "cache/official_fourroom_probing"
OUT = ROOT / "results/preflight_audit"
HF_REPO = "jepa-world-models/h-jepa"
FILES = ("fourroom_tp35_d1_probing.h5", "fourroom_tp35_d1_probing_val.h5")
PROBE_CFG = {
    "hidden": 512,
    "epochs": 60,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "drop_last": True,
    "train_role": "train",
    "val_role": "val",
    "target_columns": {"agent_xy": [0, 2], "distractor_xy": [2, 4]},
}


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in FILES + ("SHA256SUMS",):
        paths[name] = Path(hf_hub_download(HF_REPO, name, repo_type="dataset",
                                           local_dir=DATA_DIR))
    expected = {}
    for line in paths["SHA256SUMS"].read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            expected[parts[1].lstrip("*")] = parts[0]
    verified = {}
    for name in FILES:
        digest = sha256(paths[name])
        if expected.get(name) != digest:
            raise ValueError(f"Official dataset hash mismatch for {name}")
        verified[name] = {"sha256": digest, "bytes": paths[name].stat().st_size,
                          "path": str(paths[name])}
    return verified


def episode_indices(data, episodes, role_offset=0):
    lengths = data["ep_len"][:]
    offsets = data["ep_offset"][:]
    chosen = np.sort(np.asarray(episodes, dtype=np.int64))
    rows = [np.arange(offsets[i], offsets[i] + lengths[i], dtype=np.int64) for i in chosen]
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--level", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--train-episodes", type=int, default=399)
    parser.add_argument("--val-episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    verified = download()
    train_path = Path(verified[FILES[0]]["path"])
    val_path = Path(verified[FILES[1]]["path"])

    with h5py.File(train_path, "r") as f:
        train_lengths = f["ep_len"][:]
        train_offsets = f["ep_offset"][:]
        n_train_eps = len(train_lengths)
        train_order = np.random.default_rng(args.seed).permutation(n_train_eps)
        train_eps = train_order[:args.train_episodes]
        audit_eps = train_order[args.train_episodes:]
        train_indices = episode_indices(f, train_eps)
        audit_indices = episode_indices(f, audit_eps)
        logging.info("official probing train: %s episodes, %s used for train, %s audit frames",
                     n_train_eps, len(train_eps), audit_indices.size)
    with h5py.File(val_path, "r") as f:
        val_lengths = f["ep_len"][:]
        val_eps = np.arange(min(args.val_episodes, len(val_lengths)))
        val_indices = episode_indices(f, val_eps)
        logging.info("official probing val: %s episodes, %s used", len(val_lengths), len(val_eps))

    cfg = {"model_tag": "fourroom_hjepa_l2"}
    model = run_v2.load_backbone(cfg, args.seed, args.device)
    with h5py.File(train_path, "r") as f:
        train_features, _ = run_v2.encode_roles(
            model, f, {"train": train_indices, "audit": audit_indices},
            ("train", "audit"), args.device, 256)
        train_truth = {"train": run_v2.states(f, train_indices),
                       "audit": run_v2.states(f, audit_indices)}
    with h5py.File(val_path, "r") as f:
        val_features, _ = run_v2.encode_roles(
            model, f, {"val": val_indices}, ("val",), args.device, 256)
        val_truth = {"val": run_v2.states(f, val_indices)}
    del model

    normalizer = run_v2.load_normalizer(cfg, args.seed)
    results = {"seed": args.seed, "dataset_revision": "main", "files": verified,
               "train_episodes": len(train_eps), "audit_episodes": len(audit_eps),
               "val_episodes": len(val_eps), "levels": {},
               "note": "Official probing HDF5s, audited official 60-epoch two-head probe."}
    for level in args.level:
        folder = OUT / f"official_probing_seed{args.seed}_level{level}"
        folder.mkdir(parents=True, exist_ok=True)
        artifact = run_v2.fit_probe_160k_official(
            train_features["train"][level - 1], train_truth["train"].reshape(-1, 4),
            val_features["val"][level - 1], val_truth["val"].reshape(-1, 4),
            normalizer, 42, PROBE_CFG, args.device, folder)
        prediction = run_v2.predict_probe_160k_official(
            artifact, train_features["audit"][level - 1], args.device).reshape(
                *audit_indices.shape, 4)
        r2 = run_v2.r2_per_axis(prediction, train_truth["audit"])
        results["levels"][str(level)] = {
            "audit_r2": dict(zip(run_v2.AXES, r2.tolist())),
            "minimum_audit_r2": float(np.min(r2)),
            "distractor_r2_min": float(min(r2[2], r2[3])),
        }
        logging.info("official-data level %s audit %s", level,
                     {k: round(float(v), 4) for k, v in zip(run_v2.AXES, r2)})
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"official_probing_replication_seed{args.seed}.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    logging.info("Wrote %s", destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
