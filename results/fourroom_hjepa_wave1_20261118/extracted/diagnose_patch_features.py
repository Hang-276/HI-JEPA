"""Stage-1 follow-up: does the Level-1 ViT carry the distractor outside the CLS token?

Uses the OFFICIAL probing HDF5s (same 399/201/100 episode split as the
replication) and the seed-43 fourroom_hjepa_l2 checkpoint. For each frame it
extracts:
  cls              official ProjectedEncoder output (baseline)
  patch_mean       projector(mean over 256 patch tokens)
  patch_max        projector(max over 256 patch tokens)
  grid4x4          projector applied per patch, then 4x4 spatial mean-pool
  patch_mean_raw   mean over patch tokens BEFORE the projector

Each feature set gets the audited 60-epoch two-head probe (distractor head
only, since the agent axes are already R2~1.0) and a closed-form ridge.

CUDA_VISIBLE_DEVICES=7 .venv/bin/python -B diagnose_patch_features.py --seed 43
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

import run_v2

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "assets/cache/official_fourroom_probing"
OUT = ROOT / "results/preflight_audit"
TRAIN_FILE = DATA_DIR / "fourroom_tp35_d1_probing.h5"
VAL_FILE = DATA_DIR / "fourroom_tp35_d1_probing_val.h5"
PROBE_CFG = {
    "hidden": 512,
    "epochs": 60,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "drop_last": True,
    "target_columns": {"distractor_xy": [2, 4]},
}
FEATURE_ORDER = ("cls", "patch_mean", "patch_max", "grid4x4", "patch_mean_raw")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_official_data():
    sums = {}
    for line in (DATA_DIR / "SHA256SUMS").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0]
    for name in ("fourroom_tp35_d1_probing.h5", "fourroom_tp35_d1_probing_val.h5"):
        if sha256(DATA_DIR / name) != sums[name]:
            raise ValueError(f"Official dataset hash mismatch: {name}")
    return {name: sums[name] for name in
            ("fourroom_tp35_d1_probing.h5", "fourroom_tp35_d1_probing_val.h5")}


def episode_indices(data, episodes):
    lengths = data["ep_len"][:]
    offsets = data["ep_offset"][:]
    chosen = np.sort(np.asarray(episodes, dtype=np.int64))
    rows = [np.arange(offsets[i], offsets[i] + lengths[i], dtype=np.int64) for i in chosen]
    return np.stack(rows)


@torch.inference_mode()
def extract_features(model, data, indices, device, batch_size):
    """One ViT pass per batch, returning every feature set at once."""
    level = model.get_level(1)
    projected = level.encoder
    vit = projected.encoder
    projector = projected.projector
    frames = np.asarray(indices, dtype=np.int64).ravel()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    collected = {name: [] for name in FEATURE_ORDER}
    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]
        pixels = torch.from_numpy(np.ascontiguousarray(
            run_v2.read_pixels(data["pixels"], batch)))
        pixels = pixels.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255
        x = (pixels - mean) / std
        hidden = vit(x, interpolate_pos_encoding=True).last_hidden_state
        if hidden.shape[1] != 257:
            raise ValueError(f"Expected CLS + 256 patch tokens, got {tuple(hidden.shape)}")
        cls = projector(hidden[:, 0])
        patches = hidden[:, 1:]
        # The projector contains BatchNorm1d(2048); flatten tokens for it.
        projected_patches = projector(
            patches.reshape(-1, patches.shape[-1])).reshape(patches.shape)
        if projected_patches.shape[-1] != 192:
            raise ValueError("Unexpected projector output width")
        collected["cls"].append(cls.cpu().numpy())
        collected["patch_mean"].append(projected_patches.mean(1).cpu().numpy())
        collected["patch_max"].append(projected_patches.max(1).values.cpu().numpy())
        collected["patch_mean_raw"].append(patches.mean(1).cpu().numpy())
        grid = projected_patches.reshape(-1, 16, 16, 192)
        grid = grid.reshape(-1, 4, 4, 4, 4, 192).mean(dim=(2, 4)).reshape(-1, 4, 4, 192)
        collected["grid4x4"].append(grid.reshape(-1, 3072).cpu().numpy())
        if start and start % (batch_size * 200) == 0:
            logging.info("extracted %s / %s frames", start, len(frames))
    return {name: np.concatenate(parts).astype(np.float32, copy=False)
            for name, parts in collected.items()}


def ridge_with_intercept(x, y, lam):
    y_mean = y.mean(0)
    xtx = x.T @ x
    weights = np.linalg.solve(xtx + np.eye(xtx.shape[0], dtype=x.dtype) * lam,
                              x.T @ (y - y_mean))
    return weights, y_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--train-episodes", type=int, default=399)
    parser.add_argument("--val-episodes", type=int, default=100)
    parser.add_argument("--ridge-frames", type=int, default=40000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    hashes = verify_official_data()
    with h5py.File(TRAIN_FILE, "r") as f:
        lengths = f["ep_len"][:]
        order = np.random.default_rng(args.seed).permutation(len(lengths))
        train_eps = order[:args.train_episodes]
        audit_eps = order[args.train_episodes:]
        train_indices = episode_indices(f, train_eps)
        audit_indices = episode_indices(f, audit_eps)
        logging.info("train %s episodes (%s frames), audit %s episodes (%s frames)",
                     len(train_eps), train_indices.size, len(audit_eps), audit_indices.size)
    with h5py.File(VAL_FILE, "r") as f:
        val_indices = episode_indices(f, np.arange(args.val_episodes))

    model = run_v2.load_backbone({"model_tag": "fourroom_hjepa_l2"}, args.seed, args.device)
    logging.info("projector: %s", model.get_level(1).encoder.projector)
    with h5py.File(TRAIN_FILE, "r") as f:
        train_features = extract_features(model, f, train_indices, args.device, 128)
        audit_features = extract_features(model, f, audit_indices, args.device, 128)
        train_truth = run_v2.states(f, train_indices)
        audit_truth = run_v2.states(f, audit_indices)
    with h5py.File(VAL_FILE, "r") as f:
        val_features = extract_features(model, f, val_indices, args.device, 128)
        val_truth = run_v2.states(f, val_indices)
    del model

    normalizer = run_v2.load_normalizer({"model_tag": "fourroom_hjepa_l2"}, args.seed)
    y_train = train_truth.reshape(-1, 4)
    y_val = val_truth.reshape(-1, 4)
    y_audit = audit_truth.reshape(-1, 4)
    report = {
        "seed": args.seed,
        "dataset": "official fourroom_tp35_d1_probing{,_val}",
        "dataset_sha256": hashes,
        "episodes": {"train": len(train_eps), "audit": len(audit_eps), "val": args.val_episodes},
        "feature_dims": {name: int(train_features[name].shape[1]) for name in FEATURE_ORDER},
        "variants": {},
    }
    rng = np.random.default_rng(args.seed)
    for name in FEATURE_ORDER:
        logging.info("feature set %s (%s-D)", name, train_features[name].shape[1])
        folder = OUT / f"patch_diag_seed{args.seed}_{name}"
        folder.mkdir(parents=True, exist_ok=True)
        artifact = run_v2.fit_probe_160k_official(
            train_features[name], y_train, val_features[name], y_val,
            normalizer, 42, PROBE_CFG, args.device, folder)
        prediction = run_v2.predict_probe_160k_official(
            artifact, audit_features[name], args.device).reshape(*audit_indices.shape, 2)
        r2 = run_v2.r2_per_axis(prediction, audit_truth[..., 2:4])
        ridge_r2 = None
        try:
            ids = np.sort(rng.choice(len(train_features[name]), args.ridge_frames, replace=False))
            xr = train_features[name][ids]
            mean, std = xr.mean(0), np.maximum(xr.std(0), 1e-6)
            ztr = (xr - mean) / std
            za = (audit_features[name] - mean) / std
            w, y_mean = ridge_with_intercept(ztr, y_train[ids][:, 2:4], 10.0)
            ridge_r2 = run_v2.r2_per_axis(
                za @ w + y_mean, y_audit[:, 2:4]).tolist()
        except np.linalg.LinAlgError as exc:  # pragma: no cover - diagnostic
            ridge_r2 = f"LinAlgError: {exc}"
        report["variants"][name] = {
            "audit_distractor_r2": r2.tolist(),
            "linear_ridge_distractor_r2": ridge_r2,
        }
        logging.info("%s audit distractor R2=%s ridge=%s", name,
                     np.round(r2, 4).tolist(), ridge_r2)
        del artifact

    destination = OUT / f"patch_feature_diagnosis_seed{args.seed}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    logging.info("Wrote %s", destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
