"""Validate the spatial-grid readout on the Stage-1 dev_preflight data.

For each model seed this extracts the official CLS feature and the projected
4x4 spatial grid (3072-D), trains the audited two-head probe on both, and
reports all four AUDIT axes. Diagnostic only: no TEST role exists here.

CUDA_VISIBLE_DEVICES=6 .venv/bin/python -B validate_grid_readout.py --seed 42
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -B validate_grid_readout.py --seed 43
"""

import os

_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if _vis not in {"6", "7", "6,7", "7,6"}:
    raise RuntimeError(
        f"FATAL GPU ISOLATION GUARD: CUDA_VISIBLE_DEVICES={_vis!r}; only GPUs 6 and 7 are allowed!"
    )

import argparse
import json
import logging
from pathlib import Path
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

import run_v2
from diagnose_patch_features import extract_features

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "assets/data/dev_preflight"
OUT = ROOT / "results/preflight_audit"
PROBE_CFG = {
    "hidden": 512,
    "epochs": 60,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "drop_last": True,
    "target_columns": {"agent_xy": [0, 2], "distractor_xy": [2, 4]},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 43))
    parser.add_argument("--model-tag", default="fourroom_hjepa_l2",
                        choices=("fourroom_lewm", "fourroom_hjepa_l2", "fourroom_hjepa_l4"))
    parser.add_argument("--features", nargs="*", default=["cls", "grid4x4"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    _, indices, _ = run_v2.load_splits(DATA / "splits_v2.json")
    roles = ("train", "val", "audit")
    model = run_v2.load_backbone({"model_tag": args.model_tag}, args.seed, args.device)
    with h5py.File(DATA / "fourroom.h5", "r") as f:
        features = {feature: {} for feature in args.features}
        for role in roles:
            extracted = extract_features(model, f, indices[role], args.device, 128)
            for feature in args.features:
                features[feature][role] = extracted[feature]
        truth = {role: run_v2.states(f, indices[role]) for role in roles}
    del model

    normalizer = run_v2.load_normalizer({"model_tag": args.model_tag}, args.seed)
    results = {"seed": args.seed, "model_tag": args.model_tag,
               "dataset": "dev_preflight", "features": {}}
    for feature in args.features:
        logging.info("seed %s feature %s (%s-D)", args.seed, feature,
                     features[feature]["train"].shape[1])
        folder = OUT / f"grid_validation_seed{args.seed}_{feature}"
        folder.mkdir(parents=True, exist_ok=True)
        artifact = run_v2.fit_probe_160k_official(
            features[feature]["train"], truth["train"].reshape(-1, 4),
            features[feature]["val"], truth["val"].reshape(-1, 4),
            normalizer, 42, PROBE_CFG, args.device, folder)
        prediction = run_v2.predict_probe_160k_official(
            artifact, features[feature]["audit"], args.device).reshape(
                *indices["audit"].shape, 4)
        r2 = run_v2.r2_per_axis(prediction, truth["audit"])
        results["features"][feature] = {
            "audit_r2": dict(zip(run_v2.AXES, r2.tolist())),
            "minimum_audit_r2": float(np.min(r2)),
            "distractor_r2_min": float(min(r2[2], r2[3])),
        }
        logging.info("seed %s %s audit %s", args.seed, feature,
                     {k: round(float(v), 4) for k, v in zip(run_v2.AXES, r2)})
        del artifact
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"grid_readout_preflight_{args.model_tag}_seed{args.seed}.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    logging.info("Wrote %s", destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
