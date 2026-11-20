"""Diagnose the low Level-1 distractor R2 in the Stage-1 preflight.

Read-only diagnosis on dev_preflight, seed 43, fourroom_hjepa_l2 level 1.
It does not change the registered protocol; it only asks whether a missed
official step (feature scaling, capacity, epochs) could explain the gap.

CUDA_VISIBLE_DEVICES=6 .venv/bin/python -B diagnose_preflight.py
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
from torch import nn
from torch.nn import functional as F

import run_v2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "assets/data/dev_preflight"
OUT = ROOT / "results/preflight_audit"
ROLES = ("train", "val", "audit")


def r2(prediction, truth):
    var = truth.var(0)
    return 1 - ((prediction - truth) ** 2).mean(0) / var


def ridge_fit(x, y, lam):
    """Ridge with an explicit intercept: y is centered, x is standardized."""
    y_mean = y.mean(0)
    xtx = x.T @ x
    reg = np.eye(xtx.shape[0], dtype=x.dtype) * lam
    weights = np.linalg.solve(xtx + reg, x.T @ (y - y_mean))
    return weights, y_mean


def train_mlp(x_train, y_train, x_val, y_val, x_audit, y_audit, *, hidden, depth,
              epochs, batch, lr, wd, zscore, seed, device):
    torch.manual_seed(seed)
    if zscore:
        mean, std = x_train.mean(0), np.maximum(x_train.std(0), 1e-6)
    else:
        mean, std = np.zeros(x_train.shape[1], np.float32), np.ones(x_train.shape[1], np.float32)
    xt = torch.as_tensor((x_train - mean) / std, dtype=torch.float32, device=device)
    xv = torch.as_tensor((x_val - mean) / std, dtype=torch.float32, device=device)
    xa = torch.as_tensor((x_audit - mean) / std, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    layers, last = [], x_train.shape[1]
    for _ in range(depth):
        layers += [nn.Linear(last, hidden), nn.LayerNorm(hidden), nn.GELU()]
        last = hidden
    layers += [nn.Linear(last, y_train.shape[1])]
    net = nn.Sequential(*layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    n = len(xt)
    steps = n // batch
    final_train = None
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=gen, device=device)
        total = 0.0
        for b in range(steps):
            ids = perm[b * batch:(b + 1) * batch]
            loss = F.mse_loss(net(xt[ids]), yt[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        final_train = total / steps
        if epoch == 1 or epoch % max(1, epochs // 4) == 0:
            logging.info("  mlp h%d d%d %s ep%d/%d train=%.5f", hidden, depth,
                         "z" if zscore else "raw", epoch, epochs, final_train)
    net.eval()
    with torch.no_grad():
        train_pred = net(xt).cpu().numpy()
        audit_pred = net(xa).cpu().numpy()
    return {
        "train_r2": r2(train_pred, y_train).tolist(),
        "audit_r2": r2(audit_pred, y_audit).tolist(),
        "final_train_loss": final_train,
        "settings": {"hidden": hidden, "depth": depth, "epochs": epochs,
                     "batch": batch, "lr": lr, "wd": wd, "zscore": zscore},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--train-frames", type=int, default=60000)
    parser.add_argument("--ridge-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    _, indices, _ = run_v2.load_splits(DATA / "splits_v2.json")
    model = run_v2.load_backbone({"model_tag": "fourroom_hjepa_l2"}, args.seed, args.device)
    with h5py.File(DATA / "fourroom.h5", "r") as data:
        truth = {role: run_v2.states(data, indices[role]) for role in ROLES}
        sliced, _ = run_v2.encode_roles(model, data, indices, ROLES, args.device, 256)
    del model
    level = args.level - 1
    rng = np.random.default_rng(args.seed)
    n_train_frames = int(np.prod(indices["train"].shape))
    n_val_frames = int(np.prod(indices["val"].shape))
    train_ids = np.sort(rng.choice(n_train_frames, args.train_frames, replace=False))
    val_ids = np.sort(rng.choice(n_val_frames, 20000, replace=False))
    x_train = sliced["train"][level][train_ids]
    y_train = truth["train"].reshape(-1, 4)[train_ids]
    x_val = sliced["val"][level][val_ids]
    y_val = truth["val"].reshape(-1, 4)[val_ids]
    x_audit = sliced["audit"][level]
    y_audit = truth["audit"].reshape(-1, 4)

    results = {"seed": args.seed, "level": args.level, "train_frames": int(len(x_train)),
               "val_frames": int(len(x_val)), "audit_windows": int(len(x_audit)),
               "target_variance": y_train.var(0).tolist(), "variants": {}}

    mean, std = x_train.mean(0), np.maximum(x_train.std(0), 1e-6)
    ztr = (x_train - mean) / std
    # Closed-form ridge, per target column, on standardized 192-D features.
    ridge = {}
    for name, cols in (("agent_xy", [0, 1]), ("distractor_xy", [2, 3])):
        for lam in (1.0, 10.0, 100.0, 1000.0):
            w, y_mean = ridge_fit(ztr, y_train[:, cols], lam)
            pred = ((x_audit - mean) / std) @ w + y_mean
            ridge[f"{name}_lambda{lam:g}"] = r2(pred, y_audit[:, cols]).tolist()
    results["variants"]["ridge_standardized"] = ridge
    if args.ridge_only:
        OUT.mkdir(parents=True, exist_ok=True)
        destination = OUT / f"distractor_ridge_seed{args.seed}_level{args.level}.json"
        destination.write_text(json.dumps(results, indent=2) + "\n")
        logging.info("Wrote %s", destination)
        return 0

    variants = [
        ("official_60_raw", dict(hidden=512, depth=1, epochs=60, batch=128, lr=1e-3, wd=1e-4, zscore=False)),
        ("official_60_zscore", dict(hidden=512, depth=1, epochs=60, batch=128, lr=1e-3, wd=1e-4, zscore=True)),
        ("official_200_zscore", dict(hidden=512, depth=1, epochs=200, batch=128, lr=1e-3, wd=1e-4, zscore=True)),
        ("mlp2_512_60_zscore", dict(hidden=512, depth=2, epochs=60, batch=128, lr=1e-3, wd=1e-4, zscore=True)),
    ]
    for name, kwargs in variants:
        logging.info("variant %s", name)
        results["variants"][name] = train_mlp(
            x_train, y_train, x_val, y_val, x_audit, y_audit,
            seed=42, device=args.device, **kwargs)
        logging.info("  audit %s", np.round(results["variants"][name]["audit_r2"], 4).tolist())

    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"distractor_diagnosis_seed{args.seed}_level{args.level}.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    logging.info("Wrote %s", destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
