#!/usr/bin/env python3
"""E2-pilot: Overcomplete 32x32 Two-Sprite Disentanglement & Slow-Subspace Harmonic Crowding.

Compares matched pixel Autoencoders (d_h in {4, 8}) and symmetric lag-1 LeJEPA encoders
(d_h in {4, 8, 16}, SIGReg weight=0.05) on the 32x32 two-Gaussian-sprite environment under:
  1. Standard CRL / L1 metrics (Linear R^2, Linear MCC, Raw Perm MCC, ZCA-Whitened Perm MCC, MLP R^2).
  2. Unsupervised slow-subspace discovery (SFA lag-1 & Cardoso SOBI on unlabeled CAL):
     slow_subspace_lin_R2 of top-2 slowest axes predicting (z_0, z_1), orthogonal degree-2..4
     harmonic variance share in top-2 axes, and count of top-4 slowest slots with >20% harmonic energy.
  3. Downstream K_W + own_cal g50 verification (coverage, selective error, CP upper bound, pass flag,
     probe residual lag-1 autocorrelation, and fast-axis harmonic leakage) via Linear & MLP probes.

Usage:
  CPU smoke test:
    .venv_visual_cpu/bin/python -B scripts/wave3_week1_20261120/e2_pilot_sprites_overcomplete.py --smoke --device cpu
  Formal GPU run:
    python -B scripts/wave3_week1_20261120/e2_pilot_sprites_overcomplete.py --formal --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "src/slowrepr/metrics.py"

IMG = 32
LAGS = (4, 16, 32)
SOBI_LAGS = (1, 4, 16, 32)
FAST_AXES = (2, 3)
SIGMA = 2.2
AMP = 0.86
PIXEL_NOISE = 0.01
RHO_A = (0.964, 0.961, 0.78, 0.76)
MAX_LAG = 10
ROLES = ("pretrain", "train", "val", "audit", "ref", "cal", "test")

ROLE_SHAPES = {
    "pretrain": (1024, 128),
    "train": (256, 193),
    "val": (64, 193),
    "audit": (128, 193),
    "ref": (256, 193),
    "cal": (1024, 193),
    "test": (2048, 193),
}

# T=49 ensures all evaluation lags=(4, 16, 32) and SOBI_LAGS=(1, 4, 16, 32) remain valid during smoke test.
SMOKE_SHAPES = {
    "pretrain": (16, 48),
    "train": (12, 49),
    "val": (6, 49),
    "audit": (8, 49),
    "ref": (8, 49),
    "cal": (16, 49),
    "test": (16, 49),
}

ARMS = [
    {"name": "autoencoder_d4", "kind": "autoencoder", "dim": 4, "lag": 0, "w_sig": 0.0},
    {"name": "autoencoder_d8", "kind": "autoencoder", "dim": 8, "lag": 0, "w_sig": 0.0},
    {"name": "lejepa_d4", "kind": "lejepa", "dim": 4, "lag": 1, "w_sig": 0.05},
    {"name": "lejepa_d8", "kind": "lejepa", "dim": 8, "lag": 1, "w_sig": 0.05},
    {"name": "lejepa_d16", "kind": "lejepa", "dim": 16, "lag": 1, "w_sig": 0.05},
]


def load_metrics():
    spec = importlib.util.spec_from_file_location("slowrepr_metrics", METRICS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ar1(n_episodes: int, length: int, rho: tuple[float, ...], rng: np.random.Generator) -> np.ndarray:
    z = np.empty((n_episodes, length, 4), dtype=np.float32)
    rho_arr = np.asarray(rho, dtype=np.float32).reshape(1, 4)
    scale = np.sqrt(1.0 - rho_arr ** 2)
    z[:, 0] = rng.standard_normal((n_episodes, 4)).astype(np.float32)
    for t in range(1, length):
        z[:, t] = rho_arr * z[:, t - 1] + scale * rng.standard_normal((n_episodes, 4)).astype(np.float32)
    return z


def generate_splits(data_seed: int, shapes: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
    splits = {}
    for index, role in enumerate(ROLES):
        n_episodes, length = shapes[role]
        rng = np.random.default_rng([int(data_seed), index])
        splits[role] = _ar1(n_episodes, length, RHO_A, rng)
    return splits


_RENDER_GRID_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _render_grids(device: torch.device):
    key = str(device)
    cached = _RENDER_GRID_CACHE.get(key)
    if cached is None:
        axis = torch.arange(IMG, dtype=torch.float32, device=device)
        x = (axis / (IMG - 1)).view(1, IMG)
        y = (axis / (IMG - 1)).view(IMG, 1)
        bg_red = 0.045 + 0.015 * (x * y)
        bg_blue = 0.050 + 0.015 * ((1.0 - x) * y)
        bg_green = 0.065 + 0.020 * torch.sin(float(np.pi) * (x + y))
        cached = (axis, bg_red, bg_green, bg_blue)
        _RENDER_GRID_CACHE[key] = cached
    return cached


def render_states_torch(states_t: torch.Tensor, device: torch.device, noise_gen: torch.Generator | None = None) -> torch.Tensor:
    axis, bg_red, bg_green, bg_blue = _render_grids(device)
    pts = states_t.to(device=device, dtype=torch.float32).reshape(-1, 2, 2)
    centers = 16.0 + 11.0 * torch.tanh(0.45 * pts)
    delta = axis.view(1, 1, 1, IMG) - centers.unsqueeze(-1)
    gauss = torch.exp(-0.5 * (delta / SIGMA).square())
    sprites = gauss[:, :, 1, :, None] * gauss[:, :, 0, None, :]
    red = bg_red + AMP * sprites[:, 0]
    green = bg_green + 0.035 * (sprites[:, 0] + sprites[:, 1])
    blue = bg_blue + AMP * sprites[:, 1]
    images = torch.stack([red, green, blue], dim=1)
    if noise_gen is not None and PIXEL_NOISE > 0:
        images.add_(
            PIXEL_NOISE * torch.randn(images.shape, generator=noise_gen, device=device, dtype=torch.float32)
        )
    return images.clamp_(0.0, 1.0)


class SpriteEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(512, 64), nn.LayerNorm(64), nn.Linear(64, latent_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class SpriteDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, 512), nn.SiLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents)


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer (Balestriero & LeCun 2025) with 2*dt trapezoidal weights."""

    def __init__(self, dim: int, knots: int = 17, n_slices: int = 256, t_max: float = 3.0):
        super().__init__()
        self.dim = int(dim)
        self.n_slices = int(n_slices)
        t = torch.linspace(0.0, t_max, knots)
        dt = float(t_max) / (knots - 1) if knots > 1 else 1.0
        w = torch.full((knots,), 2.0 * dt)
        w[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", torch.exp(-(t ** 2) / 2.0))
        self.register_buffer("weights", w * torch.exp(-(t ** 2) / 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        a = F.normalize(
            torch.randn(flat.size(-1), self.n_slices, device=flat.device, dtype=flat.dtype),
            dim=0,
        )
        xt = (flat @ a).unsqueeze(-1) * self.t
        err = (xt.cos().mean(0) - self.phi).square() + xt.sin().mean(0).square()
        return (err @ self.weights).mean() * flat.size(0)


class Probe(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pretrain_arm(
    pretrain_states: np.ndarray,
    *,
    arm: dict,
    seed: int,
    device: torch.device,
    steps: int,
    batch: int,
    lr: float = 1e-3,
    wd: float = 1e-4,
) -> tuple[SpriteEncoder, dict]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    dim = int(arm["dim"])
    kind = str(arm["kind"])
    encoder = SpriteEncoder(dim).to(device)
    decoder = SpriteDecoder(dim).to(device) if kind == "autoencoder" else None
    sigreg = SIGReg(dim, knots=17, n_slices=256, t_max=3.0).to(device) if kind == "lejepa" else None
    params = list(encoder.parameters()) + (list(decoder.parameters()) if decoder is not None else [])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    n_episodes, seq_len, _ = pretrain_states.shape
    anchors = seq_len - MAX_LAG if kind == "lejepa" else seq_len
    if anchors < 1:
        raise ValueError(f"Sequence length {seq_len} too short for anchors")

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 7919)
    flat_states = torch.as_tensor(np.asarray(pretrain_states, dtype=np.float32).reshape(-1, 4), device=device)
    n_total = flat_states.shape[0]
    pretrain_imgs = None
    states_dev = None
    if n_total <= 200_000:
        pretrain_imgs = torch.empty((n_total, 3, IMG, IMG), dtype=torch.float32, device=device)
        for start in range(0, n_total, 16384):
            end = min(start + 16384, n_total)
            pretrain_imgs[start:end] = render_states_torch(flat_states[start:end], device=device, noise_gen=gen)
        pretrain_imgs = pretrain_imgs.view(n_episodes, seq_len, 3, IMG, IMG)
    else:
        states_dev = flat_states.view(n_episodes, seq_len, 4)
    del flat_states

    ep_all = torch.as_tensor(rng.integers(0, n_episodes, (steps, batch)), dtype=torch.long, device=device)
    anc_all = torch.as_tensor(rng.integers(0, anchors, (steps, batch)), dtype=torch.long, device=device)
    encoder.train()
    if decoder is not None:
        decoder.train()

    final_loss = float("nan")
    for step in range(1, steps + 1):
        ep = ep_all[step - 1]
        anc = anc_all[step - 1]
        if pretrain_imgs is not None:
            current = pretrain_imgs[ep, anc]
        else:
            current = render_states_torch(states_dev[ep, anc], device=device, noise_gen=gen)

        if kind == "autoencoder":
            z = encoder(current)
            recon = decoder(z)
            loss = F.mse_loss(recon, current)
        else:
            lag = int(arm["lag"])
            if pretrain_imgs is not None:
                target = pretrain_imgs[ep, anc + lag]
            else:
                target = render_states_torch(states_dev[ep, anc + lag], device=device, noise_gen=gen)
            latents = encoder(torch.cat([current, target], dim=0)).reshape(2, len(current), -1)
            matching = F.mse_loss(latents[0], latents[1])
            loss = (1.0 - float(arm["w_sig"])) * matching + float(arm["w_sig"]) * sigreg(latents)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == steps:
            loss_val = float(loss.detach())
            if not np.isfinite(loss_val):
                raise ValueError(f"Nonfinite pretrain loss for {arm['name']} seed={seed} at step {step}")
            final_loss = loss_val

    del pretrain_imgs, states_dev, ep_all, anc_all
    encoder.eval()
    return encoder, {"final_loss": final_loss, "steps": steps, "dim": dim, "kind": kind}


@torch.inference_mode()
def encode_states(encoder: SpriteEncoder, states: np.ndarray, device: torch.device, batch: int = 4096, noise_seed: int = 0) -> np.ndarray:
    states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(noise_seed))
    eff_batch = max(int(batch), 4096) if device.type == "cuda" else min(int(batch), 1024)
    chunks = []
    for start in range(0, len(states_t), eff_batch):
        img = render_states_torch(states_t[start:start + eff_batch], device=device, noise_gen=gen)
        chunks.append(encoder(img).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _sample_frames(states: np.ndarray, per_episode: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = min(int(per_episode), int(states.shape[1]))
    indices = np.stack([
        np.sort(rng.choice(states.shape[1], count, replace=False))
        for _ in range(states.shape[0])
    ])
    targets = np.stack([states[i, indices[i]] for i in range(states.shape[0])])
    return targets.reshape(-1, 4)


def fit_ridge_probe(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> dict:
    x_mean = train_x.mean(0)
    x_std = np.maximum(train_x.std(0), 1e-6)
    y_mean = train_y.mean(0)
    y_std = np.maximum(train_y.std(0), 1e-6)
    X_tr = ((train_x - x_mean) / x_std).astype(np.float64)
    Y_tr = ((train_y - y_mean) / y_std).astype(np.float64)
    X_va = ((val_x - x_mean) / x_std).astype(np.float64)
    Y_va = ((val_y - y_mean) / y_std).astype(np.float64)

    d = X_tr.shape[1]
    XtX = X_tr.T @ X_tr
    XtY = X_tr.T @ Y_tr
    best_alpha, best_mse, best_W = None, float("inf"), None
    for alpha in (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0):
        W = np.linalg.solve(XtX + alpha * np.eye(d), XtY)
        mse = float(np.mean((X_va @ W - Y_va) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_W = W
    return {
        "W": best_W, "alpha": best_alpha, "val_mse": best_mse,
        "x_mean": x_mean.astype(np.float64), "x_std": x_std.astype(np.float64),
        "y_mean": y_mean.astype(np.float64), "y_std": y_std.astype(np.float64),
    }


def predict_ridge_probe(art: dict, feats: np.ndarray) -> np.ndarray:
    Xn = (feats.astype(np.float64) - art["x_mean"]) / art["x_std"]
    return (Xn @ art["W"]) * art["y_std"] + art["y_mean"]


def fit_mlp_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    steps: int,
    batch: int,
    lr: float = 1e-3,
    wd: float = 1e-4,
) -> dict:
    set_seed(seed)
    x_mean = train_x.mean(0)
    x_std = np.maximum(train_x.std(0), 1e-6)
    y_mean = train_y.mean(0)
    y_std = np.maximum(train_y.std(0), 1e-6)
    x = torch.as_tensor((train_x - x_mean) / x_std, dtype=torch.float32, device=device)
    y = torch.as_tensor((train_y - y_mean) / y_std, dtype=torch.float32, device=device)
    vx = torch.as_tensor((val_x - x_mean) / x_std, dtype=torch.float32, device=device)
    vy = torch.as_tensor((val_y - y_mean) / y_std, dtype=torch.float32, device=device)
    probe = Probe(x.shape[1], 128, 4).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    best_loss, best_sd = float("inf"), None
    eff_batch = min(int(batch), len(x))
    for step in range(1, steps + 1):
        ids = torch.randint(len(x), (eff_batch,), device=device)
        loss = (probe(x[ids]) - y[ids]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite MLP probe train loss")
            with torch.no_grad():
                val_loss = float((probe(vx) - vy).square().mean().item())
            if not np.isfinite(val_loss):
                raise ValueError("Nonfinite MLP probe val loss")
            if val_loss < best_loss:
                best_loss = val_loss
                best_sd = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
    return {
        "state_dict": best_sd, "input_dim": int(x.shape[1]),
        "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
        "validation_loss": best_loss,
    }


@torch.inference_mode()
def predict_mlp_probe(art: dict, feats: np.ndarray, device: torch.device) -> np.ndarray:
    probe = Probe(art["input_dim"], 128, 4).to(device)
    probe.load_state_dict(art["state_dict"])
    probe.eval()
    mean, std = art["x_mean"], art["x_std"]
    chunks = []
    for start in range(0, len(feats), 16384):
        x = torch.as_tensor((feats[start:start + 16384] - mean) / std, dtype=torch.float32, device=device)
        chunks.append(probe(x).cpu().numpy())
    return np.concatenate(chunks, axis=0) * art["y_std"] + art["y_mean"]


def compute_perm_mcc(feats: np.ndarray, truths: np.ndarray) -> float:
    """Hungarian max-abs correlation across latent coords [d_h, 4] without linear mixing."""
    f = feats.astype(np.float64)
    z = truths.astype(np.float64)
    f_std = np.maximum(f.std(0), 1e-12)
    z_std = np.maximum(z.std(0), 1e-12)
    fn = (f - f.mean(0)) / f_std
    zn = (z - z.mean(0)) / z_std
    corr = np.abs((fn.T @ zn) / len(fn))  # [d_h, 4]
    r_idx, c_idx = linear_sum_assignment(-corr)
    return float(np.mean(corr[r_idx, c_idx]))


def compute_zca_whitened_perm_mcc(train_feats: np.ndarray, audit_feats: np.ndarray, audit_truths: np.ndarray) -> float:
    mu = train_feats.astype(np.float64).mean(0)
    Xc = train_feats.astype(np.float64) - mu
    cov = (Xc.T @ Xc) / len(Xc)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-7)
    W_zca = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    H_w = (audit_feats.astype(np.float64) - mu) @ W_zca
    return compute_perm_mcc(H_w, audit_truths)


def joint_diagonalize(matrices: np.ndarray, tol: float = 1e-9, max_sweeps: int = 200) -> np.ndarray:
    A = np.array(matrices, dtype=np.float64, copy=True)
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    k_cnt, d, _ = A.shape
    w = np.full(k_cnt, 1.0 / k_cnt, dtype=np.float64)
    V = np.eye(d, dtype=np.float64)
    for _ in range(max_sweeps):
        max_sin = 0.0
        for p in range(d - 1):
            for q in range(p + 1, d):
                g0 = A[:, p, p] - A[:, q, q]
                g1 = 2.0 * A[:, p, q]
                g00 = float(np.dot(w, g0 * g0))
                g01 = float(np.dot(w, g0 * g1))
                g11 = float(np.dot(w, g1 * g1))
                G = np.array([[g00, g01], [g01, g11]], dtype=np.float64)
                _, eigvecs = np.linalg.eigh(G)
                u = eigvecs[:, 1]
                if u[0] < 0.0:
                    u = -u
                norm = float(np.hypot(u[0], u[1]))
                if norm < 1e-15:
                    continue
                u0 = min(1.0, max(0.0, float(u[0] / norm)))
                u1 = float(u[1] / norm)
                c = float(np.sqrt(0.5 * (1.0 + u0)))
                s = u1 / (2.0 * c)
                if abs(s) < 1e-15:
                    continue
                max_sin = max(max_sin, abs(s))

                cp = A[:, :, p].copy()
                cq = A[:, :, q].copy()
                A[:, :, p] = c * cp + s * cq
                A[:, :, q] = -s * cp + c * cq

                rp = A[:, p, :].copy()
                rq = A[:, q, :].copy()
                A[:, p, :] = c * rp + s * rq
                A[:, q, :] = -s * rp + c * rq

                vp = V[:, p].copy()
                vq = V[:, q].copy()
                V[:, p] = c * vp + s * vq
                V[:, q] = -s * vp + c * vq
        if max_sin < tol:
            break
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    diag_mean = np.mean(np.diagonal(A, axis1=-2, axis2=-1), axis=0)
    order = np.argsort(diag_mean)[::-1]
    return V[:, order]


def fit_unmixing(h_cal: np.ndarray, lags: tuple[int, ...] = SOBI_LAGS) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit ZCA-PCA whitening + SFA_lag1 + SOBI_4lags on unlabeled [N, T, d_h] CAL representations."""
    d_h = h_cal.shape[-1]
    X = h_cal.reshape(-1, d_h).astype(np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov0 = (Xc.T @ Xc) / len(X)
    evals, evecs = np.linalg.eigh(cov0)
    evals = np.maximum(evals, 1e-7)
    W_whiten = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    Xw = (h_cal.astype(np.float64) - mu[None, None, :]) @ W_whiten

    valid_lags = tuple(h for h in lags if h < h_cal.shape[1])
    if not valid_lags:
        valid_lags = (1,)
    C_lags = []
    for h in valid_lags:
        A = Xw[:, :-h, :].reshape(-1, d_h)
        B = Xw[:, h:, :].reshape(-1, d_h)
        C = (A.T @ B) / len(A)
        C_lags.append(0.5 * (C + C.T))
    C_stack = np.stack(C_lags, axis=0)
    _, vecs_lag1 = np.linalg.eigh(C_stack[0])
    V_sfa = vecs_lag1[:, ::-1]
    V_sobi = joint_diagonalize(C_stack)
    return mu, W_whiten, V_sfa, V_sobi


def _monomials_slow(z_slow: np.ndarray) -> np.ndarray:
    z0, z1 = z_slow[:, 0], z_slow[:, 1]
    return np.stack([
        z0 ** 2, z0 * z1, z1 ** 2,
        z0 ** 3, (z0 ** 2) * z1, z0 * (z1 ** 2), z1 ** 3,
        z0 ** 4, (z0 ** 3) * z1, (z0 ** 2) * (z1 ** 2), z0 * (z1 ** 3), z1 ** 4,
    ], axis=1)


def _r2(y: np.ndarray, design: np.ndarray) -> tuple[float, np.ndarray]:
    D = np.column_stack([np.ones(len(design)), design])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    proj = D @ beta
    var = float(np.var(y))
    return (float(np.var(proj) / var) if var > 1e-15 else 0.0), proj


def _residualize(design: np.ndarray, controls: np.ndarray) -> np.ndarray:
    C = np.column_stack([np.ones(len(controls)), controls])
    beta, *_ = np.linalg.lstsq(C, design, rcond=None)
    return design - C @ beta


def evaluate_unsupervised_subspace(
    u_train: np.ndarray,
    z_train: np.ndarray,
    u_audit: np.ndarray,
    z_audit: np.ndarray,
) -> dict:
    """Evaluate zero-label slow subspace lin R^2 and degree-2..4 slow harmonic crowding."""
    # 1. Slow subspace linear R^2: predict true slow pair (z0, z1) from top-2 slowest unmixed axes U[:,:,:2]
    ut2 = u_train[:, :2].astype(np.float64)
    zt2 = z_train[:, :2].astype(np.float64)
    mu_u = ut2.mean(0)
    sd_u = np.maximum(ut2.std(0), 1e-6)
    X_tr = (ut2 - mu_u) / sd_u
    D_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
    beta, *_ = np.linalg.lstsq(D_tr, zt2, rcond=None)

    ua = u_audit.reshape(-1, u_audit.shape[-1]).astype(np.float64)
    za = z_audit.reshape(-1, 4).astype(np.float64)
    X_au = (ua[:, :2] - mu_u) / sd_u
    pred_slow = np.column_stack([np.ones(len(X_au)), X_au]) @ beta
    mse = np.mean((pred_slow - za[:, :2]) ** 2, axis=0)
    var_slow = np.maximum(np.var(za[:, :2], axis=0), 1e-12)
    slow_subspace_lin_r2 = float(np.mean(1.0 - mse / var_slow))

    # 2. Orthogonal harmonic decomposition on top-4 slowest unmixed axes U_audit[:,:,:min(4, d_h)]
    def std_cols(M: np.ndarray) -> np.ndarray:
        return (M - M.mean(0)) / np.maximum(M.std(0), 1e-8)

    fast_s = std_cols(za[:, list(FAST_AXES)])
    slow_s = std_cols(za[:, [0, 1]])
    harm_s = std_cols(_monomials_slow(za[:, [0, 1]]))
    slow_res = _residualize(slow_s, fast_s)
    harm_res = _residualize(harm_s, np.column_stack([fast_s, slow_s]))

    n_slots = min(4, ua.shape[1])
    lin_slow_fracs, harm_fracs = [], []
    for m in range(n_slots):
        y = ua[:, m] - ua[:, m].mean()
        _, proj_fast = _r2(y, fast_s)
        lin_slow_r2, _ = _r2(y, slow_res)
        harm_r2, _ = _r2(y, harm_res)
        lin_slow_fracs.append(lin_slow_r2)
        harm_fracs.append(harm_r2)

    return {
        "slow_subspace_lin_r2": slow_subspace_lin_r2,
        "top2_lin_slow_frac": float(np.mean(lin_slow_fracs[:2])),
        "top2_quad_harmonic_frac": float(np.mean(harm_fracs[:2])),
        "top4_harmonic_slots": int(sum(h > 0.20 for h in harm_fracs)),
        "slot_harmonic_fracs": [float(h) for h in harm_fracs],
    }


def fast_variance_decomposition(pred_ref: np.ndarray, true_ref: np.ndarray) -> dict:
    """Orthogonal decomposition of fast axes (2, 3) + residual lag-1 AC across all 4 factors."""
    pred = pred_ref.reshape(-1, 4).astype(np.float64)
    truth = true_ref.reshape(-1, 4).astype(np.float64)

    def std_cols(M: np.ndarray) -> np.ndarray:
        return (M - M.mean(0)) / np.maximum(M.std(0), 1e-8)

    fast_s = std_cols(truth[:, list(FAST_AXES)])
    slow_s = std_cols(truth[:, [0, 1]])
    harm_s = std_cols(_monomials_slow(truth[:, [0, 1]]))
    slow_res = _residualize(slow_s, fast_s)
    harm_res = _residualize(harm_s, np.column_stack([fast_s, slow_s]))

    harm_fast_list = []
    for j in FAST_AXES:
        y = pred[:, j] - pred[:, j].mean()
        harm_r2, _ = _r2(y, harm_res)
        harm_fast_list.append(harm_r2)

    # All-4 residual lag-1 autocorrelation across episodes on REF
    res_all = pred_ref.astype(np.float64) - true_ref.astype(np.float64)
    ac_per_axis = []
    for j in range(4):
        x0 = res_all[:, :-1, j].ravel()
        x1 = res_all[:, 1:, j].ravel()
        s0, s1 = float(np.std(x0)), float(np.std(x1))
        ac_per_axis.append(float(np.corrcoef(x0, x1)[0, 1]) if (s0 > 1e-12 and s1 > 1e-12) else 0.0)

    return {
        "mean_harm_fast": float(np.mean(harm_fast_list)),
        "mean_residual_lag1_ac": float(np.mean(ac_per_axis)),
        "per_axis_residual_lag1_ac": ac_per_axis,
    }


def eval_downstream_probe(
    metrics,
    predictions: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    alpha_fwer: float,
) -> dict:
    audit_truth = data["audit"]
    audit_pred = predictions["audit"]
    audit_r2 = 1.0 - np.mean((audit_pred - audit_truth) ** 2, axis=(0, 1)) / np.maximum(audit_truth.reshape(-1, 4).var(0), 1e-12)
    min_r2 = float(np.min(audit_r2))
    mean_r2 = float(np.mean(audit_r2))

    # Linear correlation per axis on AUDIT
    ap = audit_pred.reshape(-1, 4)
    at = audit_truth.reshape(-1, 4)
    mcc = float(np.mean([abs(float(np.corrcoef(ap[:, j], at[:, j])[0, 1])) for j in range(4)]))

    phys_var = data["ref"].reshape(-1, 4).var(0)
    pred_var = np.maximum(predictions["ref"].reshape(-1, 4).var(0), 1e-12)
    cal_sc = metrics.score_windows(predictions["cal"], predictions["cal"], pred_var, pred_var, LAGS)
    tau = float(np.quantile(cal_sc["gap"], 0.50))
    test_sc = metrics.score_windows(predictions["test"], data["test"], phys_var, pred_var, LAGS)
    rk = metrics.risk(test_sc["gap"], test_sc["wrong"], tau, alpha_fwer, minimum_coverage=0.35, max_error=0.01)
    eligible = bool(min_r2 >= 0.85)
    decomp = fast_variance_decomposition(predictions["ref"], data["ref"])
    return {
        "audit_min_r2": min_r2,
        "audit_mean_r2": mean_r2,
        "audit_mcc": mcc,
        "audit_eligible": eligible,
        "all_err": float(test_sc["wrong"].mean()),
        "cov": float(rk["coverage"]),
        "sel_err": float(rk["selective_error"]) if rk["selective_error"] is not None else 1.0,
        "cp_upper": float(rk["error_upper"]),
        "passes": bool(eligible and rk["passes"]),
        **decomp,
    }


def write_results_md(out_dir: Path, summary: dict) -> None:
    lines = [
        f"# E2-pilot ({summary['mode'].upper()}): Overcomplete 32x32 Sprites Disentanglement & Slow-Subspace Crowding",
        "",
        f"- Mode: `{summary['mode']}`, Device: `{summary['device']}`, Seeds: `{summary['seeds']}`, "
        f"Pretrain steps: `{summary['pretrain_steps']}`, Probe steps: `{summary['probe_steps']}`, "
        f"Elapsed: `{summary['elapsed_seconds']:.1f}s`",
        "",
        "## 1. Per-Arm Mean CRL & Linear/MLP Recovery (`L1`)",
        "",
        "| Arm | Final Loss | `Lin min R2` | `Lin MCC` | `Raw Perm MCC` | `ZCA Perm MCC` | `MLP min R2` | `Lin PASS` | `MLP PASS` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name, agg in summary["arm_aggregates"].items():
        lines.append(
            f"| `{arm_name}` | {agg['final_loss']:.4f} | {agg['lin_min_r2']:.4f} | {agg['lin_mcc']:.4f} | "
            f"{agg['raw_perm_mcc']:.4f} | {agg['zca_perm_mcc']:.4f} | {agg['mlp_min_r2']:.4f} | "
            f"{agg['lin_pass_count']}/{agg['n_seeds']} | {agg['mlp_pass_count']}/{agg['n_seeds']} |"
        )

    lines += [
        "",
        "## 2. Unsupervised Zero-Label Slow Subspace (`SFA_lag1` & `SOBI_4lags` on `CAL`) & Harmonic Crowding",
        "",
        "| Arm | Method | `slow_subspace_lin_R2` | `top2_lin_slow_frac` | `top2_quad_harm_frac` | `top4_harm_slots` |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm_name, agg in summary["arm_aggregates"].items():
        for meth in ("sfa", "sobi"):
            lines.append(
                f"| `{arm_name}` | `{meth.upper()}` | {agg[f'{meth}_sub_r2']:.4f} | {agg[f'{meth}_top2_lin']:.4f} | "
                f"{agg[f'{meth}_top2_harm']:.4f} | {agg[f'{meth}_top4_slots']:.2f} |"
            )

    lines += [
        "",
        "## 3. Downstream `K_W + own_cal g50` (`Linear Probe` vs `MLP Probe` on `TEST`)",
        "",
        "| Arm | Probe | `all_err` | `cov` | `sel_err` | `cp_upper` | `res_lag1_AC` | `harm_fast` |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name, agg in summary["arm_aggregates"].items():
        for pkey, plabel in (("lin", "Linear"), ("mlp", "MLP")):
            lines.append(
                f"| `{arm_name}` | `{plabel}` | {agg[f'{pkey}_all_err']*100:.2f}% | {agg[f'{pkey}_cov']*100:.1f}% | "
                f"{agg[f'{pkey}_sel_err']*100:.2f}% | {agg[f'{pkey}_cp_upper']*100:.2f}% | "
                f"{agg[f'{pkey}_res_ac']:.4f} | {agg[f'{pkey}_harm_fast']:.4f} |"
            )

    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument("--smoke", action="store_true", help="Fast CPU smoke run")
    mode_grp.add_argument("--formal", action="store_true", help="Full formal GPU run")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--out", default=None, help="Override output directory")
    args = parser.parse_args()

    is_formal = bool(args.formal)
    mode = "formal" if is_formal else "smoke"
    device_str = args.device or ("cuda" if (is_formal and torch.cuda.is_available()) else "cpu")
    device = torch.device(device_str)
    shapes = ROLE_SHAPES if is_formal else SMOKE_SHAPES
    pretrain_steps = 6000 if is_formal else 120
    probe_steps = 3000 if is_formal else 150
    batch = 256 if is_formal else 128
    seeds = [101, 102, 103] if is_formal else [101]
    default_out = ROOT / ("results/wave3_week1_20261120/e2_pilot_formal" if is_formal else "results/wave3_week1_20261120/e2_pilot_smoke")
    out_dir = Path(args.out) if args.out else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()
    t0 = time.perf_counter()
    alpha_fwer = 0.05 / max(1, len(ARMS) * len(seeds) * 2)
    cell_rows = []

    print(f"=== E2-pilot ({mode}) device={device} seeds={seeds} pretrain={pretrain_steps} probe={probe_steps} ===", flush=True)
    for seed in seeds:
        data = generate_splits(data_seed=seed * 10 + 7, shapes=shapes)
        train_states = _sample_frames(data["train"], 32, seed + 11)
        val_states = _sample_frames(data["val"], 32, seed + 17)

        for arm in ARMS:
            t_arm = time.perf_counter()
            encoder, train_meta = pretrain_arm(
                data["pretrain"], arm=arm, seed=seed, device=device, steps=pretrain_steps, batch=batch
            )
            train_x = encode_states(encoder, train_states, device, noise_seed=seed + 101)
            val_x = encode_states(encoder, val_states, device, noise_seed=seed + 102)
            latents = {
                role: encode_states(encoder, data[role].reshape(-1, 4), device, noise_seed=seed + 200 + idx).reshape(
                    data[role].shape[0], data[role].shape[1], arm["dim"]
                )
                for idx, role in enumerate(("audit", "ref", "cal", "test"))
            }

            raw_perm_mcc = compute_perm_mcc(latents["audit"].reshape(-1, arm["dim"]), data["audit"].reshape(-1, 4))
            zca_perm_mcc = compute_zca_whitened_perm_mcc(train_x, latents["audit"].reshape(-1, arm["dim"]), data["audit"].reshape(-1, 4))

            # Linear Ridge probe + MLP probe
            lin_art = fit_ridge_probe(train_x, train_states, val_x, val_states)
            lin_preds = {
                role: predict_ridge_probe(lin_art, latents[role].reshape(-1, arm["dim"])).reshape(
                    data[role].shape[0], data[role].shape[1], 4
                )
                for role in ("audit", "ref", "cal", "test")
            }
            lin_eval = eval_downstream_probe(metrics, lin_preds, data, alpha_fwer)

            mlp_art = fit_mlp_probe(
                train_x, train_states, val_x, val_states,
                seed=seed + 300, device=device, steps=probe_steps, batch=batch,
            )
            mlp_preds = {
                role: predict_mlp_probe(mlp_art, latents[role].reshape(-1, arm["dim"]), device).reshape(
                    data[role].shape[0], data[role].shape[1], 4
                )
                for role in ("audit", "ref", "cal", "test")
            }
            mlp_eval = eval_downstream_probe(metrics, mlp_preds, data, alpha_fwer)

            # Unsupervised slow subspace (SFA_lag1 & SOBI_4lags fit on CAL with 0 labels)
            mu, W_w, V_sfa, V_sobi = fit_unmixing(latents["cal"], SOBI_LAGS)
            unmix = {}
            for mname, V in (("sfa", V_sfa), ("sobi", V_sobi)):
                A = W_w @ V
                u_tr = (train_x.astype(np.float64) - mu[None, :]) @ A
                u_au = (latents["audit"].astype(np.float64) - mu[None, None, :]) @ A
                unmix[mname] = evaluate_unsupervised_subspace(u_tr, train_states, u_au, data["audit"])

            row = {
                "arm": arm["name"], "kind": arm["kind"], "dim": arm["dim"], "seed": seed,
                "final_loss": train_meta["final_loss"],
                "raw_perm_mcc": raw_perm_mcc, "zca_perm_mcc": zca_perm_mcc,
                "ridge_alpha": lin_art["alpha"],
                "linear_probe": lin_eval, "mlp_probe": mlp_eval,
                "unsupervised": unmix,
                "arm_elapsed_seconds": time.perf_counter() - t_arm,
            }
            cell_rows.append(row)
            print(
                f"  seed={seed} {arm['name']:15s} loss={train_meta['final_loss']:.4f} | "
                f"LinR2={lin_eval['audit_min_r2']:.3f} MLPR2={mlp_eval['audit_min_r2']:.3f} "
                f"PermMCC(raw/zca)={raw_perm_mcc:.3f}/{zca_perm_mcc:.3f} | "
                f"SFA(subR2={unmix['sfa']['slow_subspace_lin_r2']:.3f}, harm={unmix['sfa']['top2_quad_harmonic_frac']:.3f}, slots={unmix['sfa']['top4_harmonic_slots']}) "
                f"({row['arm_elapsed_seconds']:.1f}s)",
                flush=True,
            )

    arm_aggs = {}
    for arm in ARMS:
        sub = [r for r in cell_rows if r["arm"] == arm["name"]]
        arm_aggs[arm["name"]] = {
            "n_seeds": len(sub),
            "final_loss": float(np.mean([r["final_loss"] for r in sub])),
            "raw_perm_mcc": float(np.mean([r["raw_perm_mcc"] for r in sub])),
            "zca_perm_mcc": float(np.mean([r["zca_perm_mcc"] for r in sub])),
            "lin_min_r2": float(np.mean([r["linear_probe"]["audit_min_r2"] for r in sub])),
            "lin_mcc": float(np.mean([r["linear_probe"]["audit_mcc"] for r in sub])),
            "mlp_min_r2": float(np.mean([r["mlp_probe"]["audit_min_r2"] for r in sub])),
            "lin_pass_count": int(sum(r["linear_probe"]["passes"] for r in sub)),
            "mlp_pass_count": int(sum(r["mlp_probe"]["passes"] for r in sub)),
            "lin_all_err": float(np.mean([r["linear_probe"]["all_err"] for r in sub])),
            "lin_cov": float(np.mean([r["linear_probe"]["cov"] for r in sub])),
            "lin_sel_err": float(np.mean([r["linear_probe"]["sel_err"] for r in sub])),
            "lin_cp_upper": float(np.mean([r["linear_probe"]["cp_upper"] for r in sub])),
            "lin_res_ac": float(np.mean([r["linear_probe"]["mean_residual_lag1_ac"] for r in sub])),
            "lin_harm_fast": float(np.mean([r["linear_probe"]["mean_harm_fast"] for r in sub])),
            "mlp_all_err": float(np.mean([r["mlp_probe"]["all_err"] for r in sub])),
            "mlp_cov": float(np.mean([r["mlp_probe"]["cov"] for r in sub])),
            "mlp_sel_err": float(np.mean([r["mlp_probe"]["sel_err"] for r in sub])),
            "mlp_cp_upper": float(np.mean([r["mlp_probe"]["cp_upper"] for r in sub])),
            "mlp_res_ac": float(np.mean([r["mlp_probe"]["mean_residual_lag1_ac"] for r in sub])),
            "mlp_harm_fast": float(np.mean([r["mlp_probe"]["mean_harm_fast"] for r in sub])),
            "sfa_sub_r2": float(np.mean([r["unsupervised"]["sfa"]["slow_subspace_lin_r2"] for r in sub])),
            "sfa_top2_lin": float(np.mean([r["unsupervised"]["sfa"]["top2_lin_slow_frac"] for r in sub])),
            "sfa_top2_harm": float(np.mean([r["unsupervised"]["sfa"]["top2_quad_harmonic_frac"] for r in sub])),
            "sfa_top4_slots": float(np.mean([r["unsupervised"]["sfa"]["top4_harmonic_slots"] for r in sub])),
            "sobi_sub_r2": float(np.mean([r["unsupervised"]["sobi"]["slow_subspace_lin_r2"] for r in sub])),
            "sobi_top2_lin": float(np.mean([r["unsupervised"]["sobi"]["top2_lin_slow_frac"] for r in sub])),
            "sobi_top2_harm": float(np.mean([r["unsupervised"]["sobi"]["top2_quad_harmonic_frac"] for r in sub])),
            "sobi_top4_slots": float(np.mean([r["unsupervised"]["sobi"]["top4_harmonic_slots"] for r in sub])),
        }

    summary = {
        "experiment": "wave3_week1_e2_pilot_sprites_overcomplete",
        "mode": mode,
        "device": str(device),
        "seeds": seeds,
        "pretrain_steps": pretrain_steps,
        "probe_steps": probe_steps,
        "batch": batch,
        "metrics_sha256": hashlib.sha256(METRICS_PATH.read_bytes()).hexdigest(),
        "arm_aggregates": arm_aggs,
        "cells": cell_rows,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_results_md(out_dir, summary)
    print(f"Wrote {out_dir / 'SUMMARY.json'} and {out_dir / 'RESULTS.md'} in {summary['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
