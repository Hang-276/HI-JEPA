"""Deterministic stationary AR(1) panels and a batched, channel-separated renderer."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


ROLE_IDS = {"pretrain": 1, "train": 2, "val": 3, "audit": 4, "ref": 5, "cal": 6, "test": 7}
STREAM_IDS = {"states": 1, "frame_selection": 2, "pixel_noise": 3}


def _rng(cfg, role, stream, *, development):
    if development and role == "test":
        raise ValueError("Development cannot access the TEST role")
    seed_key = "development_seed" if development else "data_seed"
    return np.random.default_rng(
        np.random.SeedSequence([cfg[seed_key], ROLE_IDS[role], STREAM_IDS[stream]])
    )


def _apply_ar_in_place(values, rho):
    innovation = np.sqrt(1 - rho**2)
    for t in range(1, len(values)):
        values[t] = rho * values[t - 1] + innovation * values[t]
    return values


def _draw_innovations(cfg, role, *, development):
    shape = cfg["roles"][role]
    rng = _rng(cfg, role, "states", development=development)
    return rng.standard_normal((shape["frames"], shape["episodes"], 4))


def sample_states(cfg, role, *, development=False):
    """Draw one stationary, unit-marginal-variance trajectory per episode."""
    rho = np.asarray(cfg["rho"], dtype=np.float64)
    if rho.shape != (4,) or not np.all(np.isfinite(rho) & (np.abs(rho) < 1)):
        raise ValueError("Specify four finite stationary AR coefficients")
    states = _apply_ar_in_place(_draw_innovations(cfg, role, development=development), rho)
    return np.ascontiguousarray(states.transpose(1, 0, 2))


def sample_frames(states, cfg, role, *, development=False):
    """Choose a fixed, sorted sample of frames separately for each TRAIN/VAL episode."""
    count = cfg["roles"][role].get("sample_frames")
    if count is None:
        return states
    rng = _rng(cfg, role, "frame_selection", development=development)
    frame_ids = np.stack([
        np.sort(rng.choice(states.shape[1], size=count, replace=False)) for _ in states
    ])
    return states[np.arange(len(states))[:, None], frame_ids]


def image_rng(cfg, role, *, development=False):
    return _rng(cfg, role, "pixel_noise", development=development)


def canvas_coordinates(states, cfg):
    spec = cfg["renderer"]
    left, right = spec["coordinate_bounds"]
    return (left + right) / 2 + ((right - left) / 2) * np.tanh(spec["tanh_scale"] * states)


def render_frames(states, cfg, rng=None):
    """Render coordinates as float32 NCHW frames. A missing RNG disables noise.

    Red and blue each carry one Gaussian plus a known background. In the
    noiseless, real-valued renderer, the adjacent-pixel ratios of each
    background-subtracted Gaussian uniquely determine its center. Overlapping
    sprites do not mask each other. Noise and finite precision limit exact
    recovery in generated observations.
    """
    spec = cfg["renderer"]
    points = np.asarray(states, dtype=np.float32)
    if points.ndim < 2 or points.shape[-1] != 4 or not np.isfinite(points).all():
        raise ValueError("Expected finite states with four physical coordinates")
    points = points.reshape(-1, 2, 2)
    centers = canvas_coordinates(points, cfg)
    size = spec["image_size"]
    axis = np.arange(size, dtype=np.float32)
    delta = axis[None, None, None, :] - centers[..., None]
    gaussian = np.exp(-0.5 * (delta / spec["sprite_sigma"])**2)
    sprites = gaussian[:, :, 1, :, None] * gaussian[:, :, 0, None, :]
    x = axis[None, :] / (size - 1)
    y = axis[:, None] / (size - 1)
    images = np.empty((len(points), 3, size, size), dtype=np.float32)
    images[:, 0] = 0.045 + 0.015 * (x * y) + spec["sprite_amplitude"] * sprites[:, 0]
    images[:, 2] = 0.050 + 0.015 * ((1 - x) * y) + spec["sprite_amplitude"] * sprites[:, 1]
    images[:, 1] = 0.065 + 0.020 * np.sin(np.pi * (x + y)) + 0.035 * sprites.sum(axis=1)
    if rng is not None and spec["pixel_noise_sigma"]:
        images += spec["pixel_noise_sigma"] * rng.standard_normal(images.shape, dtype=np.float32)
    np.clip(images, 0, 1, out=images)
    return images


def iter_role_images(cfg, role, batch_size, *, development=False):
    """Stream images in episode/frame order without returning physical labels."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    states = sample_frames(sample_states(cfg, role, development=development), cfg, role,
                           development=development).reshape(-1, 4)
    rng = image_rng(cfg, role, development=development)
    for start in range(0, len(states), batch_size):
        yield render_frames(states[start:start + batch_size], cfg, rng)


def development_flip_check(cfg):
    """Check the proposed dynamics on independent development audit and REF streams.

    This NumPy-only check computes the physical half of the frozen scorer. It
    leaves the formal data streams untouched and never evaluates a model.
    """
    dev = deepcopy(cfg)
    dev["roles"]["audit"]["episodes"] = cfg["development_flip_check_windows"]
    states = sample_states(dev, "audit", development=True)
    reference = sample_states(dev, "ref", development=True)
    variance = reference.reshape(-1, 4).var(axis=0, ddof=0)
    numerators = np.zeros((len(states), 4), dtype=np.float64)
    for lag in dev["lags"]:
        numerators += np.mean((states[:, lag:] - states[:, :-lag])**2, axis=1)
    scores = (numerators / len(dev["lags"])) / variance
    top2 = np.sort(np.argsort(scores, kind="stable", axis=1)[:, :2], axis=1)
    flips = int(np.any(top2 != [0, 1], axis=1).sum())
    flip_rate = flips / len(states)
    low, high = cfg["development_flip_rate_range"]
    rho = np.asarray(cfg["rho"])
    population_scores = np.mean([2 * (1 - rho**lag) for lag in cfg["lags"]], axis=0)
    return {
        "development_seed": cfg["development_seed"],
        "rho": cfg["rho"],
        "windows": len(states),
        "reference_episodes": len(reference),
        "reference_variance": variance.tolist(),
        "population_scores": population_scores.tolist(),
        "mean_window_scores": scores.mean(axis=0).tolist(),
        "flip_windows": flips,
        "flip_rate": flip_rate,
        "pass": bool(low <= flip_rate <= high and np.max(population_scores[:2]) < np.min(population_scores[2:])),
    }


def _time_major_score(trajectories, reference, lags):
    numerator = np.zeros(trajectories.shape[1], dtype=np.float64)
    for lag in lags:
        numerator += np.mean((trajectories[lag:] - trajectories[:-lag])**2, axis=0)
    return (numerator / len(lags)) / reference.var(ddof=0)


def development_flip_scan(cfg):
    """Scan the permitted rho grid on the same development-only noise streams."""
    dev = deepcopy(cfg)
    windows = cfg["development_flip_check_windows"]
    dev["roles"]["audit"]["episodes"] = windows
    innovations = _draw_innovations(dev, "audit", development=True)
    ref_innovations = _draw_innovations(dev, "ref", development=True)

    def scores(axis, rho):
        states = _apply_ar_in_place(innovations[..., axis].copy(), rho)
        ref = _apply_ar_in_place(ref_innovations[..., axis].copy(), rho)
        return np.stack([
            _time_major_score(states, ref, dev["lags"]),
            _time_major_score(canvas_coordinates(states, dev), canvas_coordinates(ref, dev), dev["lags"]),
        ])

    slow_cutoff = np.maximum(scores(0, cfg["rho"][0]), scores(1, cfg["rho"][1]))
    grid = np.arange(840, 891, 2, dtype=np.float64) / 1000
    outside_2 = np.stack([scores(2, rho) < slow_cutoff for rho in grid])
    outside_3 = np.stack([scores(3, rho) < slow_cutoff for rho in grid])
    defaults = (scores(2, cfg["rho"][2]) < slow_cutoff) | (scores(3, cfg["rho"][3]) < slow_cutoff)
    checks = {}
    target_min, target_max = cfg["development_flip_rate_range"]
    for index, name in enumerate(("gaussian_z_primary", "mapped_canvas_coordinates_diagnostic")):
        # With both slow axes selected, their maximum score precedes both fast axes.
        # Axis-index tie-breaking puts slow axes before an exactly tied fast axis.
        flips = np.count_nonzero(outside_2[:, index, None, :] | outside_3[None, :, index, :], axis=2)
        min_i, min_j = np.unravel_index(flips.argmin(), flips.shape)
        max_i, max_j = np.unravel_index(flips.argmax(), flips.shape)
        default_count = int(defaults[index].sum())
        passing = (flips / windows >= target_min) & (flips / windows <= target_max)
        checks[name] = {
            "default_flips": default_count,
            "default_rate": default_count / windows,
            "minimum_flips": int(flips[min_i, min_j]),
            "minimum_rate": int(flips[min_i, min_j]) / windows,
            "minimum_rho_pair": [float(grid[min_i]), float(grid[min_j])],
            "maximum_flips": int(flips[max_i, max_j]),
            "maximum_rate": int(flips[max_i, max_j]) / windows,
            "maximum_rho_pair": [float(grid[max_i]), float(grid[max_j])],
            "passing_grid_pairs": int(passing.sum()),
            "grid_flip_counts": flips.tolist(),
        }
    primary = development_flip_check(cfg)
    if primary["flip_windows"] != checks["gaussian_z_primary"]["default_flips"]:
        raise RuntimeError("Grid diagnostic disagrees with the primary development check")
    return {
        "status": "DESIGN_GATE_PASSED" if primary["pass"] else "DESIGN_GATE_FAILED",
        "data_scope": "Only development AUDIT and independent REF streams were sampled; no formal TEST access.",
        "development_seed": cfg["development_seed"],
        "rng": "NumPy PCG64 SeedSequence([development_seed, role_id, 1]); AUDIT role 4; REF role 5.",
        "numpy_version": np.__version__,
        "windows": windows,
        "frames": dev["roles"]["audit"]["frames"],
        "reference_episodes": dev["roles"]["ref"]["episodes"],
        "lags": dev["lags"],
        "target_flip_rate_range": cfg["development_flip_rate_range"],
        "default": primary,
        "distractor_rho_grid": grid.tolist(),
        "grid_pairs": int(len(grid)**2),
        "comparison": checks,
        "grid_scope": "Paired Monte Carlo at sampled coefficient pairs, not an analytic bound on the continuum.",
        "formal_blind_gate_passed": bool(primary["pass"]),
        "decision": "BLOCK_FORMAL_PENDING_DESIGN_REVISION" if not primary["pass"] else "CONTINUE_DEVELOPMENT",
    }


def main():
    parser = argparse.ArgumentParser(description="Run the independent NumPy development flip-rate check")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] /
                        "configs/visual_sprites_lejepa/default.json")
    parser.add_argument("--scan", action="store_true", help="Also check the full allowed rho grid")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    result = development_flip_scan(cfg) if args.scan else development_flip_check(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
