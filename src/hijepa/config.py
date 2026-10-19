"""Frozen JSON protocols. Engineering choices are distinct from proposal mandates."""
from __future__ import annotations

import copy
from pathlib import Path

from .io import read_json


DEFAULTS = {
    "schema_version": 1, "name": "hijepa", "stage": "local_integration",
    "system": {"family": "real", "dimension": 4,
               "eigenvalues": [0.92, 0.90, 0.45, 0.30], "angle": 0.25,
               "coupling": 0.02, "observation_strength": 0.15,
               "observation_seed": 51, "noise_std": 0.0},
    "data": {"train_trajectories": 256, "calibration_trajectories": 64,
             "test_trajectories": 64, "length": 40, "seed": 20261019,
             "initial_radius": 0.5},
    "selection": {"local_radius": 0.02, "fd_step": 0.001,
                  "query_noise_bound": 0.0, "slow_modulus_min": 0.895,
                  "cluster_tolerance": 0.015, "certificate_margin": 0.00001},
    "model": {"variant": "full", "z_dim": 16, "width": 64, "depth": 2,
              "activation": "tanh", "target_mode": "shared_stopgrad",
              "weight_frobenius_bound": 8.0, "bias_norm_bound": 8.0},
    "losses": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "kappa": 0.1,
               "k": 4, "sigreg_slices": 64, "sigreg_points": 17,
               "fine_weight": 1.0},
    "training": {"steps": 256, "batch_size": 128, "learning_rate": 0.001,
                 "seed": 0, "device": "cpu", "dtype": "float32",
                 "checkpoint_every": 128, "threads": 2,
                 "deterministic": True, "phase": "source"},
    "evaluation": {"horizon": 16, "ridge": 0.0001,
                   "subspace_threshold": 0.25, "fine_forecast_threshold": 0.25,
                   "probe_max_samples": 2048},
    "transfer": {"enabled": False, "target_coupling": 0.04, "steps": 128,
                 "evaluation_every": 64, "target_train_trajectories": 64,
                 "coarse_threshold": 0.25, "fine_threshold": 0.25},
    "anticollapse": {"minimum_fine_effective_rank": 1.5,
                     "coarse_floor_ratio": 0.9, "minimum_dynamic_fraction": 0.01,
                     "acceptance_seeds": [0, 1]},
}


def merge(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate(config: dict) -> dict:
    cfg = merge(DEFAULTS, config)
    if cfg["schema_version"] != 1:
        raise ValueError("Unsupported config schema")
    if cfg["stage"] not in ("local_integration", "formal"):
        raise ValueError("stage must identify local integration or formal evidence")
    if cfg["model"]["variant"] not in ("full", "block_only", "hierarchical", "direct", "no_fine"):
        raise ValueError("Unknown model variant")
    if cfg["model"]["target_mode"] != "shared_stopgrad":
        raise ValueError("This protocol implements shared-current-parameter stop-gradient targets")
    if cfg["training"]["dtype"] not in ("float32", "float64"):
        raise ValueError("Only explicit float32/float64 is allowed; AMP is disabled")
    for key in ("steps", "batch_size", "threads", "checkpoint_every"):
        if cfg["training"][key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    for key in ("train_trajectories", "calibration_trajectories", "test_trajectories"):
        if cfg["data"][key] < 4:
            raise ValueError(f"data.{key} must provide independent trajectories")
    if cfg["data"]["length"] <= max(2 * cfg["losses"]["k"], cfg["evaluation"]["horizon"]):
        raise ValueError("Trajectories must cover training and evaluation horizons")
    if cfg["transfer"]["enabled"] and cfg["model"]["variant"] not in ("full", "no_fine"):
        raise ValueError("Matched transfer comparison is defined for full and no_fine")
    if cfg["system"]["noise_std"] < 0:
        raise ValueError("Observation noise must be nonnegative")
    return cfg


def load_config(path: Path | str) -> dict:
    path = Path(path).resolve()
    raw = read_json(path)
    if "extends" in raw:
        parent = load_config(path.parent / raw.pop("extends"))
        return validate(merge(parent, raw))
    return validate(raw)


def expand_matrix(path: Path | str) -> list[dict]:
    path = Path(path).resolve()
    matrix = read_json(path)
    base = load_config(path.parent / matrix["base"])
    configs = []
    for job in matrix["jobs"]:
        for seed in matrix.get("seeds", [0, 1, 2]):
            cfg = merge(base, job)
            cfg["name"] = f"{job['name']}_seed{seed}"
            cfg["training"]["seed"] = seed
            if "reuse_from" in cfg:
                cfg["reuse_from"] = f"{cfg['reuse_from']}_seed{seed}"
            configs.append(validate(cfg))
    names = [cfg["name"] for cfg in configs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate matrix run names")
    return configs
