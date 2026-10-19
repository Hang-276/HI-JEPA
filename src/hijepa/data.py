"""Independent trajectory splits, portable arrays, and content-addressed provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(system, data_config, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    length = int(data_config.get("length", 32))
    radius = float(data_config.get("initial_radius", 1.0))
    if length < 1 or radius <= 0:
        raise ValueError("length >= 1 and initial_radius > 0 are required.")
    seed = int(data_config.get("seed", 1729))
    children = np.random.SeedSequence(seed).spawn(3)
    datasets, splits = {}, {}
    for split_index, split in enumerate(("train", "calibration", "test")):
        count = int(data_config.get(split + "_trajectories", 32))
        if count < 1:
            raise ValueError(f"{split}_trajectories must be positive.")
        rng = np.random.default_rng(children[split_index])
        states = np.empty((count, length+1, system.dimension), dtype=np.float64)
        # Sampling in conjugate mode coordinates supplies declared full coverage.
        initials = rng.uniform(-radius, radius, size=(count, system.dimension))
        states[:, 0] = system.from_modes(initials)
        for t in range(length):
            states[:, t+1] = system.step(states[:, t])
        clean = system.observe(states)
        observations = clean + system.noise_std*rng.normal(size=clean.shape)
        datasets[split] = {"states": states, "observations": observations,
                           "clean_observations": clean}
        path = output / f"{split}.npz"
        np.savez_compressed(path, **datasets[split])
        splits[split] = {"file": path.name, "sha256": _sha256(path), "shape": list(states.shape),
                         "seed_spawn_key": list(children[split_index].spawn_key),
                         "initial_modes_sha256": hashlib.sha256(initials.tobytes()).hexdigest()}
    manifest = {"schema_version": 1, "format": "npz; float64; no Python objects",
                "system": system.metadata(), "data_config": dict(data_config), "seed": seed,
                "splits": splits, "split_unit": "independent trajectories before all fitting",
                "initial_distribution": "u_0 iid Uniform([-initial_radius, initial_radius]^d); s_0=phi^-1(u_0)",
                "truth_policy": "states are for explicit state-rank/compatibility/evaluation; prediction uses observations",
                "portability": "CPU NumPy float64 arrays; reuse exact bytes and manifest on CUDA"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return datasets


def load_data(output_dir, verify=True):
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text())
    datasets = {}
    for split, item in manifest["splits"].items():
        path = output / item["file"]
        if verify and _sha256(path) != item["sha256"]:
            raise ValueError(f"Data hash mismatch: {split}")
        with np.load(path, allow_pickle=False) as archive:
            datasets[split] = {name: archive[name] for name in archive.files}
    return datasets
