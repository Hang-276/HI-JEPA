"""Train the independently seeded A4 four-dimensional model panel."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parents[1]
REPO = REPORTS.parent
ENCODERS = REPORTS / "lejepa_slow_rank_20261112/training"
PROBES = REPORTS / "lejepa_mechanisms_20261113/readout"
FROZEN_LOADER = REPORTS / "lejepa_natural_direction_repair_20261113/loaders.py"
RUNS = HERE / "runs"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


original_path = sys.path[:]
missing = object()
previous_run = sys.modules.get("run", missing)
try:
    encoder_source = load_module("a4_original_encoder_source", ENCODERS / "run.py")
    sys.modules["run"] = encoder_source
    probe_source = load_module("a4_original_probe_source", PROBES / "run.py")
finally:
    if previous_run is missing:
        sys.modules.pop("run", None)
    else:
        sys.modules["run"] = previous_run
    sys.path[:] = original_path


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as result:
        json.dump(value, result, indent=2, allow_nan=False)
        result.write("\n")


def configs():
    return read(HERE / "protocol.json"), read(ENCODERS / "protocol.json"), read(PROBES / "protocol.json")


def dependencies():
    original = encoder_source.source_provenance()
    paths = [
        HERE / "run.py", HERE / "protocol.json", HERE / "PROTOCOL.md",
        ENCODERS / "run.py", ENCODERS / "protocol.json", PROBES / "run.py", PROBES / "protocol.json",
        ENCODERS / "runs/results.json", PROBES / "runs/results.json",
        encoder_source.SOURCE / "LEJEPA_SOURCE_LOCK.json",
        encoder_source.VENDOR / "experiments/lejepa_id/__init__.py",
    ]
    paths.extend(encoder_source.VENDOR / name for name in original["sha256"])
    old = read(ENCODERS / "runs/results.json")
    b = read(PROBES / "runs/results.json")
    for path, expected in (
        (ENCODERS / "run.py", old["code_sha256"]),
        (ENCODERS / "protocol.json", old["protocol_sha256"]),
        (PROBES / "run.py", b["source_sha256"]),
        (PROBES / "protocol.json", b["protocol_sha256"]),
    ):
        if digest(path) != expected:
            raise ValueError(f"frozen configuration or original code changed: {path}")
    return {str(path.relative_to(REPO)): digest(path) for path in paths}


def schedule(config, case, seed, role):
    return np.random.SeedSequence([config["panel_master"], config["cases"].index(case), seed, role])


def data(config, case, seed, role, n):
    return np.random.default_rng(schedule(config, case, seed, role)).normal(size=(n, 4))


def probe_seed(config, case, seed):
    return int(schedule(config, case, seed, 4).generate_state(1)[0])


def panel_names(config):
    return [f"{case}_{seed}" for case in config["cases"] for seed in config["seeds"]]


def train_panel():
    config, encoder_config, probe_config = configs()
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    initial = dependencies()
    RUNS.mkdir(exist_ok=False)
    write(HERE / "DESIGN_LOCK.json", {
        "sources": initial, "config": config, "original_encoder_config": encoder_config, "original_probe_config": probe_config,
    })
    assigned_seeds = [
        probe_seed(config, case, seed) for case in config["cases"] for seed in config["seeds"]
    ]
    if len(set(assigned_seeds)) != len(assigned_seeds):
        raise ValueError("duplicate probe seed")
    start = time.monotonic()
    evidence = {}
    for case in config["cases"]:
        for seed in config["seeds"]:
            member_start = time.monotonic()
            name = f"{case}_{seed}"
            print(f"training {name}", flush=True)
            encoder, history, pair_correlation = encoder_source.train(
                encoder_config, seed, encoder_config["cases"][case], encoder_config["steps"]
            )
            z_cal = data(config, case, seed, 1, probe_config["n_calibration"])
            z_val = data(config, case, seed, 2, probe_config["n_validation"])
            states = data(config, case, seed, 3, config["n_independent_audit"])
            h_cal = encoder_source.embed(encoder, z_cal)
            h_val = encoder_source.embed(encoder, z_val)
            offset, scale = probe_source.standardizer(h_cal)
            probe_master = probe_seed(config, case, seed)
            decoder, selection = probe_source.fit_mlp(
                (h_cal - offset) / scale, z_cal, (h_val - offset) / scale, z_val,
                {**probe_config, "master": probe_master},
            )
            predicted_val = probe_source.predict_mlp((h_val - offset) / scale, decoder)
            predicted_audit = probe_source.predict_mlp((encoder_source.embed(encoder, states) - offset) / scale, decoder)
            val_r2 = probe_source.r2(z_val, predicted_val)
            audit_r2 = probe_source.r2(states, predicted_audit)
            encoder_path, readout_path = RUNS / f"{name}.pt", RUNS / f"{name}_readout.pt"
            with encoder_path.open("xb") as file:
                torch.save(encoder.state_dict(), file)
            with readout_path.open("xb") as file:
                torch.save({"state": decoder.state_dict(), "input_mean": offset.tolist(), "input_std": scale.tolist()}, file)
            evidence[name] = {
                "case": case, "seed": seed, "probe_master": probe_master, "pair_correlation": pair_correlation.tolist(),
                "training_trace": history, "probe_selection": selection,
                "validation_r2": val_r2.tolist(), "minimum_validation_r2": float(val_r2.min()),
                "audit_r2": audit_r2.tolist(), "minimum_audit_r2": float(audit_r2.min()),
                "hashes": {p.name: digest(p) for p in (encoder_path, readout_path)},
                "elapsed_seconds": time.monotonic() - member_start,
            }
            write(RUNS / f"{name}.json", evidence[name])
            print(f"{name} min VAL/AUDIT R² {val_r2.min():.5f}/{audit_r2.min():.5f}", flush=True)
    if dependencies() != initial:
        raise ValueError("one or more frozen data, source or protocol dependencies changed")
    files = {p.name: digest(p) for p in RUNS.iterdir() if p.is_file()}
    write(HERE / "RECEIPT.json", {
        "design_sha256": digest(HERE / "DESIGN_LOCK.json"), "sources": initial,
        "files": files, "members": evidence, "elapsed_seconds": time.monotonic() - start,
    })


def load_readout(name):
    """Load one sealed new panel member. Returns readout(z) and path-to-hash mapping."""
    config, encoder_config, probe_config = configs()
    if name not in panel_names(config):
        raise ValueError(f"unknown new panel member: {name}")
    receipt_path = HERE / "RECEIPT.json"
    if receipt_path.exists():
        hashes = read(receipt_path)["files"]
    else:
        hashes = read(RUNS / f"{name}.json")["hashes"]
    encoder_path, decoder_path = RUNS / f"{name}.pt", RUNS / f"{name}_readout.pt"
    for path in (encoder_path, decoder_path):
        if digest(path) != hashes[path.name]:
            raise ValueError(f"panel weights changed: {path.name}")
    encoder = encoder_source.make_mlp_encoder(4, encoder_config["hidden"], encoder_config["layers"], device="cpu").eval()
    encoder.load_state_dict(torch.load(encoder_path, weights_only=True, map_location="cpu"))
    fitted = torch.load(decoder_path, weights_only=True, map_location="cpu")
    decoder = probe_source.mlp(4, probe_config["mlp_width"]).eval()
    decoder.load_state_dict(fitted["state"])
    mean, scale = np.asarray(fitted["input_mean"]), np.asarray(fitted["input_std"])

    def readout(states):
        h = encoder_source.embed(encoder, states).reshape(-1, 4)
        return probe_source.predict_mlp((h - mean) / scale, decoder).reshape(states.shape)

    return readout, {str(path.relative_to(REPO)): hashes[path.name] for path in (encoder_path, decoder_path)}


def sources_manifest():
    """Check and return the complete panel provenance once all eight models are sealed."""
    receipt_path = HERE / "RECEIPT.json"
    receipt = read(receipt_path)
    manifest = {
        **receipt["sources"],
        str(receipt_path.relative_to(REPO)): digest(receipt_path),
        str((HERE / "DESIGN_LOCK.json").relative_to(REPO)): receipt["design_sha256"],
        **{str((RUNS / name).relative_to(REPO)): shasum for name, shasum in receipt["files"].items()},
    }
    for path, shasum in manifest.items():
        if digest(REPO / path) != shasum:
            raise ValueError(f"sealed panel asset changed: {path}")
    return manifest


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    train_panel()
