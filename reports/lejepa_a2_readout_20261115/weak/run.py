"""Fit and seal the 64-state readout variant on all frozen A4 encoders."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parents[1]
ROOT = REPORTS.parent
PANEL = REPORTS / "lejepa_a4_20261115/panel"
RUNS = HERE / "runs"
spec = importlib.util.spec_from_file_location("a2_archived_a4_panel", PANEL / "run.py")
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)
digest, read, write = panel.digest, panel.read, panel.write


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dependencies():
    result = panel.sources_manifest()
    result.update({
        str(path.relative_to(ROOT)): digest(path)
        for path in (Path(__file__), HERE / "protocol.json", HERE / "PROTOCOL.md")
    })
    return result


def frozen_encoder(name, original):
    source = panel.encoder_source
    path = PANEL / "runs" / f"{name}.pt"
    model = source.make_mlp_encoder(4, original["hidden"], original["layers"], device="cpu").eval()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    return model


def fit_weak():
    config = read(HERE / "protocol.json")
    a4_config, encoder_config, probe_config = panel.configs()
    if (
        config["n_calibration"] != 64 or config["n_validation"] != probe_config["n_validation"]
        or config["audit_role"] != 721 or config["n_audit"] != 16384
    ):
        raise ValueError("unexpected nested-label or audit design")
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    sources = dependencies()
    old = read(PANEL / "RECEIPT.json")
    RUNS.mkdir(exist_ok=False)
    design_file = HERE / "DESIGN_LOCK.json"
    write(design_file, {
        "sources": sources, "config": config, "a4_config": a4_config, "probe_config": probe_config,
    })
    records = {}
    start = time.monotonic()
    for name in panel.panel_names(a4_config):
        member = old["members"][name]
        case, seed = member["case"], member["seed"]
        encoder = frozen_encoder(name, encoder_config)
        original_cal = panel.data(a4_config, case, seed, 1, probe_config["n_calibration"])
        cal = original_cal[:config["n_calibration"]]
        val = panel.data(a4_config, case, seed, 2, config["n_validation"])
        audit = panel.data(a4_config, case, seed, config["audit_role"], config["n_audit"])
        h_cal, h_val, h_audit = [panel.encoder_source.embed(encoder, values) for values in (cal, val, audit)]
        mean, scale = panel.probe_source.standardizer(h_cal)
        init = panel.probe_seed(a4_config, case, seed)
        if init != member["probe_master"]:
            raise ValueError("original probe initialization lineage changed")
        probe, selection = panel.probe_source.fit_mlp(
            (h_cal - mean) / scale, cal, (h_val - mean) / scale, val, {**probe_config, "master": init},
        )
        weak_val = panel.probe_source.predict_mlp((h_val - mean) / scale, probe)
        weak_audit = panel.probe_source.predict_mlp((h_audit - mean) / scale, probe)
        strong, strong_sources = panel.load_readout(name)
        strong_val, strong_audit = strong(val), strong(audit)
        val_strong_r2 = panel.probe_source.r2(val, strong_val)
        np.testing.assert_allclose(val_strong_r2, member["validation_r2"], atol=1e-12, rtol=0)
        result = {
            "probe_master": init,
            "selection": selection,
            "validation": {
                "weak_r2": panel.probe_source.r2(val, weak_val).tolist(),
                "strong_r2": val_strong_r2.tolist(),
            },
            "audit": {
                "weak_r2": panel.probe_source.r2(audit, weak_audit).tolist(),
                "strong_r2": panel.probe_source.r2(audit, strong_audit).tolist(),
                "weak_mse": np.mean((weak_audit - audit)**2, axis=0).tolist(),
                "strong_mse": np.mean((strong_audit - audit)**2, axis=0).tolist(),
            },
            "data_sha256": {
                "original_CAL8192": array_sha(original_cal), "nested_CAL64": array_sha(cal),
                "VAL4096": array_sha(val), "AUDIT16384_role721": array_sha(audit),
            },
            "strong_weight_sources": strong_sources,
        }
        path = RUNS / f"{name}_readout.pt"
        with path.open("xb") as output:
            torch.save({"state": probe.state_dict(), "input_mean": mean.tolist(), "input_std": scale.tolist()}, output)
        result["weak_checkpoint_sha256"] = digest(path)
        write(RUNS / f"{name}.json", result)
        records[name] = result
        print(f"{name} min audit R² weak/strong {min(result['audit']['weak_r2']):.6f}/{min(result['audit']['strong_r2']):.6f}", flush=True)
    if dependencies() != sources:
        raise ValueError("frozen panel or protocol changed during fitting")
    write(HERE / "RECEIPT.json", {
        "design_sha256": digest(design_file), "sources": sources, "config": config,
        "files": {path.name: digest(path) for path in RUNS.iterdir() if path.is_file()},
        "members": records, "elapsed_seconds": time.monotonic() - start,
    })


def sources_manifest():
    receipt_path = HERE / "RECEIPT.json"
    receipt = read(receipt_path)
    if receipt["sources"] != dependencies():
        raise ValueError("weak-fit sources changed after the fit lock")
    manifest = {
        **receipt["sources"],
        str(receipt_path.relative_to(ROOT)): digest(receipt_path),
        str((HERE / "DESIGN_LOCK.json").relative_to(ROOT)): receipt["design_sha256"],
        **{str((RUNS / name).relative_to(ROOT)): sha for name, sha in receipt["files"].items()},
    }
    for name, sha in manifest.items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"sealed fit asset changed: {name}")
    return manifest


def load_readout(name):
    """Return a frozen weak named readout and its repo-relative weight hashes."""
    panel_config, encoder_config, probe_config = panel.configs()
    if name not in panel.panel_names(panel_config):
        raise ValueError(f"unknown fixed panel member: {name}")
    weak = RUNS / f"{name}_readout.pt"
    strong_encoder = PANEL / "runs" / f"{name}.pt"
    hashes = read(HERE / "RECEIPT.json")["files"]
    panel_hashes = read(PANEL / "RECEIPT.json")["files"]
    if digest(weak) != hashes[weak.name] or digest(strong_encoder) != panel_hashes[strong_encoder.name]:
        raise ValueError("sealed weak decoder or frozen original encoder changed")
    encoder = frozen_encoder(name, encoder_config)
    saved = torch.load(weak, weights_only=True, map_location="cpu")
    probe = panel.probe_source.mlp(4, probe_config["mlp_width"]).eval()
    probe.load_state_dict(saved["state"])
    mean, scale = np.asarray(saved["input_mean"]), np.asarray(saved["input_std"])

    def inference(physical):
        h = panel.encoder_source.embed(encoder, physical).reshape(-1, 4)
        predicted = panel.probe_source.predict_mlp((h - mean) / scale, probe)
        return predicted.reshape(physical.shape)

    return inference, {
        str(strong_encoder.relative_to(ROOT)): panel_hashes[strong_encoder.name],
        str(weak.relative_to(ROOT)): hashes[weak.name],
    }


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    fit_weak()
