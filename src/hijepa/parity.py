"""One fixed-input forward/loss/gradient check across CPU and CUDA.

This checks implementation portability, not statistical equivalence of full training.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from .io import load_checkpoint, save_checkpoint, write_json, environment, source_fingerprint
from .losses import compute_losses
from .models import build_model
from .runner import configure_runtime, sample_batch
from .systems import make_system
from .data import prepare_data


def _snapshot(model, system, config, batch, projection_state):
    parameter = next(model.parameters())
    fixed = {"x": batch["x"].to(parameter),
             "futures": {q: x.to(parameter) for q, x in batch["futures"].items()}}
    generator = torch.Generator(device="cpu")
    generator.set_state(projection_state)
    model.zero_grad(set_to_none=True)
    model.train()
    loss, parts = compute_losses(model, fixed, system, config, generator)
    loss.backward()
    output = {"coarse": model.coarse(fixed["x"]).detach().cpu().double(),
              "K": model.K_matrix().detach().cpu().double(),
              "loss.total": loss.detach().cpu().double()}
    for name, value in parts.items():
        if isinstance(value, torch.Tensor):
            output["loss." + name] = value.detach().cpu().double()
    for name, value in model.named_parameters():
        if value.grad is not None:
            output["gradient." + name] = value.grad.detach().cpu().double()
    return output


def export_bundle(checkpoint_file, output_dir):
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint.get("source_sha256") != source_fingerprint():
        raise RuntimeError("Checkpoint source differs from current implementation; parity must use identical source.")
    cfg, selection = checkpoint["config"], checkpoint["selection"]
    runtime = copy.deepcopy(cfg)
    runtime["stage"], runtime["training"]["device"] = "local_integration", "cpu"
    device, dtype = configure_runtime(runtime)
    model = build_model(cfg, selection).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model"])
    system = make_system(cfg["system"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Fixed CPU generation makes inputs independent of target backend RNG.
    data_config = copy.deepcopy(cfg["data"])
    for key in ("train_trajectories", "calibration_trajectories", "test_trajectories"):
        data_config[key] = max(4, min(32, data_config[key]))
    datasets = prepare_data(system, data_config, output_dir / "fixed_data")
    batch = sample_batch(datasets["train"], cfg, np.random.default_rng(77123), device, dtype)
    projection_state = torch.Generator(device="cpu").manual_seed(77124).get_state()
    baseline = _snapshot(model, system, cfg, batch, projection_state)
    save_checkpoint(output_dir / "bundle.pt", {"schema_version": 1, "config": cfg,
                    "selection": selection, "model": checkpoint["model"], "batch": batch,
                    "projection_state": projection_state, "baseline": baseline,
                    "baseline_device": "cpu", "checkpoint_source": str(Path(checkpoint_file).name)})
    description = {"scope": "same checkpoint, fixed inputs, each loss term and parameter gradient",
                   "full_training_equivalence_claim": False, "baseline": environment("cpu"),
                   "tensor_count": len(baseline), "dtype": str(dtype), "source_sha256": source_fingerprint()}
    # Include identity inside the portable payload, not just the sidecar.
    payload = load_checkpoint(output_dir / "bundle.pt")
    payload["source_sha256"] = source_fingerprint()
    save_checkpoint(output_dir / "bundle.pt", payload)
    write_json(output_dir / "manifest.json", description)
    return description


def check_bundle(bundle_file, device="cuda", rtol=1e-4, atol=1e-6, output=None):
    bundle = load_checkpoint(bundle_file)
    if bundle.get("source_sha256") != source_fingerprint():
        raise RuntimeError("Parity baseline and target must use identical source files.")
    cfg = bundle["config"]
    runtime = copy.deepcopy(cfg)
    runtime["stage"], runtime["training"]["device"] = "local_integration", device
    selected_device, dtype = configure_runtime(runtime)
    model = build_model(cfg, bundle["selection"]).to(device=selected_device, dtype=dtype)
    model.load_state_dict(bundle["model"])
    values = _snapshot(model, make_system(cfg["system"]), cfg, bundle["batch"], bundle["projection_state"])
    baseline = bundle["baseline"]
    rows = []
    for name in sorted(set(values) | set(baseline)):
        a, b = baseline.get(name), values.get(name)
        if a is None or b is None or a.shape != b.shape:
            rows.append({"name": name, "passed": False, "reason": "missing tensor or shape mismatch"})
            continue
        delta = (a-b).abs()
        rows.append({"name": name, "passed": bool(torch.allclose(a, b, rtol=rtol, atol=atol)),
                     "maximum_absolute_error": float(delta.max()),
                     "relative_l2_error": float(torch.linalg.vector_norm(a-b)/
                                                 torch.linalg.vector_norm(a).clamp_min(atol))})
    result = {"passed": all(row["passed"] for row in rows), "rtol": rtol, "atol": atol,
              "tolerances_are": "engineering numerical comparison thresholds; not scientific recovery thresholds",
              "device": device, "environment": environment(device), "comparisons": rows,
              "full_training_equivalence_claim": False}
    if output:
        write_json(output, result)
    return result
