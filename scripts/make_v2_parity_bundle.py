"""Create a v2 parity bundle from an initialization checkpoint (no training).

The checkpoint is a deterministic initialization under the current v2 source.
Its final G readout is deliberately made nearly collinear so the GCSB term is
active on the fixed parity batch; this is a numerical preflight construction,
not a trained model and not formal evidence.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hijepa.config import load_config  # noqa: E402
from hijepa.data import prepare_data  # noqa: E402
from hijepa.io import load_checkpoint, save_checkpoint, source_fingerprint, protocol_fingerprint, write_json  # noqa: E402
from hijepa.losses import coarse_spectral_barrier  # noqa: E402
from hijepa.models import build_model, project_network_budget  # noqa: E402
from hijepa.parity import export_bundle  # noqa: E402
from hijepa.runner import configure_runtime, sample_batch, train_run  # noqa: E402
from hijepa.systems import make_system  # noqa: E402

OUT = REPO / "runs" / "formal_v2_preflight"
INIT_RUN = OUT / "parity_init_run"
INIT_DATA = OUT / "parity_init_data"
CHECKPOINT = OUT / "parity_init_near_collinear.pt"
BUNDLE_DIR = OUT / "parity_bundle"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO / "configs" / "base_formal_v2.json")
    cfg["stage"] = "local_integration"
    cfg["training"]["device"] = "cpu"
    # This call saves initial.pt at step 0; stop_after=0 means no optimizer steps.
    train_run(cfg, INIT_RUN, INIT_DATA, stop_after=0)
    payload = load_checkpoint(INIT_RUN / "initial.pt")
    cfg, selection = payload["config"], payload["selection"]
    system = make_system(cfg["system"])
    device, dtype = configure_runtime(cfg)
    torch.manual_seed(0)
    model = build_model(cfg, selection).to(device=device, dtype=dtype)
    model.load_state_dict(payload["model"])
    project_network_budget(model)
    # Deliberate near-collinear G readout for the parity preflight.
    final = model.G[-1]
    with torch.no_grad():
        base_row = final.weight[0].detach().clone()
        final.weight[1] = base_row * 1.0001 + 0.0005 * torch.randn_like(base_row)
        final.bias[1] = final.bias[0].detach() * 1.0001 + 0.0005
    project_network_budget(model)

    # Fixed parity batch, matching parity.export_bundle's capped data/RNG.
    data_config = copy.deepcopy(cfg["data"])
    for key in ("train_trajectories", "calibration_trajectories", "test_trajectories"):
        data_config[key] = max(4, min(32, data_config[key]))
    fixed_data = OUT / "parity_init_fixed_data"
    datasets = prepare_data(system, data_config, fixed_data)
    batch = sample_batch(datasets["train"], cfg, np.random.default_rng(77123), device, dtype)
    with torch.no_grad():
        c = model.coarse(batch["x"])
        centered = c - c.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (c.shape[0] - 1)
        diagonal = torch.sqrt(torch.diag(covariance).clamp_min(1e-12))
        correlation = covariance / diagonal[:, None] / diagonal[None, :]
        eigenvalues = torch.linalg.eigvalsh(covariance)
        barrier_value = coarse_spectral_barrier(
            c.double(), mu=cfg["losses"]["coarse_barrier_mu"],
            epsilon=cfg["losses"]["coarse_barrier_epsilon"],
            lambda_floor=cfg["losses"]["coarse_barrier_lambda_floor"])
    c_graph = model.coarse(batch["x"])
    barrier_graph = coarse_spectral_barrier(
        c_graph, mu=cfg["losses"]["coarse_barrier_mu"],
        epsilon=cfg["losses"]["coarse_barrier_epsilon"],
        lambda_floor=cfg["losses"]["coarse_barrier_lambda_floor"])
    grad_e = torch.autograd.grad(barrier_graph, list(model.E.parameters()), allow_unused=True, retain_graph=True)
    grad_g = torch.autograd.grad(barrier_graph, list(model.G.parameters()), allow_unused=True)

    save_checkpoint(CHECKPOINT, {
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "config": cfg, "selection": selection,
        "protocol_sha256": protocol_fingerprint(cfg),
        "source_sha256": source_fingerprint(),
    })
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = export_bundle(CHECKPOINT, BUNDLE_DIR)
    bundle = load_checkpoint(BUNDLE_DIR / "bundle.pt")
    baseline = bundle["baseline"]
    summary = {
        "scope": "parity-only initialization; no training; not formal evidence",
        "source_sha256": source_fingerprint(),
        "checkpoint": str(CHECKPOINT),
        "bundle": str(BUNDLE_DIR / "bundle.pt"),
        "losses": {key: cfg["losses"][key] for key in (
            "coarse_floor_mode", "coarse_barrier_lambda_floor",
            "coarse_barrier_epsilon", "coarse_barrier_mu")},
        "initialization": {
            "torch_seed": 0,
            "near_collinear_construction": "G final-layer row 1 set to 1.0001 * row 0 + 0.0005 * noise; bias likewise",
        },
        "fixed_batch": {
            "n_samples": int(c.shape[0]),
            "c_std": diagonal.detach().cpu().tolist(),
            "c_correlation": correlation.detach().cpu().tolist(),
            "c_covariance_eigenvalues": eigenvalues.detach().cpu().tolist(),
            "gcsb_value": float(barrier_value.detach().cpu()),
            "gcsb_active": bool(float(barrier_value.detach().cpu()) > 0.0),
            "gcsb_gradient_norm_E": float(sum(float(g.norm()) for g in grad_e if g is not None)),
            "gcsb_gradient_norm_G": float(sum(float(g.norm()) for g in grad_g if g is not None)),
        },
        "bundle_manifest": manifest,
        "bundle_baseline_has_loss_floor": "loss.floor" in baseline,
        "bundle_baseline_floor_value": float(baseline["loss.floor"]) if "loss.floor" in baseline else None,
        "bundle_baseline_gradient_tensor_count": sum(key.startswith("gradient.") for key in baseline),
        "note": "near-collinear initialization is deliberately constructed for numerical parity, not a training result",
    }
    write_json(OUT / "parity_init_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
