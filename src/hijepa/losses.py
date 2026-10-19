"""Proposal losses, including differentiable state-composite rank constraints.

SIGReg provenance (algorithm reimplemented, not vendored):
https://github.com/galilai-group/lejepa/blob/main/lejepa/univariate/epps_pulley.py
https://github.com/galilai-group/lejepa/blob/main/lejepa/multivariate/slicing.py
The official EppsPulley default is 17 trapezoid points over [0,3], doubled
by symmetry with Gaussian weight, N scaling, and mean over unit Gaussian
slices.  We use exactly those statistical choices; CPU projection sampling
is intentional for reproducibility across CPU/CUDA.  No centering,
standardization, whitening, or variance-only surrogate is applied to z.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as functional


def sigreg(z: torch.Tensor, slices: int = 1024, points: int = 17,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """Sliced Epps-Pulley statistic against the fixed standard normal law."""
    if z.ndim != 2 or min(z.shape) < 1:
        raise ValueError("SIGReg requires nonempty [samples, features] embeddings")
    if slices < 1 or points < 3 or points % 2 != 1:
        raise ValueError("SIGReg requires positive slices and an odd point count >= 3")
    if generator is not None and str(generator.device) != "cpu":
        raise ValueError("Use a CPU torch.Generator for device-independent SIGReg projections")
    # Sampling in float64 on CPU makes the sampled directions independent of
    # model precision/device, before the explicit cast to the training dtype.
    directions = torch.randn(z.shape[1], int(slices), generator=generator,
                             device="cpu", dtype=torch.float64)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-15)
    projection = z @ directions.to(device=z.device, dtype=z.dtype)
    t = torch.linspace(0.0, 3.0, int(points), device=z.device, dtype=z.dtype)
    normal_cf = torch.exp(-0.5 * t.square())
    phase = projection.unsqueeze(-1) * t
    discrepancy = (phase.cos().mean(dim=0) - normal_cf).square() + phase.sin().mean(dim=0).square()
    # Integrating over [0,3] and multiplying by two implements symmetric
    # trapezoid integration over [-3,3].  The statistic scales by sample N.
    return (2.0 * z.shape[0] * torch.trapezoid(discrepancy * normal_cf, t, dim=-1)).mean()


def covariance_floor(c: torch.Tensor) -> torch.Tensor:
    """Fixed coefficient 1, std floor .2, epsilon 1e-6; no joint whitening."""
    if c.ndim != 2 or c.shape[0] < 2 or c.shape[1] < 1:
        raise ValueError("Covariance floor requires [B,m] with B >= 2 and m >= 1")
    centered = c - c.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (c.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    std = (eigenvalues.clamp_min(0.0) + 1e-6).sqrt()
    return functional.relu(0.2 - std).square().mean()


def block_scale(c: torch.Tensor, blocks: list[list[int]]) -> torch.Tensor:
    """Implementation choice: zero means and unit variance per real mode;
    zero block mean and total variance two per rotation block.  In particular
    no off-diagonal covariance or rotation-block axis constraint is added.
    """
    penalties = []
    for indices in blocks:
        block = c[:, indices]
        centered = block - block.mean(dim=0, keepdim=True)
        variance_sum = centered.square().sum() / (c.shape[0] - 1)
        penalties.append(block.mean(dim=0).square().sum() + (variance_sum - len(indices)).square())
    if not penalties:
        raise ValueError("Scale requires the complete selected block partition")
    return torch.stack(penalties).mean()


def state_composite_jacobian(model, system, create_graph: bool = True) -> torch.Tensor:
    """D(G o E o observation)(0), never an observation-space surrogate."""
    parameter = next(model.parameters())
    origin = torch.zeros(int(system.dimension), device=parameter.device, dtype=parameter.dtype,
                         requires_grad=True)

    def composite(state):
        return model.coarse(system.observe_torch(state.unsqueeze(0))).squeeze(0)

    # create_graph is essential: training rank must backpropagate through this
    # derivative to E/G parameters.  No detached Jacobian or silent fallback.
    with torch.enable_grad():
        return torch.autograd.functional.jacobian(composite, origin, create_graph=create_graph)


def compute_losses(model, batch: dict[str, Any], system, config: dict[str, Any],
                   generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    settings = config["losses"]
    x, futures = batch["x"], batch["futures"]
    if x.shape[0] < 2:
        raise ValueError("The fixed covariance losses require batch size >= 2")
    k = int(settings["k"])
    if k < 1:
        raise ValueError("The auxiliary horizon k must be positive")
    horizons = sorted({1, k, 2 * k})
    missing = [q for q in horizons if q not in futures]
    if missing:
        raise ValueError(f"Missing future observations for horizons {missing}")
    z = model.encode(x)
    c = model.G(z)
    zero = z.sum() * 0.0
    phase = config.get("training", {}).get("phase", "source")
    use_fine = model.fine_enabled and (model.variant != "no_fine" or phase == "transfer")
    fine_weight = float(settings.get("fine_weight", 1.0)) if use_fine else 0.0
    with torch.no_grad():
        target_z = model.encode(futures[1]) if use_fine else None
        target_c = {q: model.coarse(futures[q]) for q in horizons}
    fine = functional.mse_loss(model.predict_fine(z), target_z) if use_fine else zero
    coarse = sum(functional.mse_loss(model.predict_coarse(c, q), target_c[q]) for q in horizons)
    jacobian = state_composite_jacobian(model, system, create_graph=model.rank_enabled)
    singular_values = torch.linalg.svdvals(jacobian)
    sigma_min = singular_values[-1]
    rank = functional.relu(float(settings["kappa"]) - sigma_min).square() if model.rank_enabled else zero
    scale = block_scale(c, model.coarse_blocks)
    gaussian = sigreg(z, int(settings["sigreg_slices"]), int(settings["sigreg_points"]), generator)
    floor = covariance_floor(c)
    total = (fine_weight * fine + float(settings["alpha"]) * coarse
             + float(settings["beta"]) * rank + float(settings["gamma"]) * scale
             + 0.1 * gaussian + floor)
    components = {"fine": fine, "coarse": coarse, "rank": rank, "scale": scale,
                  "sigreg": gaussian, "floor": floor, "sigma_min": sigma_min,
                  "fine_weight": z.new_tensor(fine_weight), "total": total}
    return total, components
