"""A single integrated numerical check; no research sweep or model selection."""
from __future__ import annotations

import copy
import torch

from .config import DEFAULTS, validate
from .io import environment
from .runner import configure_runtime
from .systems import make_system, compatibility_audit
from .spectral import select_spectrum
from .models import build_model
from .losses import compute_losses, state_composite_jacobian


def doctor_check(device="cpu"):
    cfg = validate({"training": {"device": device}})
    selected_device, dtype = configure_runtime(cfg)
    system = make_system(cfg["system"])
    selection = select_spectrum(system, cfg["selection"])
    if not selection["valid_for_training"]:
        return {"passed": False, "reason": "Default spectral selection unresolved", "selection": selection}
    audit = compatibility_audit(system, selection, cfg["data"], cfg["losses"])
    for key in ("reference_contraction_upper_bound", "spectral_spread_order"):
        if key in audit:
            selection["certificate"][key] = audit[key]
    model = build_model(cfg, selection).to(device=selected_device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(123)
    state = torch.randn(32, system.dimension, generator=generator, dtype=dtype) * .1
    x = system.observe_torch(state).to(selected_device)
    batch = {"x": x, "futures": {q: system.observe_torch(state * .9**q).to(selected_device) for q in (1, 4, 8)}}
    loss, parts = compute_losses(model, batch, system, cfg, generator)
    loss.backward()
    finite = all(p.grad is None or bool(p.grad.isfinite().all()) for p in model.parameters())
    jac = state_composite_jacobian(model, system, create_graph=True)
    rank_objective = torch.relu(torch.as_tensor(cfg["losses"]["kappa"], device=selected_device, dtype=dtype) -
                                torch.linalg.svdvals(jac).min()).square()
    gradients = torch.autograd.grad(rank_objective, list(model.E.parameters()) + list(model.G.parameters()), allow_unused=True)
    rank_gradient = sum(float(g.detach().abs().sum().cpu()) for g in gradients if g is not None)
    return {"passed": bool(loss.isfinite()) and finite and rank_gradient > 0,
            "environment": environment(device), "loss": float(loss.detach().cpu()),
            "rank_parameter_gradient_l1": rank_gradient,
            "selected_m": selection["m"], "K_constraints": model.K.constraints(),
            "scope": "one integrated forward/backward and rank-gradient compatibility check; no scientific result"}
