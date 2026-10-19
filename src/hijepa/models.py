"""Smooth HI-JEPA models and target-domain constrained block dynamics.

There are no action inputs.  ``m`` is the real dimension selected by the
training-only spectral procedure, including both members of complex pairs.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn


VARIANTS = ("full", "block_only", "hierarchical", "direct", "no_fine")


def _compose_smoothness(outer: dict[str, float], inner: dict[str, float]) -> dict[str, float]:
    """Global Euclidean chain-rule bounds, including the bilinear Hessian."""
    return {
        "lipschitz_upper_bound": outer["lipschitz_upper_bound"] * inner["lipschitz_upper_bound"],
        "jacobian_lipschitz_upper_bound": (
            outer["jacobian_lipschitz_upper_bound"] * inner["lipschitz_upper_bound"] ** 2
            + outer["lipschitz_upper_bound"] * inner["jacobian_lipschitz_upper_bound"]),
    }


def _mlp_smoothness(module: nn.Sequential, weight_bound: float | None = None) -> dict[str, Any]:
    """tanh has |tanh'| <= 1 and the safe global |tanh''| <= 2.

    Affine bias does not enter these derivative bounds.  The spectral norm
    can be replaced by a fixed Frobenius cap, since ||W||2 <= ||W||F.
    """
    current = {"lipschitz_upper_bound": 1.0, "jacobian_lipschitz_upper_bound": 0.0}
    layers = []
    for layer in module:
        if isinstance(layer, nn.Linear):
            actual_frobenius = float(torch.linalg.vector_norm(layer.weight.detach().double()))
            actual_spectral = float(torch.linalg.matrix_norm(layer.weight.detach().double(), ord=2))
            norm = actual_spectral if weight_bound is None else float(weight_bound)
            current = _compose_smoothness(
                {"lipschitz_upper_bound": norm, "jacobian_lipschitz_upper_bound": 0.0}, current)
            layers.append({"kind": "linear", "spectral_norm": actual_spectral,
                           "frobenius_norm": actual_frobenius, "bound_used": norm})
        elif isinstance(layer, nn.Tanh):
            current = _compose_smoothness(
                {"lipschitz_upper_bound": 1.0, "jacobian_lipschitz_upper_bound": 2.0}, current)
            layers.append({"kind": "tanh", "first_derivative_bound": 1.0,
                           "second_derivative_bound": 2.0})
        else:
            raise TypeError(f"No declared analytic smoothness bound for {type(layer).__name__}")
    return {**current, "layers": layers}


@torch.no_grad()
def project_network_budget(model: "HIJEPA") -> None:
    """Project every affine network weight/bias onto fixed norm balls.

    This is a hard parameter constraint, not an extra loss.  Block-dynamics
    raw parameters are not nn.Linear parameters and are deliberately untouched.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.weight.requires_grad:
            weight_norm = torch.linalg.vector_norm(module.weight)
            scale = (model.weight_frobenius_bound / weight_norm.clamp_min(
                torch.finfo(module.weight.dtype).tiny)).clamp_max(1.0)
            module.weight.mul_(scale)
            if module.bias is not None and module.bias.requires_grad:
                bias_norm = torch.linalg.vector_norm(module.bias)
                scale = (model.bias_norm_bound / bias_norm.clamp_min(
                    torch.finfo(module.bias.dtype).tiny)).clamp_max(1.0)
                module.bias.mul_(scale)


def make_mlp(input_dim: int, output_dim: int, width: int, depth: int) -> nn.Sequential:
    """``depth`` counts smooth hidden layers; the output is affine."""
    if min(input_dim, output_dim, width) < 1 or depth < 0:
        raise ValueError("MLP dimensions must be positive and depth nonnegative")
    layers: list[nn.Module] = []
    previous = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(previous, width), nn.Tanh()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class DisabledFinePredictor(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("The direct control has no fine predictor or fine prediction loss")


class SpectralBlock(nn.Module):
    """One real eigenvalue or a real 2D rotation-scaling block.

    Parameter bounds are an *inner* subset of the selected region.  If a
    certified reference contraction ``a`` is supplied, |lambda| > a**order
    is enforced throughout optimization, rather than just checked at init.
    """

    def __init__(self, block: dict[str, Any], certificate: dict[str, Any], margin: float):
        super().__init__()
        self.kind = str(block["kind"])
        if self.kind not in ("real", "rotation"):
            raise ValueError(f"Unsupported spectral block {self.kind!r}")
        center = float(block["center"])
        radius = float(block["radius"])
        if not math.isfinite(center) or not math.isfinite(radius) or radius <= 0:
            raise ValueError("A spectral block needs a finite center and positive radius")
        if self.kind == "real" and abs(center) <= radius:
            raise ValueError("A real target interval may not cross zero")
        self.sign = -1.0 if center < 0 and self.kind == "real" else 1.0
        magnitude = abs(center)
        # An inner 90% interval also works for very tight certified regions.
        self.margin = min(float(margin), 0.1 * radius)
        if self.margin <= 0:
            raise ValueError("The spectral boundary margin must be positive")
        a = certificate.get("reference_contraction_upper_bound")
        order = int(certificate.get("spectral_spread_order", 2))
        if order < 1:
            raise ValueError("spectral_spread_order must be positive")
        if a is not None and not (0 < float(a) < 1):
            raise ValueError("Reference contraction must lie strictly between zero and one")
        self.reference_contraction = None if a is None else float(a)
        self.spectral_spread_order = order
        inverse_floor = 0.0 if a is None else float(a) ** order
        low = max(magnitude - radius + self.margin, inverse_floor + self.margin, self.margin)
        high = min(magnitude + radius - self.margin, 1.0 - self.margin)
        if low >= high:
            raise ValueError("Selected spectrum has no interior satisfying stable inverse contraction")
        self.register_buffer("modulus_bounds", torch.tensor([low, high], dtype=torch.float64))
        self.certified_contour_radius = block.get("certificate_contour_radius")
        self.domain_distance_upper_bound = max(abs(low - magnitude), abs(high - magnitude))
        initial_fraction = min(max((magnitude - low) / (high - low), 1e-5), 1 - 1e-5)
        self.raw_modulus = nn.Parameter(torch.tensor(math.log(initial_fraction / (1 - initial_fraction))))
        if self.kind == "rotation":
            angle = float(block["angle"])
            angle_radius = float(block["angle_radius"])
            if angle_radius <= 0 or not math.isfinite(angle_radius):
                raise ValueError("A rotation block requires a positive certified angle radius")
            self.register_buffer("angle_center", torch.tensor(angle, dtype=torch.float64))
            self.register_buffer("angle_halfwidth", torch.tensor(0.9 * angle_radius, dtype=torch.float64))
            self.raw_angle = nn.Parameter(torch.zeros(()))
            self.domain_distance_upper_bound += 2.0 * magnitude * math.sin(0.9 * angle_radius / 2.0)
        if (self.certified_contour_radius is not None
                and self.domain_distance_upper_bound >= float(self.certified_contour_radius)):
            raise ValueError("The K parameter domain is not contained in its certified spectral disk")

    @property
    def dimension(self) -> int:
        return 1 if self.kind == "real" else 2

    def modulus(self) -> torch.Tensor:
        bounds = self.modulus_bounds.to(self.raw_modulus)
        return bounds[0] + (bounds[1] - bounds[0]) * self.raw_modulus.sigmoid()

    def forward(self) -> torch.Tensor:
        radius = self.modulus()
        if self.kind == "real":
            return (self.sign * radius).reshape(1, 1)
        theta = self.angle_center.to(radius) + self.angle_halfwidth.to(radius) * self.raw_angle.tanh()
        cosine, sine = theta.cos(), theta.sin()
        return radius * torch.stack((torch.stack((cosine, -sine)), torch.stack((sine, cosine))))


class BlockDynamics(nn.Module):
    def __init__(self, selection: dict[str, Any], margin: float):
        super().__init__()
        certificate = selection.get("certificate", {})
        self.blocks = nn.ModuleList(SpectralBlock(b, certificate, margin) for b in selection["blocks"])
        if not self.blocks:
            raise ValueError("At least one selected spectral block is required")
        self.dimension = sum(b.dimension for b in self.blocks)
        if self.dimension != int(selection["m"]):
            raise ValueError("Selected m disagrees with complete real/rotation block dimensions")

    def forward(self) -> torch.Tensor:
        return torch.block_diag(*(block() for block in self.blocks))

    def constraints(self) -> dict[str, Any]:
        with torch.no_grad():
            moduli = [float(block.modulus()) for block in self.blocks]
        a = self.blocks[0].reference_contraction
        order = self.blocks[0].spectral_spread_order
        minimum, maximum = min(moduli), max(moduli)
        factor = None if a is None else a ** order / minimum
        return {
            "minimum_modulus": minimum,
            "maximum_modulus": maximum,
            "inverse_operator_norm": 1.0 / minimum,
            "reference_contraction_upper_bound": a,
            "spectral_spread_order": order,
            "inverse_contraction_factor": factor,
            "inverse_contraction_certified": factor is not None and factor < 1,
            "inverse_contraction_status": "bound_enforced" if factor is not None else "missing_reference_bound",
            "stable": maximum < 1,
            "modulus_bounds": [b.modulus_bounds.detach().cpu().tolist() for b in self.blocks],
            "spectral_disk_containment": [
                {"distance_upper_bound": b.domain_distance_upper_bound,
                 "contour_radius": b.certified_contour_radius,
                 "verified": (None if b.certified_contour_radius is None else
                              b.domain_distance_upper_bound < float(b.certified_contour_radius))}
                for b in self.blocks],
        }


class HIJEPA(nn.Module):
    def __init__(self, config: dict[str, Any], selection: dict[str, Any]):
        super().__init__()
        model_config = config["model"]
        self.variant = str(model_config["variant"])
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}")
        if model_config.get("activation", "tanh") != "tanh":
            raise ValueError("Only smooth tanh MLPs are supported by this protocol")
        if model_config.get("target_mode", "shared_stopgrad") != "shared_stopgrad":
            raise ValueError("Only the declared shared-parameter stop-gradient target is supported")
        self.state_dim = int(config["system"]["dimension"])
        self.z_dim = int(model_config["z_dim"])
        self.m = int(selection["m"])
        if not 1 <= self.m <= self.state_dim:
            raise ValueError("Selected coarse dimension must be between one and the state dimension")
        width, depth = int(model_config["width"]), int(model_config["depth"])
        self.E = make_mlp(self.state_dim, self.z_dim, width, depth)
        self.G = make_mlp(self.z_dim, self.m, width, depth)
        self.F = (DisabledFinePredictor() if self.variant == "direct"
                  else make_mlp(self.z_dim, self.z_dim, width, depth))
        margin = float(config["selection"].get("certificate_margin", 1e-5))
        self.K = BlockDynamics(selection, margin)
        self.selection = copy.deepcopy(selection)
        self.coarse_blocks: list[list[int]] = []
        cursor = 0
        for block in self.K.blocks:
            self.coarse_blocks.append(list(range(cursor, cursor + block.dimension)))
            cursor += block.dimension
        self.H = (make_mlp(self.m, self.m, width, depth) if self.variant == "hierarchical" else None)
        if self.H is not None:
            self.K.requires_grad_(False)
        self.rank_enabled = self.variant not in ("block_only", "hierarchical")
        self.fine_enabled = self.variant != "direct"
        self.depth = depth
        self.weight_frobenius_bound = float(model_config.get("weight_frobenius_bound", 8.0))
        self.bias_norm_bound = float(model_config.get("bias_norm_bound", 8.0))
        if (not math.isfinite(self.weight_frobenius_bound) or self.weight_frobenius_bound <= 0
                or not math.isfinite(self.bias_norm_bound) or self.bias_norm_bound <= 0):
            raise ValueError("Fixed network weight and bias norm budgets must be finite and positive")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # In the direct control this is an internal regularization/probe tap,
        # with exactly the same E->G capacity and only an m-dimensional output.
        return self.E(x)

    def coarse(self, x: torch.Tensor) -> torch.Tensor:
        return self.G(self.encode(x))

    def predict_fine(self, z: torch.Tensor) -> torch.Tensor:
        return self.F(z)

    def K_matrix(self) -> torch.Tensor:
        return self.K()

    def predict_coarse(self, c: torch.Tensor, q: int = 1) -> torch.Tensor:
        if not isinstance(q, int) or q < 0:
            raise ValueError("Prediction horizon must be a nonnegative integer")
        if self.H is not None:
            result = c
            for _ in range(q):
                result = self.H(result)
            return result
        return c @ torch.linalg.matrix_power(self.K_matrix(), q).T

    def parameter_counts(self) -> dict[str, Any]:
        count = lambda module: 0 if module is None else sum(p.numel() for p in module.parameters())
        high = count(self.E) + count(self.G)
        return {
            "E": count(self.E), "G": count(self.G), "F": count(self.F),
            "K": count(self.K), "nonlinear_coarse_predictor": count(self.H),
            "high_path": high, "full_high_path_reference": high,
            "high_path_linear_layers": 2 * (self.depth + 1),
            "direct_high_path_exactly_matched": self.variant == "direct",
            "direct_internal_z_tap": self.variant == "direct",
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def smoothness_budget(self, system=None) -> dict[str, Any]:
        """Conservative global C1,1 bounds; never a learned recovery certificate.

        The fixed bounds apply to every parameter point satisfying the norm
        projection, independent of a local numerical derivative scan.  Passing
        a system additionally requires its declared observation-map bounds.
        Whether a properly scaled true eigenfunction lies within this budget
        is a separate compatibility audit, not established by these bounds.
        """
        e, g = _mlp_smoothness(self.E), _mlp_smoothness(self.G)
        fixed_e = _mlp_smoothness(self.E, self.weight_frobenius_bound)
        fixed_g = _mlp_smoothness(self.G, self.weight_frobenius_bound)
        ge, fixed_ge = _compose_smoothness(g, e), _compose_smoothness(fixed_g, fixed_e)
        affine_checks = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                weight_norm = float(torch.linalg.vector_norm(module.weight.detach().double()))
                bias_norm = (float(torch.linalg.vector_norm(module.bias.detach().double()))
                             if module.bias is not None else 0.0)
                # The tolerance only accounts for finite-precision projection.
                affine_checks.append({"name": name, "weight_frobenius": weight_norm,
                                      "bias_l2": bias_norm,
                                      "within_budget": (weight_norm <= self.weight_frobenius_bound * (1 + 1e-6)
                                                        and bias_norm <= self.bias_norm_bound * (1 + 1e-6))})
        observation = None
        if system is not None and hasattr(system, "observation_smoothness_budget"):
            observation = system.observation_smoothness_budget()
        if observation is None:
            h = {"available": False, "reason": "No declared observation-map analytic smoothness budget"}
            fixed_h = dict(h)
        else:
            for key in ("lipschitz_upper_bound", "jacobian_lipschitz_upper_bound"):
                if key not in observation or not math.isfinite(float(observation[key])) or float(observation[key]) < 0:
                    raise ValueError(f"Invalid observation-map smoothness bound: {key}")
            h = {"available": True, **_compose_smoothness(ge, observation)}
            fixed_h = {"available": True, **_compose_smoothness(fixed_ge, observation)}
        return {
            "norm": "Euclidean vector norm and its induced/bilinear operator norms",
            "scope": "global analytic C1,1 bounds from affine norms, tanh bounds, and composition",
            "method": "L(f o g)<=L(f)L(g); Lip D(f o g)<=Lip Df*Lg^2+Lf*Lip Dg",
            "tanh_first_derivative_bound": 1.0, "tanh_second_derivative_bound": 2.0,
            "weight_frobenius_bound": self.weight_frobenius_bound,
            "bias_norm_bound": self.bias_norm_bound,
            "projection_enforced": all(item["within_budget"] for item in affine_checks),
            "parameter_checks": affine_checks,
            "E": e, "G": g, "G_after_E": ge, "observation": observation,
            "h_state_composite": h,
            "uniform_fixed_parameter_budget": {"E": fixed_e, "G": fixed_g,
                                               "G_after_E": fixed_ge, "h_state_composite": fixed_h},
            "learned_recovery_certificate": False,
            "true_mode_compatibility": "separate audit required; no true-function compatibility claim follows from this bound",
            "adds_training_loss": False,
        }


def build_model(config: dict[str, Any], selection: dict[str, Any]) -> HIJEPA:
    model = HIJEPA(config, selection)
    dtype_name = config.get("training", {}).get("dtype", "float32")
    if dtype_name not in ("float32", "float64"):
        raise ValueError("Only float32 and float64 are supported")
    return model.to(dtype=getattr(torch, dtype_name))
