"""Smooth synthetic systems with an explicit, invertible observation map.

Truth is used to generate observations and by evaluation/compatibility only.
The selector receives local observed-space queries and derivative error budgets.
"""
from __future__ import annotations

import math
import numpy as np
import torch


def _shear(x, strength):
    y = np.array(x, dtype=np.float64, copy=True)
    y[..., 1:] += strength * np.tanh(x[..., :-1])
    return y


def _inverse_shear(y, strength):
    x = np.array(y, dtype=np.float64, copy=True)
    for i in range(1, x.shape[-1]):
        x[..., i] -= strength * np.tanh(x[..., i - 1])
    return x


def _detail_conjugacy(states, strength):
    """Identity coarse coordinates; fine triangular nonlinear coupling only.

    f(t)=tanh(t)-t has f(0)=f'(0)=0, hence Dphi(0)=I and the
    true state Jacobian equals the declared normal block matrix A.
    """
    x = np.asarray(states, dtype=np.float64)
    y = np.array(x, copy=True)
    y[..., 2:] += strength*(np.tanh(x[..., 1:-1])-x[..., 1:-1])
    return y


def _inverse_detail_conjugacy(modes, strength):
    x = np.array(modes, dtype=np.float64, copy=True)
    for i in range(2, x.shape[-1]):
        x[..., i] -= strength*(np.tanh(x[..., i-1])-x[..., i-1])
    return x


def _compose_derivative_bounds(outer, inner):
    """Global induced-infinity bounds through order three (chain rule)."""
    a, b, c = outer
    u, v, w = inner
    return (a*u, b*u*u + a*v, c*u**3 + 3*b*u*v + a*w)


def _shear_bounds(dimension, strength, first_changed_index=1):
    # A simultaneous triangular map is a composition of elementary shears.
    # |tanh'| <= 1, |tanh''| <= 2, |tanh'''| <= 16 are safe global bounds.
    result = (1.0, 0.0, 0.0)
    elementary = (1.0 + abs(strength), 2*abs(strength), 16*abs(strength))
    for _ in range(dimension - first_changed_index):
        result = _compose_derivative_bounds(elementary, result)
    return result


class SyntheticSystem:
    def __init__(self, config):
        self.config = dict(config)
        self.family = config.get("family", "real")
        self.dimension = int(config.get("dimension", 4))
        if self.dimension < 2:
            raise ValueError("Synthetic systems require dimension >= 2.")
        if self.family not in {"real", "oscillatory", "near_repeated", "coupled", "resonance"}:
            raise ValueError(f"Unknown system family: {self.family}")
        self.coupling = float(config.get("coupling", .02))
        self.observation_strength = float(config.get("observation_strength", .15))
        self.noise_std = float(config.get("noise_std", 0.0))
        if self.noise_std < 0:
            raise ValueError("noise_std must be nonnegative.")
        d = self.dimension
        defaults = np.empty(d)
        defaults[:2] = [.92, .90]
        defaults[2:] = np.linspace(.45, .30, d-2)
        if self.family == "near_repeated":
            defaults[:2] = [.92, .918]
        if self.family == "resonance":
            defaults[:2] = [.92, .92**2]
        values = np.asarray(config.get("eigenvalues", defaults.tolist()), dtype=np.float64)
        if values.shape != (d,) or np.any(np.abs(values) >= 1):
            raise ValueError("eigenvalues must contain d real entries with modulus < 1.")
        self._linear_map = np.diag(values)
        if self.family == "oscillatory":
            r = abs(float(values[0]))
            theta = float(config.get("angle", .25))
            self._linear_map[:2, :2] = r*np.array([[math.cos(theta), -math.sin(theta)],
                                                              [math.sin(theta), math.cos(theta)]])
        self.nonnormal_extension = bool(config.get("nonnormal_extension", False))
        self.linear_extension_strength = float(config.get("linear_extension_strength", self.coupling*.2)) if self.nonnormal_extension else 0.
        if self.nonnormal_extension:
            # Explicit extension outside the normal-J main theory. The default
            # coupled family changes only fine nonlinear state mechanisms.
            for i in range(d-1):
                self._linear_map[i, i+1] += self.linear_extension_strength
        rng = np.random.default_rng(int(config.get("observation_seed", 51)))
        q, r = np.linalg.qr(rng.normal(size=(d, d)))
        self._observation_rotation = q * np.where(np.diag(r) < 0, -1., 1.)
        self.fixed_point = np.zeros(d, dtype=np.float64)

    def to_modes(self, states_np):
        return _detail_conjugacy(states_np, self.coupling)

    def from_modes(self, modes_np):
        return _inverse_detail_conjugacy(modes_np, self.coupling)

    def observe(self, states_np):
        return _shear(np.asarray(states_np, dtype=np.float64), self.observation_strength) @ self._observation_rotation.T

    def inverse_observe(self, observations_np):
        return _inverse_shear(np.asarray(observations_np, dtype=np.float64) @ self._observation_rotation,
                              self.observation_strength)

    def observe_torch(self, states_tensor):
        x = states_tensor
        shifted = torch.cat([torch.zeros_like(x[..., :1]), torch.tanh(x[..., :-1])], dim=-1)
        q = torch.as_tensor(self._observation_rotation, dtype=x.dtype, device=x.device)
        return (x + self.observation_strength*shifted) @ q.T

    def observation_smoothness_budget(self):
        """Global Euclidean operator bounds; Q is orthogonal by construction."""
        return {"lipschitz_upper_bound": 1.+abs(self.observation_strength),
                "jacobian_lipschitz_upper_bound": 2.*abs(self.observation_strength),
                "norm": "euclidean", "scope": "global analytic observation-map bound",
                "derivation": "subdiagonal shift norm <=1; |tanh'|<=1, |tanh''|<=2; Q orthogonal",
                "is_empirical_scan": False}

    def step(self, states_np):
        return self.from_modes(self.to_modes(states_np) @ self._linear_map.T)

    def local_query(self, x_np):
        return self.observe(self.step(self.inverse_observe(x_np)))

    def query_derivative_budget(self, radius):
        """Global analytic C3 bound; no eigenvalue labels are exposed here.

        The selector bounds *local* J(0). It does not certify a global spectral
        condition or infer such a condition from samples.
        """
        q = self._observation_rotation
        rotation = (float(np.max(np.sum(np.abs(q), axis=1))), 0., 0.)
        inverse_rotation = (float(np.max(np.sum(np.abs(q.T), axis=1))), 0., 0.)
        observation = _shear_bounds(self.dimension, self.observation_strength)
        state = _shear_bounds(self.dimension, self.coupling, first_changed_index=2)
        linear = (float(np.max(np.sum(np.abs(self._linear_map), axis=1))), 0., 0.)
        bound = (1., 0., 0.)
        # For f=tanh-id, |f'|<=1 and derivatives 2/3 retain the tanh bounds.
        # g^-1 -> phi -> A -> phi^-1 -> g; inverse elementary shears use the same bounds.
        for part in (inverse_rotation, observation, state, linear, state, observation, rotation):
            bound = _compose_derivative_bounds(part, bound)
        # Conservative operation-count enclosure for query arithmetic in f64.
        eps = np.finfo(np.float64).eps
        gamma = (100*self.dimension**2*eps)/(1-100*self.dimension**2*eps)
        query_roundoff = gamma*max(1., float(radius))*max(1., bound[0])
        return {"derivative_order": 3, "third_derivative_inf_bound": bound[2],
                "first_derivative_inf_bound": bound[0], "query_roundoff_bound": query_roundoff,
                "bound_scope": "global analytic composition bound; queried derivative is J(0)",
                "floating_point_model": "IEEE float64; elementary tanh error <= 4 ulp assumed",
                "interval_arithmetic": False}

    def _truth_rows(self, selection):
        vals, vectors = np.linalg.eig(self._linear_map.astype(np.complex128))
        left = np.linalg.inv(vectors)
        rows, used = [], set()
        for block in selection.get("blocks", []):
            eigen = block["eigenvalues"][0]
            target = complex(eigen["real"], abs(eigen["imag"]))
            candidates = [i for i in range(len(vals)) if i not in used]
            index = min(candidates, key=lambda i: abs(vals[i]-target))
            if block["kind"] == "rotation":
                if vals[index].imag < 0:
                    index = min(candidates, key=lambda i: abs(vals[i]-np.conj(vals[index])))
                row = left[index]
                # Real and imaginary rows span the complete oscillatory block.
                rows.extend([row.real, row.imag])
                used.add(index)
                used.add(int(np.argmin(np.abs(vals-np.conj(vals[index])))))
            else:
                rows.append(left[index].real)
                used.add(index)
        return np.asarray(rows, dtype=np.float64).reshape(-1, self.dimension)

    def true_modes(self, states_np, selection):
        """Evaluation-only truth; target labels are never returned to selection."""
        return self.to_modes(states_np) @ self._truth_rows(selection).T

    def metadata(self):
        return {"family": self.family, "dimension": self.dimension, "config": self.config,
                "state_dynamics": "phi^-1(A phi(s)); phi[:2]=s[:2], phi[i]=s[i]+coupling*(tanh(s[i-1])-s[i-1]) for i>=2",
                "observation": "Q triangular_tanh(s); Q orthogonal",
                "fixed_point": self.fixed_point.tolist(),
                "observation_invertible": True, "truth_use": "generation, compatibility, evaluation only",
                "linear_mode_map": self._linear_map.tolist(),
                "state_fixed_point_jacobian": self._linear_map.tolist(),
                "state_jacobian_normal_by_construction": not self.nonnormal_extension,
                "nonnormal_extension": self.nonnormal_extension,
                "coupling_change_scope": "fine nonlinear mechanisms only; coarse first two coordinates unchanged" if not self.nonnormal_extension else "explicit nonnormal linear extension; outside normal-J main theory",
                "declared_resonance_present": bool(self.family == "resonance" and
                    abs(self._linear_map[1, 1]-self._linear_map[0, 0]**2) < 1e-12),
                "spectral_selector_truth_access": False}


def make_system(system_config):
    return SyntheticSystem(system_config)


def compatibility_audit(system, selection, data_config=None, losses_config=None):
    """Separate truth audit, never an input to spectral target selection.

    Signature: (SyntheticSystem, selection_dict, data_config=None,
    losses_config=None) -> JSON-serializable dict.
    """
    data_config, losses_config = data_config or {}, losses_config or {}
    d = system.dimension
    rows = system._truth_rows(selection)
    m = len(rows)
    radius = float(data_config.get("initial_radius", 1.0))
    # Dphi(0)=I because only fine coordinates use tanh(s)-s.
    jac = rows
    raw_covariance = (radius**2/3)*(rows @ rows.T)
    length = int(data_config.get("length", 32))
    mode_covariance = np.zeros((d, d))
    power = np.eye(d)
    source_time_count = length+1-2*int(losses_config.get("k", 1))
    if source_time_count < 1:
        raise ValueError("Trajectory length must admit source times for horizons 1,k,2k.")
    for _ in range(source_time_count):
        mode_covariance += power @ power.T
        power = system._linear_map @ power
    mode_covariance *= radius**2/(3*source_time_count)
    trajectory_covariance = rows @ mode_covariance @ rows.T
    scales = np.ones(m)
    scaling_records = []
    offset = 0
    for block in selection.get("blocks", []):
        size = 2 if block["kind"] == "rotation" else 1
        variance = float(np.trace(trajectory_covariance[offset:offset+size, offset:offset+size]))
        scale = math.sqrt(size/max(variance, np.finfo(float).tiny))
        scales[offset:offset+size] = scale
        scaling_records.append({"kind": block["kind"], "coordinates": list(range(offset, offset+size)),
                                "population_mean": [0.]*size, "raw_total_variance": variance,
                                "fixed_scalar": scale, "target_total_variance": size})
        offset += size
    scaled_covariance = scales[:, None]*trajectory_covariance*scales[None, :]
    scaled_initial = scales[:, None]*raw_covariance*scales[None, :]
    covariance_min = float(np.linalg.eigvalsh(scaled_initial).min()) if m else 0.
    trajectory_floor = float(np.linalg.eigvalsh(scaled_covariance).min()) if m else 0.
    singular_min = float(np.linalg.svd(scales[:, None]*jac, compute_uv=False).min()) if m else 0.
    a_matrix = system._linear_map
    # Normal block diagonal A is the default. An explicit nonnormal extension
    # has an additional shift norm bounded by its declared coefficient.
    if system.family == "oscillatory":
        normal_norm = max(math.hypot(a_matrix[0, 0], a_matrix[1, 0]),
                          max(np.abs(np.diag(a_matrix)[2:]), default=0.))
    else:
        normal_norm = float(np.max(np.abs(np.diag(a_matrix))))
    matrix_norm = normal_norm + abs(system.linear_extension_strength)
    matrix_norm *= 1+128*d*np.finfo(float).eps
    c = abs(system.coupling)
    detail_shift = np.zeros((d, d))
    detail_shift[np.arange(2, d), np.arange(1, d-1)] = 1.
    def norm_upper(matrix):
        return math.sqrt(float(np.max(np.sum(abs(matrix), axis=0))) *
                         float(np.max(np.sum(abs(matrix), axis=1))))
    # The following refinement uses only the fine-supported derivative. For
    # B=Ldetail diag(-tanh(s)^2), AB and B'A are bounded separately.
    detail_interaction_bound = min(2*matrix_norm,
                                  norm_upper(a_matrix@detail_shift)+norm_upper(detail_shift@a_matrix))
    detail_interaction_bound += 128*d*np.finfo(float).eps*max(1., matrix_norm)
    infinity_norm = float(np.max(np.sum(abs(a_matrix), axis=1)))
    global_lipschitz = matrix_norm*(1+c)/(1-c) if c < 1 else None
    box_global_lipschitz = infinity_norm*(1+c)/(1-c) if c < 1 else None
    if box_global_lipschitz is not None and box_global_lipschitz < 1:
        state_radius = radius/(1-c)
        invariant_domain = {"kind": "convex_state_box", "radius": state_radius,
                            "image_radius_upper_bound": float(box_global_lipschitz*state_radius),
                            "invariance_proof": "global induced-infinity Lipschitz <1 and f(0)=0"}
        domain_invariant = True
    else:
        state_radius = radius*math.sqrt(d)/(1-c) if c < 1 else radius*math.sqrt(d)
        invariant_domain = {"kind": "convex_state_euclidean_ball", "radius": state_radius,
                            "image_radius_upper_bound": float(global_lipschitz*state_radius) if global_lipschitz is not None else None,
                            "invariance_proof": "global Euclidean Lipschitz <1 and f(0)=0, when certified"}
        domain_invariant = global_lipschitz is not None and global_lipschitz < 1
    contraction = None
    if c < 1 and domain_invariant:
        contraction = matrix_norm+c/(1-c)*detail_interaction_bound*math.tanh(state_radius)**2
        contraction *= 1+128*d*np.finfo(float).eps
    selected_min = min((abs(complex(v["real"], v["imag"])) for v in selection.get("selected_eigenvalues", [])),
                       default=0.)
    selected_min -= float(selection.get("certificate", {}).get("diagonal_coordinate_error_bound", 0.))
    inverse_margin = selected_min-contraction**2 if contraction is not None else None
    cert = selection.get("certificate", {})
    parameter_margin = float(cert.get("parameterization_margin", cert.get("certificate_margin", 1e-5)))
    all_truth_eigenvalues = np.linalg.eigvals(a_matrix.astype(np.complex128))
    used = set()
    allowed_checks = []
    for block in selection.get("blocks", []):
        center, block_radius = abs(float(block["center"])), float(block["radius"])
        margin = min(parameter_margin, .1*block_radius)
        inverse_floor = contraction**2 if contraction is not None else 0.
        low = max(center-block_radius+margin, inverse_floor+margin, margin)
        high = min(center+block_radius-margin, 1.-margin)
        estimated = block["eigenvalues"][0]
        target = complex(estimated["real"], abs(estimated["imag"]))
        candidates = [i for i in range(d) if i not in used]
        index = min(candidates, key=lambda i: abs(all_truth_eigenvalues[i]-target))
        value = all_truth_eigenvalues[index]
        used.add(index)
        polar_ok = low < abs(value) < high
        if block["kind"] == "rotation":
            opposite = min((i for i in range(d) if i not in used),
                           key=lambda i: abs(all_truth_eigenvalues[i]-np.conj(value)))
            used.add(opposite)
            angle_distance = abs(float(np.angle(value))-float(block["angle"]))
            angular_halfwidth = .9*float(block["angle_radius"])
            polar_ok = polar_ok and angle_distance < angular_halfwidth
        else:
            angle_distance, angular_halfwidth = 0., 0.
            polar_ok = polar_ok and abs(value.imag) < 1e-12 and np.sign(value.real) == np.sign(float(block["center"]))
        disk_distance = abs(value-target)
        cartesian_ok = disk_distance < float(block.get("certificate_contour_radius", block_radius))
        allowed_checks.append({"kind": block["kind"], "reference_eigenvalue": {"real": float(value.real), "imag": float(value.imag)},
                               "actual_inner_modulus_bounds": [low, high], "parameterization_margin": margin,
                               "reference_angle_distance": angle_distance, "actual_inner_angle_halfwidth": angular_halfwidth,
                               "in_actual_polar_parameter_domain": bool(polar_ok),
                               "reference_cartesian_disk_distance": float(disk_distance),
                               "in_certified_cartesian_disk": bool(cartesian_ok),
                               "reference_in_allowed_K_domain": bool(polar_ok and cartesian_ok and contraction is not None)})
    scale_penalties = []
    for item in scaling_records:
        coordinates = item["coordinates"]
        total_variance = float(np.trace(scaled_covariance[np.ix_(coordinates, coordinates)]))
        item["scaled_total_variance"] = total_variance
        scale_penalties.append((total_variance-len(coordinates))**2)
    reference_scale = float(np.mean(scale_penalties)) if scale_penalties else None
    reference_rank = max(float(losses_config.get("kappa", .1))-singular_min, 0.)**2
    reference_floor = float(np.mean(np.maximum(.2-np.sqrt(np.maximum(np.linalg.eigvalsh(scaled_covariance), 0.)+1e-6), 0.)**2)) if m else None
    higher_cost = (float(losses_config.get("beta", 1.))*reference_rank +
                   float(losses_config.get("gamma", 1.))*reference_scale + reference_floor) if m else None
    return {"scope": "truth compatibility only; not learned-model certification",
            "initial_distribution": "independent uniform linear-mode coordinates",
            "coarse_scaling": scales.tolist(), "block_scaling": scaling_records,
            "scaling_rule": "population zero means; each real variance one; each rotation total variance two; no floor-driven rescaling",
            "source_training_time_indices": [0, source_time_count-1],
            "covariance_min_eigenvalue_initial": covariance_min,
            "covariance_min_eigenvalue_trajectory_mixture": trajectory_floor,
            "floor_standard_deviation": .2,
            "initial_floor_compatible": covariance_min + 1e-6 >= .2**2,
            "trajectory_floor_compatible": trajectory_floor + 1e-6 >= .2**2,
            "local_state_jacobian_sigma_min": singular_min,
            "rank_kappa": float(losses_config.get("kappa", .1)),
            "local_rank_compatible": m > 0 and singular_min >= float(losses_config.get("kappa", .1)),
            "linear_mode_operator_norm": float(matrix_norm),
            "linear_mode_contraction_certified": bool(matrix_norm < 1.),
            "state_jacobian_at_zero": a_matrix.tolist(),
            "state_jacobian_normal_by_construction": not system.nonnormal_extension,
            "state_jacobian_normality_residual_fro": float(np.linalg.norm(a_matrix@a_matrix.T-a_matrix.T@a_matrix)),
            "normality_residual_evidence": "floating-point diagnostic; main normality claim follows from block construction and Dphi(0)=I",
            "coarse_mechanism_preserved_when_coupling_changes": not system.nonnormal_extension,
            "coupling_change_scope": "only indices >=2 nonlinear conjugacy; first two physical/coarse coordinates identical" if not system.nonnormal_extension else "explicit nonnormal linear extension; outside normal-J main theory",
            "global_state_lipschitz_upper_bound": float(global_lipschitz) if global_lipschitz is not None else None,
            "reference_contraction_upper_bound": float(contraction) if contraction is not None else None,
            "spectral_spread_order": 2,
            "inverse_contraction_margin": float(inverse_margin) if inverse_margin is not None else None,
            "inverse_contraction_compatible": bool(contraction is not None and contraction < 1 and inverse_margin > 0),
            "reference_in_allowed_K_domain": bool(allowed_checks and all(item["reference_in_allowed_K_domain"] for item in allowed_checks)),
            "reference_K_domain_checks": allowed_checks,
            "reference_regularizer_comparison": {"rank": reference_rank, "scale": reference_scale, "floor": reference_floor,
                "weighted_higher_regularizer_cost": higher_cost,
                "coarse_prediction_cost_clean_observations": 0. if m else None,
                "lower_SIGReg": None, "full_joint_comparison_cost": None,
                "reason_full_cost_unavailable": "No complete finite-network E/F/G comparison solution with declared norm budgets and lower SIGReg has been certified.",
                "population_moment_scope": "uniform source initial modes, equally weighted source training times; minibatch scale/floor costs need not equal population costs",
                "all_losses_zero_claim": False},
            "state_euclidean_global_contraction_certified": bool(contraction is not None and contraction < 1),
            "contraction_domain": invariant_domain, "domain_invariant": bool(domain_invariant),
            "detail_interaction_norm_upper_bound": detail_interaction_bound,
            "contraction_note": "Analytic Jacobian upper bound on the declared convex invariant state domain; no scan-based global assertion.",
            "contraction_bound_formula": "||A||2 + |c|/(1-|c|)*min(2||A||2,||A Ldetail||2+||Ldetail A||2)*tanh(S)^2",
            "covariance_scope": "population covariance of uniform initial modes, mixed equally over source training times t=0..length-2k; finite-sample values are diagnostics",
            "rank_scope": "at state fixed point only", "truth_used": True}
