"""Empirical diagnostics and explicit analytic pseudo-solution audits.

Finite samples, floating point derivatives, and numerical scans in this module
are diagnostics.  They do not certify a global identifiability theorem.
"""
from __future__ import annotations

import numpy as np
import torch
import itertools


def _array(x):
    return np.asarray(x, dtype=np.float64)


def covariance_diagnostics(values):
    """Statistics on rows; zero/missing values are not silently interchanged."""
    x = _array(values)
    if x.ndim != 2 or len(x) < 2:
        return {"available": False, "reason": "need at least two sample rows"}
    covariance = np.atleast_2d(np.cov(x, rowvar=False))
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    if total > 0:
        probabilities = eigenvalues[eigenvalues > 0] / total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    else:
        effective_rank = 0.0
    return {
        "available": True,
        "n_samples": int(len(x)),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "minimum_covariance_eigenvalue": float(eigenvalues.min()),
        "effective_rank": effective_rank,
        "floor_penalty": float(np.mean(np.maximum(.2 - np.sqrt(eigenvalues + 1e-6), 0) ** 2)),
        "mean_squared_norm": float(np.mean(np.sum(x*x, axis=1))),
        "numerical_rank": int(np.linalg.matrix_rank(covariance)),
        "evidence": "finite_sample_diagnostic",
    }


def trajectory_diagnostics(values):
    """Separate trajectory diversity from within-trajectory temporal change."""
    x = _array(values)
    if x.ndim != 3 or x.shape[1] < 2:
        return {"available": False, "reason": "need [trajectory,time,feature]"}
    result = covariance_diagnostics(x.reshape(-1, x.shape[-1]))
    independent = covariance_diagnostics(x[:, 0])
    centered = x - x.mean(axis=1, keepdims=True)
    dynamic_variance = float(np.mean(centered ** 2))
    total_variance = float(np.mean((x - x.mean(axis=(0, 1), keepdims=True)) ** 2))
    result.update({
        "independent_trajectory_initial_states": independent,
        "pooled_samples_are_temporally_correlated": True,
        "within_trajectory_variance": dynamic_variance,
        "one_step_change_mse": float(np.mean(np.diff(x, axis=1) ** 2)),
        "dynamic_fraction_of_total_variance": dynamic_variance / total_variance if total_variance > 0 else None,
        "interpretation": "small temporal change alone is not failure: true slow modes can change little",
    })
    return result


def state_jacobian(model, system):
    """Return Dh(0), differentiating through the declared state observation map."""
    parameter = next(model.parameters())
    zero = torch.zeros(system.dimension, device=parameter.device, dtype=parameter.dtype, requires_grad=True)
    def representation(state):
        observed = system.observe_torch(state.unsqueeze(0))
        return model.coarse(observed).reshape(-1)
    with torch.enable_grad():
        derivative = torch.autograd.functional.jacobian(representation, zero)
    return derivative.detach().cpu().double().numpy()


def _finite_jacobian(function, dimension, step=1e-5):
    points = np.eye(dimension, dtype=np.float64) * step
    plus, minus = _array(function(points)), _array(function(-points))
    return ((plus - minus) / (2 * step)).T


def theory_diagnostics(model, system, selection, config):
    """Local BJ-KB residuals are empirical and distinct from selection certificates."""
    try:
        b = state_jacobian(model, system)
        step = float(config.get("evaluation", {}).get("diagnostic_fd_step", 1e-5))
        j = _finite_jacobian(system.step, system.dimension, step)
        j_half = _finite_jacobian(system.step, system.dimension, step / 2)
        parameter = next(model.parameters())
        x0 = torch.as_tensor(system.observe(np.zeros((1, system.dimension))), device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            c0 = model.coarse(x0)
            k = model.K_matrix().detach().cpu().double().numpy()
            equilibrium_residual = (model.predict_coarse(c0, 1)-c0).detach().cpu().double().numpy()
        singular = np.linalg.svd(b, compute_uv=False)
        result = {
            "available": True,
            "state_representation_jacobian": b.tolist(),
            "state_dynamics_jacobian_fd": j.tolist(),
            "state_dynamics_fd_step": step,
            "fd_step_halving_difference_fro": float(np.linalg.norm(j-j_half)),
            "minimum_state_representation_singular_value": float(singular.min()),
            "rank_penalty": float(max(float(config.get("losses", {}).get("kappa", .1))-float(singular.min()), 0)**2),
            "bj_minus_kb_fro": float(np.linalg.norm(b @ j-k @ b)),
            "bj_minus_kb_relative": float(np.linalg.norm(b @ j-k @ b)/(np.linalg.norm(b @ j)+1e-12)),
            "equilibrium_prediction_residual_l2": float(np.linalg.norm(equilibrium_residual)),
            "evidence": "empirical_local_diagnostic",
            "is_global_certificate": False,
            "fd_step_halving_is_error_bound": False,
            "selection_certificate_status": selection.get("status", "unknown"),
            "selection_certificate_scope": "declared local spectral selection only; does not certify learned representation recovery",
        }
        if config.get("model", {}).get("variant") == "hierarchical":
            with torch.enable_grad():
                coarse0 = c0.detach().reshape(-1).requires_grad_(True)
                predictor_derivative = torch.autograd.functional.jacobian(
                    lambda value: model.predict_coarse(value.unsqueeze(0), 1).reshape(-1), coarse0)
            a = predictor_derivative.detach().cpu().double().numpy()
            result["bj_minus_predictor_jacobian_b_fro"] = float(np.linalg.norm(b @ j-a @ b))
            result["k_residual_applicable_to_predictor"] = False
        else:
            result["k_residual_applicable_to_predictor"] = True
        return result
    except (RuntimeError, ValueError, TypeError, AttributeError) as error:
        return {"available": False, "reason": str(error), "evidence": "empirical_local_diagnostic", "is_global_certificate": False}


def target_spectrum_diagnostics(model, selection, config=None):
    """Numerical target-domain checks; never a substitute for selection proof."""
    with torch.no_grad():
        k = model.K_matrix().detach().cpu().double().numpy()
    learned = np.linalg.eigvals(k)
    selected = [complex(value["real"], value.get("imag", 0.)) for value in selection.get("selected_eigenvalues", [])]
    match_error = None
    if len(selected)==len(learned) and len(learned)<=8:
        match_error = float(min(sum(abs(learned[i]-selected[j])**2 for i, j in enumerate(order)) for order in itertools.permutations(range(len(selected))))/max(1, len(selected)))
    blocks, start = [], 0
    for block in selection.get("blocks", []):
        size = 2 if block["kind"]=="rotation" else 1
        eigen = np.linalg.eigvals(k[start:start+size, start:start+size])
        if size==1:
            discrepancy = abs(float(eigen[0].real)-float(block["center"]))
            inside = discrepancy <= float(block["radius"])+1e-7
            details = {"center_distance": discrepancy, "radius": float(block["radius"])}
        else:
            representative = eigen[np.argmax(eigen.imag)]
            radial = abs(abs(representative)-float(block["center"]))
            angle = abs(abs(float(np.angle(representative)))-abs(float(block.get("angle", 0.))))
            inside = radial<=float(block["radius"])+1e-7 and angle<=float(block.get("angle_radius", np.pi))+1e-7
            details = {"radial_center_distance": float(radial), "angle_center_distance": angle,
                       "radius": float(block["radius"]), "angle_radius": float(block.get("angle_radius", np.pi))}
        blocks.append({"kind": block["kind"], "inside_declared_target_domain_numeric": bool(inside), **details})
        start += size
    singular = np.linalg.svd(k, compute_uv=False)
    return {"learned_eigenvalues": [{"real": float(value.real), "imag": float(value.imag)} for value in learned],
            "matched_estimated_target_spectrum_mse": match_error,
            "block_domain_checks": blocks, "all_blocks_in_domain_numeric": all(block["inside_declared_target_domain_numeric"] for block in blocks),
            "spectral_radius": float(np.max(np.abs(learned))),
            "minimum_singular_value": float(singular.min()),
            "inverse_operator_norm": float(1/singular.min()) if singular.min()>0 else None,
            "applicable_to_prediction": (config or {}).get("model", {}).get("variant")!="hierarchical",
            "evidence": "numerical_parameter_diagnostic; local target selection certificate stored separately"}


def pseudo_solution_audit(system=None, config=None):
    """Canonical analytic u^3 / repetition / resonant-mixture audit.

    This intentionally audits the explicitly stated canonical system, not an
    arbitrary supplied simulator.  ``system`` is accepted for CLI compatibility
    and its identity is recorded. Scalar/block scales use one fixed audit sample;
    no cross-mode whitening is performed. All variants share the same floor.
    """
    config = config or {}
    rng = np.random.default_rng(193)
    states = rng.uniform(-1., 1., size=(4096, 2))
    future = states * np.array([.5, .25])
    kappa = float(config.get("losses", {}).get("kappa", .1))
    cases = [
        ("true_u", lambda s: s[:, :1], np.array([[1., 0.]]), [.5], [.5]),
        ("u_cubed", lambda s: s[:, :1] ** 3, np.array([[0., 0.]]), [.125], [.5]),
        ("repeated_encoding", lambda s: np.column_stack((s[:, 0], s[:, 0])), np.array([[1., 0.], [1., 0.]]), [.5, .5], [.5, .25]),
        ("resonant_mix", lambda s: (s[:, 1]+.5*s[:, 0]**2)[:, None], np.array([[0., 1.]]), [.25], [.25]),
    ]
    rows = []
    for name, mapping, derivative, candidate_eigs, target_eigs in cases:
        raw = mapping(states)
        scales = np.sqrt(np.mean(raw**2, axis=0)).clip(1e-12)
        values, later = raw/scales, mapping(future)/scales
        b = derivative/scales[:, None]
        sigma = float(np.linalg.svd(b, compute_uv=False).min())
        statistics = covariance_diagnostics(values)
        floor = statistics["floor_penalty"]
        rank_penalty = max(kappa-sigma, 0.)**2
        outside = not np.allclose(sorted(candidate_eigs), sorted(target_eigs), atol=1e-10, rtol=0)
        floor_rejects = floor > 1e-12
        rank_rejects = rank_penalty > 1e-12
        inverse_contraction_margin = float(min(abs(value) for value in target_eigs)-.5**2)
        rows.append({
            "candidate": name,
            "candidate_eigenvalues": candidate_eigs,
            "certified_target_eigenvalues": target_eigs,
            "free_dynamics_prediction_mse": float(np.mean((later-values*np.asarray(candidate_eigs))**2)),
            "target_dynamics_prediction_mse": float(np.mean((later-values*np.asarray(target_eigs))**2)),
            "outside_target_spectral_domain": outside,
            "shared_covariance_floor_penalty": floor,
            "minimum_local_singular_value": sigma,
            "rank_penalty": rank_penalty,
            "rank_rejects": rank_rejects,
            "already_excluded_by_spectrum_or_floor": bool(outside or floor_rejects),
            "isolated_rank_benefit_demonstrated": bool(rank_rejects and not outside and not floor_rejects),
            "accepted_by_these_constraints": bool(not outside and not floor_rejects and not rank_rejects),
            "acceptance_scope": "target spectral domain, covariance floor and local rank only; not the complete strict theory assumptions",
            "accepted_by_spectrum_floor_rank": bool(not outside and not floor_rejects and not rank_rejects),
            "strict_inverse_contraction_margin": inverse_contraction_margin,
            "strict_inverse_contraction_compatible": inverse_contraction_margin>0,
            "accepted_by_spectrum_floor_rank_and_strict_inverse_condition": bool(not outside and not floor_rejects and not rank_rejects and inverse_contraction_margin>0),
        })
    return {
        "system": "u_next=0.5*u; v_next=0.25*v; fixed point zero",
        "supplied_system_metadata": system.metadata() if system is not None and hasattr(system, "metadata") else None,
        "scope": "canonical analytic counterexamples; not a trained-network experiment",
        "sample_seed": 193,
        "scale_rule": "per-coordinate RMS from fixed audit sample; no joint whitening",
        "kappa": kappa,
        "rows": rows,
        "interpretation": "u^3/repetition do not isolate rank gains when spectrum/floor already reject them; resonant mix passes spectrum/floor/rank but is on the excluded strict inverse-contraction boundary (0.25=0.5^2)",
        "certificate_claim": "analytic candidate identities only; sampled penalties are numerical diagnostics",
    }


def evaluate_cross_seed(runs):
    """Aggregate frozen-evaluation scalar metrics without fitting alignments on test."""
    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            path = prefix+str(key)
            if isinstance(item, dict):
                result.update(flatten(item, path+"."))
            elif isinstance(item, (float, int)) and not isinstance(item, bool) and np.isfinite(item):
                result[path] = float(item)
        return result
    flat = [flatten(run) for run in runs]
    metrics = {}
    for key in sorted({key for row in flat for key in row}):
        values = [row[key] for row in flat if key in row]
        metrics[key] = {"n": len(values), "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values)>1 else None,
                        "min": float(min(values)), "max": float(max(values))}
    return {"n_runs": len(runs), "metrics": metrics,
            "evidence": "cross_seed_aggregate_of_frozen_evaluations",
            "representation_alignment_fitted_on_test": False,
            "note": "dispersion of truth-calibrated recovery metrics; pairwise representation consistency requires saved calibration-fitted matches"}
