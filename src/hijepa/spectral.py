"""Local-query spectral selection with explicit perturbation/count certificates.

No true eigenvalues or labels are accessed. A finite-difference enclosure for
J(0) is transformed into a verified approximate eigenbasis. Disjoint circular
contours certify multiplicities by homotopy from an exact diagonal matrix.
"""
from __future__ import annotations

import math
import numpy as np


def _complex_json(z):
    return {"real": float(np.real(z)), "imag": float(np.imag(z))}


def _inf(a):
    value = float(np.max(np.sum(np.abs(a), axis=1)))
    # Inflate abs/summation roundoff; matrices are small and finite here.
    n = a.shape[1]
    u = np.finfo(np.float64).eps
    return value/(1-8*max(1, n)*u)


def _roundoff_product(a, b):
    # Complex dot products use at most 8n real arithmetic operations per term.
    n = a.shape[1]
    u = np.finfo(np.float64).eps
    gamma = 8*n*u/(1-8*n*u)
    return gamma*_inf(np.abs(a) @ np.abs(b))/(1-gamma)


def select_spectrum(system, selection_config, train_split=None):
    cfg = dict(selection_config)
    d = int(system.dimension)
    h = float(cfg.get("fd_step", 1e-4))
    local_radius = float(cfg.get("local_radius", .01))
    query_noise = float(cfg.get("query_noise_bound", 0.0))
    threshold = float(cfg.get("slow_modulus_min", .75))
    cluster_tol = float(cfg.get("cluster_tolerance", .025))
    margin = float(cfg.get("certificate_margin", 1e-5))
    result = {"status": "unresolved", "valid_for_training": False, "m": 0,
              "blocks": [], "clusters": [], "selected_eigenvalues": [],
              "estimated_jacobian": None, "error_bound": None,
              "certificate": {"scope": "local observed-space Jacobian at the fixed point only",
                              "truth_labels_used": False, "train_labels_used": False,
                              "normality_assumed": False, "global_spectrum_certified": False,
                              "contours": [], "unresolved_reasons": []}}
    cert = result["certificate"]
    if h <= 0 or local_radius < h or query_noise < 0 or margin <= 0:
        cert["unresolved_reasons"].append("Invalid finite-difference radius, noise bound, or margin.")
        return result
    if not hasattr(system, "query_derivative_budget"):
        cert["unresolved_reasons"].append("No analytic C3/query arithmetic budget is available.")
        return result
    budget = system.query_derivative_budget(local_radius)
    origin = np.zeros(d, dtype=np.float64)
    eye = np.eye(d)
    jac = np.column_stack([(np.asarray(system.local_query(origin+h*eye[j]), dtype=np.float64)
                            - np.asarray(system.local_query(origin-h*eye[j]), dtype=np.float64))/(2*h)
                           for j in range(d)])
    if not np.all(np.isfinite(jac)):
        cert["unresolved_reasons"].append("Nonfinite local queries.")
        return result
    truncation = float(budget["third_derivative_inf_bound"])*h*h/6
    query_error = query_noise + float(budget["query_roundoff_bound"])
    arithmetic = 4*np.finfo(np.float64).eps*max(1., _inf(jac))
    epsilon = d*(truncation + query_error/h) + arithmetic
    result["estimated_jacobian"] = jac.tolist()
    result["error_bound"] = float(epsilon)
    cert.update({"query_center": origin.tolist(), "query_count": 2*d, "fd_step": h,
                 "certificate_margin": margin,
                 "declared_query_radius": local_radius, "matrix_error_norm": "induced infinity",
                 "matrix_error_bound": float(epsilon), "per_entry_truncation_bound": float(truncation),
                 "per_query_noise_bound": query_noise, "derivative_budget": budget,
                 "floating_point_qualification": "Conditional on stated IEEE/tanh error model; no directed-rounding interval arithmetic.",
                 "proof": "For B=V^-1 J V, ||B-D||inf<=eta. On each contour min_i|z-Dii|>eta; Neumann invertibility along D+t(B-D) preserves algebraic multiplicities."})
    try:
        values, vectors = np.linalg.eig(jac.astype(np.complex128))
        inverse = np.linalg.inv(vectors)
    except np.linalg.LinAlgError:
        cert["unresolved_reasons"].append("Estimated Jacobian has no numerically invertible eigenbasis.")
        return result
    identity_residual = _inf(inverse@vectors-np.eye(d)) + _roundoff_product(inverse, vectors)
    if identity_residual >= 1:
        cert["unresolved_reasons"].append("Could not enclose the inverse eigenbasis norm.")
        return result
    inverse_bound = _inf(inverse)/(1-identity_residual)
    residual = _inf(jac@vectors-vectors*values[None, :])
    residual += _roundoff_product(jac, vectors)
    residual += 4*np.finfo(float).eps*max(1., _inf(vectors*values[None, :]))
    eta = inverse_bound*(epsilon*_inf(vectors) + residual)
    eta += 32*d*np.finfo(float).eps*max(1., float(np.max(np.abs(values))))
    cert.update({"eigenbasis_inverse_residual_bound": float(identity_residual),
                 "eigenbasis_inverse_norm_upper_bound": float(inverse_bound),
                 "eigenbasis_condition_inf_upper_bound": float(inverse_bound*_inf(vectors)),
                 "eigenpair_residual_bound": float(residual), "diagonal_coordinate_error_bound": float(eta),
                 "estimated_eigenvalues": [_complex_json(v) for v in values]})
    if not np.isfinite(eta):
        cert["unresolved_reasons"].append("Unbounded perturbation in diagonal coordinates.")
        return result
    selected = []
    for i, value in enumerate(values):
        lower, upper = abs(value)-eta, abs(value)+eta
        if lower > threshold + margin:
            selected.append(i)
        elif upper >= threshold-margin:
            cert["unresolved_reasons"].append("A spectral uncertainty disk crosses the slow-mode threshold.")
    if cert["unresolved_reasons"]:
        return result
    if not selected:
        cert["unresolved_reasons"].append("No complete slow spectral cluster was selected.")
        return result
    selected_set = set(selected)
    remaining = set(range(d))
    spectral_components = []
    # Connected components of inflated Gershgorin disks carry full multiplicity.
    adjacency_radius = 4*eta + 2*margin
    while remaining:
        component, frontier = set(), {min(remaining)}
        while frontier:
            i = frontier.pop()
            if i in component:
                continue
            component.add(i)
            frontier.update(j for j in remaining if j not in component and abs(values[i]-values[j]) <= adjacency_radius)
        remaining -= component
        spectral_components.append(sorted(component))
    # Certify individual blocks only when their real/complex type is robust.
    block_records = []
    used = set()
    for i in sorted(selected, key=lambda j: (-abs(values[j]), -values[j].imag)):
        if i in used:
            continue
        value = values[i]
        real_like = abs(value.imag) < max(eta*.1, 1e-12)
        if real_like:
            members = [i]
            nearest = min((abs(value-values[j]) for j in range(d) if j != i), default=2.)
            if nearest <= 2*eta + margin:
                cert["unresolved_reasons"].append("A real eigenvalue multiplicity/type cannot be resolved into a certified real block.")
                continue
            contour_radius = min(nearest*.35, max(4*eta, 4*margin, .002))
            center = complex(value.real, 0.)
            block = {"kind": "real", "eigenvalues": [_complex_json(center)],
                     "center": float(value.real), "radius": float(contour_radius*.8)}
        else:
            if value.imag < 0:
                continue
            pair = min((j for j in range(d) if j != i), key=lambda j: abs(values[j]-np.conj(value)))
            if pair not in selected_set or abs(values[pair]-np.conj(value)) > max(eta, 1e-10):
                cert["unresolved_reasons"].append("A conjugate pair is incomplete or unresolved.")
                continue
            members = [i, pair]
            nearest = min((abs(value-values[j]) for j in range(d) if j != i), default=2.)
            contour_radius = min(nearest*.35, max(4*eta, 4*margin, .002))
            center = value
            # Polar rectangle stays strictly inside the Cartesian disk:
            # |delta r| + 2*r*sin(|delta theta|/2) <= contour_radius/2.
            radial = contour_radius/4
            angular = 2*math.asin(min(.25, contour_radius/(8*abs(value))))
            block = {"kind": "rotation", "eigenvalues": [_complex_json(value), _complex_json(np.conj(value))],
                     "center": float(abs(value)), "radius": float(radial),
                     "angle": float(np.angle(value)), "angle_radius": float(angular)}
        # Circle avoids all diagonal values; hence all homotopy matrices too.
        distances = [abs(center-values[j]) for j in range(d)]
        lower_resolvent = min(abs(distance-contour_radius) for distance in distances)
        inside = [j for j, distance in enumerate(distances) if distance < contour_radius]
        if lower_resolvent <= eta + margin or inside != [i] or contour_radius <= eta:
            cert["unresolved_reasons"].append("An isolated block contour lacks sufficient resolvent margin.")
            continue
        for member in members:
            member_center = values[member]
            cert["contours"].append({"center": _complex_json(member_center), "radius": float(contour_radius),
                                      "multiplicity": 1, "algebraic_multiplicity": 1,
                                      "resolvent_margin": float(lower_resolvent-eta),
                                      "proof_scope": "J(0) spectrum under local-query error model"})
        block["certificate_contour_radius"] = float(contour_radius)
        block["spectral_uncertainty_radius"] = float(eta)
        block_records.append((members, block))
        used.update(members)
    if used != selected_set or cert["unresolved_reasons"]:
        return result
    output_indices, blocks, offset = {}, [], 0
    for members, block in block_records:
        blocks.append(block)
        for member in members:
            output_indices[member] = offset
            offset += 1
    # Evaluation clusters join conjugate blocks and nearby real eigenvalues.
    coordinate_clusters, unassigned = [], set(selected)
    while unassigned:
        component, frontier = set(), {min(unassigned)}
        while frontier:
            i = frontier.pop()
            if i in component:
                continue
            component.add(i)
            for j in unassigned-component:
                same_pair = abs(values[i]-np.conj(values[j])) <= max(eta, 1e-10)
                if abs(values[i]-values[j]) <= cluster_tol or same_pair:
                    frontier.add(j)
        unassigned -= component
        coordinate_clusters.append(sorted(output_indices[j] for j in component))
    selected_values = [z for block in blocks for z in block["eigenvalues"]]
    minimum_modulus = min(abs(complex(z["real"], z["imag"]))-eta for z in selected_values)
    maximum_modulus = max(abs(complex(z["real"], z["imag"]))+eta for z in selected_values)
    excluded = [abs(values[j])+eta for j in range(d) if j not in selected_set]
    cert.update({"selected_multiplicity": offset, "total_multiplicity": d,
                 "excluded_multiplicity_by_complement": d-offset,
                 "selected_modulus_lower_bound": float(minimum_modulus),
                 "selected_modulus_upper_bound": float(maximum_modulus),
                 "excluded_modulus_upper_bound": float(max(excluded, default=0.)),
                 "all_multiplicities_accounted_for": sum(len(c) for c in spectral_components) == d,
                 "complete_conjugate_pairs": True,
                 "unresolved_reasons": [], "global_normality_status": "not asserted",
                 "global_target_cluster_status": "not asserted by local queries"})
    result.update({"status": "certified", "valid_for_training": True, "m": offset,
                   "blocks": blocks, "clusters": coordinate_clusters,
                   "selected_eigenvalues": selected_values,
                   "selected_modulus_lower_bound": float(minimum_modulus)})
    return result
