"""Fixed analytic matrix checks. No PRNG, samples, scientific arrays or models."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHECKS = []


def record(name, passed, **values):
    CHECKS.append({"name": name, "passed": bool(passed), **values})
    if not passed:
        raise AssertionError(name)


def near(left, right, tol=2e-12):
    return np.max(np.abs(np.asarray(left) - np.asarray(right))) <= tol


def rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, s], [-s, c]], dtype=float)


def block_rotation(theta1, theta2):
    out = np.zeros((4, 4))
    out[:2, :2] = rotation(theta1)
    out[2:, 2:] = rotation(theta2)
    return out


def exact_correction(M, T):
    E = np.eye(len(M)) - T
    sy = np.diag(M)
    sz = np.diag(T @ M @ T.T)
    R = np.diag(E @ M @ E.T)
    C = np.diag(M @ E.T)
    return {"SY": sy, "SZ": sz, "R": R, "C": C,
            "correction": sz - sy, "state_error2": np.diag(E @ E.T),
            "state_Y_error_cov": np.diag(E.T)}


# 1. Round2 variance correction exactly cancels a pure affine row scaling.
scale_results = []
for t in [-0.5, 0.25, 2.0]:
    V, SX = 1.7, 0.65
    AX = V * SX
    c0, vr = t * V, t * t * V
    cdelta, rdelta = t * AX, t * t * AX
    numerator = 2 * cdelta + rdelta - SX * (2 * c0 + vr)
    denominator = V + 2 * c0 + vr
    sy = (AX + 2 * cdelta + rdelta) / denominator
    scale_results.append({"t": t, "signed_numerator": numerator,
                          "positive_variance_denominator": denominator, "score": sy})
    record("scale_normalization_cancellation_" + str(t),
           abs(numerator) < 2e-12 and denominator > 0 and abs(sy - SX) < 2e-12,
           **scale_results[-1])


# 2. Same static L2 discrepancy permits opposite signed dynamic corrections.
M = np.array([[0.8, 0.4, 0, 0], [0.4, 1.2, 0, 0],
              [0, 0, 0.85, 0], [0, 0, 0, 1.8]], dtype=float)
A = np.eye(4) - M / 2
innovation = np.eye(4) - A @ A.T
record("signed_ambiguity_moments_are_stationary_realizable",
       np.linalg.eigvalsh(innovation).min() > 0 and np.linalg.eigvalsh(M).min() > 0 and
       np.linalg.eigvalsh(M).max() < 4,
       increment_eigenvalues=np.linalg.eigvalsh(M).tolist(),
       innovation_eigenvalues=np.linalg.eigvalsh(innovation).tolist())

ambiguous = []
for theta in [0.1, -0.1]:
    T = np.eye(4)
    T[:2, :2] = rotation(theta)
    values = exact_correction(M, T)
    gap = float(values["SZ"][2] - values["SZ"][0])
    statics = values["state_error2"]
    record("signed_identity_and_normalization_" + str(theta),
           near(values["correction"], values["R"] - 2 * values["C"]) and
           near(statics, 2 * values["state_Y_error_cov"]) and
           near(statics[:2], [2 * (1 - math.cos(0.1))] * 2),
           theta=theta, state_axis_error2=statics.tolist(),
           signed_dynamic_C=values["C"].tolist(), dynamic_R=values["R"].tolist(),
           score=values["SZ"].tolist(), pair_0_to_2_gap=gap)
    ambiguous.append({"theta": theta, "gap": gap,
                      "correction": values["correction"], "eta2": statics})

g_fine = float(M[2, 2] - M[0, 0])
direction = np.array([-1.0, 0, 1, 0])
corrections = np.array([row["correction"] for row in ambiguous])
support_ab = float(np.max(corrections @ (-direction)))
support_ba = float(np.max(corrections @ direction))
lower = g_fine - support_ab
upper = g_fine + support_ba
record("support_function_bounds_keep_both_signed_possibilities",
       lower < 0 < upper and near(lower, min(row["gap"] for row in ambiguous)) and
       near(upper, max(row["gap"] for row in ambiguous)) and
       near(ambiguous[0]["eta2"], ambiguous[1]["eta2"]),
       fine_pair_gap=g_fine, robust_lower=lower, robust_upper=upper,
       note="Moment ambiguity only, not a claim of two full-rank equivalent generative laws.")


# 3. A genuine PSD stationary temporal Gram for explicit candidate functions.
theta = 0.1
T = np.eye(4)
T[:2, :2] = rotation(theta)
F = np.vstack([np.eye(4), T])
G0 = F @ F.T
Gcross = F @ A @ F.T
Gh = np.block([[G0, Gcross], [Gcross.T, G0]])
Dop = np.hstack([-np.eye(8), np.eye(8)])
increment_gram = Dop @ Gh @ Dop.T
expected_increment_gram = F @ M @ F.T
linear_pair_gap = increment_gram[6, 6] - increment_gram[4, 4]
direct_pair_gap = (T @ M @ T.T)[2, 2] - (T @ M @ T.T)[0, 0]
record("joint_state_temporal_PSD_and_linear_gap_objective",
       np.linalg.eigvalsh(G0).min() > -2e-12 and np.linalg.eigvalsh(Gh).min() > -2e-12 and
       near(np.diag(G0), np.ones(8)) and near(Gh[:8, :8], G0) and
       near(Gh[8:, 8:], G0) and near(increment_gram, expected_increment_gram) and
       near(linear_pair_gap, direct_pair_gap),
       state_eigenvalue_min=float(np.linalg.eigvalsh(G0).min()),
       temporal_eigenvalue_min=float(np.linalg.eigvalsh(Gh).min()),
       linear_pair_gap=float(linear_pair_gap), direct_pair_gap=float(direct_pair_gap))


# 4. Pair-ellipsoid support can cancel a common uncertain component exactly.
ell = np.array([2.0, -1.0, -2.0, 1.0])
shared = np.array([1.0, 0.0, 1.0, 0.0])
Gamma = np.outer(shared, shared)
w_hat = np.array([0.0, 0.01, 0.0, 0.01])
radius = 0.02
joint_width = radius * math.sqrt(max(0, float(ell @ Gamma @ ell)))
box_width = radius * float(np.sum(np.abs(ell) * np.sqrt(np.diag(Gamma))))
record("joint_ellipsoid_pair_cancellation",
       np.linalg.eigvalsh(Gamma).min() > -2e-12 and near(ell @ shared, 0) and
       joint_width == 0 and box_width > 0,
       correction_center=float(ell @ w_hat), joint_width=joint_width,
       coordinatewise_box_width=box_width,
       note="Abstract confidence-envelope identity; coverage is an independent assumption.")


# 5. Block-internal rotations: every angle preserves the score matrix.
Mblock = np.diag([0.2, 0.2, 1.6, 1.6])
slow_projector = np.diag([1, 1, 0, 0])
record("all_angle_block_invariance_algebra",
       near(Mblock, 0.2 * slow_projector + 1.6 * (np.eye(4) - slow_projector)),
       reason="Any orthogonal block-diagonal T commutes with the block-scalar M.")
for angles in [(math.pi / 4, math.pi / 4), (0, 0.4), (0.2, 0.7)]:
    T = block_rotation(*angles)
    values = exact_correction(Mblock, T)
    record("block_rotation_fixed_fixture_" + str(angles),
           near(T @ T.T, np.eye(4)) and near(T @ Mblock, Mblock @ T) and
           near(values["correction"], np.zeros(4)) and
           near(values["R"], 2 * values["C"]),
           angles=list(angles), correction=values["correction"].tolist(),
           R=values["R"].tolist(), C=values["C"].tolist())

T = block_rotation(math.pi / 4, math.pi / 4)
values = exact_correction(Mblock, T)
eta2 = 2 - math.sqrt(2)
eta = math.sqrt(eta2)
exact_R_CS = 2 * np.sqrt(values["SY"] * values["R"]) + values["R"]
K = 4.0
norm_box = 2 * np.sqrt(K * values["SY"]) * eta + K * eta2
gap = 1.6 - 0.2
record("strict_gain_over_even_exact_R_Cauchy_bound",
       near(values["R"], values["SY"] * eta2) and
       near(values["C"], 0.5 * values["SY"] * eta2) and
       exact_R_CS[0] + exact_R_CS[2] > gap and
       norm_box[0] + norm_box[2] > gap and near(values["correction"], np.zeros(4)),
       axis_L2_eta=eta, axis_L2_eta2=eta2, robust_signed_lower_gap=gap,
       exact_R_CS_pair_budget=float(exact_R_CS[0] + exact_R_CS[2]),
       norm_box_pair_budget=float(norm_box[0] + norm_box[2]),
       note="Conditional block-class selection, not individual-axis recovery or actual Fine evidence.")


# Source identities are Markdown only. Never inspect arrays/checkpoints here.
sources = [
    "reports/stage_a_plus/round6/ROUND6_SCOPE.md",
    "reports/stage_a_plus/round3/AFFINE_LOCAL_THEORY.md",
    "reports/stage_a_plus/round3/ROUND3_REPORT.md",
    "reports/stage_a_plus/round2/NONLINEAR_THEORY_REVIEW.md",
    "reports/stage_a_plus/round5/TRANSPORT_AND_SELECTION.md",
]
source_hashes = [
    {"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
    for name in sources
]

result = {
    "scope": "Hand-specified covariance/temporal-Gram matrices and exact formulas only.",
    "scientific_execution": {
        "scientific_arrays_read": False, "model_or_checkpoint_load": False,
        "NN_forward": False, "fitting": False, "PRNG": False,
        "samples_or_trajectories_generated": False, "server_or_GPU": False,
        "SDP_solver_run": False,
    },
    "input_markdown_sha256": source_hashes,
    "all_passed": all(row["passed"] for row in CHECKS),
    "check_count": len(CHECKS),
    "checks": CHECKS,
}
path = HERE / "CORRELATION_SELECTION_CHECKS.json"
path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"all_passed": result["all_passed"], "check_count": len(CHECKS),
                  "output": str(path)}, ensure_ascii=False))
