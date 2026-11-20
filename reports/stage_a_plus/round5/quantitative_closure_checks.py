"""Fixed Gaussian polynomial identities only; no data, model, PRNG or samples."""

from __future__ import annotations

import json
import math
from pathlib import Path


D = 4
ZERO = (0,) * D


def const(x):
    return {ZERO: float(x)} if x else {}


def variable(i):
    power = [0] * D
    power[i] = 1
    return {tuple(power): 1.0}


def add(*polynomials):
    out = {}
    for polynomial in polynomials:
        for power, value in polynomial.items():
            out[power] = out.get(power, 0.0) + value
    return {power: value for power, value in out.items() if value != 0}


def scale(polynomial, value):
    return {power: value * coefficient for power, coefficient in polynomial.items()}


def mul(left, right):
    out = {}
    for pa, ca in left.items():
        for pb, cb in right.items():
            power = tuple(a + b for a, b in zip(pa, pb))
            out[power] = out.get(power, 0.0) + ca * cb
    return {power: value for power, value in out.items() if value != 0}


def power(polynomial, n):
    out = const(1)
    for _ in range(n):
        out = mul(out, polynomial)
    return out


def gaussian_moment(n, variance):
    if n % 2:
        return 0.0
    out = 1.0
    for k in range(1, n, 2):
        out *= k * variance
    return out


def mean(polynomial, variances=None):
    variances = variances or [1.0] * D
    result = 0.0
    for powers, coefficient in polynomial.items():
        term = coefficient
        for n, variance in zip(powers, variances):
            term *= gaussian_moment(n, variance)
        result += term
    return result


def inner(left, right, variances=None):
    return mean(mul(left, right), variances)


def norm2(polynomial):
    return max(0.0, inner(polynomial, polynomial))


V = [variable(i) for i in range(D)]
H2 = [add(power(v, 2), const(-1)) for v in V]
BASIS = [const(1)] + V + H2
BASIS_NORMS = [1.0] + [1.0] * D + [2.0] * D


def project(polynomial):
    return add(*[
        scale(basis, inner(polynomial, basis) / basis_norm)
        for basis, basis_norm in zip(BASIS, BASIS_NORMS)
    ])


def normalize(polynomial):
    centered = add(polynomial, const(-mean(polynomial)))
    return scale(centered, 1 / math.sqrt(norm2(centered)))


def hermite(i, degree):
    t = scale(V[i], math.sqrt(2))
    if degree == 1:
        return t
    if degree == 2:
        return add(power(t, 2), const(-1))
    if degree == 3:
        return add(power(t, 3), scale(t, -3))
    if degree == 4:
        return add(power(t, 4), scale(power(t, 2), -6), const(3))
    raise ValueError(degree)


CHECKS = []


def check(name, passed, **details):
    CHECKS.append({"name": name, "passed": bool(passed), **details})
    if not passed:
        raise AssertionError(name)


def close(actual, expected, tolerance=2e-10):
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


# 1. Full polynomial coefficient identities under one-/two-coordinate tilts.
ells = [0.31, -0.27, 0.19, 0.43]
aa = [0.07, -0.11, 0.05, 0.09]
p = add(const(-0.23), *[scale(V[i], ells[i]) for i in range(D)],
        *[scale(power(V[i], 2), aa[i]) for i in range(D)])
q = add(const(0.41), *[scale(V[i], (i - 2) / 13) for i in range(D)],
        *[scale(power(V[i], 2), (i + 1) / 17) for i in range(D)])
P = add(power(p, 2), scale(q, -1))
identity_errors = {"quartic": [], "cross_linear": [], "cubic_same": [], "cubic_mixed": []}
for i in range(D):
    variances = [1.0] * D
    variances[i] = 0.5
    got4 = inner(P, hermite(i, 4), variances)
    got3 = inner(P, hermite(i, 3), variances)
    identity_errors["quartic"].append(abs(got4 - 6 * aa[i] ** 2))
    identity_errors["cubic_same"].append(abs(got3 - 3 * math.sqrt(2) * ells[i] * aa[i]))
    for k in range(D):
        if i == k:
            continue
        pairvars = [1.0] * D
        pairvars[i] = pairvars[k] = 0.5
        got11 = inner(P, mul(hermite(i, 1), hermite(k, 1)), pairvars)
        got12 = inner(P, mul(hermite(i, 1), hermite(k, 2)), pairvars)
        identity_errors["cross_linear"].append(abs(got11 - ells[i] * ells[k]))
        identity_errors["cubic_mixed"].append(abs(got12 - math.sqrt(2) * ells[i] * aa[k]))
for name, errors in identity_errors.items():
    check("bounded_tilt_identity_" + name, max(errors) < 2e-12, max_absolute_error=max(errors))


# 2. Witness suprema from explicit extrema, not sampled grids.
root22 = math.sqrt(22)
quartic_candidates = [
    3.0,
    8 * (root22 - 4) * math.exp(-(7 - root22) / 4),
    8 * (4 + root22) * math.exp(-(7 + root22) / 4),
]
c4_exact = math.sqrt(2) / 6 * max(quartic_candidates)
c3_same = ((6 / math.e) ** 1.5 + 3 * math.sqrt(2 / math.e)) / 3
c3_mixed = 2 / math.sqrt(math.e) * max(1, 4 * math.exp(-1.25))
k_cross = 4 / math.e
check("witness_constants_and_d4_coefficient",
      c4_exact < 1 and c3_same < 2 and c3_mixed < 2,
      C4_exact=c4_exact, C4_theorem=1, K3_same_upper=c3_same,
      K3_mixed_upper=c3_mixed, K3_theorem=2, K_cross=k_cross,
      C_d4=128 + 192 / math.e ** 2)


# 3. Fixed normalized polynomials, including a nonzero degree-3 remainder.
# These coefficients describe mathematical fixtures, never a learned model.
Y = []
for i in range(D):
    other = (i + 1) % D
    h3 = add(power(V[i], 3), scale(V[i], -3))
    fixture = add(V[i], scale(V[other], 0.0003 * (i + 1)),
                  scale(H2[i], 0.0002 * (i + 1)), scale(h3, 0.00005))
    Y.append(normalize(fixture))

row_results = []
for j, y in enumerate(Y):
    pj = project(y)
    qj = project(power(y, 2))
    residual = add(y, scale(pj, -1))
    alpha2 = norm2(residual)
    alpha = math.sqrt(alpha2)
    beta = math.sqrt(norm2(add(power(y, 2), scale(qj, -1))))
    ell = [inner(y, V[i]) for i in range(D)]
    aquad = [inner(y, H2[i]) / 2 for i in range(D)]
    dominant = max(range(D), key=lambda i: abs(ell[i]))
    eps = beta + 2 * alpha
    L = 1 - alpha2 - 2 * D * eps
    Cd = 8 * D ** 2 + 16 * D * (D - 1) / math.e ** 2
    E = alpha2 + Cd * eps ** 2 / L
    amplitude_error2 = norm2(add(y, scale(V[dominant], -ell[dominant])))
    signed_error2 = norm2(add(y, scale(V[dominant], -math.copysign(1, ell[dominant]))))
    signed_bound2 = 2 * (1 - math.sqrt(1 - E))
    decomposition = alpha2 + 2 * sum(a * a for a in aquad) + sum(
        ell[i] ** 2 for i in range(D) if i != dominant)
    orth_error = max(abs(inner(residual, basis)) for basis in BASIS)
    row_results.append({
        "row": j, "alpha": alpha, "beta": beta, "epsilon": eps, "L": L,
        "dominant_source": dominant, "core_amplitude": ell[dominant],
        "amplitude_error2": amplitude_error2, "E_bound": E,
        "signed_error2": signed_error2, "signed_bound2": signed_bound2,
        "orthogonality_max_error": orth_error,
    })
    check("projection_and_finite_row_" + str(j),
          L > 0 and E < 1 and orth_error < 2e-12 and
          close(amplitude_error2, decomposition) and
          close(amplitude_error2, 1 - ell[dominant] ** 2) and
          amplitude_error2 <= E + 1e-12 and signed_error2 <= signed_bound2 + 1e-12,
          **row_results[-1])

cov = [[inner(left, right) for right in Y] for left in Y]
lambda_gershgorin = min(cov[i][i] - sum(abs(cov[i][k]) for k in range(D) if k != i)
                         for i in range(D))
max_pair_E = max(row_results[i]["E_bound"] + row_results[j]["E_bound"]
                 for i in range(D) for j in range(i))
check("noncollapse_gate_and_permutation", max_pair_E < lambda_gershgorin and
      len(set(row["dominant_source"] for row in row_results)) == D,
      covariance=cov, lambda_gershgorin=lambda_gershgorin, max_pair_E=max_pair_E)


# 4. Fixed collapse family: invertible for each t, while covariance degenerates.
collapse_results = []
for t in [0.1, 0.01, 0.001]:
    yy = [V[0], scale(add(V[0], scale(V[1], t)), 1 / math.sqrt(1 + t * t)), V[2], V[3]]
    square_residual2 = sum(norm2(add(power(y, 2), scale(project(power(y, 2)), -1))) for y in yy)
    delta = math.sqrt(square_residual2)
    lambda_min = 1 - 1 / math.sqrt(1 + t * t)
    row2_distance_if_not_source1 = 2 * (1 - t / math.sqrt(1 + t * t))
    collapse_results.append({"t": t, "span_delta": delta, "lambda_min": lambda_min,
                             "row2_distance2_if_source2": row2_distance_if_not_source1})
    check("invertible_not_quantitative_noncollapse_" + str(t),
          close(delta, 2 * t / (1 + t * t)) and lambda_min > 0 and
          row2_distance_if_not_source1 > 1.7,
          **collapse_results[-1])


# 5. Rotation gives the linear-in-delta rate obstruction.
rotation_results = []
for theta in [0.01, 0.001]:
    cosine, sine = math.cos(theta), math.sin(theta)
    yy = [add(scale(V[0], cosine), scale(V[1], sine)),
          add(scale(V[0], -sine), scale(V[1], cosine)), V[2], V[3]]
    delta2 = sum(norm2(add(power(y, 2), scale(project(power(y, 2)), -1))) for y in yy)
    axis2 = sum(norm2(add(yy[i], scale(V[i], -1))) for i in range(D))
    ratio = math.sqrt(axis2 / delta2)
    rotation_results.append({"theta": theta, "span_delta2": delta2, "axis_distance2": axis2,
                             "distance_over_span_delta": ratio})
    check("O_delta_rate_rotation_" + str(theta),
          close(delta2, 2 * math.sin(2 * theta) ** 2) and
          close(axis2, 4 * (1 - math.cos(theta))) and abs(ratio - 0.5) < 0.001,
          **rotation_results[-1])


# 6. Numerical nonempty theorem budget; this is not an actual Fine budget.
delta, lam = 0.005, 0.5
L = 1 - delta * delta - 24 * delta
E = delta * delta + 9 * (128 + 192 / math.e ** 2) * delta * delta / L
signed_eta = math.sqrt(2 * (1 - math.sqrt(1 - E)))
check("d4_nonempty_fixed_budget", L > 0 and E < 1 and 2 * E < lam,
      span_delta=delta, covariance_lambda_assumed=lam, L=L, E=E,
      signed_axis_eta=signed_eta, pair_E=2 * E)


# 7. Output renormalization changes both linear and square-statistic errors.
z = add(const(1.7), scale(Y[0], 2.3))
pz = project(z)
qz = project(power(z, 2))
mu = mean(z)
sigma = math.sqrt(norm2(add(z, const(-mu))))
y = scale(add(z, const(-mu)), 1 / sigma)
py = scale(add(pz, const(-mu)), 1 / sigma)
qy = scale(add(qz, scale(pz, -2 * mu), const(mu * mu)), 1 / sigma ** 2)
az = math.sqrt(norm2(add(z, scale(pz, -1))))
bz = math.sqrt(norm2(add(power(z, 2), scale(qz, -1))))
ay = math.sqrt(norm2(add(y, scale(py, -1))))
by = math.sqrt(norm2(add(power(y, 2), scale(qy, -1))))
check("output_gauge_error_budget", close(ay, az / sigma) and
      by <= (bz + 2 * abs(mu) * az) / sigma ** 2 + 1e-12,
      mu=mu, sigma=sigma, alpha_Z=az, beta_Z=bz, alpha_Y=ay, beta_Y=by,
      beta_Y_bound=(bz + 2 * abs(mu) * az) / sigma ** 2)


result = {
    "scope": "Deterministic hand-specified polynomial Gaussian moments and formulas only.",
    "prohibitions_observed": {
        "scientific_arrays_read": False, "checkpoint_or_model_loaded": False,
        "NN_forward": False, "fitting": False, "PRNG": False,
        "samples_or_trajectories_generated": False, "server_or_GPU": False,
    },
    "all_passed": all(item["passed"] for item in CHECKS),
    "check_count": len(CHECKS),
    "checks": CHECKS,
}
destination = Path(__file__).with_name("QUANTITATIVE_CLOSURE_CHECKS.json")
destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"all_passed": result["all_passed"], "check_count": len(CHECKS),
                  "output": str(destination)}, ensure_ascii=False))
