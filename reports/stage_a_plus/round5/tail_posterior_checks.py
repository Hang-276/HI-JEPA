#!/usr/bin/env python3
"""Deterministic probability/analytic Gaussian arithmetic; no model or samples."""
import hashlib
import json
import math
from pathlib import Path


def phi(z):
    if z == 0:
        return 1.0
    h = z - 1
    if abs(h) < 1e-3:
        return math.fsum((-1) ** n * h ** n / (n * (n - 1))
                         for n in range(2, 14))
    return z * math.log(z) - z + 1


def kappa(delta):
    """Exact inf over p>=2 of (p-1)*delta**(1/2-1/p)."""
    if delta == 0:
        return 0.0, None
    assert 0 < delta <= 1
    log_inv = -math.log(delta)
    if log_inv <= 4:
        return 1.0, 2.0
    exponent = (log_inv + math.sqrt(log_inv * (log_inv - 4))) / 2
    value = (exponent - 1) * math.exp((0.5 - 1 / exponent) * math.log(delta))
    return value, exponent


def scalar_checks():
    records = []
    for alpha in [0.25, 0.5, 0.75, 0.9]:
        c = math.log(alpha) ** 2 / phi(alpha)
        zs = [alpha, (1 + alpha) / 2, 0.99, 0.9999, 1.0,
              1.0001, 1.01, 1.5, 2.0, 10.0, 100.0]
        ratios = []
        for z in zs:
            if z < alpha:
                continue
            lhs = math.log(z) ** 2
            rhs = c * phi(z)
            assert lhs <= rhs + 1e-12
            ratios.append(lhs / rhs if rhs else 1.0)
        records.append({"alpha": alpha, "C_alpha": c,
                        "fixed_z_count": len(ratios), "max_lhs_rhs": max(ratios)})

    weighted = []
    for eps in [1 / 9, 0.05, 0.01, 0.001]:
        cap = 1 / eps
        constant = cap * math.log(cap) ** 2 / phi(cap)
        zs = [0, math.exp(-200), 1e-6, 0.01, 0.25, 0.99, 1.0,
              1.01, 2.0, cap / 2, cap]
        for z in zs:
            lhs = z * math.log(z) ** 2 if z > 0 else 0
            assert lhs <= constant * phi(z) + 1e-12
        weighted.append({"epsilon": eps, "D_epsilon": constant,
                         "fixed_z_count": len(zs)})
    return records, weighted


def vector_checks():
    uniform = [1 / 9] * 9
    cases = [
        ("identical", uniform, uniform, 0.1),
        ("small_pivot_transfer", uniform,
         [1 / 9 + 0.001, 1 / 9 - 0.001] + [1 / 9] * 7, 0.1),
        ("true_rare_pivot", [0.001] + [0.999 / 8] * 8,
         [0.02] + [0.98 / 8] * 8, 0.01),
        ("opposite_peaks", [0.6] + [0.05] * 8,
         [0.05, 0.6] + [0.05] * 7, 0.05),
    ]
    alpha = 0.5
    output = []
    for name, p, q, eps in cases:
        assert abs(math.fsum(p) - 1) < 1e-14
        assert abs(math.fsum(q) - 1) < 1e-14
        assert min(q) >= eps - 1e-15
        kl = math.fsum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
        kl = max(kl, 0.0)
        phi_sum = math.fsum(qi * phi(pi / qi) for pi, qi in zip(p, q))
        assert abs(kl - phi_sum) < 1e-13
        logdiff = [math.log(pi / qi) for pi, qi in zip(p, q)]
        odds_sq = math.fsum((x - logdiff[0]) ** 2 for x in logdiff[1:])
        good = min(pi / qi for pi, qi in zip(p, q)) >= alpha
        good_rhs = 9 * math.log(alpha) ** 2 / phi(alpha) * kl / eps
        if good:
            assert odds_sq <= good_rhs + 1e-12
        else:
            assert kl >= eps * phi(alpha) - 1e-14
        ce = -math.fsum(pi * math.log(qi) for pi, qi in zip(p, q))
        floor = -math.log1p(-8 * eps)
        assert ce >= floor - 1e-14
        assert floor > eps * phi(alpha)
        output.append({"name": name, "epsilon": eps, "true_p_min": min(p),
                       "q_min": min(q), "posterior_kl": kl, "good_set": good,
                       "odds_error_squared": odds_sq, "good_set_rhs_squared": good_rhs,
                       "bad_event_KL_threshold": eps * phi(alpha),
                       "cross_entropy": ce, "cross_entropy_floor": floor,
                       "CE_only_bad_mass_bound_saturates": True})
    return output


def rare_event_counterexample():
    records = []
    for eta in [0.1, 0.02, 0.005]:
        tiny = math.exp(-1 / eta)
        other = (1 - tiny) / 8
        kl_atom = tiny * math.log(9 * tiny) + 8 * other * math.log(9 * other)
        odds_atom_sq = 8 * (math.log1p(-tiny) - math.log(8) + 1 / eta) ** 2
        records.append({"rare_event_mass": eta, "posterior_KL_expectation": eta * kl_atom,
                        "odds_squared_expectation": eta * odds_atom_sq,
                        "asymptotic_KL_over_eta": math.log(9 / 8),
                        "asymptotic_eta_times_odds_squared": 8})
    assert records[-1]["posterior_KL_expectation"] < records[0]["posterior_KL_expectation"]
    assert records[-1]["odds_squared_expectation"] > records[0]["odds_squared_expectation"]
    return records


def analytic_gaussian_case(radius, eta):
    """4D/9-regime analytic mean/variance example; NO draws or trajectory data."""
    m, pi0, tau, alpha = 9, 1 / 9, 0.1, 0.5
    difference = tau * eta
    head_mean = tau + difference
    quadratic = tau / (2 * (1 + tau))
    variance_constant = -0.5 * math.log1p(tau)
    mean_constant = -0.5 * head_mean ** 2

    # True P has 4 shifted-mean contexts and 4 inflated-variance contexts.
    # q only changes shifted means. Joint generative KL upper-bounds posterior KL.
    kl_upper = 2 * difference ** 2 / 9
    contrast_min = min(head_mean, quadratic)
    true_rank = 8
    head_rank = 8
    assert contrast_min > 0

    # Omega is a fixed cube. A head polynomial bound certifies q>=eps there.
    logits_sup = max(head_mean * radius + abs(mean_constant),
                     quadratic * radius ** 2 + abs(variance_constant))
    eps = math.exp(-logits_sup) / (1 + 8 * math.exp(logits_sup))
    c_alpha = math.log(alpha) ** 2 / phi(alpha)
    beta = eps * phi(alpha)
    one_tail = math.erfc(radius / math.sqrt(2))
    normal_density = math.exp(-radius ** 2 / 2) / math.sqrt(2 * math.pi)
    second_tail = 2 * radius * normal_density + one_tail
    fourth_tail = 2 * (radius ** 3 + 3 * radius) * normal_density + 3 * one_tail
    cube_tail = 4 * one_tail - 6 * one_tail ** 2 + 4 * one_tail ** 3 - one_tail ** 4
    other_axes_tail = 3 * one_tail - 3 * one_tail ** 2 + one_tail ** 3

    mean_norm_sq = head_mean ** 2 + mean_constant ** 2
    variance_norm_sq = (3 * quadratic ** 2 + variance_constant ** 2
                        + 2 * quadratic * variance_constant)
    head_norm_sq = 4 * (mean_norm_sq + variance_norm_sq)
    mean_tail_sq = head_mean ** 2 * second_tail + mean_constant ** 2 * one_tail
    variance_tail_sq = (quadratic ** 2 * fourth_tail + variance_constant ** 2 * one_tail
                        + 2 * quadratic * variance_constant * second_tail)
    assert mean_tail_sq >= 0 and variance_tail_sq >= 0
    head_tail_sq = (4 * (mean_tail_sq + variance_tail_sq) * (1 - one_tail) ** 3
                    + head_norm_sq * other_axes_tail)

    mismatch_mass = min(1 - cube_tail, kl_upper / (pi0 * beta))
    total_bad = min(1.0, cube_tail + mismatch_mass)
    contraction, p_star = kappa(total_bad)
    good_sq = m * c_alpha * kl_upper / (pi0 * eps)
    head_bad_cap_sq = 8 * math.log((1 - 8 * eps) / eps) ** 2
    bad_head_sq = head_tail_sq + head_bad_cap_sq * mismatch_mass
    exact_odds_sq = 4 * (difference ** 2 + (0.5 * difference * (head_mean + tau)) ** 2)
    exact_span_sq = exact_odds_sq / head_mean ** 2
    assert contraction < 1
    odds_bound = (math.sqrt(good_sq) + contraction * math.sqrt(head_norm_sq)
                  + math.sqrt(bad_head_sq)) / (1 - contraction)
    span_bound = odds_bound / contrast_min
    assert math.sqrt(exact_odds_sq) <= odds_bound + 1e-12
    assert math.sqrt(exact_span_sq) <= span_bound + 1e-12

    ce_floor = -math.log1p(-8 * eps)
    assert ce_floor > beta
    return {
        "kind": "fixed analytic Gaussian formula, not data or a trained flow",
        "dimension": 4, "regimes": m, "tau": tau, "anchor_weight": pi0,
        "true_natural_rank": true_rank, "head_natural_rank": head_rank,
        "head_mean_relative_bias": eta, "cube_radius": radius,
        "head_epsilon_on_cube": eps, "Gaussian_anchor_tail_mass": cube_tail,
        "known_generative_joint_KL_upper_for_posterior_KL": kl_upper,
        "bad_mass_from_posterior_KL_bound": mismatch_mass,
        "total_bad_mass_upper": total_bad, "optimal_hypercontractive_exponent": p_star,
        "small_set_contraction": contraction, "good_error_squared_bound": good_sq,
        "head_odds_L2_squared": head_norm_sq,
        "head_odds_tail_energy": head_tail_sq,
        "head_bad_energy_squared_upper": bad_head_sq,
        "head_contrast_sigma_min_fixed_basis": contrast_min,
        "exact_odds_L2_error": math.sqrt(exact_odds_sq),
        "absorbed_odds_L2_bound": odds_bound,
        "exact_statistic_span_L2_error": math.sqrt(exact_span_sq),
        "statistic_span_L2_bound": span_bound,
        "span_bound_below_predeclared_0_01": span_bound < 0.01,
        "CE_only_bound_cannot_enter_absorption": True,
        "scope": "Known true parameters provide A in this arithmetic example; not an observable current-Fine guarantee.",
    }


def main():
    scalars, weighted = scalar_checks()
    gaussians = [analytic_gaussian_case(4.0, 1e-3),
                 analytic_gaussian_case(6.0, 1e-6),
                 analytic_gaussian_case(8.0, 1e-9)]
    assert gaussians[-1]["span_bound_below_predeclared_0_01"]
    threshold_checks = []
    for delta in [0.1, math.exp(-4), 0.01, 1e-4, 1e-8, 1e-14]:
        value, p = kappa(delta)
        assert abs((p - 1) * delta ** (0.5 - 1 / p) - value) < 1e-12
        threshold_checks.append({"delta": delta, "k": value, "optimal_p": p,
                                 "absorption_allowed": value < 1})
    script = Path(__file__).resolve()
    payload = {
        "scope": "Fixed vectors and exact Gaussian moments only; no physical samples/model/data files.",
        "scalar_good_set_checks": scalars, "weighted_score_checks": weighted,
        "fixed_probability_vectors": vector_checks(),
        "general_probability_counterexample": rare_event_counterexample(),
        "hypercontractive_threshold_checks": threshold_checks,
        "analytic_Gaussian_absorption_cases": gaussians,
        "all_checks_pass": True,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "execution_counts": {"random_draws": 0, "physical_trajectories_generated": 0,
                             "scientific_array_files_read": 0, "model_loads": 0,
                             "model_forwards": 0, "training_jobs": 0, "server_connections": 0,
                             "gpu_jobs": 0},
    }
    target = script.with_name("TAIL_POSTERIOR_CHECKS.json")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(target), "all_checks_pass": True,
                      "Gaussian_span_bounds": [x["statistic_span_L2_bound"] for x in gaussians]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
