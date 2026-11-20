"""Deterministic checks for the directional rank thresholds. No draws or training."""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
LAGS = (4, 16, 32)


def population_check():
    rho = np.array([0.93, 0.9051, 0.9049, 0.88])
    physical = 2 * (1 - np.mean(rho[None] ** np.array(LAGS)[:, None], axis=0))
    gamma = physical[2] - physical[1]
    limit = np.sqrt(gamma / (physical[1] - physical[0]))
    rows = []
    for donor in (0, 3):
        for a in (-0.10, 0.10):
            adjusted = physical.copy()
            adjusted[2] = (physical[2] + a**2 * physical[donor]) / (1 + a**2)
            result = sorted(np.argsort(adjusted)[:2].tolist())
            expected = [0, 2] if donor == 0 else [0, 1]
            assert result == expected
            rows.append({
                "donor_one_based": donor + 1,
                "amplitude": a,
                "axis_mse": 2 - 2 / np.sqrt(1 + a**2),
                "named_top2_one_based": [axis + 1 for axis in result],
                "signed_gap": float(adjusted[2] - adjusted[1]),
            })
    return {
        "rho": rho.tolist(),
        "physical_scores": physical.tolist(),
        "slow_donor_threshold": float(limit),
        "arms": rows,
    }


def window_check():
    points = np.arange(193)
    k = np.zeros((len(points), len(points)))
    for lag in LAGS:
        difference = np.eye(len(points))[lag:] - np.eye(len(points))[:-lag]
        k += difference.T @ difference / (3 * (len(points) - lag))
    z = np.sin(points / 8.5) + 0.2 * np.cos(points / 3.1)
    other = np.sin(points / 8.5 + 0.1) + 0.15 * np.cos(points / 3.3)
    competitor = np.sin(points / 8.5 + 0.3)
    competitor *= np.sqrt(0.98 * (z @ k @ z) / (competitor @ k @ competitor))
    d_j, d_i, d_v = (path @ k @ path for path in (z, competitor, other))
    gamma = d_j - d_i
    cross = z @ k @ other
    q = d_v - d_i
    a = 0.10
    direct = np.array([
        (z + sign * a * other) @ k @ (z + sign * a * other) / (1 + a**2) - d_i for sign in (1, -1)
    ])
    formula = (gamma + a**2 * q + np.array([1, -1]) * 2 * a * cross) / (1 + a**2)
    np.testing.assert_allclose(direct, formula, atol=1e-14)
    assert direct[0] > 0 > direct[1]
    temporal_z = 0.9049 ** np.abs(points[:, None] - points[None, :])
    temporal_v = 0.93 ** np.abs(points[:, None] - points[None, :])
    expected_variance = np.trace(k @ temporal_v @ k @ temporal_z)
    root_z = np.linalg.cholesky(temporal_z)
    root_v = np.linalg.cholesky(temporal_v)
    equivalent_variance = np.sum((root_z.T @ k @ root_v)**2)
    np.testing.assert_allclose(expected_variance, equivalent_variance, rtol=1e-14)
    return {
        "true_pair_gap": float(gamma),
        "signed_cross": float(cross),
        "a_plus_then_a_minus_pair_gaps": direct.tolist(),
        "same_sign_flip_probability_claim": "symmetry theorem, not a simulation estimate",
        "independent_ar_cross_variance": float(expected_variance),
        "trace_vs_factorized_variance_difference": float(expected_variance - equivalent_variance),
    }


def general_check():
    state_matrix = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, -0.1], [0.3, -0.1, 1.0]])
    dynamic_matrix = np.array([[1.3, 0.22, 0.31], [0.22, 1.1, -0.11], [0.31, -0.11, 1.02]])
    assert min(np.linalg.eigvalsh(state_matrix)) > 0
    assert min(np.linalg.eigvalsh(dynamic_matrix)) > 0
    a = -0.20
    vector = np.array([1.0, 0.0, a])
    learned = vector + np.array([0.01, -0.04, 0.03])
    other = np.array([-0.02, 1.01, 0.04])
    d_j, d_i, d_v = np.diag(dynamic_matrix)
    c, r = dynamic_matrix[0, 2], state_matrix[0, 2]
    v0, n0 = vector @ state_matrix @ vector, vector @ dynamic_matrix @ vector
    v, n = learned @ state_matrix @ learned + 0.007, learned @ dynamic_matrix @ learned
    dv, dn = v - v0, n - n0
    readout_i = (other @ dynamic_matrix @ other) / (other @ state_matrix @ other)
    score_i_error = readout_i - d_i
    ideal = d_j - d_i + 2 * a * (c - r * d_i) + a**2 * (d_v - d_i)
    calculated = ideal + dn - d_i * dv - v * score_i_error
    actual = v * (n / v - readout_i)
    np.testing.assert_allclose(ideal / v0, n0 / v0 - d_i, atol=2e-15)
    np.testing.assert_allclose(calculated, actual, atol=2e-15)
    eta = np.sqrt((learned - vector) @ dynamic_matrix @ (learned - vector))
    assert abs(dn) <= 2 * np.sqrt(n0) * eta + eta**2
    analytic_error_bound = (abs(dn) + abs(n0 / v0) * abs(dv)) / (v0 - abs(dv))
    assert abs(n / v - n0 / v0) <= analytic_error_bound
    return {
        "same_reference_rescaled_correlated_gap": float(ideal / v0),
        "learned_pair_gap": float(actual / v),
        "learned_numerator_formula_difference": float(calculated - actual),
        "axis_score_error": float(abs(n / v - n0 / v0)),
        "axis_score_error_bound": float(analytic_error_bound),
    }


def main():
    result = {"population": population_check(), "window": window_check(), "general": general_check()}
    with (HERE / "CHECKS.json").open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
