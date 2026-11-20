"""Deterministic quadrature check of the exact-Gaussian flow expansion."""

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermegauss


def field(x):
    first, second = x.T
    return np.column_stack((-first * second, first**2 - 1))


def flow(x, time, steps):
    dt = time / steps
    for _ in range(steps):
        k1 = field(x)
        k2 = field(x + dt * k1 / 2)
        k3 = field(x + dt * k2 / 2)
        k4 = field(x + dt * k3)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


def verify(nodes, steps):
    points, weights = hermegauss(nodes)
    weights = weights / weights.sum()
    normal = np.stack(np.meshgrid(*([points] * 4), indexing="ij"), axis=-1).reshape(-1, 4)
    product_weights = np.einsum("i,j,k,l->ijkl", *([weights] * 4)).reshape(-1)
    past = normal[:, :2]
    results = {}
    for b in (0.60, 0.76):
        rho = np.array([0.90, b])
        future = rho * past + np.sqrt(1 - rho**2) * normal[:, 2:]
        observations = []
        for time in (0.04, 0.02, 0.01):
            left = flow(past, time, steps)
            right = flow(future, time, steps)
            gain = np.sum(product_weights * np.sum(left * right, axis=1)) - rho.sum()
            observations.append(
                {
                    "t": time,
                    "correlation_gain": float(gain),
                    "gain_over_t_squared": float(gain / time**2),
                    "marginal_mean_max_abs": float(np.max(np.abs(product_weights @ left))),
                    "marginal_second_moment_max_error": float(
                        np.max(np.abs((left.T * product_weights) @ left - np.eye(2)))
                    ),
                }
            )
        a = rho[0]
        coefficient = 2 * a**2 - a - (2 - a) * b
        assert abs(observations[-1]["gain_over_t_squared"] - coefficient) < 2e-4
        results[str(b)] = {"predicted_coefficient": float(coefficient), "observations": observations}
    return results


def main():
    coarse = verify(12, 4)
    fine = verify(20, 8)
    for b in fine:
        for lower, upper in zip(coarse[b]["observations"], fine[b]["observations"]):
            assert abs(lower["correlation_gain"] - upper["correlation_gain"]) < 2e-8
            assert upper["marginal_mean_max_abs"] < 2e-8
            assert upper["marginal_second_moment_max_error"] < 2e-8
    output = {"fine_quadrature_nodes": 20, "fine_flow_steps": 8, "coarse": coarse, "fine": fine}
    Path(__file__).with_name("EXACT_GAUSSIAN_NUMERICS.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
