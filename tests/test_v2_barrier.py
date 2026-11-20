"""Numerical, gradient and configuration checks for the v2 GCSB."""
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hijepa.config import DEFAULTS, expand_matrix, validate
from hijepa.losses import coarse_spectral_barrier, covariance_floor


class BarrierTests(unittest.TestCase):
    def test_barrier_formula_collinear(self):
        c = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float64)
        value = coarse_spectral_barrier(c, mu=0.10, epsilon=0.005, lambda_floor=0.04)
        # Covariance eigenvalues are 0 and 2; only the zero eigenvalue is below floor.
        expected = (0.10 / 2.0) * math.log((0.04 + 0.005) / (0.0 + 0.005))
        self.assertAlmostEqual(float(value), expected, places=6)

    def test_barrier_zero_when_all_eigenvalues_above_floor(self):
        c = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
                         dtype=torch.float64)
        value = coarse_spectral_barrier(c)
        self.assertEqual(float(value), 0.0)

    def test_barrier_active_gradient_is_finite_and_nonzero(self):
        c = torch.tensor([[1.0, 1.01], [2.0, 1.99], [3.0, 3.02], [4.0, 3.98]],
                         dtype=torch.float64, requires_grad=True)
        value = coarse_spectral_barrier(c)
        self.assertGreater(float(value.detach()), 0.0)
        value.backward()
        self.assertIsNotNone(c.grad)
        self.assertTrue(bool(torch.isfinite(c.grad).all()))
        self.assertGreater(float(c.grad.abs().sum()), 0.0)

    def test_near_collinear_is_active(self):
        base = torch.linspace(-2.0, 2.0, 256, dtype=torch.float64)
        c = torch.stack([base, base + 0.01 * torch.sin(base)], dim=1)
        centered = c - c.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (len(c) - 1)
        correlation = covariance[0, 1] / torch.sqrt(covariance[0, 0] * covariance[1, 1])
        eigenvalues = torch.linalg.eigvalsh(covariance)
        self.assertGreater(float(correlation.abs()), 0.99)
        self.assertLess(float(eigenvalues[0]), 0.04)
        self.assertGreater(float(coarse_spectral_barrier(c)), 0.0)

    def test_v1_default_unchanged(self):
        self.assertNotIn("coarse_floor_mode", DEFAULTS["losses"])
        v1 = validate({})
        self.assertNotIn("coarse_floor_mode", v1["losses"])
        # The v1 hinge path is still available and unchanged.
        c = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float64)
        self.assertGreaterEqual(float(covariance_floor(c)), 0.0)

    def test_v2_config_accepted(self):
        cfg = validate({"losses": {
            "coarse_floor_mode": "spectral_barrier",
            "coarse_barrier_lambda_floor": 0.04,
            "coarse_barrier_epsilon": 0.005,
            "coarse_barrier_mu": 0.10,
        }})
        self.assertEqual(cfg["losses"]["coarse_floor_mode"], "spectral_barrier")
        self.assertEqual(cfg["losses"]["coarse_barrier_mu"], 0.10)
        self.assertNotIn("coarse_floor_mode", DEFAULTS["losses"])

    def test_v2_matrices_and_v1_default(self):
        repo = Path(__file__).resolve().parents[1]
        baseline = expand_matrix(repo / "configs" / "matrix_baseline_v2.json")
        self.assertEqual([cfg["name"] for cfg in baseline],
                         ["e1_baseline_seed0", "e1_baseline_seed1"])
        self.assertTrue(all(cfg["losses"]["coarse_floor_mode"] == "spectral_barrier"
                            for cfg in baseline))
        full = expand_matrix(repo / "configs" / "matrix_formal_v2.json")
        self.assertEqual(len(full), 57)
        self.assertTrue(all(cfg["losses"]["coarse_floor_mode"] == "spectral_barrier"
                            for cfg in full))
        v1 = expand_matrix(repo / "configs" / "matrix_formal.json")
        self.assertEqual(len(v1), 57)
        self.assertTrue(all("coarse_floor_mode" not in cfg["losses"] for cfg in v1))

    def test_invalid_v2_configs_rejected(self):
        with self.assertRaises(ValueError):
            validate({"losses": {"coarse_floor_mode": "unknown"}})
        with self.assertRaises(ValueError):
            validate({"losses": {"coarse_floor_mode": "spectral_barrier"}})
        with self.assertRaises(ValueError):
            validate({"losses": {
                "coarse_floor_mode": "spectral_barrier",
                "coarse_barrier_lambda_floor": 0.0,
                "coarse_barrier_epsilon": 0.005,
                "coarse_barrier_mu": 0.10,
            }})


if __name__ == "__main__":
    unittest.main()
