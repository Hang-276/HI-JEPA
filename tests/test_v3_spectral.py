"""Numerical and configuration checks for the v3 post-training spectral readout."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "diagnostics"))

from hijepa.config import load_config, validate  # noqa: E402
from hijepa.data import prepare_data  # noqa: E402
from hijepa.models import build_model  # noqa: E402
from hijepa.runner import sample_batch  # noqa: E402
import spectral_readout as sr  # noqa: E402
import local_modal as lm  # noqa: E402
import readout_runtime as rr  # noqa: E402


def small_config():
    selection = {
        "status": "certified", "valid_for_training": True, "m": 2,
        "blocks": [
            {"kind": "real", "eigenvalues": [{"real": 0.9, "imag": 0.0}],
             "center": 0.9, "radius": 0.02, "certificate_contour_radius": 0.05},
            {"kind": "real", "eigenvalues": [{"real": 0.8, "imag": 0.0}],
             "center": 0.8, "radius": 0.02, "certificate_contour_radius": 0.05},
        ],
        "clusters": [[0], [1]],
        "selected_eigenvalues": [{"real": 0.9, "imag": 0.0}, {"real": 0.8, "imag": 0.0}],
        "selected_modulus_lower_bound": 0.78,
        "certificate": {"parameterization_margin": 1e-5, "spectral_spread_order": 2,
                        "selected_modulus_upper_bound": 0.92},
    }
    config = validate({
        "system": {"dimension": 2, "eigenvalues": [0.9, 0.8], "coupling": 0.0},
        "data": {"train_trajectories": 8, "calibration_trajectories": 8,
                 "test_trajectories": 8, "length": 16},
        "selection": {"certificate_margin": 1e-5},
        "model": {"variant": "full", "z_dim": 3, "width": 6, "depth": 2,
                  "weight_frobenius_bound": 8.0, "bias_norm_bound": 8.0},
        "losses": {"k": 2},
        "evaluation": {"horizon": 8},
    })
    return config, selection


def small_model(config, selection, seed=0):
    torch.manual_seed(seed)
    model = build_model(config, selection).double()
    model.eval()
    return model


def synthetic_operator_features(dimension=10, trajectories=300, times=16, seed=4):
    rng = np.random.default_rng(seed)
    operator = np.zeros((dimension, dimension), dtype=np.float64)
    operator[0, 0] = 1.0
    operator[1, 1] = 0.9
    operator[2, 2] = 0.8
    angle = 0.2
    rotation = 0.85 * np.array([[math.cos(angle), -math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]])
    operator[3:5, 3:5] = rotation
    operator[5:, 5:] = 0.5 * np.eye(dimension - 5)
    phi = np.empty((trajectories, times, dimension), dtype=np.float64)
    phi[:, 0] = rng.normal(size=(trajectories, dimension))
    phi[:, 0, 0] = 1.0
    for t in range(1, times):
        phi[:, t] = phi[:, t - 1] @ operator.T
    return phi, operator


class FeatureMapTests(unittest.TestCase):
    def test_phi_layout(self):
        config, selection = small_config()
        model = small_model(config, selection)
        x = torch.randn(7, 2, dtype=torch.float64)
        phi = sr.feature_map(model, x)
        z = model.encode(x)
        u = model.G[:-1](z)
        expected = torch.cat([torch.ones_like(z[..., :1]), z, u], dim=-1)
        torch.testing.assert_close(phi, expected)
        self.assertEqual(phi.shape, (7, 1 + config["model"]["z_dim"] + config["model"]["width"]))
        self.assertTrue(bool(torch.all(phi[..., 0] == 1.0)))

    def test_feature_jacobian_constant_row_is_zero(self):
        config, selection = small_config()
        model = small_model(config, selection)
        system = sr.make_system(config["system"])
        jacobian = sr._feature_jacobian_at_origin(model, system)
        self.assertEqual(jacobian.shape[0], 1 + config["model"]["z_dim"] + config["model"]["width"])
        self.assertTrue(np.allclose(jacobian[0], 0.0))


class MomentOperatorTests(unittest.TestCase):
    def test_source_support_matches_generated_data_and_latest_sampled_start(self):
        config, _ = small_config()
        config["data"]["length"] = 64
        config["losses"]["k"] = 4
        system = sr.make_system(config["system"])
        with tempfile.TemporaryDirectory() as directory:
            train = prepare_data(system, config["data"], directory)["train"]
        self.assertEqual(train["observations"].shape[1], 65)

        class LastAllowedTime:
            """Exercise the actual sampler's upper endpoint deterministically."""
            def choice(self, n, size, replace):
                return np.arange(size)

            def integers(self, low, high, size):
                self.bounds = (low, high)
                return np.full(size, high - 1, dtype=np.int64)

        rng = LastAllowedTime()
        batch = sample_batch(train, config, rng, torch.device("cpu"), torch.float64)
        self.assertEqual(rng.bounds, (0, 57))
        n = len(batch["x"])
        torch.testing.assert_close(batch["x"], torch.as_tensor(train["observations"][:n, 56]))
        for q in (1, 4, 8):
            torch.testing.assert_close(batch["futures"][q],
                                       torch.as_tensor(train["observations"][:n, 56 + q]))
        for support in (sr._source_training_support(config),
                        sr._source_training_support(config, n_observation_times=65)):
            self.assertEqual(support["last_start"], 56)
            self.assertEqual(support["n_feature_times"], 65)
            self.assertEqual(support["n_pairs"], 57)

    def test_multihorizon_residual_uses_identical_current_times(self):
        # Deliberately nonstationary residuals make dropping late current times
        # change each q's MSE, rather than accidentally passing on exact dynamics.
        t = np.arange(9, dtype=np.float64)
        c = np.stack([np.stack([t ** 2 + 1, 2 * t ** 2 + 3]),
                      np.stack([t ** 3 - 2, 0.5 * t ** 3 + 4])])
        k = np.diag([0.8, 0.6])
        last_start, horizons = 4, [1, 4]
        result = sr._block_residual(c, k, last_start, horizons)
        expected_sum = 0.0
        for q in horizons:
            prediction = np.einsum("ab,bnt->ant", np.linalg.matrix_power(k, q),
                                   c[:, :, :last_start + 1])
            target = c[:, :, q:q + last_start + 1]
            expected = float(np.mean((prediction - target) ** 2))
            self.assertAlmostEqual(result["mse_by_horizon"][str(q)], expected, places=10)
            self.assertEqual(result["support_by_horizon"][str(q)], {
                "time_support": [0, last_start],
                "target_support": [q, last_start + q],
                "pairs_per_trajectory": last_start + 1,
            })
            expected_sum += expected
        self.assertAlmostEqual(result["sum_mse"], expected_sum, places=10)
        self.assertEqual(set(result["support_by_horizon"]), {"1", "4"})

    def test_multihorizon_residual_rejects_missing_future_times(self):
        # q=4 needs time 8 for the final common current time t=4.
        with self.assertRaises(ValueError):
            sr._block_residual(np.ones((2, 3, 8)), np.eye(2), last_start=4,
                               horizons=[1, 4])

    def test_moment_solve_recovers_known_operator(self):
        phi, operator = synthetic_operator_features()
        moments = sr._moments(phi, last_start=10)
        solved = sr._solve_operator(moments["C00"], moments["C10"])
        self.assertEqual(solved["C00_retained_rank"], phi.shape[-1])
        self.assertTrue(np.allclose(solved["operator"], operator, atol=1e-11, rtol=1e-11))
        self.assertFalse(solved["C00_threshold_searched"])

    def test_c00_cutoff_is_relative_and_fixed(self):
        phi, _ = synthetic_operator_features()
        moments = sr._moments(phi, last_start=10)
        eigenvalues = np.linalg.eigvalsh(moments["C00"])
        # Force a near-zero direction: the fixed relative cutoff must drop it.
        c00 = moments["C00"].copy()
        c00 += np.diag(np.r_[np.full(8, 1e-12), np.zeros(2)])
        solved = sr._solve_operator(c00, moments["C10"])
        expected = 1e-8 * float(np.linalg.eigvalsh(c00).max())
        self.assertAlmostEqual(solved["C00_absolute_threshold"], expected, places=15)
        self.assertEqual(solved["C00_relative_threshold"], 1e-8)
        self.assertLessEqual(solved["C00_retained_rank"], len(eigenvalues))

    def test_left_eigenvector_identities(self):
        phi, _ = synthetic_operator_features()
        solved = sr._solve_operator(*[sr._moments(phi, 10)[key] for key in ("C00", "C10")])
        spectrum = sr._left_spectrum(solved["operator"])
        spectrum["operator"] = solved["operator"]
        selection = {"selected_modulus_lower_bound": 0.78,
                     "certificate": {"selected_modulus_upper_bound": 0.92}}
        candidates = sr._candidate_blocks(spectrum, selection, (0.78, 0.92))
        kinds = sorted(candidate["kind"] for candidate in candidates)
        self.assertIn("real", kinds)
        self.assertIn("rotation", kinds)
        for candidate in candidates:
            self.assertLess(candidate["operator_residual_relative"], 1e-10)

    def test_rms_scaling_is_multiplicative_without_centering(self):
        phi, operator = synthetic_operator_features()
        moments = sr._moments(phi, 10)
        solved = sr._solve_operator(moments["C00"], moments["C10"])
        spectrum = sr._left_spectrum(solved["operator"])
        spectrum["operator"] = solved["operator"]
        candidate = sr._candidate_blocks(
            spectrum, {"selected_modulus_lower_bound": 0.78,
                       "certificate": {"selected_modulus_upper_bound": 0.92}},
            (0.78, 0.92))[0]
        scaled = sr._scale_candidate(candidate, phi, 10)
        self.assertGreater(scaled["rms_scale"], 0)
        support = scaled["c_scaled"][:, :, :11]
        self.assertAlmostEqual(float(np.sqrt(np.mean(support ** 2))), 1.0, places=12)
        raw_values = candidate["W"] @ phi[:, :11].reshape(-1, phi.shape[-1]).T
        raw_mean = float(np.mean(raw_values))
        scaled_mean = float(np.mean(support))
        self.assertAlmostEqual(scaled_mean, raw_mean / scaled["rms_scale"], places=12)


class CandidateConstraintTests(unittest.TestCase):
    def setUp(self):
        self.config, self.selection = small_config()
        self.phi, self.operator = synthetic_operator_features()
        moments = sr._moments(self.phi, 10)
        solved = sr._solve_operator(moments["C00"], moments["C10"])
        spectrum = sr._left_spectrum(solved["operator"])
        spectrum["operator"] = solved["operator"]
        self.candidates = sr._candidate_blocks(
            spectrum, self.selection, (0.78, 0.92))
        self.candidates = [sr._scale_candidate(candidate, self.phi, 10)
                           for candidate in self.candidates]
        self.domains = sr._original_domains(self.selection, self.config)

    def _combination(self, kinds):
        selected = []
        for kind in kinds:
            selected.append(next(candidate for candidate in self.candidates
                                 if candidate["kind"] == kind
                                 and all(candidate is not other for other in selected)))
        W = np.vstack([candidate["W_scaled"] for candidate in selected])
        K = np.zeros((2, 2))
        offset = 0
        for candidate in selected:
            size = candidate["dimension"]
            K[offset:offset + size, offset:offset + size] = candidate["K"]
            offset += size
        c = np.concatenate([candidate["c_scaled"] for candidate in selected], axis=0)
        return {"indices": [], "blocks": selected, "W": W, "K": K, "c": c,
                "kind_signature": tuple(candidate["kind"] for candidate in selected)}

    def test_two_real_blocks_are_legal(self):
        record = sr._k_domain_check(self._combination(["real", "real"]),
                                    self.domains, sr.REAL_IMAGINARY_TOLERANCE)
        self.assertTrue(record["legal"])
        self.assertTrue(record["matching_kind_and_dimension"])

    def test_rotation_block_not_in_two_real_domains(self):
        record = sr._k_domain_check(self._combination(["rotation"]),
                                    self.domains, sr.REAL_IMAGINARY_TOLERANCE)
        self.assertFalse(record["legal"])
        self.assertFalse(record["matching_kind_and_dimension"])
        self.assertTrue(record["certified_disk_diagnostics"])
        # A modulus in the selector window cannot override a block-type mismatch.

    def test_conjugate_pair_spectral_distance_is_zero(self):
        value = 0.85 * np.exp(0.2j)
        selection = {"selected_eigenvalues": [
            {"real": value.real, "imag": value.imag},
            {"real": value.real, "imag": -value.imag},
        ]}
        combination = {"blocks": [{"eigenvalues": [value, value.conjugate()]}]}
        self.assertAlmostEqual(sr._spectral_distance(combination, selection), 0.0, places=15)

    def test_negative_real_domain_preserves_signed_center(self):
        config, selection = small_config()
        selection["blocks"][0]["center"] = -0.9
        selection["blocks"][0]["eigenvalues"] = [{"real": -0.9, "imag": 0.0}]
        domains = sr._original_domains(selection, config)
        self.assertEqual(domains[0]["complex_center"], complex(-0.9, 0.0))

    def test_legal_alignment_reorders_weights_and_dynamics_with_blocks(self):
        original = self._combination(["real", "real"])
        reverse = [1, 0]
        combination = {
            **original,
            "indices": [20, 10],
            "blocks": [original["blocks"][i] for i in reverse],
            "W": original["W"][reverse],
            "K": original["K"][np.ix_(reverse, reverse)],
            "c": original["c"][reverse],
            "kind_signature": tuple(reversed(original["kind_signature"])),
        }
        record = sr._evaluate_combinations(
            [combination], self.selection, self.config, self.phi,
            last_start=10, horizons=[1, 2, 4],
            feature_jacobian=np.zeros((10, 2)), variant="block_only")[0]
        self.assertTrue(record["k_domain"]["legal"])
        self.assertEqual(tuple(record["k_domain"]["permutation"]), (1, 0))
        aligned = sr._align_legal_combination(record)
        np.testing.assert_array_equal(aligned["W_array"], original["W"])
        np.testing.assert_array_equal(aligned["K_matrix_array"], original["K"])
        np.testing.assert_array_equal(np.asarray(aligned["W"]), original["W"])
        np.testing.assert_array_equal(np.asarray(aligned["K_matrix"]), original["K"])
        self.assertEqual(aligned["indices"], [10, 20])
        self.assertEqual([block["eigenvalues"][0]["real"] for block in aligned["blocks"]],
                         [block["representative"].real for block in original["blocks"]])
        self.assertEqual(tuple(aligned["k_domain"]["permutation"]), (0, 1))

    def test_weight_budget_is_joint_and_not_projected(self):
        small = np.zeros((2, 10))
        small[:, 0] = 0.1
        small[0, 1] = 0.1
        self.assertTrue(sr.weight_budget_check(small, self.config)["weight_within_budget"])
        large = np.full((2, 10), 5.0)
        large[:, 0] = 0.0
        result = sr.weight_budget_check(large, self.config)
        self.assertFalse(result["weight_within_budget"])
        self.assertFalse(result["projection_applied"])
        self.assertAlmostEqual(result["joint_weight_frobenius"], 5.0 * math.sqrt(18), places=12)

    def test_no_legal_candidate_is_recorded(self):
        combination = self._combination(["rotation"])
        result = sr._evaluate_combinations(
            [combination], self.selection, self.config, self.phi,
            last_start=10, horizons=[1, 2, 4],
            feature_jacobian=np.ones((10, 2)), variant="full")
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["hard_constraints_passed"])
        self.assertFalse(result[0]["k_domain"]["legal"])

    def test_full_local_rank_check(self):
        combination = self._combination(["real", "real"])
        jacobian = np.zeros((10, 2))
        jacobian[1, 0] = 1.0
        jacobian[2, 1] = 1.0
        record = sr._evaluate_combinations(
            [combination], self.selection, self.config, self.phi,
            last_start=10, horizons=[1, 2, 4],
            feature_jacobian=jacobian, variant="full")[0]
        self.assertTrue(record["local_rank"]["passed"])
        rank = sr._local_rank_from_readout(combination["W"], jacobian, 2, 0.1)
        self.assertTrue(rank["passed"])

    def test_rank_requirement_not_applied_to_no_rank_variant(self):
        combination = self._combination(["real", "real"])
        record = sr._evaluate_combinations(
            [combination], self.selection, self.config, self.phi,
            last_start=10, horizons=[1, 2, 4],
            feature_jacobian=np.zeros((10, 2)), variant="block_only")[0]
        self.assertIsNone(record["local_rank"]["passed"])
        self.assertFalse(record["local_rank"]["applicable"])


class ConstrainedReadoutTests(unittest.TestCase):
    def test_paired_moment_objective_and_gradients_match_direct_pairs(self):
        # Nonstationary features and unequal horizons expose using different
        # start-time supports, averaging horizons, or detaching future W.
        rng = np.random.default_rng(27)
        phi = rng.normal(size=(5, 9, 6))
        phi[..., 0] = 1.0
        phi += np.arange(9, dtype=np.float64)[None, :, None] * 0.2
        last_start, horizons = 4, [1, 4]
        angle = 0.31
        rotation = 0.84 * np.array([[math.cos(angle), -math.sin(angle)],
                                  [math.sin(angle), math.cos(angle)]])
        k = np.zeros((3, 3), dtype=np.float64)
        k[0, 0] = -0.87
        k[1:, 1:] = rotation
        w = rng.normal(size=(3, phi.shape[-1]))
        W = torch.tensor(w, requires_grad=True)
        K = torch.tensor(k, requires_grad=True)
        moments = sr._paired_moments(phi, last_start, horizons)
        objective = sr._moment_residual(W, K, moments)
        actual_gradient = torch.autograd.grad(objective, (W, K))

        explicit_W = torch.tensor(w, requires_grad=True)
        explicit_K = torch.tensor(k, requires_grad=True)
        current = torch.tensor(phi[:, :last_start + 1].reshape(-1, phi.shape[-1]))
        explicit = sum(
            torch.mean((current @ explicit_W.T
                        @ torch.linalg.matrix_power(explicit_K, q).T
                        - torch.tensor(phi[:, q:q + last_start + 1]
                                       .reshape(-1, phi.shape[-1])) @ explicit_W.T) ** 2)
            for q in horizons)
        expected_gradient = torch.autograd.grad(explicit, (explicit_W, explicit_K))
        torch.testing.assert_close(objective, explicit, rtol=1e-12, atol=1e-12)
        for actual, expected in zip(actual_gradient, expected_gradient):
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)

    def test_paired_moments_reject_missing_common_future(self):
        with self.assertRaises(ValueError):
            sr._paired_moments(np.ones((4, 8, 3)), last_start=4, horizons=[1, 4])

    def test_raw_readout_exactly_embeds_original_coarse_head(self):
        config, selection = small_config()
        model = small_model(config, selection)
        W = sr._raw_readout(model)
        x = torch.randn(11, 2, dtype=torch.float64)
        torch.testing.assert_close(
            sr.feature_map(model, x) @ torch.as_tensor(W).T, model.coarse(x),
            rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(W[:, 1:1 + config["model"]["z_dim"]], 0.0)

    def test_projection_enforces_joint_weight_and_bias_after_float32_cast(self):
        config, _ = small_config()
        # Both rows violate the *joint* balls even though individual entries
        # and a separate per-row norm could pass.
        W = torch.full((2, 10), 5.0, dtype=torch.float64, requires_grad=True)
        sr._project_readout_(W, config)
        deployed = W.detach().float().double().numpy()
        record = sr.weight_budget_check(deployed, config)
        self.assertTrue(record["weight_within_budget"])
        self.assertLessEqual(float(np.linalg.norm(deployed[:, 0])),
                             config["model"]["bias_norm_bound"])
        interior = torch.zeros((2, 10), dtype=torch.float64)
        interior[0, 0], interior[1, 5] = 0.25, 0.125
        original = interior.clone()
        sr._project_readout_(interior, config)
        torch.testing.assert_close(interior, original, rtol=0, atol=0)

    def test_sampler_preserves_unique_trajectories_and_common_support(self):
        phi = np.empty((7, 12, 3), dtype=np.float64)
        phi[:, :, 0] = np.arange(7)[:, None]
        phi[:, :, 1] = np.arange(12)[None, :]
        phi[:, :, 2] = 10 * phi[:, :, 0] + phi[:, :, 1]
        first = sr._sample_features(phi, 5, 4, np.random.default_rng(12))
        repeat = sr._sample_features(phi, 5, 4, np.random.default_rng(12))
        np.testing.assert_array_equal(first, repeat)
        self.assertEqual(first.shape, (4, 3))
        self.assertEqual(len(np.unique(first[:, 0])), 4)
        self.assertTrue(np.all((first[:, 1] >= 0) & (first[:, 1] <= 5)))
        all_trajectories = sr._sample_features(phi, 5, 256, np.random.default_rng(12))
        self.assertEqual(all_trajectories.shape, (7, 3))
        self.assertEqual(len(np.unique(all_trajectories[:, 0])), 7)

    def test_tiny_fit_is_finite_domain_constrained_and_source_frozen(self):
        config, selection = small_config()
        # Exercise the same v2 anti-collapse term without changing its values.
        config["losses"].update({"coarse_floor_mode": "spectral_barrier",
                                 "coarse_barrier_mu": 0.10,
                                 "coarse_barrier_epsilon": 0.005,
                                 "coarse_barrier_lambda_floor": 0.04})
        config["training"]["batch_size"] = 4
        source = small_model(config, selection)
        source_state = {key: value.detach().clone()
                        for key, value in source.state_dict().items()}
        system = sr.make_system(config["system"])
        with tempfile.TemporaryDirectory() as directory:
            train = prepare_data(system, config["data"], directory)["train"]
        phi = sr._feature_array(source, train["observations"],
                                n_times=train["observations"].shape[1])
        jacobian = sr._feature_jacobian_at_origin(source, system)
        fit = sr.fit_constrained_readout(source, phi, jacobian, config, selection,
                                        {"solver_steps": 8})
        self.assertEqual(fit["optimization"]["steps"], 8)
        self.assertEqual(fit["optimization"]["policy"], "final_iterate")
        self.assertEqual(fit["W_array"].shape, (selection["m"], phi.shape[-1]))
        self.assertEqual(fit["K_matrix_array"].shape, (selection["m"], selection["m"]))
        self.assertTrue(np.isfinite(fit["W_array"]).all())
        self.assertTrue(np.isfinite(fit["K_matrix_array"]).all())
        self.assertTrue(fit["k_domain"]["legal"])
        self.assertTrue(sr.weight_budget_check(fit["W_array"], config)["weight_within_budget"])
        for key, original in source_state.items():
            torch.testing.assert_close(source.state_dict()[key], original, rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in source.parameters()))

    def test_solver_rejects_unimplemented_hierarchical_variant(self):
        config, selection = small_config()
        source = small_model(config, selection)
        config["model"]["variant"] = "hierarchical"
        with self.assertRaises(ValueError):
            sr.fit_constrained_readout(source, np.ones((4, 17, 10)),
                                       np.zeros((10, 2)), config, selection,
                                       {"solver_steps": 1})

    def test_tiny_cli_smoke_exports_and_reloads_invalid_diagnostic_endpoint(self):
        config, selection = small_config()
        # Deliberately impossible engineering rank requirement exercises an
        # invalid export; this fixture is not a scientific recovery experiment.
        config["losses"]["kappa"] = 1000.0
        config["losses"].update({"coarse_floor_mode": "spectral_barrier",
                                 "coarse_barrier_mu": 0.10,
                                 "coarse_barrier_epsilon": 0.005,
                                 "coarse_barrier_lambda_floor": 0.04})
        config["training"]["batch_size"] = 4
        source = small_model(config, selection)
        analysis_config = copy.deepcopy(config)
        analysis_config["spectral_readout"] = {
            "method": "post_training_constrained_readout", "solver": "constrained_readout",
            "solver_steps": 8, "seeds": [0], "device": "cpu",
            "trajectory_residual_horizons": [1, 2, 4, 8],
        }
        with tempfile.TemporaryDirectory(prefix="hijepa-tiny-smoke-") as directory:
            root = Path(directory)
            source_root, output = root / "source", root / "output"
            data_dir = source_root / "data" / sr.dataset_key(config)
            data = prepare_data(sr.make_system(config["system"]), config["data"], data_dir)
            checkpoint = source_root / "e1_baseline_seed0" / "checkpoint.pt"
            # The source reader expects complete-step metadata. This explicitly
            # marked initialized checkpoint only tests that reader contract;
            # no source optimizer or source training is invoked here.
            sr.save_checkpoint(checkpoint, {
                "model": source.state_dict(), "config": config, "selection": selection,
                "phase": "source", "step": config["training"]["steps"],
                "source_sha256": sr.source_fingerprint(),
                "protocol_sha256": sr.protocol_fingerprint(config),
                "smoke_fixture_only": True, "source_training_performed": False,
            })
            config_path = root / "analysis.json"
            sr.write_json(config_path, analysis_config)
            protected_files = [checkpoint, *sorted(data_dir.iterdir())]
            original_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in protected_files}
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    sr, "_solve_operator", side_effect=AssertionError("EDMD must not run in constrained fit")):
                report = sr.main(["--config", str(config_path),
                                  "--source-root", str(source_root), "--output", str(output),
                                  "--device", "cpu", "--no-parity"])
            seed_dir = output / "seed0"
            optimization = json.loads((seed_dir / "optimization.json").read_text())
            result = json.loads((seed_dir / "result.json").read_text())
            saved_report = json.loads((output / "report.json").read_text())
            rendered_report = (output / "REPORT.md").read_text()
            self.assertEqual(optimization["steps"], 8)
            self.assertTrue(optimization["engineering_budget_override"])
            for terms in (optimization["initial_fixed_batch_terms"],
                          optimization["final_fixed_batch_terms"]):
                self.assertTrue(all(math.isfinite(value) for value in terms.values()))
            self.assertFalse(result["conclusion"]["valid_main_model"])
            constraints = result["spectral_readout"]["constraints"]
            self.assertTrue(constraints["k_parameter_domain"]["legal"])
            self.assertTrue(constraints["weight_budget"]["weight_within_budget"])
            self.assertIn("actual exported", constraints["checked_on"])
            self.assertIn("fit_residual_float64", result["spectral_readout"])
            self.assertIn("exported_train_fit_residual", result["spectral_readout"])
            self.assertIn("final_endpoint_fails_local_rank_kappa",
                          result["conclusion"]["failure_reasons"])
            self.assertFalse(saved_report["conclusion"]["all_valid_main_models"])
            self.assertEqual(saved_report["conclusion"], report["conclusion"])
            self.assertIn("constrained readout", rendered_report)
            self.assertTrue(result["recovery"]["fine_path"]["fine_probe_identical"])
            endpoint_path = seed_dir / "spec_endpoint.pt"
            with self.assertRaisesRegex(ValueError, "diagnostic-only"):
                sr.load_spec_endpoint(endpoint_path)
            loaded = sr.load_spec_endpoint(endpoint_path, allow_diagnostic=True)
            endpoint = sr.load_checkpoint(endpoint_path)
            reference = copy.deepcopy(source).float()
            x = torch.tensor(data["train"]["observations"][:, 0], dtype=torch.float32)
            W = torch.cat([endpoint["readout_bias"][:, None], endpoint["readout_weight"]],
                          dim=1).float()
            with torch.no_grad():
                torch.testing.assert_close(loaded.coarse(x), sr.feature_map(reference, x) @ W.T)
                torch.testing.assert_close(loaded.predict_fine(loaded.encode(x)),
                                           reference.predict_fine(reference.encode(x)),
                                           rtol=0, atol=0)
            for path, original_hash in original_hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_hash)
        self.assertFalse(root.exists())


class LocalModalTests(unittest.TestCase):
    """Small exact fixtures check the new constraint, never a recovery claim."""

    @staticmethod
    def fixture(jacobian=None, error=1e-4, rotation=False, center=None):
        jacobian = np.diag([0.9, 0.8]) if jacobian is None else np.asarray(jacobian)
        dimension = len(jacobian)
        model = torch.nn.Linear(dimension, dimension).double()
        with torch.no_grad():
            model.weight.copy_(torch.eye(dimension, dtype=torch.float64))
            model.bias.copy_(torch.arange(1, dimension + 1, dtype=torch.float64) / 3)

        def features(network, x):
            values = network(x)
            return torch.cat([torch.ones_like(values[..., :1]), values], dim=-1)

        selection = {
            "valid_for_training": True, "m": dimension,
            "estimated_jacobian": jacobian.tolist(),
            "blocks": ([{"kind": "rotation"}] if rotation else
                       [{"kind": "real"} for _ in range(dimension)]),
            "certificate": {
                "scope": "local observed-space Jacobian at the fixed point only",
                "truth_labels_used": False, "normality_assumed": False,
                "matrix_error_norm": "induced infinity", "matrix_error_bound": error,
                "query_center": [0.0] * dimension if center is None else list(center),
            },
        }
        context = lm.build_context(model, selection, features)
        return model, selection, features, context

    def test_context_is_observed_derivative_and_converts_certified_norm(self):
        model, _, _, context = self.fixture(center=[0.2, -0.4], error=0.003)
        torch.testing.assert_close(context["feature_jacobian_obs"][1:], model.weight)
        self.assertEqual(context["coordinate_system"], "observations")
        self.assertEqual(context["query_center"], [0.2, -0.4])
        self.assertAlmostEqual(context["error_bound_spectral"], math.sqrt(2) * 0.003)
        self.assertFalse(context["normality_assumed"])

    def test_context_rejects_missing_or_incompatible_query_certificate(self):
        model, selection, features, _ = self.fixture()
        bad_records = []
        for key in ("matrix_error_bound", "query_center", "truth_labels_used"):
            bad = copy.deepcopy(selection)
            bad["certificate"].pop(key)
            bad_records.append(bad)
        for key, value in (("matrix_error_norm", "frobenius"),
                           ("scope", "local state Jacobian at the fixed point"),
                           ("matrix_error_bound", -1.0),
                           ("truth_labels_used", True)):
            bad = copy.deepcopy(selection)
            bad["certificate"][key] = value
            bad_records.append(bad)
        for bad in bad_records:
            with self.subTest(certificate=bad["certificate"]), self.assertRaises(ValueError):
                lm.build_context(model, bad, features)

    def test_center_and_budget_retraction_preserve_fixed_point_anchor(self):
        _, _, _, context = self.fixture(center=[0.4, -0.2])
        config, _ = small_config()
        W = torch.full((2, 3), 20.0, dtype=torch.float64, requires_grad=True)
        centered = lm.center_readout(W, context["phi0"])
        torch.testing.assert_close(centered @ context["phi0"],
                                   torch.zeros(2, dtype=torch.float64), atol=1e-13, rtol=0)
        self.assertTrue(bool(torch.count_nonzero(centered[:, 0])))
        centered.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(W.grad).all()))
        lm.project_centered_readout_(W, config, context["phi0"])
        torch.testing.assert_close(W @ context["phi0"],
                                   torch.zeros(2, dtype=torch.float64), atol=1e-13, rtol=0)
        budget = sr.weight_budget_check(W.detach().numpy(), config)
        self.assertTrue(budget["weight_within_budget"])
        self.assertTrue(budget["bias_within_budget"])

    def test_observed_jacobian_pairing_handles_nonnormal_coordinate_change(self):
        # B=C T is the state-composite derivative. Multiplying that B by J_x
        # would mix coordinate systems even though both matrices are 2 by 2.
        transform = torch.tensor([[1.0, 3.0], [0.0, 1.0]], dtype=torch.float64)
        K = torch.diag(torch.tensor([0.9, 0.8], dtype=torch.float64))
        Jx = transform @ K @ torch.linalg.inv(transform)
        _, _, _, context = self.fixture(Jx.numpy())
        C = torch.linalg.inv(transform)
        W = lm.center_readout(torch.cat([torch.zeros((2, 1), dtype=torch.float64), C], 1),
                              context["phi0"])
        correct = lm.report(W, K, context)
        self.assertTrue(correct["passed"])
        self.assertFalse(correct["normality_assumed"])
        self.assertFalse(correct["angle_certificate_claimed"])
        self.assertFalse(correct["global_recovery_certificate_claimed"])
        state_B = C @ transform
        wrong = lm.center_readout(torch.cat([torch.zeros((2, 1), dtype=torch.float64), state_B], 1),
                                context["phi0"])
        self.assertFalse(lm.report(wrong, K, context)["passed"])

    def test_row_scaling_cannot_hide_a_wrong_mode(self):
        _, _, _, context = self.fixture()
        K = torch.diag(torch.tensor([0.9, 0.8], dtype=torch.float64))
        C = torch.tensor([[1.0, 0.1], [0.2, 1.0]], dtype=torch.float64)
        W = lm.center_readout(torch.cat([torch.zeros((2, 1), dtype=torch.float64), C], 1),
                              context["phi0"])
        original, _ = lm.constraint_values(W, K, context)
        rescaled, _ = lm.constraint_values(torch.tensor([[100.0], [-0.02]]) * W, K, context)
        torch.testing.assert_close(rescaled, original, atol=1e-10, rtol=1e-10)
        self.assertTrue(bool(torch.all(original > 0)))

    def test_shared_first_mode_fails_even_when_original_rank_passes(self):
        _, _, _, context = self.fixture()
        K = torch.diag(torch.tensor([0.9, 0.8], dtype=torch.float64))
        C = torch.tensor([[1.0, 0.0], [1.0, 0.2]], dtype=torch.float64)
        W = lm.center_readout(torch.cat([torch.zeros((2, 1), dtype=torch.float64), C], 1),
                              context["phi0"])
        rank = sr._local_rank_from_readout(W.numpy(),
                                           context["feature_jacobian_obs"].numpy(), 2, 0.1)
        self.assertTrue(rank["passed"])
        checked = lm.report(W, K, context)
        self.assertTrue(checked["blocks"][0]["passed"])
        self.assertFalse(checked["blocks"][1]["passed"])
        self.assertFalse(checked["passed"])

    def test_rotation_remains_one_block_and_accepts_orthogonal_block_coordinates(self):
        angle = 0.3
        J = torch.tensor([[math.cos(angle), -math.sin(angle)],
                          [math.sin(angle), math.cos(angle)]], dtype=torch.float64) * 0.85
        _, _, _, context = self.fixture(J.numpy(), rotation=True)
        angle_basis = 0.7
        Q = torch.tensor([[math.cos(angle_basis), -math.sin(angle_basis)],
                          [math.sin(angle_basis), math.cos(angle_basis)]], dtype=torch.float64)
        K = Q @ J @ Q.T
        W = lm.center_readout(torch.cat([torch.zeros((2, 1), dtype=torch.float64), Q], 1),
                              context["phi0"])
        checked = lm.report(W, K, context)
        self.assertTrue(checked["passed"])
        self.assertEqual(len(checked["blocks"]), 1)
        self.assertEqual(checked["blocks"][0]["dimension"], 2)
        rescaled = lm.report(-0.03 * W, K, context)
        self.assertTrue(rescaled["passed"])
        singular = W.clone()
        singular[1] = singular[0]
        failed = lm.report(singular, K, context)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["evaluated"])

    def test_query_error_is_a_residual_allowance_not_a_zero_residual_demand(self):
        truth = np.diag([0.9, 0.8])
        query = truth.copy()
        query[0, 1] = 0.0005
        _, _, _, context = self.fixture(query, error=0.001)
        W = lm.center_readout(torch.cat([torch.zeros((2, 1), dtype=torch.float64),
                                         torch.eye(2, dtype=torch.float64)], 1), context["phi0"])
        checked = lm.report(W, torch.as_tensor(truth), context)
        self.assertTrue(checked["passed"])
        self.assertGreater(checked["blocks"][0]["relative_residual"], 0.0)

    def test_nonzero_fixed_point_output_fails_even_with_exact_modal_derivative(self):
        _, _, _, context = self.fixture()
        W = torch.cat([torch.zeros((2, 1), dtype=torch.float64),
                       torch.eye(2, dtype=torch.float64)], 1)
        K = torch.diag(torch.tensor([0.9, 0.8], dtype=torch.float64))
        checked = lm.report(W, K, context)
        self.assertFalse(checked["anchor_passed"])
        self.assertTrue(all(block["passed"] for block in checked["blocks"]))
        self.assertFalse(checked["passed"])

    def test_constraint_gradients_and_declared_al_updates_are_finite(self):
        _, _, _, context = self.fixture()
        W = torch.tensor([[0.0, 1.0, 0.1], [0.0, 0.2, 1.0]],
                         dtype=torch.float64, requires_grad=True)
        K = torch.diag(torch.tensor([0.9, 0.8], dtype=torch.float64)).requires_grad_()
        g, _ = lm.constraint_values(lm.center_readout(W, context["phi0"]), K, context)
        solver = lm.AugmentedLagrangian(context)
        solver.penalty(g).backward()
        self.assertTrue(bool(torch.isfinite(W.grad).all()))
        self.assertTrue(bool(torch.isfinite(K.grad).all()))
        self.assertGreater(float(torch.linalg.vector_norm(W.grad)), 0.0)
        self.assertGreater(float(torch.linalg.vector_norm(K.grad)), 0.0)
        self.assertIsNone(solver.update(g, 1))
        self.assertIsNotNone(solver.update(g, context["options"]["al_update_every"]))
        state = solver.state_dict()
        self.assertFalse(state["scientific_loss_weight_added"])
        self.assertEqual(state["selection_policy"], "fixed-budget final iterate only")

        def residual_values(weight, diagonal):
            return lm.constraint_values(lm.center_readout(weight, context["phi0"]),
                                        torch.diag(diagonal), context)[0]

        self.assertTrue(torch.autograd.gradcheck(
            residual_values,
            (W.detach().clone().requires_grad_(),
             torch.tensor([0.9, 0.8], dtype=torch.float64, requires_grad=True)),
            eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_cross_block_dynamics_cannot_pass_only_diagonal_block_checks(self):
        _, _, _, context = self.fixture()
        W = lm.center_readout(torch.cat([torch.zeros((2, 1), dtype=torch.float64),
                                         torch.eye(2, dtype=torch.float64)], 1), context["phi0"])
        K = torch.tensor([[0.9, 0.01], [0.0, 0.8]], dtype=torch.float64)
        checked = lm.report(W, K, context)
        self.assertTrue(all(block["passed"] for block in checked["blocks"]))
        self.assertFalse(checked["K_block_structure_passed"])
        self.assertAlmostEqual(checked["K_off_block_max_absolute"], 0.01)
        self.assertFalse(checked["passed"])

    def test_new_configuration_is_explicit_and_old_configuration_stays_disabled(self):
        old = sr._analysis_config(load_config(REPO / "configs" / "base_formal_v3.json"))
        new = sr._analysis_config(load_config(REPO / "configs" / "base_formal_v3_modal.json"))
        self.assertFalse(old["local_modal"]["enabled"])
        self.assertTrue(new["local_modal"]["enabled"])
        self.assertEqual(new["local_modal"]["method"], "block_local_dynamics_v1")
        self.assertEqual(old["optimizer_state_policy"], "retain")
        self.assertEqual(new["optimizer_state_policy"], "retain")
        self.assertEqual(new["solver_revision"], "feasible_block_sqp_v1")
        for key in ("system", "data", "selection", "model", "losses", "training",
                    "evaluation", "anticollapse"):
            self.assertEqual(load_config(REPO / "configs" / "base_formal_v3.json")[key],
                             load_config(REPO / "configs" / "base_formal_v3_modal.json")[key])
        with self.assertRaises(ValueError):
            lm.validated_options({"fp32_tolerance": 0.1})
        with self.assertRaises(ValueError):
            lm.validated_options({"method": "unregistered_method"})
        for policy in ("reset_every_step", "reset_on_al_change"):
            incompatible = load_config(REPO / "configs" / "base_formal_v3.json")
            incompatible["spectral_readout"]["optimizer_state_policy"] = policy
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                sr._analysis_config(incompatible)

    @staticmethod
    def source_fixture():
        config, selection = small_config()
        config["training"]["batch_size"] = 4
        source = small_model(config, selection)
        system = sr.make_system(config["system"])
        h, eye = 1e-5, np.eye(system.dimension)
        observed_jacobian = np.column_stack([
            (system.local_query(h * eye[j]) - system.local_query(-h * eye[j])) / (2 * h)
            for j in range(system.dimension)])
        selection = copy.deepcopy(selection)
        selection["estimated_jacobian"] = observed_jacobian.tolist()
        selection["certificate"].update({
            "scope": "local observed-space Jacobian at the fixed point only",
            "truth_labels_used": False, "normality_assumed": False,
            "matrix_error_norm": "induced infinity", "matrix_error_bound": 1e-4,
            "query_center": [0.0] * system.dimension,
        })
        return config, selection, source, system

    def test_tiny_modal_fit_keeps_source_frozen_and_reports_final_constraint(self):
        config, selection, source, system = self.source_fixture()
        before = {key: value.detach().clone() for key, value in source.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            train = prepare_data(system, config["data"], directory)["train"]
        phi = sr._feature_array(source, train["observations"],
                                n_times=train["observations"].shape[1])
        state_jacobian = sr._feature_jacobian_at_origin(source, system)
        fit = sr.fit_constrained_readout(source, phi, state_jacobian, config, selection,
                                        {"solver_steps": 8, "local_modal": {"enabled": True}})
        optimization = fit["optimization"]
        self.assertEqual(optimization["steps"], 8)
        self.assertTrue(optimization["engineering_budget_override"])
        self.assertTrue(optimization["selector_jacobian_used_for_readout_training"])
        self.assertFalse(optimization["global_identifiability_certified"])
        self.assertIn("not proof", optimization["failure_scope"])
        self.assertTrue(np.isfinite(fit["W_array"]).all())
        self.assertTrue(np.isfinite(fit["K_matrix_array"]).all())
        for key in ("initial_fixed_batch_gradients", "final_fixed_batch_gradients"):
            for name in ("W_gradient_norm", "K_matrix_gradient_norm", "K_raw_gradient_norm"):
                self.assertTrue(math.isfinite(optimization[key][name]))
        self.assertEqual(optimization["local_constraint_satisfied"],
                         fit["local_modal_consistency"]["passed"])
        self.assertTrue(fit["local_modal_consistency"]["anchor_passed"])
        self.assertTrue(fit["k_domain"]["legal"])
        budget = sr.weight_budget_check(fit["W_array"], config)
        self.assertTrue(budget["weight_within_budget"])
        self.assertTrue(budget["bias_within_budget"])
        for key, value in before.items():
            torch.testing.assert_close(source.state_dict()[key], value, rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in source.parameters()))

    def test_actual_endpoint_check_and_reload_reject_forged_validity_metadata(self):
        config, selection, source, _ = self.source_fixture()
        W = sr._raw_readout(source)
        # A small bias perturbation is within budget but violates the actual
        # fixed-point anchor. Stored "passed" metadata cannot override it.
        W[:, 0] += 0.25
        K = source.K_matrix().detach().numpy()
        model = sr.build_spec_model(source, W, K, config["model"]["z_dim"], dtype=torch.float32)
        actual = sr._endpoint_modal_check(model, selection, {"enabled": True})
        self.assertFalse(actual["anchor_passed"])
        self.assertFalse(actual["passed"])
        self.assertEqual(actual["checked_endpoint_dtype"], "float32")
        self.assertIn("actual exported endpoint", actual["checked_on"])
        readout_config = {"spectral_readout": {"local_modal": {"enabled": True}}}
        payload = {
            "kind": "hijepa_v3_spectral_endpoint", "config": config,
            "selection": selection, "source_model": source.state_dict(),
            "readout_weight": torch.as_tensor(W[:, 1:]),
            "readout_bias": torch.as_tensor(W[:, 0]), "k_matrix": torch.as_tensor(K),
            "z_dim": config["model"]["z_dim"], "endpoint_dtype": "float32",
            "valid_main_model": True, "diagnostic_only": False,
            "readout_method": {"method": "post_training_local_modal_readout",
                               "local_modal": {"enabled": True}},
            "readout_config": readout_config,
            "readout_config_sha256": sr.fingerprint(readout_config),
            "selection_sha256": sr.fingerprint(selection),
            "constraints": {"local_modal_consistency": {"passed": True}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged_endpoint.pt"
            sr.save_checkpoint(path, payload)
            with self.assertRaisesRegex(ValueError, "Actual endpoint fails"):
                sr.load_spec_endpoint(path)
            loaded = sr.load_spec_endpoint(path, allow_diagnostic=True)
            self.assertEqual(loaded.readout_weight.dtype, torch.float32)
            # Missing method metadata must not route a new endpoint through
            # legacy compatibility while its saved config enables constraints.
            del payload["readout_method"]
            sr.save_checkpoint(path, payload)
            with self.assertRaisesRegex(ValueError, "matching validity metadata"):
                sr.load_spec_endpoint(path)

    def test_new_parity_snapshot_exercises_active_gradients_on_cpu(self):
        config, selection, source, system = self.source_fixture()
        with tempfile.TemporaryDirectory() as directory:
            data = prepare_data(system, config["data"], directory)
        source_run = {"config": config, "selection": selection, "data": data,
                      "payload": {"model": source.state_dict()}}
        snapshot = sr._constrained_parity_snapshot(
            source_run, sr._raw_readout(source), source.K_matrix().detach().numpy(),
            "cpu", {"local_modal": {"enabled": True}})
        for name in ("endpoint.fixed_point.float32", "endpoint.observed_coarse_jacobian.float32",
                     "modal.observed_feature_jacobian.float64", "modal.activated_fixture.gradient_W",
                     "modal.activated_fixture.gradient_K"):
            self.assertTrue(np.isfinite(snapshot[name]).all(), name)
        self.assertGreater(float(snapshot["modal.activated_fixture.loss"]), 0)
        self.assertGreater(np.linalg.norm(snapshot["modal.activated_fixture.gradient_K"]), 0)

    def test_tiny_modal_cli_writes_reports_exports_fp32_and_reloads_diagnostic(self):
        config, selection, source, system = self.source_fixture()
        # Impossible original rank budget makes this an explicitly invalid
        # engineering endpoint regardless of short numerical solver progress.
        config["losses"]["kappa"] = 1000.0
        analysis_config = copy.deepcopy(config)
        analysis_config["spectral_readout"] = {
            "method": "post_training_local_modal_readout", "solver": "constrained_readout",
            "solver_steps": 101, "seeds": [0], "device": "cpu",
            "optimizer_state_policy": "reset_on_al_change",
            "trajectory_residual_horizons": [1, 2, 4, 8],
            "local_modal": {"enabled": True, "method": "block_local_dynamics_v1"},
        }
        with tempfile.TemporaryDirectory(prefix="hijepa-modal-cli-") as directory:
            root = Path(directory)
            source_root, output = root / "source", root / "output"
            data_dir = source_root / "data" / sr.dataset_key(config)
            data = prepare_data(system, config["data"], data_dir)
            checkpoint = source_root / "e1_baseline_seed0" / "checkpoint.pt"
            sr.save_checkpoint(checkpoint, {
                "model": source.state_dict(), "config": config, "selection": selection,
                "phase": "source", "step": config["training"]["steps"],
                "source_sha256": sr.source_fingerprint(),
                "protocol_sha256": sr.protocol_fingerprint(config),
                "smoke_fixture_only": True, "source_training_performed": False,
            })
            protected_files = [checkpoint, *sorted(data_dir.iterdir())]
            original_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in protected_files}
            config_path = root / "analysis.json"
            sr.write_json(config_path, analysis_config)
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    sr, "_solve_operator", side_effect=AssertionError("EDMD must not run")):
                returned = sr.main(["--config", str(config_path), "--source-root", str(source_root),
                                    "--output", str(output), "--device", "cpu", "--no-parity"])
            seed_dir = output / "seed0"
            result = json.loads((seed_dir / "result.json").read_text())
            saved = json.loads((output / "report.json").read_text())
            self.assertEqual(saved["conclusion"], returned["conclusion"])
            self.assertIn("local", (output / "REPORT.md").read_text().lower())
            self.assertFalse(result["conclusion"]["valid_main_model"])
            constraints = result["spectral_readout"]["constraints"]
            self.assertIn("local_modal_fit_float64", constraints)
            self.assertIn("local_modal_consistency", constraints)
            actual = constraints["local_modal_consistency"]
            self.assertEqual(actual["checked_endpoint_dtype"], "float32")
            self.assertIn("actual exported endpoint", actual["checked_on"])
            endpoint_path = seed_dir / "spec_endpoint.pt"
            payload = sr.load_checkpoint(endpoint_path)
            self.assertEqual(payload["endpoint_dtype"], "float32")
            self.assertTrue(payload["readout_method"]["local_modal"]["enabled"])
            with self.assertRaisesRegex(ValueError, "diagnostic-only"):
                sr.load_spec_endpoint(endpoint_path)
            reloaded = sr.load_spec_endpoint(endpoint_path, allow_diagnostic=True)
            self.assertEqual(reloaded.readout_weight.dtype, torch.float32)
            x = torch.as_tensor(data["train"]["observations"][:, 0], dtype=torch.float32)
            with torch.no_grad():
                expected = sr.feature_map(reloaded.source, x) @ torch.cat([
                    payload["readout_bias"][:, None], payload["readout_weight"]], dim=1).float().T
                torch.testing.assert_close(reloaded.coarse(x), expected)
            optimization = json.loads((seed_dir / "optimization.json").read_text())
            self.assertEqual(optimization["steps"], 101)
            self.assertTrue(optimization["engineering_budget_override"])
            self.assertFalse(optimization["global_identifiability_certified"])
            self.assertEqual(optimization["optimizer_state_policy"], "reset_on_al_change")
            self.assertEqual(optimization["al_update_count"], 1)
            self.assertEqual(optimization["loss_log_count"], len(optimization["history"]))
            self.assertEqual(optimization["optimizer_state_reset_count"], 1)
            self.assertEqual(optimization["optimizer_state_reset_steps"], [100])
            self.assertEqual(len(optimization["optimizer_state_events"]), 1)
            self.assertTrue(optimization["optimizer_state_events"][0]["reset_performed"])
            state_metadata = optimization["optimizer_state_artifact"]
            state_path = seed_dir / state_metadata["path"]
            self.assertEqual(state_metadata["path"], "readout_optimizer_state.pt")
            self.assertEqual(hashlib.sha256(state_path.read_bytes()).hexdigest(),
                             state_metadata["sha256"])
            self.assertIn("resume not implemented", state_metadata["scope"])
            state_payload = sr.load_checkpoint(state_path)
            self.assertEqual(state_payload["completed_steps"], 101)
            self.assertEqual(state_payload["optimizer_state_policy"], "reset_on_al_change")
            layout = state_payload["parameter_layout"]
            self.assertEqual(layout[0]["name"], "W")
            self.assertEqual(set(state_payload["primal_parameters"]),
                             {row["name"] for row in layout})
            for row in layout:
                self.assertEqual(list(state_payload["primal_parameters"][row["name"]].shape),
                                 row["shape"])
            saved_readout = torch.cat([payload["readout_bias"][:, None],
                                       payload["readout_weight"]], dim=1)
            torch.testing.assert_close(state_payload["effective_readout"].float(),
                                       saved_readout.float(), rtol=0, atol=0)
            torch.testing.assert_close(state_payload["K_matrix"].float(),
                                       payload["k_matrix"].float(), rtol=0, atol=0)
            fitted_dynamics = copy.deepcopy(source.K)
            dynamics_state = fitted_dynamics.state_dict()
            dynamics_state.update({name.removeprefix("K."): value for name, value in
                                   state_payload["primal_parameters"].items()
                                   if name.startswith("K.")})
            fitted_dynamics.load_state_dict(dynamics_state)
            torch.testing.assert_close(fitted_dynamics(), state_payload["K_matrix"],
                                       rtol=0, atol=0)
            saved_states = state_payload["optimizer_state_dict"]["state"]
            self.assertTrue(saved_states)
            for state in saved_states.values():
                self.assertEqual(float(state["step"]), 1.0)
            self.assertFalse(list(seed_dir.glob("*.tmp")))
            for path, digest in original_hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertFalse(root.exists())


class AugmentedLagrangianAdamStateTests(unittest.TestCase):
    @staticmethod
    def optimizer_fixture(betas=(0.9, 0.999)):
        parameter = torch.nn.Parameter(torch.tensor([0.3], dtype=torch.float64))
        optimizer = torch.optim.Adam([parameter], lr=0.002, betas=betas)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        return parameter, optimizer

    def test_capped_rho_multiplier_change_still_resets_adam(self):
        _, _, _, context = LocalModalTests.fixture()
        solver = lm.AugmentedLagrangian(context)
        solver.rho = context["options"]["al_rho_max"]
        solver.previous_violation = 0.1
        update = solver.update(torch.tensor([0.1, 0.2], dtype=torch.float64), 100)
        self.assertTrue(update["multipliers_changed"])
        self.assertFalse(update["rho_changed"])
        self.assertTrue(update["objective_changed"])
        _, optimizer = self.optimizer_fixture()
        decision = lm.reset_adam_after_al_update(optimizer, update, 100, 101)
        self.assertTrue(decision["reset_performed"])
        self.assertEqual(len(optimizer.state), 0)

    def test_no_objective_change_no_update_retain_and_final_step_do_not_reset(self):
        _, _, _, context = LocalModalTests.fixture()
        solver = lm.AugmentedLagrangian(context)
        unchanged = solver.update(-torch.ones(2, dtype=torch.float64), 100)
        self.assertFalse(unchanged["multipliers_changed"])
        self.assertFalse(unchanged["rho_changed"])
        self.assertFalse(unchanged["objective_changed"])
        changed = solver.update(torch.ones(2, dtype=torch.float64), 200)
        self.assertTrue(changed["objective_changed"])
        cases = ((None, 1, 101, "reset_on_al_change"),
                 (unchanged, 100, 101, "reset_on_al_change"),
                 (changed, 200, 201, "retain"),
                 (changed, 200, 200, "reset_on_al_change"))
        for update, step, total_steps, policy in cases:
            with self.subTest(step=step, policy=policy, update=update):
                parameter, optimizer = self.optimizer_fixture()
                before = copy.deepcopy(optimizer.state[parameter])
                decision = lm.reset_adam_after_al_update(
                    optimizer, update, step, total_steps, policy)
                self.assertFalse(decision["reset_performed"])
                self.assertEqual(set(optimizer.state[parameter]), set(before))
                for name, value in before.items():
                    torch.testing.assert_close(optimizer.state[parameter][name], value,
                                               rtol=0, atol=0)

    def test_reset_preserves_parameters_lr_groups_and_rng_then_restarts_step_one(self):
        parameter, optimizer = self.optimizer_fixture()
        value_before = parameter.detach().clone()
        gradient_before = parameter.grad.detach().clone()
        groups_before = [{key: value for key, value in group.items() if key != "params"}
                         for group in optimizer.param_groups]
        random_before = torch.random.get_rng_state().clone()
        update = {"objective_changed": True, "multipliers_changed": True,
                  "rho_changed": False}
        decision = lm.reset_adam_after_al_update(optimizer, update, 100, 101)
        self.assertTrue(decision["reset_performed"])
        self.assertIs(optimizer.param_groups[0]["params"][0], parameter)
        torch.testing.assert_close(parameter, value_before, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, gradient_before, rtol=0, atol=0)
        self.assertEqual([{key: value for key, value in group.items() if key != "params"}
                          for group in optimizer.param_groups], groups_before)
        torch.testing.assert_close(torch.random.get_rng_state(), random_before, rtol=0, atol=0)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertEqual(float(optimizer.state[parameter]["step"]), 1.0)

    def test_large_previous_gradient_suppresses_new_gradient_until_state_reset(self):
        # beta1=0 isolates Adam's second-moment suppression from momentum.
        # This scalar regression checks the numerical mechanism, not recovery.
        stale = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
        fresh = torch.nn.Parameter(stale.detach().clone())
        stale_optimizer = torch.optim.Adam([stale], lr=0.001, betas=(0.0, 0.999))
        fresh_optimizer = torch.optim.Adam([fresh], lr=0.001, betas=(0.0, 0.999))
        for parameter, optimizer in ((stale, stale_optimizer), (fresh, fresh_optimizer)):
            parameter.grad = torch.full_like(parameter, 1e9)
            optimizer.step()
        stale_before, fresh_before = stale.detach().clone(), fresh.detach().clone()
        reset = lm.reset_adam_after_al_update(
            fresh_optimizer, {"objective_changed": True, "multipliers_changed": True,
                              "rho_changed": False}, 100, 101)
        self.assertTrue(reset["reset_performed"])
        for parameter, optimizer in ((stale, stale_optimizer), (fresh, fresh_optimizer)):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        stale_delta = float(torch.abs(stale.detach() - stale_before))
        fresh_delta = float(torch.abs(fresh.detach() - fresh_before))
        self.assertLess(stale_delta, fresh_delta * 1e-7)
        self.assertAlmostEqual(fresh_delta, 0.001, places=9)
        diagnostics = lm.adam_diagnostics(
            stale_optimizer, [("scalar", stale)], {"scalar": torch.ones_like(stale)})
        record = diagnostics["parameters"]["scalar"]
        self.assertEqual(record["step"], 2)
        self.assertGreater(record["exp_avg_sq_max"], 1e12)
        self.assertTrue(math.isfinite(record["effective_lr_max"]))
        self.assertLess(record["audit_preconditioned_gradient_norm"], 1e-7)
        raw_second_moment = float(stale_optimizer.state[stale]["exp_avg_sq"])
        expected_scale = 1.0 / (math.sqrt(raw_second_moment / (1 - 0.999 ** 2)) + 1e-8)
        self.assertAlmostEqual(record["effective_lr_max"], 0.001 * expected_scale, places=20)

    def test_all_al_updates_are_counted_independently_of_sparse_loss_logs(self):
        _, _, _, context = LocalModalTests.fixture()
        solver = lm.AugmentedLagrangian(context)
        for step in range(100, 20_001, 100):
            solver.update(torch.ones(2, dtype=torch.float64), step)
        self.assertEqual(len(solver.history), 200)
        self.assertEqual(solver.history[-1]["step"], 20_000)
        self.assertGreater(len(solver.history), 21)


class FullRankReadoutCoordinateTests(unittest.TestCase):
    @staticmethod
    def fixture(condition_number=1e5):
        generator = torch.Generator().manual_seed(317)
        p, dimension, m = 9, 3, 2
        U = torch.linalg.qr(torch.randn(p, p, dtype=torch.float64,
                                       generator=generator))[0]
        V = torch.linalg.qr(torch.randn(dimension, dimension, dtype=torch.float64,
                                       generator=generator))[0]
        singular = torch.logspace(0, -math.log10(condition_number), dimension,
                                  dtype=torch.float64)
        A = (U[:, :dimension] * singular) @ V.T
        phi0 = torch.cat([torch.ones(1, dtype=torch.float64),
                           torch.randn(p, dtype=torch.float64, generator=generator)])
        W = torch.randn(m, p + 1, dtype=torch.float64, generator=generator)
        context = {"phi0": phi0,
                   "feature_jacobian_obs": torch.cat([
                       torch.zeros(1, dimension, dtype=torch.float64), A], dim=0)}
        coordinates = lm.FullRankReadoutCoordinates(context, W)
        return coordinates, context, W

    def test_complete_ill_conditioned_readout_class_has_invertible_coordinates(self):
        coordinates, context, W = self.fixture()
        original = W.clone()
        C, Z = coordinates.encode(W[:, 1:])
        T, orthogonal_Z = coordinates.encode_orthogonal(W[:, 1:])
        self.assertEqual(C.shape, (2, 3))
        self.assertEqual(Z.shape, (2, 6))
        self.assertEqual(coordinates.N.shape, (9, 6))
        self.assertEqual(coordinates.U_t.shape[1] + coordinates.N.shape[1], 9)
        expected = lm.center_readout(W, context["phi0"])
        torch.testing.assert_close(coordinates.decode(C, Z), expected, rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(coordinates.decode_orthogonal(T, orthogonal_Z), expected,
                                   rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(coordinates.decode(C, Z)[:, 1:] @ coordinates.A, C,
                                   rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(coordinates.tangent_from_orthogonal(T), C,
                                   rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(coordinates.orthogonal_from_tangent(C), T,
                                   rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(T.square().sum() + Z.square().sum(),
                                   W[:, 1:].square().sum(), rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(W, original, rtol=0, atol=0)
        audit = coordinates.audit_metadata()
        self.assertTrue(audit["full_centered_readout_class_preserved"])
        self.assertFalse(audit["rank_truncation_used"])
        self.assertTrue(audit["svd_full_matrices"])
        self.assertEqual(audit["null_dimension"], 6)
        self.assertAlmostEqual(audit["feature_jacobian_condition_number"] / 1e5, 1.0, places=9)
        self.assertFalse(audit["representation_forward_changed"])
        json.dumps(audit, allow_nan=False)

    def test_coordinate_decoders_have_correct_gradients_in_both_full_bases(self):
        coordinates, _, W = self.fixture()
        C, Z = (tensor.detach().clone().requires_grad_()
                for tensor in coordinates.encode(W[:, 1:]))
        T, orthogonal_Z = (tensor.detach().clone().requires_grad_()
                           for tensor in coordinates.encode_orthogonal(W[:, 1:]))
        self.assertTrue(torch.autograd.gradcheck(coordinates.decode, (C, Z),
                                                eps=1e-6, atol=2e-5, rtol=1e-4))
        self.assertTrue(torch.autograd.gradcheck(coordinates.decode_orthogonal,
                                                (T, orthogonal_Z), eps=1e-6,
                                                atol=1e-6, rtol=1e-5))

    def test_nonlinear_null_features_survive_without_changing_local_tangent(self):
        coordinates, context, W = self.fixture()
        C, _ = coordinates.encode(W[:, 1:])
        Z = torch.zeros(2, coordinates.null_dimension, dtype=torch.float64)
        Z[0, 0], Z[1, 1] = 0.7, -0.4
        reconstructed = coordinates.decode(C, Z)
        linear_only = coordinates.decode(C, torch.zeros_like(Z))
        x = torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64)
        quadratic = torch.zeros(coordinates.null_dimension, dtype=torch.float64)
        quadratic[:3] = x.square()
        nonlinear_features = torch.cat([
            torch.ones(1, dtype=torch.float64),
            context["phi0"][1:] + coordinates.A @ x + coordinates.N @ quadratic])
        nonlinear_output = reconstructed @ nonlinear_features
        tangent_output = linear_only @ nonlinear_features
        torch.testing.assert_close(nonlinear_output, C @ x + Z @ quadratic,
                                   rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(tangent_output, C @ x, rtol=1e-9, atol=1e-10)
        self.assertGreater(float((nonlinear_output - tangent_output).norm()), 0.01)
        torch.testing.assert_close(reconstructed[:, 1:] @ coordinates.A,
                                   linear_only[:, 1:] @ coordinates.A,
                                   rtol=1e-10, atol=1e-11)

    def test_coupled_budget_retraction_matches_full_readout_retraction(self):
        coordinates, context, W = self.fixture()
        config, _ = small_config()
        config["model"]["bias_norm_bound"] = 0.2
        W = W * 100
        expected = W.clone()
        lm.project_centered_readout_(expected, config, context["phi0"])
        for encode, decode, project in (
                (coordinates.encode, coordinates.decode, coordinates.project_coordinates_),
                (coordinates.encode_orthogonal, coordinates.decode_orthogonal,
                 coordinates.project_orthogonal_)):
            with self.subTest(project=project.__name__):
                first, Z = encode(W[:, 1:])
                before_first, before_Z = first.clone(), Z.clone()
                scale = project(first, Z, config)
                self.assertGreater(scale, 0)
                self.assertLess(scale, 1)
                torch.testing.assert_close(first, before_first * scale, rtol=0, atol=0)
                torch.testing.assert_close(Z, before_Z * scale, rtol=0, atol=0)
                actual = decode(first, Z)
                torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
                torch.testing.assert_close(actual @ context["phi0"],
                                           torch.zeros(2, dtype=torch.float64),
                                           rtol=0, atol=1e-12)
                budget = sr.weight_budget_check(actual.float().double().numpy(), config)
                self.assertTrue(budget["weight_within_budget"])
                self.assertTrue(budget["bias_within_budget"])

    def test_rank_deficiency_is_rejected_without_truncating_feature_class(self):
        _, context, W = self.fixture()
        deficient = copy.deepcopy(context)
        deficient["feature_jacobian_obs"][:, 2] = deficient["feature_jacobian_obs"][:, 1]
        with self.assertRaisesRegex(lm.LocalModalConstraintError, "full column rank"):
            lm.FullRankReadoutCoordinates(deficient, W)
        too_few_features = {"phi0": context["phi0"][:3],
                            "feature_jacobian_obs": context["feature_jacobian_obs"][:3]}
        with self.assertRaisesRegex(lm.LocalModalConstraintError, "full column rank"):
            lm.FullRankReadoutCoordinates(too_few_features, W[:, :3])

    def test_legacy_modal_fit_does_not_enter_new_coordinate_solver(self):
        config, selection, source, system = LocalModalTests.source_fixture()
        with tempfile.TemporaryDirectory() as directory:
            train = prepare_data(system, config["data"], directory)["train"]
        phi = sr._feature_array(source, train["observations"],
                                n_times=train["observations"].shape[1])
        state_jacobian = sr._feature_jacobian_at_origin(source, system)
        with mock.patch.object(lm, "FullRankReadoutCoordinates", side_effect=AssertionError(
                "Legacy readout fit must retain the original parameterization")):
            fit = sr.fit_constrained_readout(
                source, phi, state_jacobian, config, selection,
                {"solver_steps": 2, "local_modal": {"enabled": True}})
        self.assertEqual(fit["optimization"]["steps"], 2)
        self.assertTrue(np.isfinite(fit["W_array"]).all())


class ReadoutFeasibilityRestorationTests(unittest.TestCase):
    @staticmethod
    def fixture(kappa=0.1):
        _, _, _, context = LocalModalTests.fixture()
        config, selection = small_config()
        config["losses"]["kappa"] = kappa
        source = small_model(config, selection)
        W = lm.center_readout(torch.cat([
            torch.zeros(2, 1, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64)], dim=1), context["phi0"])
        coordinates = lm.FullRankReadoutCoordinates(context, W)
        restorer = lm.ReadoutFeasibilityRestorer(
            coordinates, context, source.K,
            context["feature_jacobian_obs"].detach().clone(), config)
        return config, context, source.K, coordinates, restorer, W

    def test_already_feasible_point_is_not_changed(self):
        _, _, dynamics, coordinates, restorer, W = self.fixture()
        T, Z = coordinates.encode_orthogonal(W[:, 1:])
        before_T, before_Z = T.clone(), Z.clone()
        before_K = {name: value.detach().clone()
                    for name, value in dynamics.state_dict().items()}
        result = restorer.restore_(T, Z)
        self.assertTrue(result["success"])
        self.assertEqual(result["iterations"], 0)
        torch.testing.assert_close(T, before_T, rtol=0, atol=0)
        torch.testing.assert_close(Z, before_Z, rtol=0, atol=0)
        for name, value in dynamics.state_dict().items():
            torch.testing.assert_close(value, before_K[name], rtol=0, atol=0)

    def test_mixed_mode_candidate_is_repaired_or_rejected_without_relaxing_checks(self):
        config, context, dynamics, coordinates, restorer, _ = self.fixture()
        M = torch.tensor([[1.0, 0.0], [1.0, 0.2]], dtype=torch.float64)
        T, Z = coordinates.encode_orthogonal(M)
        before_T, before_Z = T.clone(), Z.clone()
        before_K = {name: value.detach().clone()
                    for name, value in dynamics.state_dict().items()}
        self.assertFalse(lm.report(coordinates.decode_orthogonal(T, Z),
                                    dynamics(), context)["passed"])
        result = restorer.restore_(T, Z)
        if result["success"]:
            actual = coordinates.decode_orthogonal(T, Z)
            self.assertTrue(lm.report(actual, dynamics(), context)["passed"])
            self.assertTrue(sr._local_rank_from_readout(
                actual.detach().numpy(), context["feature_jacobian_obs"].numpy(),
                2, config["losses"]["kappa"])["passed"])
            budgets = sr.weight_budget_check(actual.detach().numpy(), config)
            self.assertTrue(budgets["weight_within_budget"])
            self.assertTrue(budgets["bias_within_budget"])
        else:
            torch.testing.assert_close(T, before_T, rtol=0, atol=0)
            torch.testing.assert_close(Z, before_Z, rtol=0, atol=0)
            for name, value in dynamics.state_dict().items():
                torch.testing.assert_close(value, before_K[name], rtol=0, atol=0)

    def test_nearby_mixed_candidate_is_repaired_and_committed(self):
        _, context, dynamics, coordinates, restorer, _ = self.fixture()
        M = torch.tensor([[1.0, 0.002], [0.002, 1.0]], dtype=torch.float64)
        T, Z = coordinates.encode_orthogonal(M)
        before = T.clone()
        self.assertFalse(lm.report(coordinates.decode_orthogonal(T, Z),
                                    dynamics(), context)["passed"])
        result = restorer.restore_(T, Z)
        self.assertTrue(result["success"])
        self.assertGreater(result["iterations"], 0)
        self.assertLessEqual(result["iterations"], 25)
        self.assertGreater(float((T - before).norm()), 0)
        self.assertTrue(lm.report(coordinates.decode_orthogonal(T, Z),
                                   dynamics(), context)["passed"])

    def test_impossible_original_rank_budget_is_rejected_transactionally(self):
        _, _, dynamics, coordinates, restorer, W = self.fixture(kappa=1000.0)
        T, Z = coordinates.encode_orthogonal(W[:, 1:])
        before_T, before_Z = T.clone(), Z.clone()
        before_K = {name: value.detach().clone()
                    for name, value in dynamics.state_dict().items()}
        result = restorer.restore_(T, Z)
        self.assertFalse(result["success"])
        self.assertLessEqual(result["iterations"], 25)
        torch.testing.assert_close(T, before_T, rtol=0, atol=0)
        torch.testing.assert_close(Z, before_Z, rtol=0, atol=0)
        for name, value in dynamics.state_dict().items():
            torch.testing.assert_close(value, before_K[name], rtol=0, atol=0)


class NullspaceReadoutFitTests(unittest.TestCase):
    @staticmethod
    def options():
        return {"solver_steps": 8, "local_modal": {"enabled": True},
                "solver_revision": "exact_nullspace_projection_v1",
                "optimizer_state_policy": "retain"}

    def test_new_fit_preserves_source_and_saves_complete_matching_coordinates(self):
        config, selection, source, system = LocalModalTests.source_fixture()
        before = {name: tensor.detach().clone()
                  for name, tensor in source.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            train = prepare_data(system, config["data"], directory)["train"]
        phi = sr._feature_array(source, train["observations"],
                                n_times=train["observations"].shape[1])
        state_jacobian = sr._feature_jacobian_at_origin(source, system)
        fit = sr.fit_constrained_readout(source, phi, state_jacobian, config,
                                         selection, self.options())
        optimization = fit["optimization"]
        self.assertEqual(optimization["solver_revision"], "exact_nullspace_projection_v1")
        self.assertEqual(optimization["steps"], 8)
        self.assertEqual(optimization["al_update_count"], 0)
        self.assertEqual(optimization["optimizer_state_reset_count"], 0)
        self.assertIsNone(optimization["augmented_lagrangian"])
        self.assertTrue(optimization["initial_restoration"]["success"])
        self.assertEqual(optimization["restoration"]["accepted_updates"]
                         + optimization["restoration"]["rejected_updates"], 8)
        self.assertTrue(fit["hard_constraints_passed"])
        state = fit["optimizer_state_payload"]
        parameters, basis = state["primal_parameters"], state["coordinate_basis"]
        self.assertIn("T", parameters)
        self.assertIn("Z", parameters)
        self.assertNotIn("W", parameters)
        self.assertEqual(parameters["Z"].shape[1],
                         phi.shape[-1] - 1 - config["system"]["dimension"])
        M = parameters["T"] @ basis["U_t"].T + parameters["Z"] @ basis["N"].T
        reconstructed = torch.cat([-(M @ basis["phi0"][1:])[:, None], M], dim=1)
        torch.testing.assert_close(reconstructed, torch.as_tensor(fit["W_array"]),
                                   rtol=0, atol=1e-14)
        torch.testing.assert_close(state["effective_readout"], reconstructed,
                                   rtol=0, atol=1e-14)
        torch.testing.assert_close(state["local_observed_jacobian"],
                                   (parameters["T"] * basis["singular_values"]) @ basis["Vh"],
                                   rtol=0, atol=1e-14)
        for parameter_state in state["optimizer_state_dict"]["state"].values():
            self.assertEqual(float(parameter_state["step"]), 8)
        exported = sr.build_spec_model(copy.deepcopy(source), fit["W_array"], fit["K_matrix_array"],
                                        config["model"]["z_dim"], dtype=torch.float32)
        actual = sr._endpoint_modal_check(exported, selection, {"enabled": True})
        self.assertEqual(actual["checked_endpoint_dtype"], "float32")
        self.assertTrue(actual["passed"])
        for name, tensor in source.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in source.parameters()))

    def test_query_initializer_keeps_negative_real_and_rotation_block_domains(self):
        for kind in ("negative_real", "rotation"):
            with self.subTest(kind=kind):
                config, selection = small_config()
                if kind == "negative_real":
                    J = torch.diag(torch.tensor([-0.9, 0.8], dtype=torch.float64))
                    selection["blocks"][0]["center"] = -0.9
                    selection["blocks"][0]["eigenvalues"][0]["real"] = -0.9
                    _, _, _, context = LocalModalTests.fixture(J.numpy())
                else:
                    angle, modulus = 0.3, 0.85
                    J = modulus * torch.tensor([[math.cos(angle), -math.sin(angle)],
                                                 [math.sin(angle), math.cos(angle)]],
                                                dtype=torch.float64)
                    selection["blocks"] = [{
                        "kind": "rotation", "center": modulus, "radius": 0.02,
                        "angle": angle, "angle_radius": 0.02,
                        "certificate_contour_radius": 0.05,
                        "eigenvalues": [{"real": modulus * math.cos(angle),
                                         "imag": modulus * math.sin(angle)},
                                        {"real": modulus * math.cos(angle),
                                         "imag": -modulus * math.sin(angle)}]}]
                    _, _, _, context = LocalModalTests.fixture(J.numpy(), rotation=True)
                source = small_model(config, selection)
                W = lm.center_readout(torch.cat([
                    torch.zeros(2, 1, dtype=torch.float64),
                    torch.eye(2, dtype=torch.float64)], dim=1), context["phi0"])
                coordinates = lm.FullRankReadoutCoordinates(context, W)
                _, Z = coordinates.encode_orthogonal(W[:, 1:])
                before_Z = Z.clone()
                T = sr._initialize_query_tangent(coordinates, context, source.K,
                                                 context["feature_jacobian_obs"],
                                                 selection, config, Z)
                initialized = coordinates.decode_orthogonal(T, Z)
                self.assertTrue(lm.report(initialized, source.K_matrix(), context)["passed"])
                torch.testing.assert_close(source.K_matrix(), J, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(Z, before_Z, rtol=0, atol=0)
                domain = sr._k_domain_check(
                    {"blocks": sr._blocks_from_matrix(source.K_matrix().detach().numpy(),
                                                       selection)},
                    sr._original_domains(selection, config), sr.REAL_IMAGINARY_TOLERANCE)
                self.assertTrue(domain["legal"])
                self.assertTrue(all(parameter.requires_grad for parameter in source.K.parameters()))

    def test_new_solver_cli_saves_coordinate_state_and_rechecks_actual_fp32_endpoint(self):
        config, selection, source, system = LocalModalTests.source_fixture()
        analysis_config = copy.deepcopy(config)
        analysis_config["spectral_readout"] = {
            "method": "post_training_local_modal_readout", "solver": "constrained_readout",
            "seeds": [0], "device": "cpu", "trajectory_residual_horizons": [1, 2, 4, 8],
            **self.options(),
        }
        with tempfile.TemporaryDirectory(prefix="hijepa-nullspace-cli-") as directory:
            root = Path(directory)
            source_root, output = root / "source", root / "output"
            data_dir = source_root / "data" / sr.dataset_key(config)
            prepare_data(system, config["data"], data_dir)
            checkpoint = source_root / "e1_baseline_seed0" / "checkpoint.pt"
            sr.save_checkpoint(checkpoint, {
                "model": source.state_dict(), "config": config, "selection": selection,
                "phase": "source", "step": config["training"]["steps"],
                "source_sha256": sr.source_fingerprint(),
                "protocol_sha256": sr.protocol_fingerprint(config),
                "smoke_fixture_only": True, "source_training_performed": False,
            })
            original_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [checkpoint, *sorted(data_dir.iterdir())]}
            config_path = root / "analysis.json"
            sr.write_json(config_path, analysis_config)
            with contextlib.redirect_stdout(io.StringIO()):
                sr.main(["--config", str(config_path), "--source-root", str(source_root),
                         "--output", str(output), "--device", "cpu", "--no-parity"])
            seed_dir = output / "seed0"
            optimization = json.loads((seed_dir / "optimization.json").read_text())
            self.assertEqual(optimization["steps"], 8)
            self.assertEqual(optimization["solver_revision"], "exact_nullspace_projection_v1")
            metadata = optimization["optimizer_state_artifact"]
            state_path = seed_dir / metadata["path"]
            self.assertEqual(hashlib.sha256(state_path.read_bytes()).hexdigest(),
                             metadata["sha256"])
            state = sr.load_checkpoint(state_path)
            endpoint = sr.load_checkpoint(seed_dir / "spec_endpoint.pt")
            self.assertEqual(state["completed_steps"], 8)
            self.assertEqual(state["solver_revision"], "exact_nullspace_projection_v1")
            self.assertIn("T", state["primal_parameters"])
            self.assertIn("Z", state["primal_parameters"])
            self.assertIn("coordinate_basis", state)
            torch.testing.assert_close(state["effective_readout"].float(),
                                       torch.cat([endpoint["readout_bias"][:, None],
                                                  endpoint["readout_weight"]], dim=1).float(),
                                       rtol=0, atol=0)
            result = json.loads((seed_dir / "result.json").read_text())
            constraints = result["spectral_readout"]["constraints"]
            actual = constraints["local_modal_consistency"]
            self.assertEqual(actual["checked_endpoint_dtype"], "float32")
            self.assertTrue(actual["passed"])
            self.assertTrue(constraints["local_rank_kappa"]["passed"])
            self.assertTrue(constraints["k_parameter_domain"]["legal"])
            self.assertTrue(constraints["weight_budget"]["weight_within_budget"])
            self.assertTrue(constraints["weight_budget"]["bias_within_budget"])
            loaded = sr.load_spec_endpoint(seed_dir / "spec_endpoint.pt", allow_diagnostic=True)
            self.assertEqual(loaded.readout_weight.dtype, torch.float32)
            for path, digest in original_hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertFalse(list(seed_dir.glob("*.tmp")))
        self.assertFalse(root.exists())


class LocalDirectionProjectionTests(unittest.TestCase):
    @staticmethod
    def fixture(kind="real"):
        config, selection = small_config()
        if kind == "negative_real":
            J = torch.diag(torch.tensor([-0.9, 0.8], dtype=torch.float64))
            selection["blocks"][0]["center"] = -0.9
            selection["blocks"][0]["eigenvalues"][0]["real"] = -0.9
            _, _, _, context = LocalModalTests.fixture(J.numpy())
        elif kind == "rotation":
            angle, modulus = 0.3, 0.85
            J = modulus * torch.tensor([[math.cos(angle), -math.sin(angle)],
                                         [math.sin(angle), math.cos(angle)]],
                                        dtype=torch.float64)
            selection["blocks"] = [{
                "kind": "rotation", "center": modulus, "radius": 0.02,
                "angle": angle, "angle_radius": 0.02,
                "certificate_contour_radius": 0.05,
                "eigenvalues": [{"real": modulus * math.cos(angle),
                                 "imag": modulus * math.sin(angle)},
                                {"real": modulus * math.cos(angle),
                                 "imag": -modulus * math.sin(angle)}]}]
            _, _, _, context = LocalModalTests.fixture(J.numpy(), rotation=True)
        else:
            J = torch.diag(torch.tensor([0.9, 0.8], dtype=torch.float64))
            _, _, _, context = LocalModalTests.fixture(J.numpy())
        source = small_model(config, selection)
        W = lm.center_readout(torch.cat([
            torch.zeros(2, 1, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64)], dim=1), context["phi0"])
        coordinates = lm.FullRankReadoutCoordinates(context, W)
        T, Z = coordinates.encode_orthogonal(W[:, 1:])
        projector = lm.LocalDirectionProjector(
            coordinates, context, source.K,
            context["feature_jacobian_obs"].detach().clone(), config)
        return config, selection, context, source.K, coordinates, T, Z, projector

    def test_zero_modal_residual_still_constrains_transverse_matrix_direction(self):
        _, _, context, dynamics, coordinates, T, Z, projector = self.fixture()
        before_T, before_Z = T.clone(), Z.clone()
        before_K = {name: value.detach().clone()
                    for name, value in dynamics.state_dict().items()}
        desired = torch.tensor([[0.0, 0.1], [0.1, 0.0]], dtype=torch.float64)
        raw = {name: torch.zeros_like(parameter)
               for name, parameter in dynamics.named_parameters()}
        original_raw = {name: direction.clone() for name, direction in raw.items()}
        actual, actual_raw, diagnostics = projector.project(T, Z, desired, raw)
        self.assertTrue(diagnostics["success"])
        self.assertTrue(diagnostics["model_feasible"])
        self.assertTrue(diagnostics["requires_original_constraint_and_descent_backtracking"])
        self.assertLessEqual(diagnostics["iterations"], 64)
        self.assertEqual(set(actual_raw), set(raw))
        # d||F|| at F=0 is insufficient: the full matrix linearization must
        # see these off-diagonal directions in two distinct real eigenblocks.
        self.assertLess(float(actual.norm()), float(desired.norm()) * 0.1)
        self.assertFalse(lm.report(coordinates.decode_orthogonal(T + desired, Z),
                                    dynamics(), context)["passed"])
        torch.testing.assert_close(T, before_T, rtol=0, atol=0)
        torch.testing.assert_close(Z, before_Z, rtol=0, atol=0)
        for name, value in dynamics.state_dict().items():
            torch.testing.assert_close(value, before_K[name], rtol=0, atol=0)
        for name, direction in raw.items():
            torch.testing.assert_close(direction, original_raw[name], rtol=0, atol=0)

    def test_block_kinds_parameter_layout_and_norm_bounds_are_preserved(self):
        for kind in ("negative_real", "rotation"):
            with self.subTest(kind=kind):
                config, _, _, dynamics, coordinates, T, Z, projector = self.fixture(kind)
                raw = {name: torch.zeros_like(parameter)
                       for name, parameter in dynamics.named_parameters()}
                actual, actual_raw, diagnostics = projector.project(
                    T, Z, torch.eye(2, dtype=torch.float64) * 20, raw)
                self.assertTrue(diagnostics["success"])
                self.assertEqual(actual.shape, T.shape)
                self.assertEqual(set(actual_raw), set(raw))
                for name, direction in actual_raw.items():
                    self.assertEqual(direction.shape, raw[name].shape)
                    self.assertTrue(bool(torch.isfinite(direction).all()))
                W = coordinates.decode_orthogonal(T + actual, Z)
                if (float(W[:, 1:].norm()) > config["model"]["weight_frobenius_bound"]
                        or float(W[:, 0].norm()) > config["model"]["bias_norm_bound"]):
                    self.assertFalse(diagnostics["model_feasible"])
                model = projector.linearization(T, Z)
                packed = torch.cat([actual.reshape(-1),
                                    *[actual_raw[name].reshape(-1)
                                      for name in projector.parameter_names]])
                value = model["a"] + model["B"] @ packed
                independent = projector._model_violations(value, model["blocks"])
                self.assertAlmostEqual(diagnostics["max_constraint_violation"],
                                       max(independent.values()), places=12)
                projected_product = projector._prox_product(value, model["blocks"])
                self.assertLessEqual(max(projector._model_violations(
                    projected_product, model["blocks"]).values()), 1e-12)
                self.assertTrue(diagnostics["requires_original_constraint_and_descent_backtracking"])
                self.assertTrue(all(parameter.requires_grad for parameter in dynamics.parameters()))

    def test_admissible_raw_directions_keep_every_original_k_parameter_variable(self):
        for kind in ("real", "negative_real", "rotation"):
            with self.subTest(kind=kind):
                _, _, _, dynamics, _, T, Z, projector = self.fixture(kind)
                raw = {name: torch.full_like(parameter, 1e-4)
                       for name, parameter in dynamics.named_parameters()}
                actual, actual_raw, diagnostics = projector.project(
                    T, Z, torch.eye(2, dtype=torch.float64) * 0.1, raw)
                self.assertTrue(diagnostics["success"])
                self.assertTrue(diagnostics["model_feasible"])
                for name, direction in actual_raw.items():
                    self.assertGreater(float(direction.abs().max()), 0)
                    torch.testing.assert_close(direction, raw[name], rtol=0, atol=0)

    def test_normalized_matrix_linearization_differentiates_the_cholesky_denominator(self):
        for kind in ("real", "rotation"):
            with self.subTest(kind=kind):
                _, _, _, dynamics, _, T, Z, projector = self.fixture(kind)
                T = T + torch.tensor([[0.0, 0.001], [0.002, 0.0]], dtype=torch.float64)
                model = projector.linearization(T, Z)
                modal_stop = max(block["stop"] for block in model["blocks"]
                                 if block["kind"] == "spectral_ball")
                scale_direction = torch.cat([T.reshape(-1),
                                             *[torch.zeros_like(parameter).reshape(-1)
                                               for parameter in dynamics.parameters()]])
                self.assertGreater(float(model["a"][:modal_stop].norm()), 0.1)
                # F(s C,K) = F(C,K) for positive uniform s. Omitting the
                # Cholesky denominator derivative would yield dF=F here.
                directional = model["B"][:modal_stop] @ scale_direction
                torch.testing.assert_close(directional, torch.zeros_like(directional),
                                           rtol=0, atol=1e-9)


class ReadoutRuntimeTests(unittest.TestCase):
    @staticmethod
    def payload(step=1):
        return {"completed_steps": step, "planned_steps": 8,
                "parameters": {"T": torch.eye(2, dtype=torch.float64)},
                "optimizer_states": {"null": None, "local": None},
                "coordinates": {"U_t": torch.eye(2, dtype=torch.float64)},
                "rng_state": {"numpy": {"state": 12},
                              "torch": torch.random.get_rng_state().clone(), "python": None},
                "statistics": {"accepted_updates": step}}

    def test_latest_checkpoint_is_atomic_and_identity_changes_refuse_resume(self):
        identity = {"solver_revision": "feasible_block_sqp_v1", "source_sha256": "a" * 64,
                    "train_sha256": "b" * 64, "parameter_layout": ["T", "Z", "K.raw"]}
        with tempfile.TemporaryDirectory(prefix="hijepa-runtime-") as directory:
            root = Path(directory)
            runtime = rr.ReadoutRuntime(root, identity, emit_stdout=False)
            payload = self.payload()
            before_payload = copy.deepcopy(payload)
            runtime.save(payload, {"completed_steps": 1, "elapsed_seconds": 0.2,
                                   "accepted_updates": 1, "rejected_updates": 0})
            checkpoint_path = root / "readout_latest.pt"
            before_checkpoint = checkpoint_path.read_bytes()
            before_progress = (root / "progress.json").read_bytes()
            self.assertEqual(runtime.load()["completed_steps"], 1)
            torch.testing.assert_close(payload["parameters"]["T"],
                                       before_payload["parameters"]["T"], rtol=0, atol=0)
            self.assertNotIn("kind", payload)
            with mock.patch.object(rr.os, "replace", side_effect=OSError("simulated disk failure")):
                with self.assertRaisesRegex(OSError, "simulated disk failure"):
                    runtime.save(self.payload(2), {"completed_steps": 2})
            self.assertEqual(checkpoint_path.read_bytes(), before_checkpoint)
            self.assertEqual((root / "progress.json").read_bytes(), before_progress)
            self.assertFalse(list(root.glob("*.tmp")))
            for key in ("source_sha256", "train_sha256", "parameter_layout", "solver_revision"):
                changed = copy.deepcopy(identity)
                changed[key] = ["other"] if key == "parameter_layout" else "incompatible"
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity"):
                    rr.ReadoutRuntime(root, changed, emit_stdout=False).load()


class FeasibleBlockReadoutTests(unittest.TestCase):
    @staticmethod
    def fixture():
        config, selection, source, system = LocalModalTests.source_fixture()
        config["training"]["checkpoint_every"] = 4
        with tempfile.TemporaryDirectory() as directory:
            train = prepare_data(system, config["data"], directory)["train"]
        phi = sr._feature_array(source, train["observations"],
                                n_times=train["observations"].shape[1])
        state_jacobian = sr._feature_jacobian_at_origin(source, system)
        return config, selection, source, phi, state_jacobian

    @staticmethod
    def options():
        return {"solver_steps": 8, "local_modal": {"enabled": True},
                "solver_revision": "feasible_block_sqp_v1", "optimizer_state_policy": "retain"}

    def assert_nested_identical(self, first, second):
        if torch.is_tensor(first):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_identical(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_identical(a, b)
        else:
            self.assertEqual(first, second)

    def test_real_updates_reduce_independently_recomputed_fixed_train_objective(self):
        config, selection, source, phi, state_jacobian = self.fixture()
        before_source = {name: value.detach().clone()
                         for name, value in source.state_dict().items()}
        fixed = phi[:config["training"]["batch_size"], 0].copy()
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                sr, "_sample_features", side_effect=lambda *args, **kwargs: fixed.copy()):
            fit = sr.fit_constrained_readout(source, phi, state_jacobian, config,
                                             selection, self.options())
        optimization = fit["optimization"]
        updates = optimization["block_updates"]
        self.assertGreater(updates["accepted_updates"], 0)
        state = fit["optimizer_state_payload"]
        initial_W = state["statistics"]["initial_weight"]
        initial_K = state["statistics"]["initial_K"]
        final_W = torch.as_tensor(fit["W_array"])
        final_K = torch.as_tensor(fit["K_matrix_array"])
        delta = float((final_W - initial_W).norm() + (final_K - initial_K).norm())
        self.assertGreater(delta, 1e-10)
        support = sr._source_training_support(config, phi.shape[1])
        horizons = sorted({1, int(config["losses"]["k"]), 2 * int(config["losses"]["k"])})
        moments = {q: torch.from_numpy(values) for q, values in
                   sr._paired_moments(phi, support["last_start"], horizons).items()}
        audit = torch.as_tensor(fixed)
        jacobian = torch.as_tensor(state_jacobian)
        initial_loss, _ = sr._readout_objective(initial_W, initial_K, moments, audit,
                                                jacobian, config, source.coarse_blocks)
        final_loss, _ = sr._readout_objective(final_W, final_K, moments, audit,
                                              jacobian, config, source.coarse_blocks)
        self.assertLess(float(final_loss), float(initial_loss) - 1e-10)
        self.assertAlmostEqual(float(initial_loss),
                               optimization["initial_fixed_batch_terms"]["total"], places=10)
        self.assertAlmostEqual(float(final_loss),
                               optimization["final_fixed_batch_terms"]["total"], places=10)
        self.assertTrue(fit["hard_constraints_passed"])
        endpoint = sr.build_spec_model(copy.deepcopy(source), fit["W_array"],
                                        fit["K_matrix_array"], config["model"]["z_dim"],
                                        dtype=torch.float32)
        self.assertTrue(sr._endpoint_modal_check(endpoint, selection, {"enabled": True})["passed"])
        self.assertTrue(sr.local_rank_check(endpoint, sr.make_system(config["system"]),
                                            config, "full")["passed"])
        actual_W = torch.cat([endpoint.readout_bias[:, None], endpoint.readout_weight], dim=1)
        budget = sr.weight_budget_check(actual_W.detach().double().numpy(), config)
        self.assertTrue(budget["weight_within_budget"])
        self.assertTrue(budget["bias_within_budget"])
        actual_K = endpoint.K_matrix().detach().double().numpy()
        domain = sr._k_domain_check({"blocks": sr._blocks_from_matrix(actual_K, selection)},
                                    sr._original_domains(selection, config),
                                    sr.REAL_IMAGINARY_TOLERANCE)
        self.assertTrue(domain["legal"])
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, before_source[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in source.parameters()))

    def test_zero_acceptance_and_fake_success_without_motion_fail_numerical_progress(self):
        config, selection, source, phi, state_jacobian = self.fixture()
        options = {**self.options(), "solver_steps": 2}
        for accepted in (False, True):
            with self.subTest(fake_accepted=accepted):
                record = {"accepted": accepted, "reason": "synthetic_no_motion",
                          "parameter_delta": 0.0, "backtracks": 0, "alpha": 0.0,
                          "direction_model": None}
                if accepted:
                    with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                            sr, "_feasible_block_update", return_value=record):
                        with self.assertRaisesRegex(RuntimeError, "acceptance without parameter movement"):
                            sr.fit_constrained_readout(source, phi, state_jacobian, config,
                                                       selection, options)
                    continue
                with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                        sr, "_feasible_block_update", return_value=record):
                    fit = sr.fit_constrained_readout(source, phi, state_jacobian, config,
                                                     selection, options)
                self.assertTrue(fit["hard_constraints_passed"])
                progress = fit["optimization"]["numerical_progress"]
                self.assertFalse(progress["passed"])
                self.assertFalse(progress["parameters_changed"])
                self.assertEqual(progress["status"], "no_effective_updates")
                initial = fit["optimizer_state_payload"]["statistics"]
                np.testing.assert_array_equal(fit["W_array"], initial["initial_weight"].numpy())
                np.testing.assert_array_equal(fit["K_matrix_array"], initial["initial_K"].numpy())

    def test_no_descent_step_is_rejected_and_local_rejection_keeps_accepted_null_step(self):
        stationary = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        optimizer = torch.optim.Adam([stationary], lr=0.01)
        result = sr._feasible_block_update(
            [stationary], optimizer, lambda: stationary.square().sum(),
            lambda: {"passed": True})
        self.assertFalse(result["accepted"])
        self.assertEqual(result["parameter_delta"], 0)
        self.assertEqual(float(stationary.detach()), 0)
        self.assertFalse(optimizer.state)

        Z = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        T = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        null_optimizer = torch.optim.Adam([Z], lr=0.01)
        local_optimizer = torch.optim.Adam([T], lr=0.01)
        objective = lambda: (Z - 1).square().sum() + (T - 1).square().sum()
        null_result = sr._feasible_block_update(
            [Z], null_optimizer, objective, lambda: {"passed": True})
        self.assertTrue(null_result["accepted"])
        saved_Z = Z.detach().clone()
        saved_null_state = copy.deepcopy(null_optimizer.state_dict())
        local_result = sr._feasible_block_update(
            [T], local_optimizer, objective, lambda: {"passed": True},
            projector=lambda directions: ([torch.zeros_like(value) for value in directions],
                                           {"success": False, "reason": "synthetic_local_rejection"}))
        self.assertFalse(local_result["accepted"])
        torch.testing.assert_close(Z, saved_Z, rtol=0, atol=0)
        self.assert_nested_identical(null_optimizer.state_dict(), saved_null_state)
        self.assertEqual(float(T.detach()), 0)
        self.assertFalse(local_optimizer.state)

    def test_explicit_resume_without_checkpoint_cannot_silently_initialize(self):
        config, selection, source, phi, state_jacobian = self.fixture()
        with tempfile.TemporaryDirectory(prefix="hijepa-missing-resume-") as directory:
            with mock.patch.object(sr, "_initialize_query_tangent", side_effect=AssertionError(
                    "Missing resume checkpoint must fail before initialization")):
                with self.assertRaisesRegex(FileNotFoundError, "missing"):
                    sr.fit_constrained_readout(source, phi, state_jacobian, config,
                                               selection, self.options(),
                                               runtime_dir=Path(directory), resume=True)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_interrupted_fresh_run_cannot_leave_a_previous_same_identity_checkpoint(self):
        config, selection, source, phi, state_jacobian = self.fixture()
        with tempfile.TemporaryDirectory(prefix="hijepa-fresh-boundary-") as directory:
            output = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                sr.fit_constrained_readout(source, phi, state_jacobian, config,
                                           selection, self.options(), runtime_dir=output)
            self.assertEqual(sr.load_checkpoint(output / "readout_latest.pt")["completed_steps"], 8)
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    sr, "_initialize_query_tangent", side_effect=RuntimeError("interrupted fresh initialization")):
                with self.assertRaisesRegex(RuntimeError, "interrupted fresh initialization"):
                    sr.fit_constrained_readout(source, phi, state_jacobian, config,
                                               selection, self.options(), runtime_dir=output)
            self.assertFalse((output / "readout_latest.pt").exists())
            with mock.patch.object(sr, "_initialize_query_tangent", side_effect=AssertionError(
                    "Resume cannot reuse stale state or silently start over")):
                with self.assertRaisesRegex(FileNotFoundError, "missing"):
                    sr.fit_constrained_readout(source, phi, state_jacobian, config,
                                               selection, self.options(), runtime_dir=output,
                                               resume=True)

    def test_cli_zero_effective_updates_reports_failure_and_preserves_source(self):
        config, selection, source, system = LocalModalTests.source_fixture()
        config["training"]["checkpoint_every"] = 1
        analysis_config = copy.deepcopy(config)
        analysis_config["spectral_readout"] = {
            "method": "post_training_local_modal_readout", "solver": "constrained_readout",
            "seeds": [0], "device": "cpu", "trajectory_residual_horizons": [1, 2, 4, 8],
            **self.options(), "solver_steps": 2,
        }
        with tempfile.TemporaryDirectory(prefix="hijepa-stalled-cli-") as directory:
            root = Path(directory)
            source_root, output = root / "source", root / "output"
            data_dir = source_root / "data" / sr.dataset_key(config)
            prepare_data(system, config["data"], data_dir)
            checkpoint = source_root / "e1_baseline_seed0" / "checkpoint.pt"
            sr.save_checkpoint(checkpoint, {
                "model": source.state_dict(), "config": config, "selection": selection,
                "phase": "source", "step": config["training"]["steps"],
                "source_sha256": sr.source_fingerprint(),
                "protocol_sha256": sr.protocol_fingerprint(config),
                "smoke_fixture_only": True, "source_training_performed": False,
            })
            hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [checkpoint, *sorted(data_dir.iterdir())]}
            config_path = root / "analysis.json"
            sr.write_json(config_path, analysis_config)
            rejected = {"accepted": False, "reason": "synthetic_no_motion",
                        "parameter_delta": 0.0, "backtracks": 0, "alpha": 0.0,
                        "direction_model": None}
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    sr, "_feasible_block_update", return_value=rejected):
                sr.main(["--config", str(config_path), "--source-root", str(source_root),
                         "--output", str(output), "--device", "cpu", "--no-parity"])
            result = json.loads((output / "seed0" / "result.json").read_text())
            self.assertFalse(result["conclusion"]["recovery_success"])
            self.assertIn("numerical_solver_no_effective_updates",
                          result["conclusion"]["failure_reasons"])
            optimization = json.loads((output / "seed0" / "optimization.json").read_text())
            self.assertFalse(optimization["numerical_progress"]["passed"])
            self.assertEqual(optimization["block_updates"]["accepted_updates"], 0)
            durable = sr.load_checkpoint(output / "seed0" / "readout_latest.pt")
            self.assertEqual(durable["completed_steps"], 2)
            for path, expected in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_new_cli_exports_verified_fp32_endpoint_report_and_resumable_state(self):
        config, selection, source, system = LocalModalTests.source_fixture()
        config["training"]["checkpoint_every"] = 4
        analysis_config = copy.deepcopy(config)
        analysis_config["spectral_readout"] = {
            "method": "post_training_local_modal_readout", "solver": "constrained_readout",
            "seeds": [0], "device": "cpu", "trajectory_residual_horizons": [1, 2, 4, 8],
            **self.options(),
        }
        with tempfile.TemporaryDirectory(prefix="hijepa-feasible-cli-") as directory:
            root = Path(directory)
            source_root, output = root / "source", root / "output"
            data_dir = source_root / "data" / sr.dataset_key(config)
            prepare_data(system, config["data"], data_dir)
            checkpoint = source_root / "e1_baseline_seed0" / "checkpoint.pt"
            sr.save_checkpoint(checkpoint, {
                "model": source.state_dict(), "config": config, "selection": selection,
                "phase": "source", "step": config["training"]["steps"],
                "source_sha256": sr.source_fingerprint(),
                "protocol_sha256": sr.protocol_fingerprint(config),
                "smoke_fixture_only": True, "source_training_performed": False,
            })
            hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [checkpoint, *sorted(data_dir.iterdir())]}
            config_path = root / "analysis.json"
            sr.write_json(config_path, analysis_config)
            with contextlib.redirect_stdout(io.StringIO()):
                returned = sr.main(["--config", str(config_path), "--source-root", str(source_root),
                                    "--output", str(output), "--device", "cpu", "--no-parity"])
            seed_dir = output / "seed0"
            optimization = json.loads((seed_dir / "optimization.json").read_text())
            self.assertEqual(optimization["solver_revision"], "feasible_block_sqp_v1")
            self.assertTrue(optimization["numerical_progress"]["passed"])
            self.assertGreater(optimization["block_updates"]["accepted_updates"], 0)
            result = json.loads((seed_dir / "result.json").read_text())
            constraints = result["spectral_readout"]["constraints"]
            self.assertTrue(constraints["local_modal_consistency"]["passed"])
            self.assertEqual(constraints["local_modal_consistency"]["checked_endpoint_dtype"],
                             "float32")
            self.assertTrue(constraints["local_rank_kappa"]["passed"])
            self.assertTrue(constraints["weight_budget"]["weight_within_budget"])
            self.assertTrue(constraints["weight_budget"]["bias_within_budget"])
            self.assertTrue(constraints["k_parameter_domain"]["legal"])
            loaded = sr.load_spec_endpoint(seed_dir / "spec_endpoint.pt")
            self.assertEqual(loaded.readout_weight.dtype, torch.float32)
            durable = sr.load_checkpoint(seed_dir / "readout_latest.pt")
            self.assertEqual(durable["completed_steps"], 8)
            self.assertIn("null", durable["optimizer_states"])
            self.assertIn("local", durable["optimizer_states"])
            self.assertIn("numpy", durable["rng_state"])
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["conclusion"], returned["conclusion"])
            self.assertTrue((output / "REPORT.md").is_file())
            for path, expected in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_intermediate_checkpoint_resume_is_exact_and_progress_is_live(self):
        config, selection, source, phi, state_jacobian = self.fixture()
        initial_torch_rng = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory(prefix="hijepa-resume-") as directory:
            root = Path(directory)
            continuous_dir, resumed_dir = root / "continuous", root / "resumed"
            with contextlib.redirect_stdout(io.StringIO()):
                continuous = sr.fit_constrained_readout(
                    source, phi, state_jacobian, config, selection, self.options(),
                    runtime_dir=continuous_dir)
            torch.random.set_rng_state(initial_torch_rng)
            real_save = rr.ReadoutRuntime.save
            observed_steps = []

            class InterruptedAfterDurableSave(RuntimeError):
                pass

            def interrupt_after_save(runtime, payload, progress):
                saved = real_save(runtime, payload, progress)
                record = json.loads((resumed_dir / "progress.json").read_text())
                self.assertEqual(record["completed_steps"], payload["completed_steps"])
                self.assertIn("elapsed_seconds", record)
                observed_steps.append(payload["completed_steps"])
                if payload["completed_steps"] == 4:
                    raise InterruptedAfterDurableSave()
                return saved

            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    rr.ReadoutRuntime, "save", new=interrupt_after_save):
                with self.assertRaises(InterruptedAfterDurableSave):
                    sr.fit_constrained_readout(
                        source, phi, state_jacobian, config, selection, self.options(),
                        runtime_dir=resumed_dir)
            self.assertIn(1, observed_steps)
            self.assertIn(4, observed_steps)
            partial = sr.load_checkpoint(resumed_dir / "readout_latest.pt")
            self.assertEqual(partial["completed_steps"], 4)
            self.assertEqual(partial["planned_steps"], 8)
            checkpoint_before_rejection = (resumed_dir / "readout_latest.pt").read_bytes()
            progress_before_rejection = (resumed_dir / "progress.jsonl").read_bytes()
            changed_phi = phi.copy()
            changed_phi[0, 0, 1] += 1e-5
            changed_config = copy.deepcopy(config)
            changed_config["training"]["learning_rate"] *= 2
            changed_selection = copy.deepcopy(selection)
            changed_selection["certificate"]["matrix_error_bound"] *= 2
            changed_source = copy.deepcopy(source)
            with torch.no_grad():
                next(changed_source.parameters()).view(-1)[0].add_(1e-5)
            for name, altered_source, altered_phi, altered_config, altered_selection, altered_options in (
                    ("train", source, changed_phi, config, selection, self.options()),
                    ("config", source, phi, changed_config, selection, self.options()),
                    ("selection", source, phi, config, changed_selection, self.options()),
                    ("source", changed_source, phi, config, selection, self.options()),
                    ("planned_budget", source, phi, config, selection,
                     {**self.options(), "solver_steps": 7})):
                with self.subTest(incompatible=name), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "identity"):
                        sr.fit_constrained_readout(
                            altered_source, altered_phi, state_jacobian, altered_config,
                            altered_selection, altered_options, runtime_dir=resumed_dir, resume=True)
                self.assertEqual((resumed_dir / "readout_latest.pt").read_bytes(),
                                 checkpoint_before_rejection)
                self.assertEqual((resumed_dir / "progress.jsonl").read_bytes(),
                                 progress_before_rejection)
            for corruption in ("coordinate_basis", "tangent_shape", "parameter_layout"):
                broken = copy.deepcopy(partial)
                if corruption == "coordinate_basis":
                    broken["coordinates"]["N"][0, 0] += 1e-5
                elif corruption == "tangent_shape":
                    broken["parameters"]["T"] = broken["parameters"]["T"][:, :-1]
                else:
                    broken["parameters"]["K.unexpected"] = torch.zeros(1, dtype=torch.float64)
                sr.save_checkpoint(resumed_dir / "readout_latest.pt", broken)
                corrupted_bytes = (resumed_dir / "readout_latest.pt").read_bytes()
                try:
                    with self.subTest(corruption=corruption), contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(ValueError):
                            sr.fit_constrained_readout(
                                source, phi, state_jacobian, config, selection, self.options(),
                                runtime_dir=resumed_dir, resume=True)
                    self.assertEqual((resumed_dir / "readout_latest.pt").read_bytes(), corrupted_bytes)
                    self.assertEqual((resumed_dir / "progress.jsonl").read_bytes(),
                                     progress_before_rejection)
                finally:
                    (resumed_dir / "readout_latest.pt").write_bytes(checkpoint_before_rejection)
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    sr, "_initialize_query_tangent", side_effect=AssertionError(
                        "Resume must load saved coordinates instead of initializing again")):
                resumed = sr.fit_constrained_readout(
                    source, phi, state_jacobian, config, selection, self.options(),
                    runtime_dir=resumed_dir, resume=True)
            np.testing.assert_array_equal(continuous["W_array"], resumed["W_array"])
            np.testing.assert_array_equal(continuous["K_matrix_array"], resumed["K_matrix_array"])
            full_state = sr.load_checkpoint(continuous_dir / "readout_latest.pt")
            resumed_state = sr.load_checkpoint(resumed_dir / "readout_latest.pt")
            self.assertEqual(resumed_state["completed_steps"], 8)
            for key in ("parameters", "optimizer_states", "coordinates", "rng_state"):
                with self.subTest(key=key):
                    self.assert_nested_identical(full_state[key], resumed_state[key])
            self.assertEqual({key: value for key, value in
                              continuous["optimization"]["block_updates"].items()
                              if key != "elapsed_seconds"},
                             {key: value for key, value in
                              resumed["optimization"]["block_updates"].items()
                              if key != "elapsed_seconds"})
            self.assertGreater(resumed["optimization"]["block_updates"]["accepted_updates"], 0)
            records = [json.loads(line) for line in
                       (resumed_dir / "progress.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["completed_steps"], 8)
            elapsed = [record["elapsed_seconds"] for record in records]
            self.assertEqual(elapsed, sorted(elapsed))


class EndpointModelTests(unittest.TestCase):
    def test_coarse_matches_feature_readout_and_k(self):
        config, selection = small_config()
        source = small_model(config, selection)
        feature_dim = 1 + config["model"]["z_dim"] + config["model"]["width"]
        W = np.random.default_rng(3).normal(size=(2, feature_dim))
        K = np.array([[0.9, -0.1], [0.1, 0.9]])
        model = sr.SpectralReadoutModel(source, W, K, config["model"]["z_dim"])
        model.eval()
        x = torch.randn(5, 2, dtype=torch.float64)
        expected = sr.feature_map(source, x) @ torch.as_tensor(W, dtype=torch.float64).T
        torch.testing.assert_close(model.coarse(x), expected)
        c = torch.randn(4, 2, dtype=torch.float64)
        torch.testing.assert_close(model.predict_coarse(c, 1), c @ torch.as_tensor(K).T)
        torch.testing.assert_close(model.predict_coarse(c, 2),
                                   c @ torch.as_tensor(K @ K).T)
        counts = model.parameter_counts()
        self.assertEqual(counts["readout_weight"], 2 * (feature_dim - 1))
        self.assertEqual(counts["readout_bias"], 2)
        count = lambda module: sum(p.numel() for p in module.parameters())
        active_network = count(source.E) + count(source.G[:-1]) + count(source.F)
        self.assertEqual(counts["registered_parameters"], count(source))
        self.assertEqual(counts["active_network_parameters"], active_network)
        self.assertEqual(counts["fixed_readout_coefficients"], W.size)
        self.assertEqual(counts["fixed_dynamics_coefficients"], K.size)
        self.assertEqual(counts["unused_source_parameters"], count(source.G[-1]) + count(source.K))
        self.assertEqual(counts["total_active"], active_network + W.size + K.size)

    def test_endpoint_loader_roundtrip(self):
        config, selection = small_config()
        source = small_model(config, selection)
        feature_dim = 1 + config["model"]["z_dim"] + config["model"]["width"]
        W = np.random.default_rng(5).normal(size=(2, feature_dim))
        K = np.diag([0.9, 0.8])
        payload = {
            "kind": "hijepa_v3_spectral_endpoint", "config": config,
            "selection": selection, "source_model": source.state_dict(),
            "readout_weight": torch.as_tensor(W[:, 1:]),
            "readout_bias": torch.as_tensor(W[:, 0]),
            "k_matrix": torch.as_tensor(K), "z_dim": config["model"]["z_dim"],
            "valid_main_model": True, "diagnostic_only": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec_endpoint.pt"
            torch.save(payload, path)
            loaded = sr.load_spec_endpoint(path, dtype=torch.float64)
        x = torch.randn(6, 2, dtype=torch.float64)
        original = sr.SpectralReadoutModel(source, W, K, config["model"]["z_dim"])
        torch.testing.assert_close(loaded.coarse(x), original.coarse(x))

    def test_invalid_endpoint_requires_explicit_diagnostic_loading(self):
        config, selection = small_config()
        source = small_model(config, selection)
        feature_dim = 1 + config["model"]["z_dim"] + config["model"]["width"]
        W = np.random.default_rng(6).normal(size=(2, feature_dim))
        K = np.array([[0.9, -0.1], [0.1, 0.9]])
        payload = {
            "kind": "hijepa_v3_spectral_endpoint", "config": config,
            "selection": selection, "source_model": source.state_dict(),
            "readout_weight": torch.as_tensor(W[:, 1:]),
            "readout_bias": torch.as_tensor(W[:, 0]),
            "k_matrix": torch.as_tensor(K), "z_dim": config["model"]["z_dim"],
            "valid_main_model": False, "diagnostic_only": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic_endpoint.pt"
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                sr.load_spec_endpoint(path, dtype=torch.float64)
            loaded = sr.load_spec_endpoint(path, dtype=torch.float64, allow_diagnostic=True)
        x = torch.randn(6, 2, dtype=torch.float64)
        original = sr.SpectralReadoutModel(source, W, K, config["model"]["z_dim"])
        torch.testing.assert_close(loaded.coarse(x), original.coarse(x))


class ParityAndConfigTests(unittest.TestCase):
    def test_cpu_cuda_feature_parity(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        config, selection = small_config()
        cpu = small_model(config, selection).double()
        cuda = small_model(config, selection).to(device="cuda", dtype=torch.float64)
        x = torch.randn(32, 2, dtype=torch.float64)
        a = sr.feature_map(cpu, x)
        b = sr.feature_map(cuda, x.cuda()).cpu()
        self.assertTrue(torch.allclose(a, b, rtol=1e-10, atol=1e-12))

    def test_v3_config_fits_only_constrained_readout_and_retains_protocol(self):
        config = load_config(REPO / "configs" / "base_formal_v3.json")
        self.assertIn("spectral_readout", config)
        self.assertEqual(config["spectral_readout"]["method"],
                         "post_training_constrained_readout")
        self.assertEqual(config["spectral_readout"]["solver"], "constrained_readout")
        self.assertEqual(config["spectral_readout"]["c00_relative_eigenvalue_threshold"], 1e-8)
        # Source losses stay unchanged; v3 now records additional readout fitting.
        self.assertEqual(config["losses"]["coarse_floor_mode"], "spectral_barrier")
        self.assertNotIn("spectral_anchor_enabled", config["losses"])
        self.assertTrue((REPO / "scripts" / "run_v3_spectral.sh").is_file())

    def test_v2_protocol_unchanged(self):
        v2 = load_config(REPO / "configs" / "base_formal_v2.json")
        self.assertEqual(v2["losses"]["coarse_floor_mode"], "spectral_barrier")
        self.assertNotIn("spectral_readout", v2)


if __name__ == "__main__":
    unittest.main()
