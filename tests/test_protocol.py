"""Behavioral/mathematical checks for the complete implementation contract."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hijepa.config import validate, expand_matrix
from hijepa.data import prepare_data, load_data
from hijepa.systems import make_system, compatibility_audit
from hijepa.spectral import select_spectrum
from hijepa.models import build_model, project_network_budget
from hijepa.losses import compute_losses, covariance_floor, state_composite_jacobian
from hijepa.evaluation import (evaluate_model, evaluate_transfer_thresholds,
                               fit_signed_permutation, apply_signed_permutation,
                               cross_seed_consistency)
from hijepa.diagnostics import pseudo_solution_audit
from hijepa.io import load_checkpoint, protocol_fingerprint, read_json
from hijepa.runner import sample_batch, train_run, configure_runtime, run_matrix
from hijepa.parity import export_bundle, check_bundle


def context(variant="full", family="real"):
    cfg = validate({"model": {"variant": variant, "width": 16, "z_dim": 8},
                    "system": {"family": family},
                    "data": {"train_trajectories": 16, "calibration_trajectories": 8,
                             "test_trajectories": 8, "length": 20},
                    "training": {"steps": 4, "batch_size": 8, "checkpoint_every": 2},
                    "evaluation": {"horizon": 4, "probe_max_samples": 128},
                    "losses": {"sigreg_slices": 8}})
    configure_runtime(cfg)
    system = make_system(cfg["system"])
    selection = select_spectrum(system, cfg["selection"])
    if not selection["valid_for_training"]:
        raise RuntimeError(selection)
    audit = compatibility_audit(system, selection, cfg["data"], cfg["losses"])
    selection["certificate"].update({key: audit[key] for key in
                                    ("reference_contraction_upper_bound", "spectral_spread_order")})
    model = build_model(cfg, selection)
    project_network_budget(model)
    return cfg, system, selection, model


class ProtocolTests(unittest.TestCase):
    def test_data_roundtrip_and_independent_splits(self):
        cfg, system, _, _ = context()
        with tempfile.TemporaryDirectory() as directory:
            first = prepare_data(system, cfg["data"], directory)
            loaded = load_data(directory)
            for split in first:
                np.testing.assert_array_equal(first[split]["observations"], loaded[split]["observations"])
            self.assertFalse(np.array_equal(first["train"]["states"][:8, 0], first["test"]["states"][:, 0]))
            file = Path(directory) / "test.npz"
            file.write_bytes(file.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                load_data(directory)

    def test_observation_inverse_and_normal_state_linearization(self):
        for family in ("real", "oscillatory", "near_repeated", "coupled", "resonance"):
            cfg, system, _, _ = context(family=family)
            states = np.random.default_rng(4).normal(size=(12, 4))*.2
            np.testing.assert_allclose(system.inverse_observe(system.observe(states)), states, atol=1e-12)
            eps = 1e-5
            j = ((system.step(eps*np.eye(4))-system.step(-eps*np.eye(4)))/(2*eps)).T
            self.assertLess(np.linalg.norm(j@j.T-j.T@j), 1e-7)

    def test_selector_has_no_truth_label_access(self):
        cfg, system, expected, _ = context()
        class QueryAccess:
            dimension = system.dimension
            local_query = staticmethod(system.local_query)
            query_derivative_budget = staticmethod(system.query_derivative_budget)
        selected = select_spectrum(QueryAccess(), cfg["selection"])
        self.assertEqual(selected["m"], expected["m"])
        self.assertTrue(selected["valid_for_training"])
        noisy = dict(cfg["selection"], query_noise_bound=.002)
        self.assertFalse(select_spectrum(QueryAccess(), noisy)["valid_for_training"])

    def test_all_variants_and_rank_gradient(self):
        for variant in ("full", "block_only", "hierarchical", "direct", "no_fine"):
            cfg, system, selection, model = context(variant)
            with tempfile.TemporaryDirectory() as directory:
                data = prepare_data(system, cfg["data"], directory)
                batch = sample_batch(data["train"], cfg, np.random.default_rng(7), torch.device("cpu"), torch.float32)
                total, parts = compute_losses(model, batch, system, cfg,
                                              torch.Generator().manual_seed(7))
                total.backward()
                self.assertTrue(bool(total.isfinite()))
                self.assertTrue(all(p.grad is None or bool(p.grad.isfinite().all()) for p in model.parameters()))
                if variant == "no_fine":
                    self.assertEqual(float(parts["fine_weight"]), 0)
                    self.assertTrue(all(p.grad is None for p in model.F.parameters()))
                    cfg["training"]["phase"] = "transfer"
                    self.assertEqual(float(compute_losses(model, batch, system, cfg)[1]["fine_weight"]), 1)
        cfg, system, _, model = context()
        jac = state_composite_jacobian(model, system)
        rank = torch.relu(torch.tensor(.5)-torch.linalg.svdvals(jac).min()).square()
        gradients = torch.autograd.grad(rank, list(model.E.parameters())+list(model.G.parameters()), allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients if g is not None), 0)

    def test_direct_matches_high_path_capacity(self):
        full = context("full")[3]
        direct = context("direct")[3]
        self.assertEqual(full.parameter_counts()["high_path"], direct.parameter_counts()["high_path"])
        self.assertEqual(direct.parameter_counts()["F"], 0)

    def test_covariance_floor_detects_duplicated_dimensions(self):
        coordinate = torch.linspace(-2, 2, 100)
        duplicated = torch.stack([coordinate, coordinate], dim=1)
        self.assertGreater(float(covariance_floor(duplicated)), .01)
        independent = torch.randn(1000, 2, generator=torch.Generator().manual_seed(5))
        self.assertEqual(float(covariance_floor(independent)), 0)

    def test_heldout_evaluation_and_frozen_transfer_monitor(self):
        for variant in ("full", "direct"):
            cfg, system, selection, model = context(variant)
            with tempfile.TemporaryDirectory() as directory:
                data = prepare_data(system, cfg["data"], directory)
                result = evaluate_model(model, system, data, selection, cfg)
                self.assertFalse(result["protocol"]["test_used_for_fit"])
                json.dumps(result, allow_nan=False)
                if variant == "full":
                    no_test = {k: v for k, v in data.items() if k != "test"}
                    first = evaluate_transfer_thresholds(model, system, no_test, selection, cfg)
                    second = evaluate_transfer_thresholds(model, system, no_test, selection, cfg, frozen=first["frozen"])
                    self.assertEqual(first["coarse_error"], second["coarse_error"])
                    self.assertFalse(second["test_accessed"])

    def test_axis_and_cluster_equivalence_are_distinct(self):
        rng = np.random.default_rng(7)
        truth = rng.normal(size=(40, 2))
        predicted = truth[:, [1, 0]]*np.array([-1, 1])
        match = fit_signed_permutation(predicted, truth)
        np.testing.assert_allclose(apply_signed_permutation(predicted, match), truth)
        mixed = truth@np.array([[1., 2.], [0., 1.]])
        record = cross_seed_consistency([truth, mixed], [truth, mixed], [[0, 1]])
        cluster = record["pairs"][0]["clusters"][0]
        self.assertLess(cluster["calibration_frozen_general_linear_mse"], 1e-20)
        self.assertFalse(cluster["axis_claim"])

    def test_expected_pseudosolution_attribution(self):
        audit = pseudo_solution_audit()
        rows = {row["candidate"]: row for row in audit["rows"]}
        self.assertTrue(rows["resonant_mix"]["accepted_by_spectrum_floor_rank"])
        self.assertFalse(rows["resonant_mix"]["strict_inverse_contraction_compatible"])
        self.assertFalse(rows["u_cubed"]["isolated_rank_benefit_demonstrated"])

    def test_resume_exact_sequence_and_parity_bundle(self):
        cfg, _, _, _ = context()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            whole = train_run(cfg, root/"whole", root/"data")
            train_run(cfg, root/"resumed", root/"data", stop_after=2)
            resumed = train_run(cfg, root/"resumed", root/"data", resume=True)
            left, right = load_checkpoint(root/"whole"/"checkpoint.pt"), load_checkpoint(root/"resumed"/"checkpoint.pt")
            self.assertTrue(all(torch.equal(v, right["model"][k]) for k, v in left["model"].items()))
            export_bundle(root/"whole"/"checkpoint.pt", root/"parity")
            self.assertTrue(check_bundle(root/"parity"/"bundle.pt", "cpu")["passed"])
            altered = copy.deepcopy(cfg)
            altered["training"]["steps"] += 1
            before = (root/"whole"/"config.json").read_bytes()
            with self.assertRaises(RuntimeError):
                train_run(altered, root/"whole", root/"data", resume=True)
            self.assertEqual(before, (root/"whole"/"config.json").read_bytes())

    def test_formal_profile_never_silently_falls_back_to_cpu(self):
        cfg = validate({"stage": "formal", "training": {"device": "cpu"}})
        with self.assertRaises(RuntimeError):
            configure_runtime(cfg)
        repo = Path(__file__).resolve().parents[1]
        formal = expand_matrix(repo/"configs"/"matrix_formal.json")
        self.assertTrue(all(c["stage"] == "formal" and c["training"]["device"] == "cuda" for c in formal))


if __name__ == "__main__":
    unittest.main()
