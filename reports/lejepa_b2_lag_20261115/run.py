"""Freeze B2 lag fits and visible predictions, then independently score physical windows."""

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANEL = HERE.parent / "lejepa_a4_20261115/panel"
PERMUTATIONS = np.asarray(list(itertools.permutations(range(4))))

spec = importlib.util.spec_from_file_location("b2_frozen_panel", PANEL / "run.py")
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)
encoder = panel.encoder_source


def rng(config, role, seed=0, process_index=0):
    return np.random.default_rng(np.random.SeedSequence([config["master"], config["roles"][role], seed, process_index]))


def trajectories(config, role, seed, process_index, rates, count):
    draws = rng(config, role, seed, process_index).normal(size=(count, config["length"], 4))
    values = draws.copy()
    rho = np.asarray(rates)
    scale = np.sqrt(1 - rho**2)
    for t in range(1, config["length"]):
        values[:, t] = rho * values[:, t - 1] + scale * draws[:, t]
    return values


def state_pool(config, role, seed, process_index, count):
    return rng(config, role, seed, process_index).normal(size=(count, 4))


def moment(values):
    return np.einsum("ntj,ntk->njk", values, values, optimize=True) / values.shape[1]


def dynamic_gram(values, lags):
    return sum(moment(values[:, h:] - values[:, :-h]) for h in lags) / len(lags)


def white_operators(latents, lags, weights):
    means = weights @ latents.mean(axis=1)
    means_outer = means[:, :, None] * means[:, None, :]
    c0 = np.einsum("bn,nij->bij", weights, moment(latents)) - means_outer
    vals, vecs = np.linalg.eigh(c0)
    if np.any(vals <= 1e-8):
        raise ValueError("Singular CAL embedding covariance")
    whitening = (vecs / np.sqrt(vals[:, None, :])) @ np.swapaxes(vecs, -1, -2)
    operators = []
    for h in lags:
        left, right = latents[:, :-h], latents[:, h:]
        raw = np.einsum("nti,ntj->nij", left, right) / left.shape[1]
        raw = (raw + np.swapaxes(raw, -1, -2)) / 2
        weighted = np.einsum("bn,nij->bij", weights, raw)
        endpoints = weights @ ((left.mean(axis=1) + right.mean(axis=1)) / 2)
        shift = endpoints[:, :, None] * means[:, None, :]
        centered = weighted - shift - np.swapaxes(shift, -1, -2) + means_outer
        operators.append(whitening @ centered @ whitening)
    operators.append(sum(operators) / len(operators))
    operators = np.stack(operators, axis=1)
    return means, c0, vals, whitening, (operators + np.swapaxes(operators, -1, -2)) / 2


def name_axes(axes, response):
    moved = response @ axes
    normalized = np.abs(moved) / np.maximum(np.linalg.norm(moved, axis=-2, keepdims=True), 1e-12)
    utilities = normalized[..., np.arange(4)[None, :], PERMUTATIONS].sum(axis=-1)
    ordered_utility = np.sort(utilities, axis=-1)
    names = PERMUTATIONS[np.argmax(utilities, axis=-1)]
    slopes = np.take_along_axis(moved, names[..., :, None], axis=-1)[..., 0]
    if np.any(np.abs(slopes) < 1e-8):
        raise ValueError("Undefined physical scaling from named response")
    named = np.take_along_axis(axes, names[..., None, :], axis=-1) / slopes[..., None, :]
    return named, names, slopes, ordered_utility[..., -1] - ordered_utility[..., -2]


def fit_axes(cal, response, weights, config, rates):
    means, c0, vals0, whitening, operators = white_operators(cal, config["cal_lags"], weights)
    values, vectors = np.linalg.eigh(operators)
    values = values[..., ::-1]
    axes = whitening[:, None] @ vectors[..., ::-1]
    named, names, slopes, assignment = name_axes(axes, response)
    radii = np.linalg.norm(operators[1:] - operators[0], ord=2, axis=(-2, -1))
    radius = np.quantile(radii, config["noise_quantile"], axis=0)
    dots = np.einsum("aij,ik,bakj->baj", named[0], c0[0], named[1:])
    norm0 = np.einsum("aij,ik,akj->aj", named[0], c0[0], named[0])
    norm_other = np.einsum("baij,ik,bakj->baj", named[1:], c0[0], named[1:])
    rotation = np.clip(1 - dots**2 / (norm0 * norm_other), 0, 1)
    rho = np.asarray(rates)
    ideal = rho[None, :] ** np.asarray(config["cal_lags"])[:, None]
    ideal = np.concatenate((ideal, ideal.mean(axis=0, keepdims=True)))
    diffs = np.abs(ideal[:, :, None] - ideal[:, None, :])
    diffs[:, np.arange(4), np.arange(4)] = np.inf
    relevant_gap = diffs[:, 1:3, :].min(axis=(1, 2))
    commutators = []
    for a, b in itertools.combinations(operators[0, :-1], 2):
        commutators.append(np.linalg.norm(a @ b - b @ a) / (np.linalg.norm(a) * np.linalg.norm(b)))
    diagnostics = {"cal_embedding_eigenvalues": vals0[0].tolist(),
                   "mean_relative_commutator": float(np.mean(commutators)), "arms": {}}
    for a, label in enumerate(config["arms"]):
        diagnostics["arms"][label] = {
            "eigenvalues": values[0, a].tolist(), "ideal_factor2_3_eigengap": float(ideal[a, 1] - ideal[a, 2]),
            "ideal_nearest_eigengap_axes2_3": float(relevant_gap[a]), "operator_radius_q90": float(radius[a]),
            "ideal_gap_to_radius": float(relevant_gap[a] / radius[a]), "named_permutation": names[0, a].tolist(),
            "signed_own_response": slopes[0, a].tolist(), "assignment_gap": float(assignment[0, a]),
            "cal_bootstrap_same_permutation": float(np.all(names[1:, a] == names[0, a], axis=1).mean()),
            "cal_bootstrap_rotation_loss_axes2_3": float(rotation[:, a, 1:3].mean()),
            "named_basis": named[0, a].tolist(),
        }
    return means[0], named[0], diagnostics


def named_audit(latent, physical, center, axes):
    latent_centered = latent - latent.mean(axis=0)
    physical_centered = physical - physical.mean(axis=0)
    forward = np.linalg.lstsq(physical_centered, latent_centered, rcond=None)[0]
    result = []
    for axis in axes:
        pred = (latent - center) @ axis
        centered = pred - pred.mean(axis=0)
        pvar = physical.var(axis=0)
        corr2 = (physical_centered * centered).mean(axis=0)**2 / (pvar * pred.var(axis=0))
        r2 = 1 - ((pred - physical)**2).mean(axis=0) / pvar
        loading = forward @ axis
        off = loading - np.diag(np.diag(loading))
        result.append({"axis_correlation_squared": corr2.tolist(), "axis_r2_named_scale": r2.tolist(),
                       "forward_linear_physical_loading": loading.tolist(),
                       "off_diagonal_loading_rms_axes2_3": float(np.sqrt((off[:, 1:3]**2).sum(axis=0).mean()))})
    return result


def sources_manifest():
    result = panel.sources_manifest()
    for file in ("config.json", "PROTOCOL.md", "run.py"):
        path = HERE / file
        result[str(path.relative_to(ROOT))] = panel.digest(path)
    return result


def forecast(config):
    start = time.monotonic()
    sources = sources_manifest()
    folder = HERE / "forecast"
    folder.mkdir(exist_ok=False)
    lock = HERE / "DESIGN_LOCK.json"
    panel.write(lock, {"config": config, "sources": sources, "environment": {"python": sys.version,
                                  "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
                                  "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}})
    n = config["cal_windows"]
    panel_config, train_config, _ = panel.configs()
    encoders = {}
    for name in panel.panel_names(panel_config):
        model = encoder.make_mlp_encoder(4, train_config["hidden"], train_config["layers"], device="cpu").eval()
        model.load_state_dict(torch.load(PANEL / f"runs/{name}.pt", weights_only=True, map_location="cpu"))
        encoders[name] = model
    records = []
    dose = config["name_dose"]
    deltas = np.eye(4)[:, None, :] * dose
    for seed in config["seeds"]:
        for process_index, (process, rates) in enumerate(config["processes"].items()):
            cal = trajectories(config, "cal", seed, process_index, rates, n)
            anchors = state_pool(config, "name", seed, process_index, config["name_anchors"])
            reference = state_pool(config, "ref", seed, process_index, config["reference_states"])
            audit = state_pool(config, "audit", seed, process_index, config["audit_states"])
            test = trajectories(config, "test", seed, process_index, rates, config["test_windows"])
            bootstrap = rng(config, "cal_bootstrap", seed, process_index)
            weights = np.full(n, 1 / n)
            weights = np.vstack([weights, bootstrap.multinomial(n, weights, size=config["cal_bootstrap_draws"]) / n])
            for training in config["training_cases"]:
                name = f"{training}_{seed}"
                model = encoders[name]
                positive = encoder.embed(model, anchors[None] + deltas)
                negative = encoder.embed(model, anchors[None] - deltas)
                response = ((positive - negative) / (2 * dose)).mean(axis=1)
                latent = encoder.embed(model, cal)
                center, transform, diagnostics = fit_axes(latent, response, weights, config, rates)
                ref_h = encoder.embed(model, reference)
                ref_var = np.einsum("ni,aij->anj", ref_h, transform).var(axis=1)
                if np.any(ref_var <= 1e-12) or not np.isfinite(ref_var).all():
                    raise ValueError("Undefined reference variance")
                state_audits = named_audit(encoder.embed(model, audit), audit, center, transform)
                test_gram = dynamic_gram(encoder.embed(model, test), config["eval_lags"])
                scores = np.einsum("aij,nik,akj->anj", transform, test_gram, transform, optimize=True) / ref_var[:, None]
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite TEST forecast")
                key = f"{name}__{process}"
                for a, label in enumerate(config["arms"]):
                    diagnostics["arms"][label]["state_audit"] = state_audits[a]
                    diagnostics["arms"][label]["reference_variance"] = ref_var[a].tolist()
                path = folder / f"{key}.npz"
                with path.open("xb") as handle:
                    np.savez_compressed(handle, scores=scores)
                records.append({"name": name, "training_case": training, "seed": seed, "process": process,
                                "process_index": process_index, "path": str(path.relative_to(HERE)),
                                "sha256": panel.digest(path), "calibration": diagnostics})
            print(f"B2 sealed {seed} / {process}", flush=True)
    if sources != sources_manifest():
        raise ValueError("Input files changed while forecasting")
    panel.write(HERE / "FORECAST.json", {"design_sha256": panel.digest(lock), "cells": records,
                                        "elapsed_seconds": time.monotonic() - start})


def verify(config):
    lock = HERE / "DESIGN_LOCK.json"
    receipt = panel.read(HERE / "FORECAST.json")
    design = panel.read(lock)
    if (panel.digest(lock) != receipt["design_sha256"] or design["config"] != config
            or design["sources"] != sources_manifest()):
        raise ValueError("Forecast source or design lock mismatch")
    expected = {(seed, process, case) for seed in config["seeds"]
                for process in config["processes"] for case in config["training_cases"]}
    processes = list(config["processes"])
    cells = {}
    for cell in receipt["cells"]:
        seed, process, case = key = tuple(cell[field] for field in ("seed", "process", "training_case"))
        if key not in expected or key in cells or cell["name"] != f"{case}_{seed}" or cell["process_index"] != processes.index(process):
            raise ValueError(f"Invalid cell ID: {key}")
        path = (HERE / cell["path"]).resolve()
        if not path.is_relative_to(HERE / "forecast") or panel.digest(path) != cell["sha256"]:
            raise ValueError(f"Forecast file mismatch: {key}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"scores"}:
                raise ValueError("Unexpected sealed prediction fields")
            scores = archive["scores"]
        if scores.shape != (len(config["arms"]), config["test_windows"], 4) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid score data: {key}")
        cells[key] = (cell, scores)
    if cells.keys() != expected:
        raise ValueError("Forecast omitted a predeclared cell")
    print(f"B2 verified {len(cells)} sealed cells. SHA256 {panel.digest(HERE / 'FORECAST.json')}", flush=True)
    return cells


def bootstrap_primary(results, config):
    conditions = config["training_cases"]
    process = config["primary_process"]
    names = config["primary_contrasts"]
    arm_indices = [config["arms"].index(name) for name in names]
    n_draws, n = config["window_bootstrap_draws"], config["test_windows"]
    draws = np.zeros((len(names), n_draws))
    by_seed = {}
    for seed in config["seeds"]:
        rng_seed = rng(config, "window_bootstrap", seed)
        stacked = np.stack([results[(seed, process, case)]["correct"] for case in conditions])
        means = stacked.mean(axis=0)
        gain = means[arm_indices] - means[0]
        by_seed[seed] = gain.mean(axis=1)
        for start in range(0, n_draws, 128):
            stop = min(start + 128, n_draws)
            rows = rng_seed.integers(0, n, size=(stop - start, n))
            draws[:, start:stop] += gain[:, rows].mean(axis=2) / len(config["seeds"])
    lower = np.quantile(draws, config["familywise_alpha"] / len(names), axis=1)
    primary = {}
    for j, arm in enumerate(names):
        seed_gain = {str(seed): float(gains[j]) for seed, gains in by_seed.items()}
        point = float(np.mean(list(seed_gain.values())))
        primary[arm] = {"gain": point, "one_sided_bonferroni_lower": float(lower[j]), "seed_gains": seed_gain,
                        "n_positive_seeds": sum(gain > 0 for gain in seed_gain.values()),
                        "passes": bool(point >= config["minimum_gain"] and lower[j] > 0)}
    return primary


def score(config, cells):
    output = HERE / "SUMMARY.json"
    if output.exists():
        raise FileExistsError(output)
    expected = {}
    results = {}
    for key, (cell, scores) in cells.items():
        seed, process, _ = key
        if (seed, process) not in expected:
            z = trajectories(config, "test", seed, cell["process_index"], config["processes"][process], config["test_windows"])
            true_scores = np.diagonal(dynamic_gram(z, config["eval_lags"]), axis1=1, axis2=2)
            expected[seed, process] = (true_scores, encoder.top_two(true_scores))
        truth, order = expected[seed, process]
        correct = np.stack([(encoder.top_two(s) == order).all(axis=1) for s in scores])
        metrics = {}
        for a, name in enumerate(config["arms"]):
            metrics[name] = {"accuracy": float(correct[a].mean()), "correct": int(correct[a].sum()),
                             "gain_vs_lag1": float(correct[a].mean() - correct[0].mean()),
                             "fixes_vs_lag1": int((correct[a] & ~correct[0]).sum()),
                             "harms_vs_lag1": int((~correct[a] & correct[0]).sum()),
                             "physical_score_mse": float(np.mean((scores[a] - truth)**2))}
        results[key] = {"correct": correct, "by_arm": metrics}
    primary = bootstrap_primary(results, config)
    by_cell = {f"{cell['name']}__{cell['process']}": {"test": results[key]["by_arm"], **cell}
               for key, (cell, _) in cells.items()}
    panel.write(output, {"forecast_sha256": panel.digest(HERE / "FORECAST.json"), "primary": primary, "by_cell": by_cell})
    print("B2 primary", primary, flush=True)


def smoke(config):
    small = np.random.default_rng(1311).normal(size=(12, 36, 4))
    response = np.eye(4)
    weights = np.full((2, len(small)), 1 / len(small))
    weights[1, :2] += np.array([-0.02, 0.02])
    mean, basis, reports = fit_axes(small, response, weights, config, list(config["processes"].values())[0])
    old_basis, old_values, _ = encoder.fit_lag_axes(small)
    old_order, _ = encoder.assign_names(response @ old_basis)
    np.testing.assert_allclose(basis[0], old_basis[:, old_order] / np.diag((response @ old_basis)[:, old_order]), atol=1e-10)
    np.testing.assert_allclose(reports["arms"]["lag1"]["eigenvalues"], old_values, atol=1e-11)
    assert basis.shape == (5, 4, 4) and mean.shape == (4,)
    print("B2 smoke matched the historical lag-one estimator and naming.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "forecast", "verify", "score"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config = panel.read(HERE / "config.json")
    if args.stage == "smoke":
        smoke(config)
    elif args.stage == "forecast":
        forecast(config)
    else:
        cells = verify(config)
        if args.stage == "score":
            score(config, cells)


if __name__ == "__main__":
    main()
