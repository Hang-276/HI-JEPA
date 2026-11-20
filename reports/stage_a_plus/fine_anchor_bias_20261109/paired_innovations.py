"""Generator-owned paired variance interventions for bias diagnosis.

The input receipt describes a passive calibration panel that was already
generated and sealed. Hold its exact FP64 state and action fixed, then apply
the same new Gaussian draw under the original baseline and variance regimes.
The returned arrays are observed siblings, never physical targets.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import operator
from pathlib import Path
import sys

import numpy as np


_HERE = Path(__file__).resolve().parent
_FRESH_PATH = _HERE.parent / "fine_independent_validation_20261109" / "fresh_data.py"
_FRESH_SPEC = importlib.util.spec_from_file_location("_paired_variance_fresh_data", _FRESH_PATH)
fresh = importlib.util.module_from_spec(_FRESH_SPEC)
sys.modules[_FRESH_SPEC.name] = fresh
_FRESH_SPEC.loader.exec_module(fresh)
_NAMESPACE = "HCF_FINE_ANCHOR_BIAS_20261109_PAIRED_VARIANCE_V1"
_VARIANCE_REGIMES = (5, 6, 7, 8)


def source_hashes():
    """All math sources plus the two frozen replay adapters."""
    return {**fresh.source_hashes(), str(Path(__file__).resolve()): fresh._sha(Path(__file__))}


def _check_variance_regimes(means, sigmas, base_sigma):
    expected = base_sigma * (1.0 + 2.0 * np.eye(4))
    np.testing.assert_array_equal(means[[0, *_VARIANCE_REGIMES]], np.zeros((5, 4)))
    np.testing.assert_array_equal(sigmas[0], np.full(4, base_sigma))
    np.testing.assert_array_equal(sigmas[list(_VARIANCE_REGIMES)], expected)
    return expected


def _exact_contexts(slot, count, root_seed, tag, times, public_identities):
    owner, _ = fresh._sources()
    instance = fresh._instance(slot)
    parameters = instance["physics"]
    environment, transform = owner._math()
    fresh._require(type(count) is int and count > 0, "Nonpositive trajectory count")
    fresh._require(
        times.ndim == 1 and len(times) > 0 and np.all(times[1:] > times[:-1])
        and times[0] >= 0 and times[-1] < instance["spec"]["T"] - 1,
        "Use sorted unique, in-range transition indices",
    )
    if slot == "ar_rich_fresh1":
        burn = parameters["burn"]
        steps = parameters["steps"]
        alpha = np.asarray(parameters["alpha"], dtype=np.float64)
        weights = np.asarray(parameters["B"], dtype=np.float64)
        fresh._require(parameters["mode"] == "rich", "AR variance semantics require rich mode")
        namespace = {
            "np": np, "data_seed": root_seed, "D": 4, "A": 2,
            "L": steps, "burn": burn, "mode": parameters["mode"],
            "alpha": alpha, "B": weights,
        }
        owner._extract(owner.AR_SOURCE, ("simulate",), namespace, parent="generate_case")
        physical, actions32, regimes = namespace["simulate"](count, tag)
        action_rng = np.random.default_rng(np.random.SeedSequence([root_seed, tag, 2]))
        actions64 = action_rng.uniform(-1, 1, (count, burn + steps, 2))[:, burn:]
        noise_rng = np.random.default_rng(np.random.SeedSequence([root_seed, tag, 4]))
        noise = noise_rng.normal(size=(count, burn + steps, 4))[:, burn + times]
        means = np.zeros((9, 4))
        means[1:5] = .6 * np.eye(4)
        sigmas = np.full((9, 4), .12)
        sigmas[5:9] *= 1.0 + 2.0 * np.eye(4)
        base_sigma = .12
        deterministic = physical[:, times] * alpha + actions64[:, times] @ weights
    else:
        cfg = dataclasses.replace(environment.Config(**parameters), seed=root_seed)
        fresh._require(
            cfg.epsilon == .08 and not cfg.rotational and cfg.measurement_noise == 0.
            and cfg.contrast_strength == 1.0 and cfg.variance_multiplier == 3.0,
            "FHN variance semantics differ from the frozen rich slot",
        )
        means, sigmas = environment.regimes(cfg)
        base_sigma = cfg.kick_std

        def guard(c, t, n, capability):
            fresh._require(c == cfg and t == tag and n == count and capability is None,
                           "Unexpected FHN physical stream")

        namespace = {
            "np": np, "copy": copy, "env": environment,
            "require": fresh._require, "ah": owner.array_sha, "_draw_guard": guard,
        }
        owner._extract(owner.FHN_SOURCE, ("_draw", "_physical"), namespace)
        original_draw = namespace["_draw"]
        captured = {}

        def capture_context(c, t, n, capability=None):
            result = original_draw(c, t, n, capability)
            captured["actions"] = result[0][:, cfg.burnin_samples:]
            captured["noise"] = result[2][:, cfg.burnin_samples + times]
            return result

        namespace["_draw"] = capture_context
        physical, actions32, regimes, _ = namespace["_physical"](cfg, tag, count)
        actions64 = captured["actions"]
        noise = captured["noise"]
        deterministic = environment.BASE.advance(
            physical[:, times], actions64[:, times], environment.mechanism_config(cfg))

    spike_sigmas = _check_variance_regimes(means, sigmas, base_sigma)
    np.testing.assert_array_equal(actions64.astype(np.float32), actions32)
    mix = {"data": {"dimension": 4, "data_seed": fresh._old_seed(slot, "mix"),
                    "observation_layers": 4, "observation_strength": .25}}
    x = np.ascontiguousarray(transform(physical, mix), dtype=np.float32)
    fresh._require(fresh._identities(dict(x=x, actions=actions32, regimes=regimes)) == public_identities,
                   "Exact passive context replay differs from the sealed observables")

    actual_regime = regimes[:, times]
    reconstructed = deterministic + means[actual_regime] + sigmas[actual_regime] * noise
    actual_next = physical[:, times + 1]
    max_error = float(np.max(np.abs(reconstructed - actual_next)))
    np.testing.assert_allclose(reconstructed, actual_next, rtol=0, atol=1e-13)
    return (
        physical[:, times], actions64[:, times], deterministic, x[:, times],
        base_sigma, spike_sigmas, transform, mix, max_error,
    )


def _paired_children(deterministic, epsilon, base_sigma, spike_sigmas):
    baseline = deterministic + base_sigma * epsilon
    spikes = deterministic[None] + spike_sigmas[:, None, None, :] * epsilon[None]
    delta = spikes - baseline[None]
    target = np.eye(4, dtype=bool)
    np.testing.assert_array_equal(np.where(target[:, None, None, :], 0., delta), 0.)
    expected = (spike_sigmas - base_sigma)[:, None, None, :] * epsilon[None]
    np.testing.assert_allclose(delta, expected, atol=1e-13, rtol=0)
    fresh._require(np.isfinite(spikes).all(), "Nonfinite paired physical response")
    return baseline, spikes


def paired_observations(receipt, seed, times):
    """Return observed current/base/spike siblings and a JSON-native audit.

    Shapes are [episode, context, 4] for current and baseline,
    [target_axis, episode, context, 4] for spikes and [episode, context, 2]
    for actions. A context index t controls the transition t → t+1.
    """
    slot = receipt["slot"]
    trajectory_seed = fresh._fresh_seed(slot, receipt["seed"])
    fresh._require(
        receipt["namespace"] == fresh.FRESH_NAMESPACE
        and receipt["trajectory_seed"] == trajectory_seed
        and receipt["trajectory_tag"] == fresh.FRESH_TAG
        and receipt["observation_mix_seed"] == fresh._old_seed(slot, "mix"),
        "Only the sealed fresh calibration stream can supply paired contexts",
    )
    fresh._require(type(seed) is int and seed >= 0, "Dedicated pair seed must be a nonnegative integer")
    times = np.asarray([operator.index(t) for t in times], dtype=np.int64)
    _, actions, deterministic, current, base_sigma, spike_sigmas, transform, mix, check = _exact_contexts(
        slot, receipt["count"], trajectory_seed, fresh.FRESH_TAG, times, receipt["public_arrays"])
    seed_bytes = hashlib.sha256(f"{_NAMESPACE}|{slot}|{seed}".encode()).digest()[:16]
    seed_operand = int.from_bytes(seed_bytes, "big")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed_operand])))
    epsilon = rng.standard_normal(deterministic.shape)
    baseline, spikes = _paired_children(deterministic, epsilon, base_sigma, spike_sigmas)
    panel = dict(
        current=np.ascontiguousarray(current),
        baseline=np.ascontiguousarray(transform(baseline, mix), dtype=np.float32),
        spikes=np.ascontiguousarray(transform(spikes, mix), dtype=np.float32),
        actions=np.ascontiguousarray(actions, dtype=np.float32),
    )
    fresh._require(all(np.isfinite(value).all() for value in panel.values()),
                   "Nonfinite observable sibling")
    owner, _ = fresh._sources()
    audit = {
        "slot": slot, "context_panel_trajectory_seed": trajectory_seed,
        "context_times": times.tolist(), "context_count": receipt["count"] * len(times),
        "rng_namespace": _NAMESPACE, "seed": seed, "seed_operand": seed_operand,
        "rng": "PCG64 shared Gaussian across baseline and all four siblings",
        "physics_context": "exact FP64 state and action replay",
        "observation_mix_seed": receipt["observation_mix_seed"], "base_regime": 0,
        "variance_regimes": list(_VARIANCE_REGIMES), "baseline_sigma": base_sigma,
        "spike_sigmas": spike_sigmas.tolist(), "replayed_transition_maxabs_fp64": check,
        "off_axis_physical_change_exactly_zero": True,
        "observations": {name: owner.identity(value) for name, value in panel.items()},
    }
    return panel, audit
