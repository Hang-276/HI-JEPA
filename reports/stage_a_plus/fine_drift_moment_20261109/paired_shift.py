"""Exact generator-side paired FHN variance responses at shifted contexts."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import operator
from pathlib import Path
import sys

import numpy as np


_HERE = Path(__file__).resolve().parent
_NAMESPACE = "HCF_FINE_DRIFT_MOMENT_20261109_PAIRED_V1"
_STREAM_HASHES = (
    "initial_preburn_sha256", "actions_preburn_sha256", "regimes_preburn_sha256",
    "standard_innovations_preburn_sha256",
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shift = _load("_paired_shift_owner", _HERE.parent / "fine_calibration_shift_20261109/shift_data.py")
sibling = _load("_paired_shift_variance", _HERE.parent / "fine_anchor_bias_20261109/paired_innovations.py")


def source_hashes():
    return {
        **shift.source_hashes(), **sibling.source_hashes(),
        str(Path(__file__).resolve()): shift.base._sha(Path(__file__)),
    }


def _exact_contexts(slot, epsilon, count, trajectory_seed, tag, times, public_ids, stream_hashes):
    owner, env, transform, cfg, spec = shift._recipe(slot, epsilon, trajectory_seed)
    shift.base._require(
        type(count) is int and count > 0 and times.ndim == 1 and len(times) > 0
        and times[0] >= 0 and times[-1] < spec["T"] - 1 and np.all(times[1:] > times[:-1]),
        "Use sorted unique, in-range transition indices",
    )

    def guard(c, t, n, capability):
        shift.base._require(c == cfg and t == tag and n == count and capability is None,
                            "Unexpected shifted FHN conditional draw")

    namespace = {
        "np": np, "copy": copy, "env": env, "require": shift.base._require,
        "ah": owner.array_sha, "_draw_guard": guard,
    }
    owner._extract(owner.FHN_SOURCE, ("_draw", "_physical"), namespace)
    original_draw = namespace["_draw"]
    captured = {}

    def capture(c, t, n, capability=None):
        values = original_draw(c, t, n, capability)
        captured["actions"] = values[0][:, cfg.burnin_samples:]
        captured["innovation"] = values[2][:, cfg.burnin_samples + times]
        return values

    namespace["_draw"] = capture
    physical, actions32, regimes, replay = namespace["_physical"](cfg, tag, count)
    actions64 = captured["actions"]
    np.testing.assert_array_equal(actions64.astype(np.float32), actions32)
    mix = {"data": {"dimension": 4, "data_seed": shift.base._old_seed(slot, "mix"),
                    "observation_layers": 4, "observation_strength": .25}}
    x = np.ascontiguousarray(transform(physical, mix), dtype=np.float32)
    actual_ids = shift.base._identities(dict(x=x, actions=actions32, regimes=regimes))
    shift.base._require(actual_ids == public_ids, "Paired context replay differs from the sealed observables")
    if stream_hashes is not None:
        shift.base._require({name: replay[name] for name in _STREAM_HASHES} == stream_hashes,
                            "Paired context replay differs from the sealed physical random streams")
    means, sigmas = env.regimes(cfg)
    spike_sigmas = sibling._check_variance_regimes(means, sigmas, cfg.kick_std)
    deterministic = env.BASE.advance(physical[:, times], actions64[:, times], env.mechanism_config(cfg))
    sampled_regime = regimes[:, times]
    reconstructed = deterministic + means[sampled_regime] + sigmas[sampled_regime] * captured["innovation"]
    true_next = physical[:, times + 1]
    np.testing.assert_allclose(reconstructed, true_next, rtol=0, atol=1e-13)
    return (
        x[:, times], actions32[:, times], deterministic, cfg.kick_std, spike_sigmas, transform, mix,
        float(np.max(np.abs(reconstructed - true_next))),
    )


def _siblings(deterministic, epsilon, base_sigma, spike_sigmas, signs):
    children = [sibling._paired_children(deterministic, sign * epsilon, base_sigma, spike_sigmas)
                for sign in signs]
    baseline = np.stack([pair[0] for pair in children], axis=0)
    spikes = np.stack([pair[1] for pair in children], axis=1)
    return baseline, spikes


def paired_observations(receipt, seed, times, antithetic=True):
    """Return observed intervention siblings with explicit sign-first axes.

    current, actions: [episode, context, channel].
    baseline: [sign, episode, context, 4].
    spikes: [target axis, sign, episode, context, 4].
    The same Gaussian array defines all five arms; sign order is +1, then -1.
    """
    slot = receipt["slot"]
    role = receipt["role"]
    trajectory_seed = shift._stream_seed(slot, receipt["seed"], role)
    shift.base._require(
        role == "calibration" and receipt["namespace"] == shift.FRESH_NAMESPACE
        and receipt["trajectory_seed"] == trajectory_seed and receipt["trajectory_tag"] == shift.FRESH_TAG
        and receipt["observation_mix_seed"] == shift.base._old_seed(slot, "mix"),
        "Use a sealed shift-calibration receipt for matched interventions",
    )
    shift.base._require(type(seed) is int and seed >= 0 and type(antithetic) is bool,
                        "Use a nonnegative dedicated seed and Boolean antithetic flag")
    times = np.asarray([operator.index(t) for t in times], dtype=np.int64)
    current, actions, deterministic, base_sigma, spike_sigmas, transform, mix, check = _exact_contexts(
        slot, receipt["epsilon"], receipt["count"], trajectory_seed, shift.FRESH_TAG, times,
        receipt["public_arrays"], receipt["physical_rng_stream_hashes"])
    identity = f"{_NAMESPACE}|{slot}|{trajectory_seed}|{seed}".encode()
    seed_operand = int.from_bytes(hashlib.sha256(identity).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed_operand])))
    gaussian = rng.standard_normal(deterministic.shape)
    signs = (1, -1) if antithetic else (1,)
    baseline, spikes = _siblings(deterministic, gaussian, base_sigma, spike_sigmas, signs)
    panel = dict(
        current=np.ascontiguousarray(current),
        actions=np.ascontiguousarray(actions),
        baseline=np.ascontiguousarray(transform(baseline, mix), dtype=np.float32),
        spikes=np.ascontiguousarray(transform(spikes, mix), dtype=np.float32),
        noise_signs=list(signs),
    )
    owner, _ = shift.base._sources()
    audit = {
        "slot": slot, "epsilon": receipt["epsilon"], "trajectory_seed": trajectory_seed,
        "seed": seed, "seed_operand": seed_operand, "rng_namespace": _NAMESPACE,
        "noise_signs": list(signs), "context_times": times.tolist(),
        "context_count": receipt["count"] * len(times),
        "base_regime": 0, "variance_regimes": [5, 6, 7, 8],
        "baseline_sigma": base_sigma, "spike_sigmas": spike_sigmas.tolist(),
        "replayed_transition_maxabs_fp64": check, "off_axis_physical_change_exactly_zero": True,
        "observation_mix_seed": receipt["observation_mix_seed"],
        "observations": {key: owner.identity(panel[key]) for key in ("current", "actions", "baseline", "spikes")},
    }
    return panel, audit
