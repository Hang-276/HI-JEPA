"""Paired FHN epsilon interventions with a frozen source observation mapping."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


_HERE = Path(__file__).resolve().parent
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_shift_original_replay",
    _HERE.parent / "fine_independent_validation_20261109" / "fresh_data.py",
)
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)

SLOTS = ("fhn_fresh_eps008", "fhn_fresh_eps05")
EPSILONS = (.08, .5)
ROLES = ("calibration", "evaluation")
FRESH_NAMESPACE = "HCF_FINE_CALIBRATION_SHIFT_20261109_V1"
FRESH_TAG = 3109


def source_hashes():
    return {**base.source_hashes(), str(Path(__file__).resolve()): base._sha(Path(__file__))}


def _instance(slot):
    base._require(slot in SLOTS, f"Unsupported frozen FHN source slot: {slot}")
    return json.loads(base.INSTANCE_LOCK.read_text())["instances"][slot]


def _stream_seed(slot, seed, role):
    base._require(slot in SLOTS and role in ROLES and type(seed) is int and seed >= 0,
                  "Use a frozen source slot, explicit role and nonnegative integer seed")
    identity = f"{FRESH_NAMESPACE}|{slot}|{role}|{seed}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1)


def _recipe(slot, epsilon, root_seed):
    base._require(epsilon in EPSILONS, "The only permitted target epsilons are .08 and .5")
    instance = _instance(slot)
    original = instance["physics"]
    owner, _ = base._sources()
    env, transform = owner._math()
    frozen = env.Config(**original)
    config = dataclasses.replace(frozen, seed=root_seed, epsilon=epsilon)
    base._require(not config.rotational and config.measurement_noise == 0.,
                  "Unexpected frozen FHN recipe")
    base._require(dataclasses.asdict(config) == {**original, "seed": root_seed, "epsilon": epsilon},
                  "A knob other than epsilon or the independent RNG seed changed")
    return owner, env, transform, config, instance["spec"]


def _draw(slot, epsilon, count, root_seed, tag):
    base._require(type(count) is int and count > 0, "Trajectory count must be a positive integer")
    owner, env, transform, config, spec = _recipe(slot, epsilon, root_seed)

    def guard(cfg, t, n, capability):
        base._require(cfg == config and t == tag and n == count and capability is None,
                      "Unexpected FHN conditional draw")

    namespace = {
        "np": np, "copy": copy, "env": env, "require": base._require,
        "ah": owner.array_sha, "_draw_guard": guard,
    }
    owner._extract(owner.FHN_SOURCE, ("_draw", "_physical"), namespace)
    physical, actions, regimes, draw_receipt = namespace["_physical"](config, tag, count)
    mix = {"data": {
        "dimension": 4, "data_seed": base._old_seed(slot, "mix"),
        "observation_layers": 4, "observation_strength": .25,
    }}
    observed = transform(physical, mix)
    inverse_error = float(np.max(np.abs(transform(observed, mix, inverse=True) - physical)))
    base._require(np.isfinite(observed).all() and inverse_error < 1e-10,
                  "Frozen observation mix inverse check failed")
    x = np.ascontiguousarray(observed, dtype=np.float32)
    base._require(x.shape == (count, spec["T"], 4)
                  and actions.shape == (count, x.shape[1] - 1, 2) and actions.dtype == np.float32
                  and regimes.shape == (count, x.shape[1] - 1) and regimes.dtype == np.int64,
                  "Unexpected FHN public array shape or dtype")
    # These hashes verify paired RNG use and cannot reveal realized state arrays.
    stream_hashes = {name: draw_receipt[name] for name in (
        "initial_preburn_sha256", "actions_preburn_sha256", "regimes_preburn_sha256",
        "standard_innovations_preburn_sha256")}
    return dict(x=x, actions=actions, regimes=regimes), physical.astype(np.float32), inverse_error, stream_hashes


def verify_historical_train(slot):
    """Match this adapter's source-epsilon replay to only archived public TRAIN."""
    instance = _instance(slot)
    count = instance["spec"]["train_n"]
    epsilon = instance["physics"]["epsilon"]
    public, _, inverse_error, _ = _draw(slot, epsilon, count, base._old_seed(slot, "trajectory"), 101)
    row = next(row for row in json.loads(base.TRAIN_RECEIPT.read_text())["records"]
               if row["slot_id"] == slot)
    expected = row["diagnostics"]["train"]["arrays"]
    with np.load(base.CONFIRMATION / "data/TRAIN_CAL" / f"{slot}.npz", allow_pickle=False) as archive:
        archived = {field: archive[f"train__{field}"] for field in base.FIELDS}
    actual = base._identities(public)
    base._require(actual == base._identities(archived) == expected,
                  f"Historical public TRAIN byte check failed: {slot}")
    return {
        "slot": slot, "epsilon": epsilon, "count": count, "public_byte_match": True,
        "public_arrays": actual, "inverse_maxabs_fp64": inverse_error,
        "archived_members_read": [f"train__{field}" for field in base.FIELDS],
    }


def fresh_observations(slot, epsilon, n, seed, role):
    """Return a sealed-replay receipt and public arrays only.

    With identical slot, n, seed and role, changing epsilon reuses the exact
    underlying initialization, actions, regimes and Gaussian innovations.
    """
    root_seed = _stream_seed(slot, seed, role)
    _, rng = base._sources()
    historical_seeds = {
        rng.seed_for("DataOwner", known_slot, 0, "data", old_role, namespace=namespace)
        for known_slot in rng.SLOTS
        for old_role in ("trajectory", "mix")
        for namespace in (rng.NAMESPACE, rng.ENGINEERING_NAMESPACE)
    }
    base._require(root_seed not in historical_seeds, "Shift stream collides with a historical root")
    public, _, inverse_error, stream_hashes = _draw(slot, epsilon, n, root_seed, FRESH_TAG)
    return public, {
        "namespace": FRESH_NAMESPACE, "slot": slot, "epsilon": epsilon, "count": n,
        "seed": seed, "role": role, "trajectory_seed": root_seed, "trajectory_tag": FRESH_TAG,
        "observation_mix_seed": base._old_seed(slot, "mix"), "public_arrays": base._identities(public),
        "inverse_maxabs_fp64": inverse_error, "physical_rng_stream_hashes": stream_hashes,
    }


def replay_physical(receipt):
    """Regenerate the frozen evaluation and verify all public hashes before reveal."""
    slot = receipt["slot"]
    role = receipt["role"]
    root_seed = _stream_seed(slot, receipt["seed"], role)
    base._require(
        role == "evaluation" and receipt["namespace"] == FRESH_NAMESPACE
        and receipt["trajectory_seed"] == root_seed and receipt["trajectory_tag"] == FRESH_TAG
        and receipt["observation_mix_seed"] == base._old_seed(slot, "mix"),
        "Physical reveal requires the exact frozen evaluation stream",
    )
    public, physical, _, stream_hashes = _draw(slot, receipt["epsilon"], receipt["count"], root_seed, FRESH_TAG)
    base._require(base._identities(public) == receipt["public_arrays"]
                  and stream_hashes == receipt["physical_rng_stream_hashes"],
                  "Shift replay differs from the sealed observables or RNG hashes")
    return physical
