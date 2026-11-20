"""Independent AR2 and FHN .5 draws from the frozen two-slot extension."""

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
    "_extension_original_replay",
    _HERE.parent / "fine_independent_validation_20261109" / "fresh_data.py",
)
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)

SLOTS = ("ar_rich_fresh2", "fhn_fresh_eps05")
FRESH_NAMESPACE = "HCF_FINE_REPLICATION_EXTENSION_20261109_V1"
FRESH_TAG = 2109


def source_hashes():
    return {**base.source_hashes(), str(Path(__file__).resolve()): base._sha(Path(__file__))}


def _instance(slot):
    base._require(slot in SLOTS, f"Unsupported extension slot: {slot}")
    return json.loads(base.INSTANCE_LOCK.read_text())["instances"][slot]


def _fresh_seed(slot, seed):
    base._require(type(seed) is int and seed >= 0, "Fresh seed must be a nonnegative integer")
    identity = f"{FRESH_NAMESPACE}|{slot}|{seed}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1)


def _draw(slot, count, trajectory_seed, tag):
    owner, _ = base._sources()
    instance = _instance(slot)
    params = instance["physics"]
    base._require(type(count) is int and count > 0, "Trajectory count must be positive")
    environment, transform = owner._math()
    if slot == "ar_rich_fresh2":
        namespace = {
            "np": np, "data_seed": trajectory_seed, "D": 4, "A": 2,
            "L": params["steps"], "burn": params["burn"], "mode": params["mode"],
            "alpha": np.asarray(params["alpha"], dtype=np.float64),
            "B": np.asarray(params["B"], dtype=np.float64),
        }
        owner._extract(owner.AR_SOURCE, ("simulate",), namespace, parent="generate_case")
        physical, actions, regimes = namespace["simulate"](count, tag)
    else:
        config = dataclasses.replace(environment.Config(**params), seed=trajectory_seed)
        base._require(config.epsilon == .5 and not config.rotational and config.measurement_noise == 0.,
                      "Unexpected FHN extension recipe")

        def guard(cfg, t, n, capability):
            base._require(cfg == config and t == tag and n == count and capability is None,
                          "Unexpected FHN physical stream")

        namespace = {
            "np": np, "copy": copy, "env": environment,
            "require": base._require, "ah": owner.array_sha, "_draw_guard": guard,
        }
        owner._extract(owner.FHN_SOURCE, ("_draw", "_physical"), namespace)
        physical, actions, regimes, _ = namespace["_physical"](config, tag, count)

    mix = {"data": {
        "dimension": 4, "data_seed": base._old_seed(slot, "mix"),
        "observation_layers": 4, "observation_strength": .25,
    }}
    x = transform(physical, mix)
    inverse_error = float(np.max(np.abs(transform(x, mix, inverse=True) - physical)))
    base._require(np.isfinite(x).all() and inverse_error < 1e-10, "Frozen mix inverse check failed")
    x = np.ascontiguousarray(x, dtype=np.float32)
    base._require(x.shape == (count, instance["spec"]["T"], 4)
                  and actions.shape == (count, x.shape[1] - 1, 2) and actions.dtype == np.float32
                  and regimes.shape == (count, x.shape[1] - 1) and regimes.dtype == np.int64,
                  "Unexpected public array shape or dtype")
    return dict(x=x, actions=actions, regimes=regimes), physical.astype(np.float32), inverse_error


def verify_historical_train(slot):
    """Verify only the three public TRAIN members, with no CAL or truth load."""
    count = _instance(slot)["spec"]["train_n"]
    public, _, inverse_error = _draw(slot, count, base._old_seed(slot, "trajectory"), 101)
    historical = next(row for row in json.loads(base.TRAIN_RECEIPT.read_text())["records"]
                      if row["slot_id"] == slot)
    expected = historical["diagnostics"]["train"]["arrays"]
    with np.load(base.CONFIRMATION / "data/TRAIN_CAL" / f"{slot}.npz", allow_pickle=False) as archive:
        archived = {field: archive[f"train__{field}"] for field in base.FIELDS}
    actual = base._identities(public)
    base._require(actual == base._identities(archived) == expected,
                  f"Historical public TRAIN members failed byte match: {slot}")
    return {
        "slot": slot, "count": count, "public_byte_match": True,
        "public_arrays": actual, "inverse_maxabs_fp64": inverse_error,
        "archived_members_read": [f"train__{field}" for field in base.FIELDS],
    }


def fresh_observations(slot, n, seed):
    """Return only observables and a write-once forecast replay receipt."""
    physical_seed = _fresh_seed(slot, seed)
    _, rng = base._sources()
    historical_seeds = {
        rng.seed_for("DataOwner", known_slot, 0, "data", role, namespace=namespace)
        for known_slot in rng.SLOTS
        for role in ("trajectory", "mix")
        for namespace in (rng.NAMESPACE, rng.ENGINEERING_NAMESPACE)
    }
    base._require(physical_seed not in historical_seeds, "Extension stream collides with a historical root")
    public, _, inverse_error = _draw(slot, n, physical_seed, FRESH_TAG)
    return public, {
        "namespace": FRESH_NAMESPACE, "slot": slot, "count": n, "seed": seed,
        "trajectory_seed": physical_seed, "trajectory_tag": FRESH_TAG,
        "observation_mix_seed": base._old_seed(slot, "mix"),
        "public_arrays": base._identities(public), "inverse_maxabs_fp64": inverse_error,
    }


def replay_physical(receipt):
    """Regenerate fresh physical truth after sealing and checking observables."""
    slot = receipt["slot"]
    physical_seed = _fresh_seed(slot, receipt["seed"])
    base._require(
        receipt["namespace"] == FRESH_NAMESPACE and receipt["trajectory_seed"] == physical_seed
        and receipt["trajectory_tag"] == FRESH_TAG
        and receipt["observation_mix_seed"] == base._old_seed(slot, "mix"),
        "Extension receipt does not match its frozen streams",
    )
    public, physical, _ = _draw(slot, receipt["count"], physical_seed, FRESH_TAG)
    base._require(base._identities(public) == receipt["public_arrays"],
                  "Extension replay differs from the sealed observables")
    return physical
