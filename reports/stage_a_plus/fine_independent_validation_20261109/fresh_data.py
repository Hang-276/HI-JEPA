"""Replay public TRAIN and generate independent trajectories on the frozen physics.

Call fresh_observations in the prediction phase. After predictions are sealed,
use replay_physical to regenerate its physical targets and verify all public
array identities before any score is calculated. No historical truth is read.
"""

from __future__ import annotations

import copy
import dataclasses
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CONFIRMATION = ROOT / "outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2"
INSTANCE_LOCK = CONFIRMATION / "registration/INSTANCE_LOCK.json"
TRAIN_RECEIPT = CONFIRMATION / "data/DATA_PREPARED.json"
OLD_OWNER = ROOT / "work/hcf_stable_id_20261027/confirmation_v1/data_owner_engineering.py"
OLD_RNG = ROOT / "work/hcf_stable_id_20261028/confirmation_v2/rng_namespace.py"
SLOTS = ("ar_rich_fresh1", "fhn_fresh_eps008")
FIELDS = ("x", "actions", "regimes")
FRESH_NAMESPACE = "HCF_FINE_INDEPENDENT_VALIDATION_20261109_V1"
FRESH_TAG = 1109


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _sources():
    instance_sha = "0800ba1873f3422b8b8f746702156faaf89f792081a4b210bdc6b06a77214065"
    owner_sha = "ead722272ccad5a2d18a2091ef25f0394ac8cda098b886b79e9471244a8677db"
    rng_sha = "a4329f53453406618be8d717ad3ec5964d746e21b2767f47c6bf0ce5c5ec7f1e"
    for path, expected in ((INSTANCE_LOCK, instance_sha), (OLD_OWNER, owner_sha), (OLD_RNG, rng_sha)):
        _require(_sha(path) == expected, f"Frozen historical source differs: {path}")
    owner = _load("_fresh_validation_original_owner", OLD_OWNER)
    rng = _load("_fresh_validation_original_rng", OLD_RNG)
    return owner, rng


def source_hashes():
    owner, _ = _sources()
    return {
        **owner.source_hashes(),
        str(OLD_RNG): _sha(OLD_RNG),
        str(INSTANCE_LOCK): _sha(INSTANCE_LOCK),
        str(TRAIN_RECEIPT): _sha(TRAIN_RECEIPT),
        str(Path(__file__).resolve()): _sha(Path(__file__)),
    }


def _instance(slot):
    _require(slot in SLOTS, f"Unsupported frozen slot: {slot}")
    return json.loads(INSTANCE_LOCK.read_text())["instances"][slot]


def _old_seed(slot, role):
    _, rng = _sources()
    return rng.seed_for("DataOwner", slot, 0, "data", role)


def _fresh_seed(slot, seed):
    _require(type(seed) is int and seed >= 0, "Fresh experiment seed must be a nonnegative integer")
    operand = f"{FRESH_NAMESPACE}|{slot}|{seed}".encode()
    return int.from_bytes(hashlib.sha256(operand).digest()[:8], "big") & ((1 << 63) - 1)


def _draw(slot, count, trajectory_seed, tag):
    owner, _ = _sources()
    instance = _instance(slot)
    config = instance["physics"]
    _require(type(count) is int and count > 0, "Trajectory count must be a positive integer")
    environment, transform = owner._math()
    if slot == "ar_rich_fresh1":
        namespace = {
            "np": np, "data_seed": trajectory_seed, "D": 4, "A": 2,
            "L": config["steps"], "burn": config["burn"], "mode": config["mode"],
            "alpha": np.asarray(config["alpha"], dtype=np.float64),
            "B": np.asarray(config["B"], dtype=np.float64),
        }
        owner._extract(owner.AR_SOURCE, ("simulate",), namespace, parent="generate_case")
        physical, actions, regimes = namespace["simulate"](count, tag)
    else:
        def check_role(cfg, actual_tag, n, capability):
            _require(
                capability is None and cfg.seed == trajectory_seed and actual_tag == tag and n == count,
                "Unexpected physical stream",
            )

        namespace = {
            "np": np, "copy": copy, "env": environment,
            "require": _require, "ah": owner.array_sha, "_draw_guard": check_role,
        }
        owner._extract(owner.FHN_SOURCE, ("_draw", "_physical"), namespace)
        config = dataclasses.replace(environment.Config(**config), seed=trajectory_seed)
        physical, actions, regimes, _ = namespace["_physical"](config, tag, count)

    mix = {"data": {
        "dimension": 4, "data_seed": _old_seed(slot, "mix"),
        "observation_layers": 4, "observation_strength": .25,
    }}
    observed = transform(physical, mix)
    inverse_error = float(np.max(np.abs(transform(observed, mix, inverse=True) - physical)))
    _require(np.isfinite(observed).all() and inverse_error < 1e-10,
             "Frozen observation transform failed its inverse check")
    x = np.ascontiguousarray(observed, dtype=np.float32)
    _require(x.shape == (count, instance["spec"]["T"], 4)
             and actions.shape == (count, x.shape[1] - 1, 2) and actions.dtype == np.float32
             and regimes.shape == (count, x.shape[1] - 1) and regimes.dtype == np.int64,
             "Frozen generator produced unexpected array shape or dtype")
    public = dict(x=x, actions=actions, regimes=regimes)
    return public, np.ascontiguousarray(physical, dtype=np.float32), inverse_error


def _identities(public):
    owner, _ = _sources()
    return {field: owner.identity(public[field]) for field in FIELDS}


def verify_historical_train(slot):
    """Compare whole regenerated TRAIN members to the archived public TRAIN."""
    train_path = CONFIRMATION / "data/TRAIN_CAL" / f"{slot}.npz"
    instance = _instance(slot)
    count = instance["spec"]["train_n"]
    public, _, inverse_error = _draw(slot, count, _old_seed(slot, "trajectory"), 101)
    owner_receipt = json.loads(TRAIN_RECEIPT.read_text())
    historical = next(row for row in owner_receipt["records"] if row["slot_id"] == slot)
    expected = historical["diagnostics"]["train"]["arrays"]
    with np.load(train_path, allow_pickle=False) as archive:
        archived = {field: archive[f"train__{field}"] for field in FIELDS}
    _require(_identities(archived) == expected, f"Archived TRAIN members differ from owner receipt: {slot}")
    actual = _identities(public)
    _require(actual == expected, f"Historical TRAIN reproduction failed byte match: {slot}")
    return {
        "slot": slot, "count": count, "public_byte_match": True,
        "public_arrays": actual, "inverse_maxabs_fp64": inverse_error,
        "archived_members_read": [f"train__{field}" for field in FIELDS],
    }


def fresh_observations(slot, n, seed):
    """Return only x/actions/regimes and the receipt needed for score-time replay."""
    physical_seed = _fresh_seed(slot, seed)
    _, rng = _sources()
    historical_seeds = {
        rng.seed_for("DataOwner", known_slot, 0, "data", role, namespace=namespace)
        for known_slot in rng.SLOTS
        for role in ("trajectory", "mix")
        for namespace in (rng.NAMESPACE, rng.ENGINEERING_NAMESPACE)
    }
    _require(physical_seed not in historical_seeds, "Fresh stream collides with a historical data root")
    public, _, inverse_error = _draw(slot, n, physical_seed, FRESH_TAG)
    return public, {
        "namespace": FRESH_NAMESPACE, "slot": slot, "count": n, "seed": seed,
        "trajectory_seed": physical_seed, "trajectory_tag": FRESH_TAG,
        "observation_mix_seed": _old_seed(slot, "mix"),
        "public_arrays": _identities(public), "inverse_maxabs_fp64": inverse_error,
    }


def replay_physical(receipt):
    """Regenerate a locked fresh panel and verify observations before giving truth."""
    slot = receipt["slot"]
    physical_seed = _fresh_seed(slot, receipt["seed"])
    _require(
        receipt["namespace"] == FRESH_NAMESPACE and receipt["trajectory_seed"] == physical_seed
        and receipt["trajectory_tag"] == FRESH_TAG and receipt["observation_mix_seed"] == _old_seed(slot, "mix"),
        "Fresh panel receipt differs from the declared streams",
    )
    public, physical, _ = _draw(slot, receipt["count"], physical_seed, FRESH_TAG)
    _require(_identities(public) == receipt["public_arrays"], "Fresh replay differs from sealed observables")
    return physical
