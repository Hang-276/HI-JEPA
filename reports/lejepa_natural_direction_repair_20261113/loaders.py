"""Read-only loading and independent data roles for the natural-residual panel."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
ROOT = REPORTS.parent
A1 = REPORTS / "lejepa_direction_a1_20261113"
CONTROLLED = A1 / "controlled"
NATURAL = A1 / "natural"
TRAIN = REPORTS / "lejepa_slow_rank_20261112/training"
READOUT = REPORTS / "lejepa_mechanisms_20261113/readout"
HISTORICAL = REPORTS / "lejepa_dynamic_transfer_20261113/analysis/HISTORICAL_GROUPS.json"
ROLES = {"state_cal": 1, "reference": 2, "test": 3, "audit_state": 4}


def _load_controlled():
    path = CONTROLLED / "controlled.py"
    original_path = sys.path[:]
    # The old B helper has an unqualified `from run import ...`. Bind it to
    # the archived training module only while executing the old imports.
    missing = object()
    previous_run = sys.modules.get("run", missing)

    def execute(name, file):
        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        training = execute("a1_training_source_for_natural_repair", TRAIN / "run.py")
        sys.modules["run"] = training
        return execute("a1_controlled_source_for_natural_repair", path)
    finally:
        if previous_run is missing:
            sys.modules.pop("run", None)
        else:
            sys.modules["run"] = previous_run
        sys.path[:] = original_path


def _expected(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Unexpected frozen source or artifact: {label}")


def _repo_key(path):
    return str(path.relative_to(ROOT))


def _artifact(path, expected, digest):
    _expected(digest(path), expected, path)
    return {_repo_key(path): expected}


def load_panel():
    """Return the controlled helpers, frozen loader identity and 60 model cells."""
    source = _load_controlled()
    original, b_config, strong, encoder_hashes, decoder_hashes = source.identity()
    natural_receipt = source.read(NATURAL / "primary/FORECAST_RECEIPT.json")
    controlled_receipt = source.read(CONTROLLED / "forecast/RECEIPT.json")
    encoder_receipt = source.read(TRAIN.parent / "prospective/pipeline_forecast_receipt.json")
    b_original = source.read(READOUT / "runs/results.json")
    b_replication = source.read(READOUT / "replication/FORECAST_RECEIPT.json")
    digest = source.digest

    provenance = {}
    for archive in (natural_receipt["source_hashes"], controlled_receipt["sources"]):
        for path, expected in archive.items():
            provenance.update(_artifact(REPORTS / path, expected, digest))
    for path, expected in (
        (TRAIN / "protocol.json", encoder_receipt["training_protocol_sha256"]),
        (TRAIN / "run.py", source.read(TRAIN / "runs/results.json")["code_sha256"]),
        (READOUT / "run.py", b_original["source_sha256"]),
        (READOUT / "protocol.json", b_original["protocol_sha256"]),
        (READOUT / "runs/results.json", b_replication["original_summary_sha256"]),
    ):
        provenance.update(_artifact(path, expected, digest))

    training = source.helper.source
    original_vendor = training.source_provenance()
    vendor_paths = [training.SOURCE / "LEJEPA_SOURCE_LOCK.json"]
    vendor_paths += [training.VENDOR / name for name in original_vendor["sha256"]]
    vendor_paths.append(training.VENDOR / "experiments/lejepa_id/__init__.py")
    anchor_manifests = [
        CONTROLLED / "forecast/RECEIPT.json",
        NATURAL / "primary/FORECAST_RECEIPT.json",
        TRAIN.parent / "prospective/pipeline_forecast_receipt.json",
        TRAIN / "runs/results.json",
        READOUT / "replication/FORECAST_RECEIPT.json",
        HISTORICAL,
    ]
    provenance.update({_repo_key(path): digest(path) for path in (*anchor_manifests, *vendor_paths)})

    processes = natural_receipt["config"]["processes"]
    expected_processes = ("separated_wide", "separated_narrow")
    if tuple(processes) != expected_processes:
        raise ValueError("Unexpected historical natural evaluation process list")
    for process in expected_processes:
        _expected(processes[process], original["cases"][process], process)
    model_names = [
        f"{case}_{seed}" for case in original["cases"] for seed in original["seeds"]
    ]
    if len(model_names) != 16 or len(strong) != 14 or not set(strong) <= set(model_names):
        raise ValueError("Unexpected archived model panel")

    identities = {}
    for name in model_names:
        encoder = TRAIN / "runs" / f"{name}.pt"
        identities[name] = _artifact(encoder, encoder_hashes[encoder.name], digest)

    cells = []
    for process in expected_processes:
        for name in model_names:
            transform = NATURAL / "primary" / f"{process}__{name}.json"
            cell_hashes = {
                **identities[name],
                **_artifact(transform, natural_receipt["parameter_sha256"][transform.name], digest),
            }
            if source.read(transform)["encoder_sha256"] != encoder_hashes[f"{name}.pt"]:
                raise ValueError(f"Unrelated named transform: {transform}")
            cells.append({"name": name, "layer": "lag", "process": process, "provenance": cell_hashes})
        for name in strong:
            decoder = READOUT / "runs" / f"{name}_readouts.pt"
            cell_hashes = {**identities[name], **_artifact(decoder, decoder_hashes[name], digest)}
            cells.append({"name": name, "layer": "mlp", "process": process, "provenance": cell_hashes})

    identity = {
        "source": source,
        "original": original,
        "b_config": b_config,
        "encoder_hashes": encoder_hashes,
        "decoder_hashes": decoder_hashes,
        "natural_parameter_hashes": natural_receipt["parameter_sha256"],
        "provenance": provenance,
    }
    return source, identity, cells


def load_readout(cell, identity):
    """Return a frozen callable on physical simulator states and its archive hashes."""
    source = identity["source"]
    name, layer, process = cell["name"], cell["layer"], cell["process"]
    original = identity["original"]
    encoder_hashes = identity["encoder_hashes"]
    encoder_path = TRAIN / "runs" / f"{name}.pt"
    expected_encoder = encoder_hashes[encoder_path.name]
    provenance = _artifact(encoder_path, expected_encoder, source.digest)
    if layer == "lag":
        case, seed = name.rsplit("_", 1)
        transform_path = NATURAL / "primary" / f"{process}__{name}.json"
        expected_transform = identity["natural_parameter_hashes"][transform_path.name]
        provenance.update(_artifact(transform_path, expected_transform, source.digest))
        parameters = source.read(transform_path)
        _expected(parameters["encoder_sha256"], expected_encoder, transform_path)
        transform = np.asarray(parameters["transform"], dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"Invalid frozen named transform: {transform_path}")
        encoder, _ = source.helper.load_encoder(case, int(seed), original, encoder_hashes)

        def readout(states):
            return source.helper.source.embed(encoder, states) @ transform

    elif layer == "mlp":
        decoder = READOUT / "runs" / f"{name}_readouts.pt"
        decoder_hashes = identity["decoder_hashes"]
        provenance.update(_artifact(decoder, decoder_hashes[name], source.digest))
        readout, _ = source.frozen_model(name, original, identity["b_config"], encoder_hashes, decoder_hashes)
    else:
        raise ValueError(f"Unknown frozen layer: {layer}")
    _expected(provenance, cell["provenance"], (layer, process, name))
    return readout, provenance


def sources_manifest():
    """Return repository-relative hashes for the loader and its frozen inputs."""
    source, identity, cells = load_panel()
    manifest = {_repo_key(Path(__file__)): source.digest(Path(__file__)), **identity["provenance"]}
    for cell in cells:
        manifest.update(cell["provenance"])
    return manifest


def _rng(master, role):
    identifier = ROLES[role] if isinstance(role, str) else role
    if identifier not in ROLES.values():
        raise ValueError(f"Unrecognized data role: {role}")
    return np.random.default_rng(np.random.SeedSequence([master, identifier]))


def paths(master, role, n, length, rates):
    """Generate stationary Gaussian AR paths, sharing innovations across rate vectors."""
    rates = np.asarray(rates, dtype=np.float64)
    if rates.shape != (4,) or not np.isfinite(rates).all() or np.any(np.abs(rates) >= 1):
        raise ValueError("Expected four stationary AR rates")
    innovations = _rng(master, role).normal(size=(n, length, 4))
    states = innovations.copy()
    noise_scale = np.sqrt(1 - rates**2)
    for step in range(1, length):
        states[:, step] = rates * states[:, step - 1] + noise_scale * innovations[:, step]
    return states


def iids(master, role, n):
    """Draw stationary iid states on the selected independent role stream."""
    return _rng(master, role).normal(size=(n, 4))
