"""Complete local integration and frozen Linux/CUDA experiment execution."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from .config import validate, merge
from .data import prepare_data, load_data
from .diagnostics import pseudo_solution_audit, evaluate_cross_seed, covariance_diagnostics, trajectory_diagnostics
from .evaluation import evaluate_model
from .io import (write_json, read_json, fingerprint, protocol_fingerprint,
                 save_checkpoint, load_checkpoint, environment, source_fingerprint)
from .losses import compute_losses
from .models import build_model, project_network_budget
from .spectral import select_spectrum
from .systems import make_system, compatibility_audit


def configure_runtime(config: dict) -> tuple[torch.device, torch.dtype]:
    options = config["training"]
    device = torch.device(options["device"])
    if config["stage"] == "formal" and device.type != "cuda":
        raise RuntimeError("Formal result generation is reserved for Linux + CUDA; use the local_integration profile on Mac.")
    if config["stage"] == "formal" and not sys.platform.startswith("linux"):
        raise RuntimeError("The frozen formal profile requires Linux.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. CPU fallback is forbidden.")
    if device.type not in ("cpu", "cuda"):
        raise RuntimeError("The validated protocol supports CPU and CUDA; no silent MPS operator fallback.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(int(options["threads"]))
    torch.use_deterministic_algorithms(bool(options["deterministic"]))
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(options["deterministic"])
    torch.manual_seed(int(options["seed"]))
    return device, getattr(torch, options["dtype"])


def dataset_key(config: dict) -> str:
    generator = hashlib.sha256()
    for name in ("systems.py", "data.py"):
        generator.update((Path(__file__).parent / name).read_bytes())
    return fingerprint({"system": config["system"], "data": config["data"],
                        "generator_source_sha256": generator.hexdigest(), "numpy": np.__version__})[:20]


def cached_data(system, config: dict, cache_dir: Path) -> dict:
    destination = cache_dir / dataset_key(config)
    manifest = destination / "manifest.json"
    if manifest.exists():
        metadata = read_json(destination / "cache_identity.json")
        if metadata["dataset_id"] != dataset_key(config):
            raise RuntimeError("Cached data generator/config identity mismatch")
        return load_data(destination)
    result = prepare_data(system, config["data"], destination)
    write_json(destination / "cache_identity.json", {"dataset_id": dataset_key(config),
               "system": config["system"], "data": config["data"], "numpy": np.__version__})
    return result


def sample_batch(split, config, rng, device, dtype):
    observations = split["observations"]
    horizons = sorted({1, int(config["losses"]["k"]), 2 * int(config["losses"]["k"])})
    n, length, dimension = observations.shape
    batch_size = min(int(config["training"]["batch_size"]), n)
    trajectories = rng.choice(n, batch_size, replace=False)
    times = rng.integers(0, length - max(horizons), size=batch_size)
    def tensor(array):
        return torch.as_tensor(np.array(array, copy=True), dtype=dtype, device=device)
    return {"x": tensor(observations[trajectories, times]),
            "futures": {q: tensor(observations[trajectories, times + q]) for q in horizons}}


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _scalars(losses):
    return {key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value
            for key, value in losses.items()}


def _memory(device):
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"process_peak_rss_bytes": int(maximum if sys.platform == "darwin" else maximum * 1024),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _snapshot(model, optimizer, config, selection, rng, generator, step, phase="source", extra=None):
    return {"schema_version": 1, "model": _cpu_state(model), "optimizer": optimizer.state_dict(),
            "config": config, "selection": selection, "protocol_sha256": protocol_fingerprint(config),
            "source_sha256": source_fingerprint(), "step": step, "phase": phase,
            "numpy_rng": rng.bit_generator.state, "projection_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(), "extra": extra or {}}


def prepare_run(config: dict, output_dir: Path, cache_dir: Path):
    system = make_system(config["system"])
    datasets = cached_data(system, config, cache_dir)
    if (output_dir / "selection.json").exists():
        frozen_environment = read_json(output_dir / "environment.json")
        if frozen_environment.get("source_sha256") != source_fingerprint():
            raise RuntimeError("Frozen preflight comes from different source; original files were preserved.")
        return system, datasets, read_json(output_dir / "selection.json"), read_json(output_dir / "compatibility.json")
    selection = select_spectrum(system, config["selection"], datasets["train"])
    selection.setdefault("certificate", {})["parameterization_margin"] = config["selection"]["certificate_margin"]
    audit = compatibility_audit(system, selection, config["data"], config["losses"])
    for key in ("reference_contraction_upper_bound", "spectral_spread_order", "inverse_contraction_margin"):
        if key in audit:
            selection.setdefault("certificate", {})[key] = audit[key]
            selection[key] = audit[key]
    selection["certificate"]["reference_bound_provenance"] = "truth_compatibility_audit_after_query_selection"
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "selection.json", selection)
    write_json(output_dir / "compatibility.json", audit)
    write_json(output_dir / "data_reference.json", {"dataset_id": dataset_key(config),
               "relative_cache": os.path.relpath(cache_dir / dataset_key(config), output_dir),
               "truth_access": "state observation map for rank shared across controls; modal truth for audit/evaluation only"})
    return system, datasets, selection, audit


def anticollapse_record(model, datasets, config):
    """Independent trajectory samples from train only; no test-based gating."""
    x = datasets["train"]["observations"]
    count = min(len(x), 4096)
    source_times = x.shape[1] - 2*config["losses"]["k"]
    rng = np.random.default_rng(552011)
    t = rng.integers(0, source_times, size=count)
    parameter = next(model.parameters())
    with torch.no_grad():
        samples = torch.as_tensor(x[np.arange(count), t], device=parameter.device, dtype=parameter.dtype)
        z = model.encode(samples).cpu().double().numpy()
        c = model.coarse(samples).cpu().double().numpy()
        dynamic_x = torch.as_tensor(x[:min(count, 128)], device=parameter.device, dtype=parameter.dtype)
        shape = dynamic_x.shape
        dynamics_z = model.encode(dynamic_x.reshape(-1, shape[-1])).cpu().double().numpy().reshape(shape[0], shape[1], -1)
        dynamics_c = model.coarse(dynamic_x.reshape(-1, shape[-1])).cpu().double().numpy().reshape(shape[0], shape[1], -1)
    fine, coarse = covariance_diagnostics(z), covariance_diagnostics(c)
    fine_dynamic = trajectory_diagnostics(dynamics_z)
    coarse_dynamic = trajectory_diagnostics(dynamics_c)
    limits = config["anticollapse"]
    checks = {"fine_effective_rank": fine["effective_rank"] >= limits["minimum_fine_effective_rank"],
              "coarse_covariance_floor": coarse["minimum_covariance_eigenvalue"] >= (.2*limits["coarse_floor_ratio"])**2-1e-6,
              "fine_dynamic_information": (fine_dynamic["dynamic_fraction_of_total_variance"] or 0) >= limits["minimum_dynamic_fraction"],
              "coarse_dynamic_information": (coarse_dynamic["dynamic_fraction_of_total_variance"] or 0) >= limits["minimum_dynamic_fraction"]}
    return {"passed": all(checks.values()), "checks": checks, "thresholds": limits,
            "threshold_provenance": "predeclared engineering acceptance choices; not proposal constants",
            "fit_and_score_split": "train only; one sampled time per independent trajectory for covariance",
            "test_accessed": False, "independent_trajectories": count,
            "fine": fine, "coarse": coarse, "fine_dynamic": fine_dynamic,
            "coarse_dynamic": coarse_dynamic, "formal_acceptance": config["stage"] == "formal"}


def train_run(config: dict, output_dir: Path | str, cache_dir: Path | str,
              resume=False, stop_after: int | None = None) -> dict:
    cfg = validate(config)
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device, dtype = configure_runtime(cfg)
    result_file = output_dir / "result.json"
    if result_file.exists():
        result = read_json(result_file)
        if result.get("protocol_sha256") != protocol_fingerprint(cfg):
            raise RuntimeError("Output already contains a different frozen protocol; choose a new directory.")
        if resume and result.get("status") in ("completed", "unresolved"):
            if result.get("source_sha256") != source_fingerprint():
                raise RuntimeError("Completed result was generated by different source; choose a new result directory.")
            return result
        if not resume:
            raise FileExistsError("Run already exists. Use --resume or a new output directory.")
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint = None
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError("Checkpoint exists; use --resume or a new output directory.")
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint["protocol_sha256"] != protocol_fingerprint(cfg):
            raise RuntimeError("Checkpoint protocol differs from requested configuration.")
        if checkpoint["source_sha256"] != source_fingerprint():
            raise RuntimeError("Source changed since checkpoint; start a new run to preserve result provenance.")
    if (output_dir / "config.json").exists() and protocol_fingerprint(read_json(output_dir / "config.json")) != protocol_fingerprint(cfg):
        raise RuntimeError("Frozen preflight configuration differs; choose a new output directory.")
    env = environment(str(device))
    if not (output_dir / "environment.json").exists():
        write_json(output_dir / "environment.json", env)
    system, datasets, selection, audit = prepare_run(cfg, output_dir, cache_dir)
    if not selection.get("valid_for_training", False):
        result = {"status": "unresolved", "name": cfg["name"], "stage": cfg["stage"],
                  "protocol_sha256": protocol_fingerprint(cfg), "selection": selection,
                  "source_sha256": source_fingerprint(),
                  "reason": "Declared spectral access does not certify a complete target; retain this experimental condition."}
        write_json(result_file, result)
        return result
    if not audit.get("local_rank_compatible", True):
        raise RuntimeError("Truth audit rejects the frozen rank setting; inspect compatibility.json.")
    if not 1 <= selection["m"] < system.dimension:
        raise RuntimeError("Partial-mode HI-JEPA requires 1 <= m < state dimension.")
    for key in ("inverse_contraction_compatible", "reference_in_allowed_K_domain",
                "state_jacobian_normal_by_construction"):
        if not audit.get(key, False):
            raise RuntimeError(f"Frozen main-theory setting fails {key}; inspect compatibility.json without silently changing the setting.")
    # Initialization always happens on CPU, and initial tensors are persisted.
    model = build_model(cfg, selection).to(dtype=dtype)
    model.to(device)
    project_network_budget(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    rng = np.random.default_rng(cfg["training"]["seed"] + 9001)
    generator = torch.Generator(device="cpu").manual_seed(cfg["training"]["seed"] + 19001)
    start = 0
    if checkpoint is not None:
        if fingerprint(checkpoint["selection"]) != fingerprint(selection):
            raise RuntimeError("Frozen spectral selection differs from checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.bit_generator.state = checkpoint["numpy_rng"]
        generator.set_state(checkpoint["projection_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        start = checkpoint["step"]
    else:
        save_checkpoint(output_dir / "initial.pt", {"model": _cpu_state(model), "config": cfg,
                        "selection": selection, "protocol_sha256": protocol_fingerprint(cfg),
                        "source_sha256": source_fingerprint()})
    total_steps = int(cfg["training"]["steps"])
    end = min(total_steps, stop_after) if stop_after is not None else total_steps
    if end < start:
        raise ValueError("stop_after precedes checkpoint step")
    model.train()
    _sync(device)
    started = time.perf_counter()
    losses = {}
    if start and (output_dir / "training.jsonl").exists():
        previous_lines = (output_dir / "training.jsonl").read_text().splitlines()
        if previous_lines:
            losses = json.loads(previous_lines[-1])["losses"]
    interval = int(cfg["training"]["checkpoint_every"])
    with (output_dir / "training.jsonl").open("a", encoding="utf-8") as log:
        for step in range(start, end):
            batch = sample_batch(datasets["train"], cfg, rng, device, dtype)
            optimizer.zero_grad(set_to_none=True)
            total, parts = compute_losses(model, batch, system, cfg, generator)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"Nonfinite total loss at step {step + 1}")
            total.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite parameter gradient at step {step + 1}")
            optimizer.step()
            project_network_budget(model)
            if step == start or (step + 1) % interval == 0 or step + 1 == end:
                _sync(device)
                losses = _scalars(parts)
                losses["total"] = float(total.detach().cpu())
                record = {"step": step + 1, "phase": "source", "losses": losses,
                          "elapsed_seconds_this_segment": time.perf_counter() - started}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                save_checkpoint(checkpoint_path, _snapshot(model, optimizer, cfg, selection,
                                rng, generator, step + 1))
    _sync(device)
    seconds = time.perf_counter() - started
    if end < total_steps:
        return {"status": "interrupted_for_resume_check", "step": end,
                "protocol_sha256": protocol_fingerprint(cfg)}
    acceptance = anticollapse_record(model, datasets, cfg)
    write_json(output_dir / "anticollapse.json", acceptance)
    evaluation = evaluate_model(model, system, datasets, selection, cfg, output_dir)
    source_counts = model.parameter_counts()
    smoothness = model.smoothness_budget(system)
    transfer = None
    if cfg["transfer"]["enabled"]:
        transfer = run_transfer(model, cfg, selection, output_dir / "transfer", cache_dir, device, dtype, resume)
    result = {"status": "completed", "name": cfg["name"], "stage": cfg["stage"],
              "evidence": "engineering_integration_only" if cfg["stage"] == "local_integration" else "formal_synthetic_mechanism",
              "protocol_sha256": protocol_fingerprint(cfg), "source_sha256": source_fingerprint(),
              "dataset_id": dataset_key(cfg), "config": cfg, "selection": selection,
              "parameter_counts": source_counts, "smoothness_budget": smoothness, "source_updates": total_steps,
              "anticollapse": acceptance,
              "last_losses": losses, "source_training_seconds_this_segment": seconds,
              "average_step_seconds_this_segment": seconds / (end - start) if end > start else None,
              "evaluation": evaluation, "transfer": transfer, "environment": env,
              "memory": _memory(device)}
    write_json(result_file, result)
    return result


def run_transfer(model, source_config, selection, output_dir, cache_dir, device, dtype, resume=False):
    from .evaluation import evaluate_transfer_thresholds
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(source_config)
    cfg["system"]["coupling"] = cfg["transfer"]["target_coupling"]
    cfg["data"]["seed"] += 100000
    cfg["data"]["train_trajectories"] = cfg["transfer"]["target_train_trajectories"]
    cfg["training"]["phase"] = "transfer"
    cfg["training"]["steps"] = cfg["transfer"]["steps"]
    target = make_system(cfg["system"])
    datasets = cached_data(target, cfg, cache_dir)
    write_json(output_dir / "config.json", cfg)
    frozen_before = {"G." + k: v.detach().cpu().clone() for k, v in model.G.state_dict().items()}
    frozen_before.update({"K." + k: v.detach().cpu().clone() for k, v in model.K.state_dict().items()})
    for module in (model.G, model.K):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    parameters = [p for module in (model.E, model.F) for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=cfg["training"]["learning_rate"])
    rng = np.random.default_rng(cfg["training"]["seed"] + 29001)
    generator = torch.Generator(device="cpu").manual_seed(cfg["training"]["seed"] + 39001)
    checkpoint_file = output_dir / "checkpoint.pt"
    start, records, frozen, first_reached = 0, [], None, None
    if resume and checkpoint_file.exists():
        checkpoint = load_checkpoint(checkpoint_file)
        if checkpoint["source_sha256"] != source_fingerprint():
            raise RuntimeError("Transfer source changed since checkpoint")
        if checkpoint["protocol_sha256"] != protocol_fingerprint(cfg):
            raise RuntimeError("Transfer protocol changed since checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.bit_generator.state = checkpoint["numpy_rng"]
        generator.set_state(checkpoint["projection_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        start = checkpoint["step"]
        state = checkpoint["extra"]
        frozen, records, first_reached = state["frozen"], state["records"], state["first_reached"]
    if frozen is None:
        initial = evaluate_transfer_thresholds(model, target, datasets, selection, cfg, frozen=None)
        frozen = initial.pop("frozen")
        records.append({"step": 0, **initial})
        if initial["reached"]:
            first_reached = 0
        write_json(output_dir / "frozen_monitor.json", frozen)
    steps, interval = cfg["transfer"]["steps"], cfg["transfer"]["evaluation_every"]
    model.train()
    started = time.perf_counter()
    for step in range(start, steps):
        optimizer.zero_grad(set_to_none=True)
        batch = sample_batch(datasets["train"], cfg, rng, device, dtype)
        total, _ = compute_losses(model, batch, target, cfg, generator)
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Nonfinite transfer objective")
        total.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError("Nonfinite transfer gradient")
        optimizer.step()
        project_network_budget(model)
        if (step + 1) % interval == 0 or step + 1 == steps:
            monitor = evaluate_transfer_thresholds(model, target, datasets, selection, cfg, frozen=frozen)
            monitor.pop("frozen", None)
            records.append({"step": step + 1, **monitor})
            if monitor["reached"] and first_reached is None:
                first_reached = step + 1
            save_checkpoint(checkpoint_file, _snapshot(model, optimizer, cfg, selection, rng, generator,
                            step + 1, "transfer", {"frozen": frozen, "records": records, "first_reached": first_reached}))
    frozen_after = {"G." + k: v.detach().cpu() for k, v in model.G.state_dict().items()}
    frozen_after.update({"K." + k: v.detach().cpu() for k, v in model.K.state_dict().items()})
    unchanged = all(torch.equal(value, frozen_after[key]) for key, value in frozen_before.items())
    if not unchanged:
        raise RuntimeError("Frozen G/K changed during adaptation")
    final_evaluation = evaluate_model(model, target, datasets, selection, cfg, output_dir)
    result = {"source_to_target": {"source_coupling": source_config["system"]["coupling"],
                                  "target_coupling": cfg["system"]["coupling"]},
              "score_split_for_cost": "calibration_monitor", "test_controls_stopping": False,
              "maximum_updates": steps, "executed_updates": steps,
              "first_reached_updates": first_reached,
              "status": "reached" if first_reached is not None else "not_reached_within_budget",
              "counts_as_equal_accuracy_cost": first_reached is not None,
              "cost_resolution_updates": interval,
              "trainable_parameters": sum(p.numel() for p in parameters),
              "G_K_bitwise_unchanged": unchanged, "records": records,
              "elapsed_seconds": time.perf_counter() - started,
              "final_test_evaluation": final_evaluation}
    write_json(output_dir / "result.json", result)
    return result


def run_matrix(configs: list[dict], output_dir: Path | str, resume=False) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source_sha256": source_fingerprint(), "runs": [
        {"name": c["name"], "protocol_sha256": protocol_fingerprint(c), "config": c} for c in configs]}
    manifest_path = output_dir / "matrix_manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        identity = lambda x: [(r["name"], r["protocol_sha256"]) for r in x["runs"]]
        if identity(old) != identity(manifest) or old.get("source_sha256") != manifest["source_sha256"]:
            raise RuntimeError("Existing matrix has a different protocol or source; original results were preserved.")
    else:
        write_json(manifest_path, manifest)
    records = []
    for index, config in enumerate(configs):
        print(f"[{index+1}/{len(configs)}] {config['name']}", flush=True)
        try:
            if config.get("reuse_from"):
                original = read_json(output_dir / config["reuse_from"] / "result.json")
                original_cfg = copy.deepcopy(original["config"])
                requested = copy.deepcopy(config)
                for item in (original_cfg, requested):
                    item.pop("name", None)
                    item.pop("reuse_from", None)
                if protocol_fingerprint(original_cfg) != protocol_fingerprint(requested):
                    raise RuntimeError("Attempted result reuse across different scientific protocols")
                if original["source_sha256"] != source_fingerprint() or original["status"] != "completed":
                    raise RuntimeError("Reused result is not complete with current source")
                record = copy.deepcopy(original)
                record.update({"name": config["name"], "config": config,
                               "protocol_sha256": protocol_fingerprint(config),
                               "reused_from": config["reuse_from"],
                               "independent_replication": False})
                destination = output_dir / config["name"]
                write_json(destination / "result.json", record)
                write_json(destination / "artifact_reference.json", {"relative_directory": "../"+config["reuse_from"]})
            else:
                record = train_run(config, output_dir / config["name"], output_dir / "data", resume=resume)
        except Exception as error:
            import traceback
            record = {"name": config["name"], "status": "failed", "error": str(error),
                      "traceback": traceback.format_exc(), "stage": config["stage"]}
            write_json(output_dir / config["name"] / "failure.json", record)
            print(record["traceback"], flush=True)
        records.append(record)
        write_json(output_dir / "progress.json", {"finished": len(records), "total": len(configs),
                   "statuses": [{"name": r["name"], "status": r["status"]} for r in records]})
        if config["stage"] == "formal" and config["name"].startswith("e1_baseline_seed"):
            required = config["anticollapse"]["acceptance_seeds"]
            acceptance_records = [r for r in records if r.get("config", {}).get("training", {}).get("seed") in required
                                  and r["name"].startswith("e1_baseline_seed") and r["status"] == "completed"]
            if len(acceptance_records) == len(required):
                accepted = all(r["anticollapse"]["passed"] for r in acceptance_records)
                write_json(output_dir / "anticollapse_acceptance.json", {"passed": accepted, "seeds": required,
                           "test_accessed": False, "records": [r["anticollapse"] for r in acceptance_records]})
                if not accepted:
                    write_json(output_dir / "halted.json", {"reason": "Fixed anti-collapse infrastructure failed the declared two-seed acceptance; remaining formal tasks were not run."})
                    break
    return summarize(output_dir)


def summarize(output_dir: Path | str) -> dict:
    output_dir = Path(output_dir)
    manifest = read_json(output_dir / "matrix_manifest.json")
    records = []
    for run in manifest["runs"]:
        path = output_dir / run["name"]
        if (path / "result.json").exists():
            saved = read_json(path / "result.json")
            if saved.get("protocol_sha256") != run["protocol_sha256"] or saved.get("source_sha256") != manifest.get("source_sha256"):
                records.append({"name": run["name"], "status": "failed", "error": "Saved result identity differs from matrix"})
            else:
                records.append(saved)
        elif (path / "failure.json").exists():
            records.append(read_json(path / "failure.json"))
        else:
            records.append({"name": run["name"], "status": "pending"})
    groups = {}
    group_records = {}
    for record in records:
        if record["status"] == "completed":
            key = record["name"].rsplit("_seed", 1)[0]
            groups.setdefault(key, []).append(record["evaluation"])
            group_records.setdefault(key, []).append(record)
    consistency = {}
    from .evaluation import cross_seed_consistency
    for key, runs in group_records.items():
        calibration, test = [], []
        for run in runs:
            path = output_dir / run.get("reused_from", run["name"]) / "seed_outputs.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as arrays:
                    calibration.append(arrays["calibration"])
                    test.append(arrays["test"])
        if len(calibration) >= 2:
            consistency[key] = cross_seed_consistency(calibration, test, runs[0]["selection"]["clusters"])
    summary = {"planned_runs": len(records),
               "completed": sum(r["status"] == "completed" for r in records),
               "unresolved": sum(r["status"] == "unresolved" for r in records),
               "failed": sum(r["status"] == "failed" for r in records),
               "pending": sum(r["status"] == "pending" for r in records),
               "evidence": "engineering_only" if all(c["config"]["stage"] == "local_integration" for c in manifest["runs"]) else "formal_synthetic_mechanism",
               "runs": [{"name": r["name"], "status": r["status"],
                          "error": r.get("error"), "source_updates": r.get("source_updates"),
                          "average_step_seconds": r.get("average_step_seconds_this_segment"),
                          "transfer_status": r.get("transfer", {}).get("status") if r.get("transfer") else None}
                         for r in records],
               "cross_seed": {key: evaluate_cross_seed(value) for key, value in groups.items()},
               "cross_seed_representation_consistency": consistency,
               "reused_results_are_independent_replications": False}
    summary["halted"] = (output_dir / "halted.json").exists()
    write_json(output_dir / "summary.json", summary)
    lines = ["# HI-JEPA 运行汇总", "", f"证据范围：{summary['evidence']}。", "",
             f"完成 {summary['completed']}，不可认证 {summary['unresolved']}，失败 {summary['failed']}，待运行 {summary['pending']}。", "",
             "| 任务 | 状态 | 秒每步 | 迁移结果 |", "| --- | --- | --- | --- |"]
    for row in summary["runs"]:
        seconds = row["average_step_seconds"]
        lines.append(f"| {row['name']} | {row['status']} | {seconds:.5f} | {row['transfer_status'] or 'N/A'} |" if seconds is not None else
                     f"| {row['name']} | {row['status']} | N/A | N/A |")
    lines += ["", "local_integration 的指标仅用于确认代码路径，不构成正式恢复或分层收益结论。",
              "不可认证的预设条件被保留；缺测使用 N/A。正式主结果应在冻结的 Linux CUDA 环境统一生成。", ""]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
