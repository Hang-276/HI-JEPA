"""Run dual development gates and the sealed joint-dynamics comparison.

The initial engineering fixture retains its original code and receipts. This
entry point reuses its training, audit, scoring, and report functions.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

import render
import run as base


HERE = Path(__file__).resolve().parent
CONFIG_DIR = base.ROOT / "configs/visual_sprites_lejepa"
FORMAL_NAME = "formal_v1"
TUNABLE = ("steps", "learning_rate", "lejepa_sigreg_weight", "predictor_sigreg_weight", "long_lag")


def config_path(path):
    path = path.absolute()
    if path.parent != CONFIG_DIR or path.resolve() != path or path.suffix != ".json":
        raise ValueError("Use a physical JSON config in configs/visual_sprites_lejepa/")
    return path


def source_hashes(config):
    paths = [config, base.METRICS, base.SIGREG, HERE / "models.py", HERE / "render.py", HERE / "run.py", Path(__file__)]
    return {str(p.relative_to(base.ROOT)): base.sha256(p) for p in paths}


def checked_sources(lock):
    path = config_path(base.ROOT / lock["config_path"])
    if source_hashes(path) != lock["source_sha256"] or base.read_json(path) != lock["config"]:
        raise ValueError("Code or configuration differs from the design lock")


def validate_config(config):
    if "fixture_only" in config or config["seeds"] != [101, 102, 103]:
        raise ValueError("The formal run requires the three preregistered seeds")
    if config["development_model_seed"] in config["seeds"] or config["development_seed"] == config["data_seed"]:
        raise ValueError("Development must use independent model and data seeds")
    if config["development_flip_rate_range"] != [.08, .12]:
        raise ValueError("The joint search targets an 8–12% development flip rate")
    ax, ay, dx, dy = config["rho"]
    if not (.955 <= ax <= .988 and .955 <= ay <= .988 and .20 <= dx <= .78 and .20 <= dy <= .78
            and ax != ay and dx != dy):
        raise ValueError("Expected distinct axis coefficients within the approved slow and fast ranges")
    slow_times = [-1 / math.log(rho) for rho in (ax, ay)]
    fast_times = [-1 / math.log(rho) for rho in (dx, dy)]
    lag = config["encoder"]["long_lag"]
    if not isinstance(lag, int) or isinstance(lag, bool) or not max(fast_times) < lag < min(slow_times):
        raise ValueError("Prediction lag must lie between the fast and slow correlation times")
    if config["lags"] != [4, 16, 32] or any(config["roles"][role]["frames"] != 193 for role in ("audit", "ref", "cal", "test")):
        raise ValueError("The scoring window and lag definitions must match the registered protocol")
    for key, expected in (("audit_min_r2", .85), ("cal_gap_quantile", .5), ("family_alpha", .05),
                          ("minimum_coverage", .35), ("maximum_selective_error", .01)):
        if config[key] != expected:
            raise ValueError(f"Cannot change the registered formal value of {key}")
    return {"pred_lag": lag, "slow_correlation_times": slow_times, "fast_correlation_times": fast_times,
            "lag_over_longest_fast_time": lag / max(fast_times), "rho_to_pred_lag": [rho**lag for rho in config["rho"]]}


def verify_attempt_two(config, sources):
    previous = base.result_path("development_joint_v1_attempt1")
    lock = base.read_json(previous / "DESIGN_LOCK.json")
    checked_sources(lock)
    receipt = base.read_json(previous / "DEVELOPMENT_RECEIPT.json")
    if (receipt["status"] != "DEVELOPMENT_GATE_FAILED" or receipt["passes"] is not False
            or receipt["design_lock_sha256"] != base.sha256(previous / "DESIGN_LOCK.json")):
        raise ValueError("A single retry requires a completed, failed first development attempt")
    old_config = lock["config"]
    changed = {key: {"before": old_config["encoder"].get(key), "after": config["encoder"].get(key)}
               for key in TUNABLE if old_config["encoder"].get(key) != config["encoder"].get(key)}
    neutral = deepcopy(config)
    for key in changed:
        neutral["encoder"][key] = old_config["encoder"][key]
    if not changed or neutral != old_config:
        raise ValueError(f"Retry must adjust only encoder fields {TUNABLE} and retain all data, seeds and gates")
    common_sources = {path: digest for path, digest in lock["source_sha256"].items() if path != lock["config_path"]}
    if any(sources.get(path) != digest for path, digest in common_sources.items()):
        raise ValueError("Retry must retain the original code bytes")
    return {"changes": changed, "previous_receipt_sha256": base.sha256(previous / "DEVELOPMENT_RECEIPT.json")}


def lock_run(out, config, source_config, sources, device, mode, preflight, **evidence):
    out.mkdir(parents=True, exist_ok=False)
    lock = {
        "mode": mode,
        "evidence_tier": "formal_visual_sprites" if mode == "formal" else "dual_arm_development",
        "created_utc": base.utc_now(),
        "device": str(device),
        "config": config,
        "config_path": str(source_config.relative_to(base.ROOT)),
        "source_sha256": sources,
        "development_preflight": preflight,
        **evidence,
    }
    if mode == "formal":
        lock["test_seal_protocol"] = (
            "A separate renderer uses transient TEST coordinates and supplies the inference process only images. "
            "The scorer regenerates TEST coordinates after the forecast receipt has been persisted and verified."
        )
    base.save_json(out / "DESIGN_LOCK.json", lock)
    checked_sources(lock)
    return lock


def development_gates(audits, seed):
    positive, negative = [audits[f"{arm}_seed{seed}"]["r2"] for arm in base.ARMS]
    criteria = {
        "visual_lejepa_all_r2_ge_0_90": all(positive[axis] >= .90 for axis in base.AXES),
        "long_horizon_pred_agent_r2_ge_0_90": all(negative[axis] >= .90 for axis in base.AXES[:2]),
        "long_horizon_pred_distractor_r2_lt_0_60": all(negative[axis] < .60 for axis in base.AXES[2:]),
    }
    return criteria, all(criteria.values())


def run_development(out, lock, config, models, torch, device):
    started = time.perf_counter()
    # The shared helper uses this in-memory flag to enumerate both arms on the
    # development data stream. No fixture configuration or blind stage is used.
    helper_config = {**config, "fixture_only": True}
    _, _, audits = base.prepare_models(out, helper_config, models, torch, device, True)
    criteria, passes = development_gates(audits, config["development_model_seed"])
    checked_sources(lock)
    design = out / "DESIGN_LOCK.json"
    receipt = {
        "status": "DEVELOPMENT_GATE_PASSED" if passes else "DEVELOPMENT_GATE_FAILED",
        "evidence_tier": "dual_arm_development",
        "passes": passes,
        "criteria": criteria,
        "audit": audits,
        "audit_sha256": {name: base.sha256(out / name / "audit.json") for name in audits},
        "design_lock_sha256": base.sha256(design),
        "source_sha256": lock["source_sha256"],
        "device": str(device),
        "data_roles": ["pretrain", "train", "val", "audit"],
        "elapsed_seconds": time.perf_counter() - started,
        "finished_utc": base.utc_now(),
    }
    base.save_json(out / "DEVELOPMENT_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


def verify_development(out, lock, config, device):
    receipt = base.read_json(out / "DEVELOPMENT_RECEIPT.json")
    if (receipt["status"] != "DEVELOPMENT_GATE_PASSED" or receipt["passes"] is not True
            or receipt["design_lock_sha256"] != base.sha256(out / "DESIGN_LOCK.json")
            or receipt["device"] != str(device) or receipt["source_sha256"] != lock["source_sha256"]):
        raise ValueError("The formal stage requires a passing development receipt for the locked design")
    for name, digest in receipt["audit_sha256"].items():
        if base.sha256(out / name / "audit.json") != digest or base.read_json(out / name / "audit.json") != receipt["audit"][name]:
            raise ValueError("An AUDIT file differs from the development receipt")
    if not development_gates(receipt["audit"], config["development_model_seed"])[1]:
        raise ValueError("Development evidence does not pass both preregistered gates")
    path = out / "DEVELOPMENT_RECEIPT.json"
    return {"path": str(path.relative_to(base.ROOT)), "sha256": base.sha256(path)}


def render_locked_test(out):
    lock = base.read_json(out / "DESIGN_LOCK.json")
    if lock["mode"] != "formal" or (out / "FORECAST_RECEIPT.json").exists():
        raise ValueError("TEST renderer requires an active formal design before forecast sealing")
    checked_sources(lock)
    config = lock["config"]
    for images in render.iter_role_images(config, "test", base.image_batch_size(config), development=False):
        sys.stdout.buffer.write(memoryview(np.ascontiguousarray(images)).cast("B"))


def blind_images(out, config, record):
    test = config["roles"]["test"]
    count = test["episodes"] * test["frames"]
    shape = (3, 32, 32)
    batch = base.image_batch_size(config)
    digest = hashlib.sha256()
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--render-locked-test"]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as renderer:
        try:
            for start in range(0, count, batch):
                size = min(batch, count - start) * math.prod(shape) * np.dtype(np.float32).itemsize
                raw = renderer.stdout.read(size)
                if len(raw) != size:
                    raise ValueError(f"TEST image stream ended at frame {start}")
                digest.update(raw)
                yield np.frombuffer(raw, dtype=np.float32).reshape(-1, *shape).copy()
                if start and start % (batch * 100) == 0:
                    logging.info("Encoded %d/%d TEST frames for all models", start, count)
            if renderer.stdout.read(1) or renderer.wait():
                raise ValueError("Unexpected renderer output or exit status")
        finally:
            if renderer.poll() is None:
                renderer.terminate()
    record.update({"frames": count, "shape_chw": list(shape), "dtype": "float32",
                   "sha256_contiguous_chw_stream": digest.hexdigest()})


def make_forecast(out, config, lock, models, metrics, device, encoders, probes):
    roles = {}
    for role in ("ref", "cal"):
        logging.info("Encoding %s for all models", role.upper())
        features = base.features_for_role(encoders, config, role, False, models, device)
        shape = config["roles"][role]
        roles[role] = {name: models.predict(probe, features[name], device).reshape(shape["episodes"], shape["frames"], 4)
                       for name, probe in probes.items()}
    physical_ref = base.role_labels(config, "ref", False)
    base.save_npz(out / "reference_truth.npz", states=physical_ref, variance=physical_ref.reshape(-1, 4).var(0, ddof=0))
    thresholds = {}
    for name, ref in roles["ref"].items():
        path = out / name
        variance = ref.reshape(-1, 4).var(0, ddof=0)
        cal = roles["cal"][name]
        scores = metrics.score_windows(cal, cal, variance, variance, config["lags"])
        thresholds[name] = float(np.quantile(scores["gap"], config["cal_gap_quantile"], method="linear"))
        base.save_npz(path / "reference.npz", prediction=ref, variance=variance)
        base.save_npz(path / "calibration.npz", prediction=cal, scores=scores["predicted_scores"], gap=scores["gap"])
        base.save_json(path / "thresholds.json", {"own_cal": thresholds[name]})
    base.save_json(out / "thresholds.json", {"own_cal": thresholds, "quantile": config["cal_gap_quantile"], "method": "linear"})

    checked_sources(lock)
    image_record = {}
    started = base.utc_now()
    logging.info("Starting blind TEST image encoding")
    features = base.features_for_images(encoders, blind_images(out, config, image_record), models, config, device)
    test = config["roles"]["test"]
    for name, probe in probes.items():
        prediction = models.predict(probe, features[name], device).reshape(test["episodes"], test["frames"], 4)
        base.save_npz(out / name / "test_prediction.npz", prediction=prediction)
    checked_sources(lock)
    paths = ["thresholds.json", "reference_truth.npz"]
    paths.extend(f"{name}/{file}" for name in probes for file in (
        "thresholds.json", "test_prediction.npz", "reference.npz", "calibration.npz", "audit.json"))
    digests = {path: base.sha256(out / path) for path in paths}
    base.save_json(out / "FORECAST_RECEIPT.json", {
        "evidence_tier": lock["evidence_tier"],
        "sealed_utc": base.utc_now(),
        "test_image_inference_started_utc": started,
        "design_lock_sha256": base.sha256(out / "DESIGN_LOCK.json"),
        "test_image_stream": image_record,
        "files_sha256": digests,
    })
    logging.info("Sealed %d forecast and calibration files", len(digests))


def verify_formal(out):
    lock = base.read_json(out / "DESIGN_LOCK.json")
    checked_sources(lock)
    path = base.ROOT / lock["development_receipt"]["path"]
    if base.sha256(path) != lock["development_receipt"]["sha256"]:
        raise ValueError("Development receipt hash no longer matches the formal lock")
    receipt, count = base.verify_forecast(out)
    summary = base.read_json(out / "SUMMARY.json")
    if (summary["status"] != "COMPLETE" or summary["test_truth_opened_utc"] < receipt["sealed_utc"]
            or summary["test_truth_sha256"] != base.sha256(out / "test_truth.npz")
            or summary["forecast_receipt_sha256"] != base.sha256(out / "FORECAST_RECEIPT.json")):
        raise ValueError("Formal completion, truth hash, or scoring order verification failed")
    print(json.dumps({"status": "verified", "files_verified": count,
                      "forecast_receipt_sha256": summary["forecast_receipt_sha256"]}), flush=True)


def train_both(args, formal):
    source_config = config_path(args.config)
    config = base.read_json(source_config)
    sources = source_hashes(source_config)
    correlations = validate_config(config)
    adjustment = verify_attempt_two(config, sources) if args.attempt == 2 else None
    preflight = render.development_flip_check(config)
    if not preflight["pass"]:
        raise ValueError(f"Development flip check failed: {preflight}")
    if formal.exists():
        raise FileExistsError(f"Formal result directory already exists: {formal}")
    if sys.prefix == sys.base_prefix:
        raise ValueError("Run training inside the isolated experiment environment")

    import models
    import torch

    device = models.configure_device(args.device, config["runtime"]["cpu_threads"])
    dev = base.result_path(f"development_joint_v1_attempt{args.attempt}")
    dev_lock = lock_run(dev, config, source_config, sources, device, "dual_development", preflight,
                       attempt=args.attempt, adjustment=adjustment, correlation_times=correlations)
    logging.info("Starting dual-arm development attempt %d; physical flip rate %.4f%%", args.attempt, 100 * preflight["flip_rate"])
    try:
        dev_receipt = run_development(dev, dev_lock, config, models, torch, device)
    except Exception as error:
        base.save_json(dev / "FAILURE.json", {"error": repr(error), "failed_utc": base.utc_now()})
        raise
    if not dev_receipt["passes"]:
        logging.info("Stopped at development recovery gate after attempt %d", args.attempt)
        return 3

    dev_record = verify_development(dev, dev_lock, config, device)
    started = time.perf_counter()
    formal_lock = lock_run(formal, config, source_config, sources, device, "formal", preflight,
                           development_receipt=dev_record, development_audit=dev_receipt["audit"],
                           correlation_times=correlations)
    logging.info("Both development arms passed. Starting six locked formal cells")
    try:
        metrics = base.load_metrics()
        encoders, probes, _ = base.prepare_models(formal, config, models, torch, device, False)
        make_forecast(formal, config, formal_lock, models, metrics, device, encoders, probes)
        checked_sources(formal_lock)
        base.score_forecast(formal, config, metrics, started)
        verify_formal(formal)
    except Exception as error:
        base.save_json(formal / "FAILURE.json", {"error": repr(error), "failed_utc": base.utc_now()})
        raise
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--attempt", type=int, default=1, choices=(1, 2))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify", action="store_true")
    modes.add_argument("--render-locked-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    formal = base.result_path(FORMAL_NAME)
    if args.render_locked_test:
        render_locked_test(formal)
        return 0
    if args.verify:
        verify_formal(formal)
        return 0
    if args.config is None:
        parser.error("Training requires --config")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return train_both(args, formal)


if __name__ == "__main__":
    raise SystemExit(main())
