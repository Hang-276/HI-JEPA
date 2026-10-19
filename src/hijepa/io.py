"""Atomic, portable experiment records with explicit missing values."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, torch.Tensor):
        return serializable(value.detach().cpu().tolist())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(serializable(value), ensure_ascii=False,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value: Any) -> str:
    raw = json.dumps(serializable(value), sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def protocol_fingerprint(config: dict) -> str:
    import copy
    cfg = copy.deepcopy(config)
    for key in ("device", "threads", "checkpoint_every"):
        cfg.get("training", {}).pop(key, None)
    return fingerprint(cfg)


def save_checkpoint(path: Path | str, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path: Path | str) -> dict:
    # Checkpoints created by this package include optimizer/RNG state.
    # Only load trusted local experiment artifacts.
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def environment(device: str = "cpu") -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "torch": torch.__version__, "numpy": np.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and
                              torch.backends.mps.is_available()),
        "device": device,
        "device_name": torch.cuda.get_device_name() if device.startswith("cuda") and
                       torch.cuda.is_available() else platform.processor(),
        "cpu_threads": torch.get_num_threads(),
        "source_sha256": source_fingerprint(),
        "pid": os.getpid(),
    }
