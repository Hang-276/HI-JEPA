"""Read-only replay of prior saved models in the current local runtime."""
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "lejepa_slow_rank_20261112" / "training"
sys.path.insert(0, str(OLD))
from run import evaluate, make_mlp_encoder, source_provenance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(old, new, name, errors):
    if isinstance(old, dict):
        assert old.keys() == new.keys(), name
        for key in old:
            compare(old[key], new[key], name + "/" + key, errors)
    elif isinstance(old, list):
        assert len(old) == len(new), name
        for i, (a, b) in enumerate(zip(old, new)):
            compare(a, b, name + f"/{i}", errors)
    elif isinstance(old, (float, int)):
        assert np.isfinite(new), name
        errors.append((abs(old - new), name))
        if "permutation" in name or name.endswith("/correct") or name.endswith("/windows"):
            assert old == new, (name, old, new)
        else:
            assert abs(old - new) < 1e-5, (name, old, new)
    else:
        assert old == new, name


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((OLD / "protocol.json").read_text())
    receipt = json.loads((OLD / "runs/results.json").read_text())
    assert digest(OLD / "protocol.json") == receipt["protocol_sha256"]
    assert digest(OLD / "run.py") == receipt["code_sha256"]
    out = {"old_environment": receipt["environment"], "current_environment": {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "cpu_count": os.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total, "mps_available": torch.backends.mps.is_available(),
        "device": "cpu", "torch_threads": 1,
    }, "author_source": source_provenance(), "checks": []}
    start = time.monotonic()
    for case, rates in config["cases"].items():
        for seed in config["seeds"]:
            checkpoint = OLD / "runs" / f"{case}_{seed}.pt"
            model = make_mlp_encoder(4, config["hidden"], config["layers"], device="cpu").eval()
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            new = evaluate(model, config, seed, np.asarray(rates))
            previous = receipt["cases"][case]["runs"][str(seed)]
            errors = []
            compare({k: previous[k] for k in new}, new, case + f"/{seed}", errors)
            worst = max(errors)
            out["checks"].append({"checkpoint": checkpoint.name, "sha256": digest(checkpoint),
                "max_abs_statistic_delta": worst[0], "worst_field": worst[1],
                "all_reported_top2_counts_and_named_permutations_exact": True})
    out["seconds"] = time.monotonic() - start
    out["max_abs_statistic_delta"] = max(r["max_abs_statistic_delta"] for r in out["checks"])
    out["status"] = "passed"
    (HERE / "ENVIRONMENT_REPLAY.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "checkpoints": len(out["checks"]),
        "max_abs_statistic_delta": out["max_abs_statistic_delta"], "seconds": out["seconds"]}))


if __name__ == "__main__":
    main()
