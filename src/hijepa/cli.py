"""Research CLI with explicit evidence profiles and no device fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config, expand_matrix
from .io import write_json, serializable


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hijepa")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Inspect and exercise the selected derivative backend once")
    doctor.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    doctor.add_argument("--output")
    run = sub.add_parser("run", help="Train and evaluate one frozen configuration")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", choices=("cpu", "cuda"))
    run.add_argument("--resume", action="store_true")
    matrix = sub.add_parser("matrix", help="Execute a complete declared local or formal matrix")
    matrix.add_argument("--config", required=True)
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--device", choices=("cpu", "cuda"))
    matrix.add_argument("--resume", action="store_true")
    dry = sub.add_parser("manifest", help="Freeze and inspect all matrix configurations without training")
    dry.add_argument("--config", required=True)
    dry.add_argument("--output", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--input", required=True)
    audit = sub.add_parser("audit-pseudosolutions")
    audit.add_argument("--output", required=True)
    export = sub.add_parser("parity-export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    parity = sub.add_parser("parity-check")
    parity.add_argument("--bundle", required=True)
    parity.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parity.add_argument("--rtol", type=float, default=1e-4)
    parity.add_argument("--atol", type=float, default=1e-6)
    parity.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        from .verification import doctor_check
        result = doctor_check(args.device)
        if args.output:
            write_json(args.output, result)
    elif args.command == "run":
        from .runner import train_run
        cfg = load_config(args.config)
        if args.device:
            cfg["training"]["device"] = args.device
        output = Path(args.output)
        result = train_run(cfg, output, output.parent / "data", resume=args.resume)
        result = {key: result.get(key) for key in ("name", "status", "stage", "evidence", "source_updates")}
    elif args.command in ("matrix", "manifest"):
        from .runner import run_matrix
        from .io import protocol_fingerprint
        configs = expand_matrix(args.config)
        if getattr(args, "device", None):
            for cfg in configs:
                cfg["training"]["device"] = args.device
        if args.command == "manifest":
            result = {"run_count": len(configs), "runs": [
                {"name": c["name"], "protocol_sha256": protocol_fingerprint(c), "config": c} for c in configs]}
            write_json(args.output, result)
        else:
            result = run_matrix(configs, args.output, resume=args.resume)
    elif args.command == "summarize":
        from .runner import summarize
        result = summarize(args.input)
    elif args.command == "audit-pseudosolutions":
        from .diagnostics import pseudo_solution_audit
        result = pseudo_solution_audit()
        write_json(args.output, result)
    elif args.command == "parity-export":
        from .parity import export_bundle
        result = export_bundle(args.checkpoint, args.output)
    else:
        from .parity import check_bundle
        result = check_bundle(args.bundle, args.device, args.rtol, args.atol, args.output)
    print(json.dumps(serializable(result), ensure_ascii=False, indent=2, allow_nan=False))
    if result.get("failed", 0) or result.get("passed") is False or result.get("halted", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
