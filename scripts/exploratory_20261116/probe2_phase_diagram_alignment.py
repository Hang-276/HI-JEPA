#!/usr/bin/env python3
"""Pure NumPy exploratory phase grid, archived-anchor overlay, and stress probe.

No model loading, training, or archived prediction evaluation. The complete
default grid keeps CAL=2048, TEST=4096, REF=32768 for all 800 phase cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = "exploratory_numerical_sandbox"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/exploratory_20261116")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / "results/exploratory_20261116").resolve()):
        parser.error("Output must stay under results/exploratory_20261116/")
    start = time.perf_counter()
    import numpy as np
    from phase_core import run_phase
    from anchors_and_stress import run_anchors, run_stress

    phase = run_phase(args.seed)
    stress = run_stress(args.seed + 2000)
    anchors = run_anchors(phase)
    sources = {}
    for relative in (
        "src/slowrepr/metrics.py",
        "reports/lejepa_slow_rank_20261112/training/run.py",
        "docs/THEORY_FORMALIZATION_20261116.md",
        "scripts/exploratory_20261116/probe2_phase_diagram_alignment.py",
        "scripts/exploratory_20261116/phase_core.py",
        "scripts/exploratory_20261116/anchors_and_stress.py",
    ):
        sources[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    result = {
        "evidence_tier": EVIDENCE,
        "paper_title": "When Does Factor Identifiability Yield Reliable Slow Representations?",
        "status": "completed_exploratory_synthetic_scan",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "seed": args.seed,
        "phase": phase,
        "anchors": anchors,
        "stress": stress,
        "source_sha256": sources,
        "limitations": [
            "Synthetic residual processes are declared hypotheses, not fitted neural error models.",
            "Population-pair inversion SNR and same-window readout error are different targets.",
            "Grid T_min and pointwise intervals are exploratory, not a selection-adjusted guarantee.",
            "Scalar audit R2 cannot identify residual spectrum, direction, or finite-REF distortion.",
            "A signal-amplitude fraction is not the state-MSE definition of audit R2.",
            "The 0.85 admission cutoff is an existing operational gate, not proved necessary or sufficient.",
        ],
    }
    result["elapsed_seconds_before_serialization"] = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    target = output / "probe2_summary.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    elapsed = time.perf_counter() - start
    result["elapsed_seconds"] = elapsed
    result["runtime_below_60_seconds"] = elapsed < 60
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"evidence_tier": EVIDENCE, "output": str(target),
                      "elapsed_seconds": elapsed,
                      "runtime_below_60_seconds": elapsed < 60}, ensure_ascii=False))
    if elapsed >= 60:
        raise RuntimeError("Full run exceeded the requested 60-second limit")


if __name__ == "__main__":
    main()
