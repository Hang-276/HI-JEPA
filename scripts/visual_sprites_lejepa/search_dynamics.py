"""Select distinct stationary AR coefficients using development panels only."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import render


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/visual_sprites_lejepa/default.json"
SEARCH_SEEDS = (2026111701, 2026111703)
CONFIRMATION_SEED = 2026111704
NATURAL_FLIP_TARGET = 176 / 2048
TARGET_RANGE = (0.08, 0.12)
SLOW_PAIRS = tuple((rho / 1000, (rho - 3) / 1000) for rho in range(958, 989))
FAST_PAIRS = tuple((rho / 1000, (rho - 20) / 1000) for rho in range(220, 781, 20))


def flip_grid(config, seed):
    """Score each axis once at every candidate, sharing innovations within a panel."""
    cfg = deepcopy(config)
    cfg["development_seed"] = seed
    cfg["roles"]["audit"]["episodes"] = cfg["development_flip_check_windows"]
    innovations = render._draw_innovations(cfg, "audit", development=True)
    ref_innovations = render._draw_innovations(cfg, "ref", development=True)

    def score(axis, rho):
        x = render._apply_ar_in_place(innovations[..., axis].copy(), rho)
        ref = render._apply_ar_in_place(ref_innovations[..., axis].copy(), rho)
        return render._time_major_score(x, ref, cfg["lags"])

    slow = [np.maximum(score(0, x), score(1, y)) for x, y in SLOW_PAIRS]
    fast = [np.minimum(score(2, x), score(3, y)) for x, y in FAST_PAIRS]
    return np.asarray([[np.count_nonzero(f < s) for f in fast] for s in slow])


def check_design(config, rho, seed):
    cfg = deepcopy(config)
    cfg.update(rho=rho, development_seed=seed, development_flip_rate_range=list(TARGET_RANGE))
    return render.development_flip_check(cfg)


def select(config):
    """Screen on two development panels, then recheck the winner on a third."""
    if config["data_seed"] in (*SEARCH_SEEDS, CONFIRMATION_SEED):
        raise ValueError("Development seeds must be disjoint from the formal data seed")
    windows = config["development_flip_check_windows"]
    counts = [flip_grid(config, seed) for seed in SEARCH_SEEDS]
    eligible = []
    for i, slow in enumerate(SLOW_PAIRS):
        for j, fast in enumerate(FAST_PAIRS):
            rho = [*slow, *fast]
            flip_windows = [int(count[i, j]) for count in counts]
            rates = [count / windows for count in flip_windows]
            if all(TARGET_RANGE[0] <= rate <= TARGET_RANGE[1] for rate in rates):
                eligible.append({"rho": rho, "flip_windows": flip_windows, "flip_rates": rates,
                                 "mean_flip_rate": float(np.mean(rates))})
    if not eligible:
        raise ValueError("No grid cell met the flip target on both development panels")

    # Preserve useful adjacent-pair signal on the faster axes when designs are
    # comparably close to FourRoom. Otherwise select the closest paired rate.
    def rank(row):
        distance = abs(row["mean_flip_rate"] - NATURAL_FLIP_TARGET)
        inside_margin = distance <= 0.01
        return (not inside_margin, -row["rho"][2] if inside_margin else 0, distance, row["rho"])

    eligible.sort(key=rank)
    winner = eligible[0]
    checks = [check_design(config, winner["rho"], seed) for seed in SEARCH_SEEDS]
    if [result["flip_windows"] for result in checks] != winner["flip_windows"]:
        raise RuntimeError("Cached per-axis scores disagree with the original development scorer")
    confirmation = check_design(config, winner["rho"], CONFIRMATION_SEED)
    return {
        "status": "DESIGN_GATE_PASSED" if confirmation["pass"] else "CONFIRMATION_GATE_FAILED",
        "scope": "All sampled data use AUDIT or REF with development=True. No formal TEST role accessed.",
        "search_seeds": list(SEARCH_SEEDS),
        "confirmation_seed": CONFIRMATION_SEED,
        "target": NATURAL_FLIP_TARGET,
        "target_range": list(TARGET_RANGE),
        "windows_per_panel": windows,
        "frames_per_window": config["roles"]["audit"]["frames"],
        "reference_episodes_per_panel": config["roles"]["ref"]["episodes"],
        "lags": config["lags"],
        "slow_pairs": SLOW_PAIRS,
        "fast_pairs": FAST_PAIRS,
        "candidate_count": len(SLOW_PAIRS) * len(FAST_PAIRS),
        "selection_rule": ("Both development panels within 8–12%. For paired mean within one percentage "
                           "point of FourRoom, prefer higher fast rho, then closeness. If none fall within "
                           "that margin, prefer closest mean. Third panel confirms the single frozen winner."),
        "search_flip_counts": [count.tolist() for count in counts],
        "eligible_candidates_ranked": eligible,
        "winner": winner,
        "primary_checks": checks,
        "confirmation": confirmation,
        "numpy_version": np.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    results = ROOT / "results/visual_sprites_lejepa/development"
    if not output.is_relative_to(results):
        raise ValueError("Write search evidence only under the visual development results directory")
    sources = (args.config.resolve(), Path(__file__).resolve(), Path(render.__file__).resolve())
    result = select(json.loads(args.config.read_text()))
    result["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    result["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "winner": result["winner"],
                      "confirmation": result["confirmation"], "output": str(output)}, indent=2))
    if not result["confirmation"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
