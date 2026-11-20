"""Lock natural-error coefficients and seal both rounds before truth scoring."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import loaders

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def forward(z, u):
    fit = np.linalg.lstsq(np.column_stack((np.ones(len(z)), z)), u, rcond=None)[0]
    return fit[0], fit[1:].T


def fit_direction(z, u, config):
    offset, matrix = forward(z, u)
    scale = matrix.diagonal().copy()
    if np.any(np.abs(scale) < 1e-8):
        raise ValueError("Undefined physical slope; stop rather than silently renormalize.")
    v = (u - offset) / scale
    mean = v.mean(axis=0)
    normalized = matrix / scale[:, None]
    halves = np.array([forward(z[sl], v[sl])[1] for sl in
                       (slice(0, len(z) // 2), slice(len(z) // 2, None))])
    pairs = [(j, d) for j in config["recipients"] for d in range(4) if j != d]
    j, d = max(pairs, key=lambda pair: abs(normalized[pair]))
    raw = float(normalized[j, d])
    active = bool(abs(raw) >= config["minimum_loading"] and
                  np.all(np.sign(halves[:, j, d]) == np.sign(raw)) and
                  np.all(np.abs(scale) >= config["minimum_diagonal_slope"]))
    c = raw if active else 0.0
    other = min(k for k in range(4) if k not in (j, d))
    covariance = np.cov(v, rowvar=False, ddof=0)
    wrong_c = c * np.sqrt(covariance[d, d] / covariance[other, other])
    matrices = np.repeat(np.eye(4)[None], len(config["arms"]), axis=0)
    intercepts = np.zeros((len(config["arms"]), 4))
    for arm, donor, coef in ((1, d, -c), (2, d, c), (3, other, -wrong_c)):
        matrices[arm, j, donor] += coef
        intercepts[arm, j] = -coef * mean[donor]
    intercepts[4], matrices[4] = forward(v, z)
    out = np.einsum("nk,ajk->anj", v, matrices) + intercepts[:, None, :]
    np.testing.assert_allclose(c*c*covariance[d, d], wrong_c*wrong_c*covariance[other, other], atol=1e-12)
    return {
        "offset": offset.tolist(), "scale": scale.tolist(), "mean_v": mean.tolist(),
        "forward": normalized.tolist(), "split_forward": halves.tolist(),
        "recipient": j, "donor": d, "wrong_donor": other,
        "raw_coefficient": raw, "active": active, "coefficient": c,
        "wrong_coefficient": float(wrong_c), "matrices": matrices.tolist(),
        "intercepts": intercepts.tolist(), "cal_mse": np.mean((out-z[None])**2, axis=1).tolist(),
        "cal_update_variance": float(c*c*covariance[d, d]),
    }


def predict(visible, ref, fit, config, moment):
    v = (visible - fit["offset"]) / fit["scale"]
    reference = (ref - fit["offset"]) / fit["scale"]
    gram = moment(v, config["lags"])
    covariance = np.cov(reference, rowvar=False, ddof=0)
    matrices = np.asarray(fit["matrices"])
    numerator = np.einsum("ajk,nkl,ajl->anj", matrices, gram, matrices)
    denominator = np.einsum("ajk,kl,ajl->aj", matrices, covariance, matrices)
    if np.any(denominator <= 0):
        raise ValueError("Nonpositive reference variance")
    scores = numerator / denominator[:, None, :]
    raw_variance = ref.var(axis=0)
    raw_scores = moment(visible, config["lags"]).diagonal(axis1=1, axis2=2) / raw_variance
    np.testing.assert_allclose(scores[0], raw_scores, atol=2e-12, rtol=2e-12)
    return scores, gram, covariance, denominator, raw_variance, raw_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((HERE / "protocol.json").read_text())
    C, identity, cells = loaders.load_panel()
    if args.smoke:
        cells = [next(cell for cell in cells if cell["layer"] == layer) for layer in ("lag", "mlp")]
        config = {**config, "rounds": {"smoke": 2026111394}, "n_state_cal": 512,
                  "n_reference": 512, "n_test": 16}
    sources = loaders.sources_manifest()
    for filename in ("run.py", "loaders.py", "score.py", "protocol.json", "PROTOCOL.md", "THEORY.md"):
        path = HERE / filename
        sources[str(path.relative_to(ROOT))] = C.digest(path)
    output = HERE / ("smoke" if args.smoke else "forecast")
    output.mkdir(exist_ok=False)
    lock = {"sources": sources, "config": config, "cells": cells}
    design_path = output / "DESIGN_LOCK.json" if args.smoke else HERE / "DESIGN_LOCK.json"
    write(design_path, lock)
    start = time.monotonic()
    for round_name, master in config["rounds"].items():
        folder = output / round_name
        folder.mkdir()
        z_cal = loaders.iids(master, "state_cal", config["n_state_cal"])
        z_ref = loaders.iids(master, "reference", config["n_reference"])
        for index, cell in enumerate(cells):
            key = f'{cell["layer"]}__{cell["process"]}__{cell["name"]}'
            readout, provenance = loaders.load_readout(cell, identity)
            fit = fit_direction(z_cal, readout(z_cal), config)
            visible_ref = readout(z_ref)
            # The simulator passes states only through the observation and frozen readout here.
            visible = readout(loaders.paths(master, "test", config["n_test"], config["length"],
                                          np.asarray(config["processes"][cell["process"]])))
            scores, gram, cov, var, raw_var, raw_scores = predict(visible, visible_ref, fit, config, C.calculate_moments)
            metadata = {"cell": cell, **fit, "reference_covariance": cov.tolist(),
                        "reference_variance": var.tolist(), "provenance": provenance}
            write(folder / f"{key}.json", metadata)
            with (folder / f"{key}.npz").open("xb") as handle:
                np.savez_compressed(handle, scores=scores, moments=gram,
                                    top2=np.stack([C.top_two(s) for s in scores]),
                                    raw_reference_variance=raw_var, raw_scores=raw_scores)
            if index % 15 == 0:
                print(f"sealed {round_name}: {index+1}/{len(cells)}", flush=True)
    for relative, expected in sources.items():
        if C.digest(ROOT / relative) != expected:
            raise ValueError(f"Source changed while predicting: {relative}")
    files = {str(path.relative_to(HERE)): C.digest(path)
             for path in sorted(output.rglob("*")) if path.is_file()}
    write(output / "RECEIPT.json", {"sources": sources, "files": files,
                                  "design_sha256": C.digest(design_path),
                                  "rounds": config["rounds"], "elapsed_seconds": time.monotonic()-start})
    print("Both rounds sealed; truth remains for independent scoring.", flush=True)


if __name__ == "__main__":
    main()
