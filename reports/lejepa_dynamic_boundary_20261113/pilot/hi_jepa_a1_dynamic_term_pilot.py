"""CAL/VAL-only A1 dynamic numerator pilot. Never uses a formal TEST role."""
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

HERE = Path.cwd() / "reports/lejepa_natural_direction_repair_20261113"
spec = importlib.util.spec_from_file_location("dynamic_term_pilot_loaders", HERE / "loaders.py")
loaders = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loaders)
MASTER = 2026111387
LAGS = (4, 16, 32)
LENGTH = 193
RATES = {
    "separated_wide": [.93, .918, .892, .88],
    "separated_narrow": [.93, .906, .904, .88],
}
ROLES = {"state": 51, "reference": 52, "fit": 53, "validation": 54}
UPPER = np.triu_indices(4)
PAIRS = np.array(list(itertools.combinations(range(4), 2)))
RIDGE = .001

def rng(role):
    return np.random.default_rng(np.random.SeedSequence([MASTER, ROLES[role]]))

def paths(role, count, rates):
    rates = np.asarray(rates)
    innovations = rng(role).normal(size=(count, LENGTH, 4))
    paths = innovations.copy()
    noise = np.sqrt(1 - rates**2)
    for t in range(1, LENGTH):
        paths[:, t] = rates * paths[:, t - 1] + noise * innovations[:, t]
    return paths

def moments(v, z):
    gram = np.zeros((len(v), 4, 4))
    cross, energy, exact = (np.zeros((len(v), 4)) for _ in range(3))
    e = v - z
    for lag in LAGS:
        dv, dz, de = (x[:, lag:] - x[:, :-lag] for x in (v, z, e))
        gram += np.einsum("ntj,ntk->njk", dv, dv) / dv.shape[1]
        cross += 2 * np.mean(dz * de, axis=1)
        energy += np.mean(de**2, axis=1)
        exact += np.mean(dz**2, axis=1)
    gram, cross, energy, exact = (x / len(LAGS) for x in (gram, cross, energy, exact))
    visible = gram.diagonal(axis1=1, axis2=2)
    np.testing.assert_allclose(visible - exact, cross + energy, rtol=0, atol=2e-14)
    return visible, cross, energy, exact, gram

def features(v, stats):
    diag, _, _, _, gram = stats
    full = gram[:, UPPER[0], UPPER[1]]
    center = v.mean(axis=1)
    shifted = v - center[:, None, :]
    covariance = np.einsum("ntj,ntk->njk", shifted, shifted) / v.shape[1]
    state = np.column_stack([full, center, covariance[:, UPPER[0], UPPER[1]]])
    return {"diag": diag, "full": full, "full_state": state}

def regress(x, y, xv):
    xmean, xscale = x.mean(axis=0), x.std(axis=0)
    standardized = (x - xmean) / xscale
    coefficients = np.linalg.solve(
        standardized.T @ standardized / len(x) + RIDGE * np.eye(x.shape[-1]),
        standardized.T @ (y - y.mean(axis=0)) / len(x),
    )
    return (xv - xmean) / xscale @ coefficients + y.mean(axis=0)

def score(estimated, truth, nominal):
    true_top = np.sort(np.argsort(truth, axis=-1, kind="stable")[:, :2], axis=-1)
    base_top = np.sort(np.argsort(nominal, axis=-1, kind="stable")[:, :2], axis=-1)
    estimated_top = np.sort(np.argsort(estimated, axis=-1, kind="stable")[:, :2], axis=-1)
    base_ok, ok = ((x == true_top).all(axis=1) for x in (base_top, estimated_top))
    pair_true = truth[:, PAIRS[:, 1]] >= truth[:, PAIRS[:, 0]]
    pair_pred = estimated[:, PAIRS[:, 1]] >= estimated[:, PAIRS[:, 0]]
    return {
        "accuracy": float(ok.mean()),
        "score_mae": float(np.abs(estimated - truth).mean()),
        "fixed_pair_23_accuracy": float((pair_true[:, 3] == pair_pred[:, 3]).mean()),
        "fixes": int((ok & ~base_ok).sum()),
        "new_errors": int((~ok & base_ok).sum()),
    }

def rmse(x):
    return np.sqrt(np.mean(x**2, axis=0))

def main():
    start = time.monotonic()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    _, ids, cells = loaders.load_panel()
    zcal = rng("state").normal(size=(32768, 4))
    zref = rng("reference").normal(size=(32768, 4))
    inputs = {
        process: (paths("fit", 512, rates), paths("validation", 256, rates))
        for process, rates in RATES.items()
    }
    output = []
    for cell in cells:
        if cell["layer"] != "mlp":
            continue
        readout, _ = loaders.load_readout(cell, ids)
        affine = np.linalg.lstsq(np.column_stack([np.ones(len(zcal)), zcal]), readout(zcal), rcond=None)[0]
        scale, offset = affine[1:].diagonal(), affine[0]
        assert np.min(abs(scale)) > .05
        standardize = lambda z: (readout(z) - offset) / scale
        variance = np.var(standardize(zref), axis=0)
        z, zv = inputs[cell["process"]]
        v, vv = standardize(z), standardize(zv)
        fitting, validation = moments(v, z), moments(vv, zv)
        x, xv = features(v, fitting), features(vv, validation)
        _, cross, _, _, _ = fitting
        nv, cv, ev, nz, _ = validation
        nominal = nv / variance
        predictors = {
            "zero": np.zeros_like(cv),
            "mean": np.broadcast_to(cross.mean(axis=0), cv.shape),
            **{key: regress(x[key], cross, xv[key]) for key in x},
            "oracle_cross": cv,
        }
        arms = {}
        for key, candidate in predictors.items():
            error = cv - candidate
            mean_rmse = rmse(cv - cross.mean(axis=0))
            vals = {
                "K_rmse": rmse(error).tolist(),
                "K_R2_against_cal_mean": (1 - (rmse(error) / mean_rmse)**2).tolist(),
                "subtract_K_keep_visible_REF": score((nv - candidate) / variance, nz, nominal),
            }
            if key in x:
                first = regress(x[key][:256], cross[:256], xv[key])
                last = regress(x[key][256:], cross[256:], xv[key])
                vals["split_prediction_rmse"] = rmse(first - last).tolist()
                vals["split_prediction_rmse_over_K_centered_rmse"] = (
                    rmse(first - last) / mean_rmse
                ).tolist()
                vals["split_gain_against_cal_mean"] = [
                    (1 - (rmse(cv - part) / mean_rmse)**2).tolist() for part in (first, last)
                ]
            arms[key] = vals
        norm_k = rmse(cv)
        norm_e = rmse(ev)
        denom_piece = nz * (1 - variance)
        output.append({
            "model": cell["name"], "process": cell["process"], "variance": variance.tolist(),
            "K_rmse_about_zero": norm_k.tolist(), "K_centered_sd": cv.std(axis=0).tolist(),
            "K_mean": cv.mean(axis=0).tolist(), "E_rmse_about_zero": norm_e.tolist(),
            "E_mean": ev.mean(axis=0).tolist(), "reference_error_numerator_scale_rmse": rmse(denom_piece).tolist(),
            "arms": arms,
            "oracle_both_terms_keep_visible_REF": score((nv - cv - ev) / variance, nz, nominal),
        })
    result = {"scope": "pilot state CAL+window CAL/VAL only", "seed": MASTER, "roles": ROLES,
              "windows": {"CAL": 512, "VAL": 256}, "ridge_alpha": RIDGE,
              "source_frozen_panel": "14 strong MLP, two prior evaluation AR processes",
              "elapsed_seconds": time.monotonic() - start, "cells": output}
    with Path("/tmp/hi_jepa_a1_dynamic_term_pilot.json").open("x") as out:
        json.dump(result, out, allow_nan=False, indent=2)

if __name__ == "__main__":
    main()
