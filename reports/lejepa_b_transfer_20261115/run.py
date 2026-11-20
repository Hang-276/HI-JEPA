"""Frozen local B1/B3 transfer: CAL, prediction, then truth scoring."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A4 = ROOT / "reports/lejepa_a4_20261115"
sys.path.insert(0, str(A4))
import forecast as old
from panel import run as panel
spec = importlib.util.spec_from_file_location("a3_score", ROOT / "reports/lejepa_a3_gap_20261115/score.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(sources):
    for path, expected in sources.items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen source mismatch: {path}")


def matrix(rates, setting):
    a = np.diag(np.asarray(rates)**setting["exponent"])
    a[1, 2] = setting["coupling"]
    q = np.eye(4) - a @ a.T
    return a, np.linalg.cholesky(q), q


def paths(cfg, seed, process, regime, role, count):
    master = cfg["master_base"] + seed * 10 + list(cfg["processes"]).index(process)
    noise = old.generator(master, role).normal(size=(count, cfg["length"], 4))
    result = noise.copy()
    a, chol, _ = matrix(cfg["processes"][process], cfg["regimes"][regime])
    for t in range(1, cfg["length"]):
        result[:, t] = result[:, t - 1] @ a.T + noise[:, t] @ chol.T
    return result


def precheck(cfg):
    diagnostics = {}
    for process, rates in cfg["processes"].items():
        for regime, setting in cfg["regimes"].items():
            a, chol, q = matrix(rates, setting)
            if not np.allclose(a @ a.T + chol @ chol.T, np.eye(4), atol=1e-14):
                raise ValueError("Stationary covariance check failed")
            ds = np.mean([2 * (1 - np.diag(np.linalg.matrix_power(a, h))) for h in cfg["lags"]], axis=0)
            expected = np.mean([2 * (1 - np.asarray(rates)**(setting["exponent"] * h)) for h in cfg["lags"]], axis=0)
            if not np.allclose(ds, expected, atol=1e-14):
                raise ValueError("Population marginal score check failed")
            diagnostics[process + "__" + regime] = {"transition": a.tolist(), "innovation_covariance": q.tolist(),
                "minimum_innovation_eigenvalue": float(np.linalg.eigvalsh(q).min()), "population_scores": ds.tolist()}
        master = cfg["master_base"] + cfg["seeds"][0] * 10 + list(cfg["processes"]).index(process)
        actual = paths(cfg, cfg["seeds"][0], process, "baseline", 819, 7)
        expected = old.trajectories(cfg, master, 819, 7, rates)
        if not np.allclose(actual, expected, atol=1e-14):
            raise ValueError("Original independent AR generator mismatch")
    return diagnostics


def models(cfg):
    return [(seed, case, f"{case}_{seed}") for seed in cfg["seeds"] for case in cfg["training_cases"]]


def freeze(cfg):
    sources = panel.sources_manifest()
    for path in (HERE / "run.py", HERE / "config.json", HERE / "PROTOCOL.md", A4 / "forecast.py",
                 A4 / "NEW_FORECAST.json", ROOT / "reports/lejepa_a3_gap_20261115/score.py",
                 ROOT / "reports/lejepa_a3_gap_20261115/THRESHOLDS.json"):
        sources[str(path.relative_to(ROOT))] = sha(path)
    locked = read(ROOT / "reports/lejepa_a3_gap_20261115/THRESHOLDS.json")["thresholds"]["g50"]
    if locked != cfg["fixed_threshold"]:
        raise ValueError("A3 threshold mismatch")
    write(HERE / "DESIGN_LOCK.json", {"config": cfg, "sources": sources, "mathematical_checks": precheck(cfg)})
    print("Frozen design and verified stationary generators", flush=True)


def calibrate(cfg):
    design = read(HERE / "DESIGN_LOCK.json")
    verify(design["sources"])
    refs = {(c["name"], c["process"]): c["reference_variance"] for c in read(A4 / "NEW_FORECAST.json")["cells"]}
    gaps = {r: [] for r in cfg["regimes"]}
    audit = {}
    common = old.generator(cfg["master_base"], cfg["audit_role"]).normal(size=(cfg["n_audit"], 4))
    start = time.monotonic()
    for seed, case, name in models(cfg):
        readout, _ = panel.load_readout(name)
        predicted = readout(common)
        audit[name] = (1 - ((predicted-common)**2).mean(axis=0) / common.var(axis=0)).tolist()
        for process in cfg["processes"]:
            for regime in cfg["regimes"]:
                z = paths(cfg, seed, process, regime, cfg["cal_role"], cfg["n_cal"])
                scores = old.slow_scores(readout(z), cfg["lags"]) / refs[name, process]
                ordered = np.sort(scores, axis=1)
                gaps[regime].append(ordered[:, 2] - ordered[:, 1])
        print(f"CAL {name}; audit min R2={min(audit[name]):.6f}", flush=True)
    values = {r: np.concatenate(v) for r, v in gaps.items()}
    with (HERE / "CAL_GAPS.npz").open("xb") as stream:
        np.savez_compressed(stream, **values)
    verify(design["sources"])
    write(HERE / "THRESHOLDS.json", {"design_sha256": sha(HERE / "DESIGN_LOCK.json"),
        "cal_sha256": sha(HERE / "CAL_GAPS.npz"), "fixed": cfg["fixed_threshold"],
        "target_median": {r: float(np.quantile(v, .5, method="higher")) for r, v in values.items()},
        "iid_audit_r2": audit, "elapsed_seconds": time.monotonic()-start})


def forecast(cfg):
    design = read(HERE / "DESIGN_LOCK.json")
    verify(design["sources"])
    thresholds = read(HERE / "THRESHOLDS.json")
    if thresholds["design_sha256"] != sha(HERE / "DESIGN_LOCK.json"):
        raise ValueError("Threshold design mismatch")
    refs = {(c["name"], c["process"]): c["reference_variance"] for c in read(A4 / "NEW_FORECAST.json")["cells"]}
    folder = HERE / "forecasts"
    folder.mkdir(exist_ok=False)
    cells = []
    start = time.monotonic()
    for seed, case, name in models(cfg):
        readout, _ = panel.load_readout(name)
        for process in cfg["processes"]:
            for regime in cfg["regimes"]:
                z = paths(cfg, seed, process, regime, cfg["test_role"], cfg["n_test"])
                prediction = old.slow_scores(readout(z), cfg["lags"]) / refs[name, process]
                file = folder / f"{name}__{process}__{regime}.npz"
                with file.open("xb") as stream:
                    np.savez_compressed(stream, predicted_scores=prediction)
                cells.append({"name": name, "seed": seed, "training_case": case, "process": process, "regime": regime,
                    "path": str(file.relative_to(HERE)), "sha256": sha(file), "reference_variance": refs[name, process]})
        print(f"TEST predictions sealed for {name}", flush=True)
    verify(design["sources"])
    write(HERE / "FORECAST.json", {"design_sha256": sha(HERE / "DESIGN_LOCK.json"),
        "threshold_sha256": sha(HERE / "THRESHOLDS.json"), "cells": cells, "elapsed_seconds": time.monotonic()-start})


def top2(s):
    return np.sort(np.argsort(s, axis=1, kind="stable")[:, :2], axis=1)


def description(mask, wrong):
    n = int(mask.sum())
    errors = int((mask & wrong).sum())
    return {"n": len(mask), "accepted": n, "errors": errors, "coverage": n/len(mask), "risk": errors/n if n else None}


def stratify(gap, wrong, cfg):
    return [{"left": a, "right": b, **description((gap >= a)&(gap < b), wrong)}
            for a, b in zip(cfg["gap_bins"][:-1], cfg["gap_bins"][1:])]


def score(cfg):
    design = read(HERE / "DESIGN_LOCK.json")
    verify(design["sources"])
    receipt = read(HERE / "FORECAST.json")
    thresholds = read(HERE / "THRESHOLDS.json")
    if receipt["threshold_sha256"] != sha(HERE / "THRESHOLDS.json") or receipt["design_sha256"] != sha(HERE / "DESIGN_LOCK.json"):
        raise ValueError("Sealed forecast mismatch")
    expected = {(name,p,r) for _,_,name in models(cfg) for p in cfg["processes"] for r in cfg["regimes"]}
    if {(c["name"],c["process"],c["regime"]) for c in receipt["cells"]} != expected or len(receipt["cells"]) != len(expected):
        raise ValueError("Incomplete panel")
    family = len(expected)*2
    rows = []
    truth = {}
    for cell in receipt["cells"]:
        file = HERE / cell["path"]
        if sha(file) != cell["sha256"]:
            raise ValueError("Prediction mismatch")
        pred = np.load(file)["predicted_scores"]
        if pred.shape != (cfg["n_test"],4) or not np.isfinite(pred).all():
            raise ValueError("Prediction invalid")
        key = (cell["seed"],cell["process"],cell["regime"])
        if key not in truth:
            z = paths(cfg, *key, cfg["test_role"], cfg["n_test"])
            truth[key] = old.slow_scores(z, cfg["lags"])
        physical = truth[key]
        wrong = np.any(top2(pred) != top2(physical), axis=1)
        pgap = np.sort(pred,axis=1)[:,2]-np.sort(pred,axis=1)[:,1]
        tgap = np.sort(physical,axis=1)[:,2]-np.sort(physical,axis=1)[:,1]
        rules = {}
        for label,t in (("fixed",thresholds["fixed"]),("recalibrated",thresholds["target_median"][cell["regime"]])):
            r = description(pgap >= t, wrong)
            upper = cp.cp_upper(r["errors"],r["accepted"],cfg["alpha"]/family)
            rules[label] = {**r,"threshold":t,"upper":upper,"pass_coverage":r["coverage"]>=cfg["minimum_coverage"],
                "pass_risk":upper<=cfg["maximum_error"],"pass":r["coverage"]>=cfg["minimum_coverage"] and upper<=cfg["maximum_error"]}
        rows.append({**cell,"all_answers":description(np.ones(len(wrong),dtype=bool),wrong),"rules":rules,
            "physical_gap_quantiles":np.quantile(tgap,[.1,.25,.5,.75,.9]).tolist(),
            "predicted_gap_quantiles":np.quantile(pgap,[.1,.25,.5,.75,.9]).tolist(),
            "score_mse_by_axis":np.mean((pred-physical)**2,axis=0).tolist(),
            "by_true_gap":stratify(tgap,wrong,cfg),"by_predicted_gap":stratify(pgap,wrong,cfg)})
    aggregate = {}
    for regime in cfg["regimes"]:
        chosen = [r for r in rows if r["regime"] == regime]
        total = sum(r["all_answers"]["n"] for r in chosen)
        aggregate[regime] = {"all_error_rate":sum(r["all_answers"]["errors"] for r in chosen)/total,
            "total_model_window_evaluations":total,"rules":{}}
        for label in ("fixed","recalibrated"):
            vals = [r["rules"][label] for r in chosen]
            n = sum(v["accepted"] for v in vals)
            k = sum(v["errors"] for v in vals)
            aggregate[regime]["rules"][label] = {"accepted":n,"errors":k,"coverage":n/total,"risk":k/n if n else None,
                "passed":sum(v["pass"] for v in vals),"passed_coverage":sum(v["pass_coverage"] for v in vals),
                "passed_risk":sum(v["pass_risk"] for v in vals),"worst_upper":max(v["upper"] for v in vals),
                "minimum_coverage":min(v["coverage"] for v in vals)}
    # Shared innovations support paired descriptive differences, without a new significance claim.
    paired = []
    for row in rows:
        if row["regime"] == "baseline":
            continue
        base = next(b for b in rows if b["regime"]=="baseline" and all(b[k]==row[k] for k in ("name","process")))
        paired.append({"name":row["name"],"process":row["process"],"regime":row["regime"],
            "all_risk_change":row["all_answers"]["risk"]-base["all_answers"]["risk"],
            "fixed_coverage_change":row["rules"]["fixed"]["coverage"]-base["rules"]["fixed"]["coverage"],
            "true_gap_median_change":row["physical_gap_quantiles"][2]-base["physical_gap_quantiles"][2]})
    write(HERE / "SUMMARY.json",{"forecast_sha256":sha(HERE/"FORECAST.json"),"family_size":family,
        "alpha":cfg["alpha"],"aggregate":aggregate,"cells":rows,"paired_descriptive_changes":paired})
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("precheck","freeze","calibrate","forecast","score"))
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config=read(HERE/"config.json")
    if args.phase=="precheck":
        print(json.dumps(precheck(config),indent=2))
    else:
        globals()[args.phase](config)
