"""Read sealed B3 outputs; replay only old physical truth, never model inference."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "reports/lejepa_b_transfer_20261115"
REGIMES = ("baseline", "slower", "faster")
COVERAGES = (0.2, 0.5, 0.8)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_scores(cfg, seed, process, regime):
    master = cfg["master_base"] + seed * 10 + list(cfg["processes"]).index(process)
    rng = np.random.default_rng(np.random.SeedSequence([master, cfg["test_role"]]))
    noise = rng.normal(size=(cfg["n_test"], cfg["length"], 4))
    x = noise.copy()
    a = np.diag(np.asarray(cfg["processes"][process]) ** cfg["regimes"][regime]["exponent"])
    a[1, 2] = cfg["regimes"][regime]["coupling"]
    chol = np.linalg.cholesky(np.eye(4) - a @ a.T)
    for t in range(1, cfg["length"]):
        x[:, t] = x[:, t-1] @ a.T + noise[:, t] @ chol.T
    return np.mean([np.mean((x[:, h:] - x[:, :-h])**2, axis=1) for h in cfg["lags"]], axis=0)


def description(mask, wrong):
    n, k = int(mask.sum()), int((mask & wrong).sum())
    return {"accepted": n, "errors": k, "coverage": n/mask.size, "risk": k/n if n else None}


def stratified(values, wrong, edges):
    return [description((values >= a) & (values < b), wrong) for a, b in zip(edges[:-1], edges[1:])]


cfg = read(SOURCE / "config.json")
old = read(SOURCE / "SUMMARY.json")
receipt = read(SOURCE / "FORECAST.json")
thresholds = read(SOURCE / "THRESHOLDS.json")
cal = dict(np.load(SOURCE / "CAL_GAPS.npz", allow_pickle=False))
assert digest(SOURCE / "FORECAST.json") == old["forecast_sha256"]
assert digest(SOURCE / "CAL_GAPS.npz") == thresholds["cal_sha256"]
assert digest(SOURCE / "THRESHOLDS.json") == receipt["threshold_sha256"]
assert digest(SOURCE / "DESIGN_LOCK.json") == receipt["design_sha256"]

new_thresholds = {
    r: {str(c): float(np.quantile(cal[r], 1-c, method="higher")) for c in COVERAGES}
    for r in REGIMES
}
for r in REGIMES:
    assert new_thresholds[r]["0.5"] == thresholds["target_median"][r]

arrays, rows, truths = {}, [], {}
source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in
                 [SOURCE / n for n in ("config.json", "run.py", "SUMMARY.json", "FORECAST.json", "CAL_GAPS.npz", "THRESHOLDS.json", "DESIGN_LOCK.json")]}
index = {(c["name"], c["process"], c["regime"]): c for c in old["cells"]}
for cell in receipt["cells"]:
    r = cell["regime"]
    if r not in REGIMES:
        continue
    path = SOURCE / cell["path"]
    assert digest(path) == cell["sha256"]
    source_hashes[str(path.relative_to(ROOT))] = cell["sha256"]
    key = (cell["name"], cell["process"], r)
    tkey = (cell["seed"], cell["process"], r)
    if tkey not in truths:
        truths[tkey] = physical_scores(cfg, *tkey)
    truth = truths[tkey]
    prediction = np.load(path, allow_pickle=False)["predicted_scores"]
    actual_set = np.sort(np.argsort(truth, axis=1, kind="stable")[:, :2], axis=1)
    predicted_set = np.sort(np.argsort(prediction, axis=1, kind="stable")[:, :2], axis=1)
    wrong = np.any(actual_set != predicted_set, axis=1)
    pgap = np.sort(prediction, axis=1)[:, 2] - np.sort(prediction, axis=1)[:, 1]
    tgap = np.sort(truth, axis=1)[:, 2] - np.sort(truth, axis=1)[:, 1]
    rank = np.searchsorted(np.sort(cal[r]), pgap, side="right") / len(cal[r])
    rank = np.minimum(rank, np.nextafter(1.0, 0.0))
    original = index[key]
    assert int(wrong.sum()) == original["all_answers"]["errors"]
    for label, threshold in (("fixed", thresholds["fixed"]), ("recalibrated", thresholds["target_median"][r])):
        stats = description(pgap >= threshold, wrong)
        assert all(stats[k] == original["rules"][label][k] for k in ("accepted", "errors"))
    np.testing.assert_allclose(np.mean((prediction-truth)**2, axis=0), original["score_mse_by_axis"], rtol=1e-10, atol=1e-14)
    arrays[key] = {"wrong": wrong, "pgap": pgap, "tgap": tgap, "rank": rank,
                   "seed": cell["seed"], "training_case": cell["training_case"]}
    rows.append({"name": cell["name"], "seed": cell["seed"], "process": cell["process"], "regime": r,
                 "all": description(np.ones_like(wrong), wrong),
                 "fixed": description(pgap >= thresholds["fixed"], wrong),
                 "coverages": {str(c): description(pgap >= new_thresholds[r][str(c)], wrong) for c in COVERAGES},
                 "by_predicted_gap": stratified(pgap, wrong, cfg["gap_bins"]),
                 "by_true_gap": stratified(tgap, wrong, cfg["gap_bins"]),
                 "by_cal_percentile": stratified(rank, wrong, np.linspace(0, 1, 6))})

summaries = {}
for r in REGIMES:
    selected = [v for (name, process, regime), v in arrays.items() if regime == r]
    wrong = np.stack([v["wrong"] for v in selected])
    gap = np.stack([v["pgap"] for v in selected])
    summaries[r] = {"all": description(np.ones_like(wrong), wrong),
                    "fixed": description(gap >= thresholds["fixed"], wrong),
                    "coverages": {str(c): description(gap >= new_thresholds[r][str(c)], wrong) for c in COVERAGES},
                    "by_seed": {}}
    for seed in cfg["seeds"]:
        cells = [v for v in selected if v["seed"] == seed]
        y = np.stack([v["wrong"] for v in cells])
        g = np.stack([v["pgap"] for v in cells])
        summaries[r]["by_seed"][str(seed)] = {str(c): description(g >= new_thresholds[r][str(c)], y) for c in COVERAGES}

# Within each fixed model/process, standardize both regimes to the same gap-bin mass.
matched = {}
for kind, edges in (("pgap", cfg["gap_bins"]), ("tgap", cfg["gap_bins"]), ("rank", np.linspace(0, 1, 6))):
    matched[kind] = {}
    for r in REGIMES[1:]:
        details = []
        for name, process, regime in arrays:
            if regime != "baseline":
                continue
            b, other = arrays[name, process, "baseline"], arrays[name, process, r]
            totals = np.zeros(3)
            bins = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m0 = (b[kind] >= lo) & (b[kind] < hi)
                m1 = (other[kind] >= lo) & (other[kind] < hi)
                n0, n1 = int(m0.sum()), int(m1.sum())
                weight = min(n0, n1)
                if weight:
                    risk0, risk1 = float(b["wrong"][m0].mean()), float(other["wrong"][m1].mean())
                    totals += (weight, weight*risk0, weight*risk1)
                    bins.append({"left": float(lo), "right": float(hi), "base_n": n0, "other_n": n1,
                                 "base_errors": int(b["wrong"][m0].sum()), "other_errors": int(other["wrong"][m1].sum()),
                                 "weight": weight, "base_risk": risk0, "other_risk": risk1})
            details.append({"name": name, "process": process, "seed": b["seed"], "support": int(totals[0]),
                            "base_standardized_risk": totals[1]/totals[0], "other_standardized_risk": totals[2]/totals[0],
                            "difference": (totals[2]-totals[1])/totals[0], "bins": bins})
        denom = sum(d["support"] for d in details)
        base_risk = sum(d["support"]*d["base_standardized_risk"] for d in details)/denom
        other_risk = sum(d["support"]*d["other_standardized_risk"] for d in details)/denom
        matched[kind][r] = {"common_support_fraction": denom/(16*cfg["n_test"]), "baseline": base_risk,
                             "other": other_risk, "difference": other_risk-base_risk,
                             "cells_higher": int(sum(d["difference"] > 0 for d in details)), "cells": details}

# Resample independent innovation rows within each seed/process, sharing draws over models/regimes.
replications = 1000
rng = np.random.default_rng(2026111603)
bootstrap_k = np.zeros((replications, len(REGIMES)))
bootstrap_n = np.zeros_like(bootstrap_k)
for seed in cfg["seeds"]:
    for process in cfg["processes"]:
        vectors_k, vectors_n = [], []
        for r in REGIMES:
            group = [arrays[f"{case}_{seed}", process, r] for case in cfg["training_cases"]]
            masks = [v["pgap"] >= new_thresholds[r]["0.5"] for v in group]
            vectors_k.append(np.sum([m & v["wrong"] for m, v in zip(masks, group)], axis=0))
            vectors_n.append(np.sum(masks, axis=0))
        vk, vn = np.stack(vectors_k, axis=1), np.stack(vectors_n, axis=1)
        for i in range(replications):
            indices = rng.integers(0, cfg["n_test"], size=cfg["n_test"])
            bootstrap_k[i] += vk[indices].sum(axis=0)
            bootstrap_n[i] += vn[indices].sum(axis=0)
risks = bootstrap_k / bootstrap_n
intervals = {r: {"difference": summaries[r]["coverages"]["0.5"]["risk"]-summaries["baseline"]["coverages"]["0.5"]["risk"],
                 "exploratory_paired_percentile_95": np.quantile(risks[:, i]-risks[:, 0], [.025,.975]).tolist()}
             for i, r in enumerate(REGIMES) if i}

result = {"status": "exploratory_reanalysis_of_seen_test", "no_new_model_inference": True,
          "truth_reconstruction": "Exact original seeds, roles, generator and scores; all original cell counts and score MSE reproduced",
          "numpy_version": np.__version__, "thresholds": new_thresholds, "summary": summaries, "cells": rows,
          "matched_gap_comparisons": matched, "paired_bootstrap_at_50_percent": {"replications": replications, "seed": 2026111603, "results": intervals},
          "source_hashes": source_hashes, "analysis_plan_sha256": digest(HERE / "ANALYSIS_PLAN.md"), "script_sha256": digest(Path(__file__))}
(HERE / "SUMMARY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
print(json.dumps({"summary": summaries, "matched": {k:{r:{a:v for a,v in x.items() if a!='cells'}for r,x in y.items()}for k,y in matched.items()}, "intervals": intervals}, indent=2))
