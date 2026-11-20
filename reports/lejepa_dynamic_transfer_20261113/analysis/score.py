"""Freeze validation matching and score frozen LeJEPA readout predictions."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXP = HERE.parent / "experiment"
TRAIN = REPO / "reports/lejepa_slow_rank_20261112/training"
CONTROLS = ("state", "shuffled_1")
BASE = "temporal_1"
BOOT_SEED = 2026111305
BOOT_COUNT = 4096
BINS = ("up_to_0.01", "0.01_to_0.03", "over_0.03")
MODES = ("eight_ref_mean", "pooled_ref", "true_n_empirical_v", "estimated_n_true_v", "true_both")
ROW_METRICS = ("accuracy", "negative_margin_burden", "signed_margin", "fastest_false_inclusion", "fastest_false_exclusion")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1 << 20), b""):
            checksum.update(data)
    return checksum.hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")


def inputs():
    historical = read(HERE / "HISTORICAL_GROUPS.json")
    if digest(REPO / historical["source_relative_to_repository"]) != historical["source_sha256"]:
        raise ValueError("Historical source hash mismatch.")
    return read(EXP / "protocol.json"), read(TRAIN / "protocol.json"), historical


def dependencies():
    paths = {
        "fits_receipt_sha256": EXP / "FITS_RECEIPT.json",
        "experiment_protocol_sha256": EXP / "protocol.json",
        "analysis_protocol_sha256": HERE / "PROTOCOL.md",
        "scorer_sha256": Path(__file__),
        "historical_groups_sha256": HERE / "HISTORICAL_GROUPS.json",
    }
    return {key: digest(path) for key, path in paths.items()}


def freeze():
    config, training, history = inputs()
    receipt = read(EXP / "FITS_RECEIPT.json")
    rows = {}
    matching = []
    for case in training["cases"]:
        for seed in training["seeds"]:
            name = f"{case}_{seed}"
            probes = []
            for probe in config["probe_seeds"]:
                values = {}
                for arm in (BASE, *CONTROLS):
                    filename = f"{name}_{probe}_{arm}.json"
                    path = EXP / "fits" / filename
                    if digest(path) != receipt["fit_records_sha256"][filename]:
                        raise ValueError(f"Unsealed VAL record: {filename}")
                    values[arm] = np.asarray(read(path)["val_final_r2"])
                    if values[arm].shape != (4,) or not np.isfinite(values[arm]).all():
                        raise ValueError(f"Invalid VAL R²: {filename}")
                deltas = {arm: values[BASE] - values[arm] for arm in CONTROLS}
                maximum = {arm: float(np.max(np.abs(delta))) for arm, delta in deltas.items()}
                probes.append({
                    "probe": probe, "val_r2": {key: val.tolist() for key, val in values.items()},
                    "t1_minus_comparator": {key: val.tolist() for key, val in deltas.items()},
                    "maximum_axis_difference": maximum,
                    "matches": maximum["state"] <= config["validation_match_max_abs_r2_gap"],
                })
            selected = all(row["matches"] for row in probes)
            rows[name] = {"case": case, "matched_both_seeds": selected, "probes": probes}
            if selected:
                matching.append(name)
    by_case = {case: sum(rows[name]["case"] == case for name in matching) for case in training["cases"]}
    write(HERE / "VAL_MATCH_LOCK.json", {
        **dependencies(), "threshold": config["validation_match_max_abs_r2_gap"],
        "groups": {"all": list(rows), "strong": history["strong"], "weak": history["weak"], "matched": matching},
        "matching_gate_passes": len(matching) >= 8 and all(by_case.values()),
        "matched_counts_by_case": by_case, "rows": rows,
    })


def membership(scores):
    order = np.argsort(scores, axis=-1, kind="stable")[..., :2]
    inside = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(inside, order, True, axis=-1)
    return inside


def dynamic(predicted, target, lags):
    numerator = np.zeros((len(target), 4))
    energy = np.zeros_like(numerator)
    cross = np.zeros_like(numerator)
    for lag in lags:
        truth = target[:, lag:] - target[:, :-lag]
        estimate = predicted[:, lag:] - predicted[:, :-lag]
        residual = estimate - truth
        numerator += np.mean(estimate**2, axis=1)
        energy += np.mean(residual**2, axis=1)
        cross += np.mean(2 * truth * residual, axis=1)
    return numerator / len(lags), energy / len(lags), cross / len(lags)


def rank(estimated, physical, inside):
    chosen = membership(estimated)
    signed_margin = np.where(~inside, estimated, np.inf).min(-1) - np.where(inside, estimated, -np.inf).max(-1)
    return {
        "accuracy": np.all(chosen == inside, axis=-1),
        "negative_margin_burden": np.maximum(-signed_margin, 0),
        "signed_margin": signed_margin,
        "fastest_false_inclusion": chosen[..., 3] & ~inside[..., 3],
        "fastest_false_exclusion": ~chosen[..., 3] & inside[..., 3],
        "score_mae_by_axis": np.abs(estimated - physical).mean(axis=(0, 1)),
        "score_bias_by_axis": (estimated - physical).mean(axis=(0, 1)),
    }


def targets(paths, config):
    result = {}
    for length in config["test_lengths"]:
        prefix = paths[:, :length]
        physical = dynamic(prefix, prefix, config["lags"])[0]
        ordered = np.sort(physical, axis=-1)
        gap_bins = np.digitize(ordered[:, 2] - ordered[:, 1], [0.01, 0.03], right=True)
        result[length] = (prefix, physical, membership(physical), gap_bins)
    return result


def read_forecast(path, config, goals):
    with np.load(path, allow_pickle=False) as source:
        test = source["test_predictions"].astype(np.float64)
        v, mean, second = (source[key] for key in ("ref_variance", "ref_mean", "ref_second_moment"))
    if test.shape != (config["n_test_trajectories"], max(config["test_lengths"]), 4):
        raise ValueError(f"Invalid forecast size: {path.name}")
    expected = (config["n_reference_banks"], 4)
    if any(value.shape != expected for value in (v, mean, second)):
        raise ValueError(f"Invalid reference size: {path.name}")
    np.testing.assert_allclose(v, second - mean**2, rtol=1e-10, atol=2e-12)
    pooled = second.mean(0) - mean.mean(0)**2
    if np.any(v <= 0) or np.any(pooled <= 0):
        raise ValueError(f"Undefined reference variance: {path.name}")
    result = {}
    for length, (truth, n_true, inside, _) in goals.items():
        n, energy, cross = dynamic(test[:, :length], truth, config["lags"])
        np.testing.assert_allclose(n - n_true, energy + cross, rtol=1e-10, atol=3e-14)
        all_scores = (n[None] / v[:, None], (n / pooled)[None], n_true[None] / v[:, None], n[None], n_true[None])
        result[length] = {
            "dynamic": {
                "increment_mse": energy.mean(0).tolist(),
                "signed_cross": cross.mean(0).tolist(),
                "numerator_mae": np.abs(n - n_true).mean(0).tolist(),
                "numerator_bias": (n - n_true).mean(0).tolist(),
                "reference_variance": v.tolist(),
                "pooled_reference_variance": pooled.tolist(),
            },
            **{mode: rank(score, n_true, inside) for mode, score in zip(MODES, all_scores)},
        }
    return result


def by_gap(data, bins):
    return {
        label: {"count": int(mask.sum()), "average": float(data[:, mask].mean()) if mask.any() else None}
        for idx, label in enumerate(BINS) for mask in [bins == idx]
    }


def summarize(arms, goals):
    reports = {}
    for length, (_, _, _, bins) in goals.items():
        reports[length] = {"dynamic": {arm: outcomes[length]["dynamic"] for arm, outcomes in arms.items()}}
        for mode in MODES:
            row = {arm: outcomes[length][mode] for arm, outcomes in arms.items()}
            per_arm = {}
            for arm, metrics in row.items():
                per_ref = metrics["accuracy"].mean(1)
                per_arm[arm] = {
                    **{metric: float(metrics[metric].mean()) for metric in ROW_METRICS},
                    "accuracy_by_ref": per_ref.tolist(),
                    "accuracy_ref_sd": float(per_ref.std(ddof=1)) if len(per_ref) > 1 else None,
                    "accuracy_by_gap": by_gap(metrics["accuracy"], bins),
                    "negative_margin_burden_by_gap": by_gap(metrics["negative_margin_burden"], bins),
                    "score_mae_by_axis": metrics["score_mae_by_axis"].tolist(),
                    "score_bias_by_axis": metrics["score_bias_by_axis"].tolist(),
                }
            pairs = {}
            for control in CONTROLS:
                delta = row[BASE]["accuracy"].astype(np.int8) - row[control]["accuracy"].astype(np.int8)
                pairs[control] = {
                    "accuracy_gain": float(delta.mean()),
                    "mean_repairs": float((delta == 1).sum(axis=1).mean()),
                    "mean_new_harms": float((delta == -1).sum(axis=1).mean()),
                    "accuracy_gain_by_ref": delta.mean(axis=1).tolist(),
                    "accuracy_gain_by_gap": by_gap(delta, bins),
                    "negative_margin_burden_delta": per_arm[BASE]["negative_margin_burden"] - per_arm[control]["negative_margin_burden"],
                }
            reports[length][mode] = {"by_arm": per_arm, "t1_minus": pairs}
    return reports


def accumulate(total, arms, weight):
    for length in arms[BASE]:
        for mode in MODES:
            for control in CONTROLS:
                for metric in ("accuracy", "negative_margin_burden"):
                    left = arms[BASE][length][mode][metric]
                    right = arms[control][length][mode][metric]
                    key = (length, mode, control, metric)
                    if key not in total:
                        total[key] = np.zeros_like(left, dtype=float)
                    total[key] += weight * (left.astype(float) - right)


def intervals(matrix, config):
    rng = np.random.default_rng(BOOT_SEED)
    n_rows = config["n_test_trajectories"]
    n_ref = config["n_reference_banks"]
    weights_test = rng.multinomial(n_rows, np.full(n_rows, 1 / n_rows), size=BOOT_COUNT) / n_rows
    weights_ref = rng.multinomial(n_ref, np.full(n_ref, 1 / n_ref), size=BOOT_COUNT) / n_ref
    output = {}
    for (length, mode, control, metric), values in matrix.items():
        rows = weights_test @ values.T
        draws = np.einsum("br,br->b", rows, weights_ref) if values.shape[0] == n_ref else rows[:, 0]
        target = output.setdefault(length, {}).setdefault(mode, {}).setdefault(control, {})
        target[metric] = {
            "estimate": float(values.mean()),
            "interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "effect_by_ref": values.mean(axis=1).tolist(),
        }
    return output


def score():
    config, training, _ = inputs()
    lock = read(HERE / "VAL_MATCH_LOCK.json")
    if any(lock[key] != value for key, value in dependencies().items()):
        raise ValueError("Analysis dependencies changed after matching.")
    forecast = read(EXP / "FORECAST_RECEIPT.json")
    reveal = read(EXP / "REVEAL_RECEIPT.json")
    forecast_receipt_digest = digest(EXP / "FORECAST_RECEIPT.json")
    if digest(EXP / "labels.npz") != reveal["labels_sha256"] or reveal["forecast_receipt_sha256"] != forecast_receipt_digest:
        raise ValueError("Unrelated TEST truth reveal.")
    groups = lock["groups"]
    groups = {**groups, **{case: [name for name in groups["all"] if name.startswith(case + "_")] for case in training["cases"]}}
    matrices = defaultdict(dict)
    panels = {}
    with np.load(EXP / "labels.npz", allow_pickle=False) as labels:
        for case in training["cases"]:
            goal = targets(labels[case], config)
            for seed in training["seeds"]:
                name = f"{case}_{seed}"
                for probe in config["probe_seeds"]:
                    prefix = f"{name}_{probe}"
                    arms = {}
                    for arm in config["arms"]:
                        file = f"{prefix}_{arm}.npz"
                        path = EXP / "forecast" / file
                        if digest(path) != forecast["forecast_files_sha256"][file]:
                            raise ValueError(f"Forecast not locked: {file}")
                        arms[arm] = read_forecast(path, config, goal)
                    panels[prefix] = summarize(arms, goal)
                    for group, names in groups.items():
                        if name in names:
                            accumulate(matrices[group], arms, 1 / (len(names) * len(config["probe_seeds"])))
                    print(prefix, flush=True)
    aggregated = {group: intervals(value, config) for group, value in matrices.items()}
    primary = aggregated["strong"][193]["eight_ref_mean"]
    passed = all(row["accuracy"]["estimate"] >= 0.01 and row["accuracy"]["interval_95"][0] > 0 for row in primary.values())
    write(HERE / "SUMMARY.json", {
        "val_lock_sha256": digest(HERE / "VAL_MATCH_LOCK.json"),
        "forecast_receipt_sha256": forecast_receipt_digest,
        "labels_sha256": reveal["labels_sha256"],
        "resampling_seed": BOOT_SEED,
        "resampling_draws": BOOT_COUNT,
        "primary": primary,
        "primary_gate_passes": passed,
        "val_matching_gate_passes": lock["matching_gate_passes"],
        "paired_group_bootstrap": aggregated,
        "panels": panels,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "score"))
    args = parser.parse_args()
    {"freeze": freeze, "score": score}[args.stage]()


if __name__ == "__main__":
    main()
