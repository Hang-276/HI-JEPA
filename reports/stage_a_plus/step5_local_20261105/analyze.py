"""Deterministic evaluation of the frozen step-5 episode summaries."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from shared import candidates, scores, tcrit

RUN = Path(__file__).resolve().parent
TARGET = [1, 3]
# Physical target axes (zero based), then a nontarget. Gap = score_b - score_a.
PAIRS = [(0, 1), (0, 3), (2, 1), (2, 3)]
PAIR_NAMES = [f"S{b + 1}-S{a + 1}" for a, b in PAIRS]
PHIS = [0, 60, 80]
PREFIXES = [32, 128, 256]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(split, phi, size):
    path = RUN / f"{split}_{phi}.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in (
            "episode_index", "seed_keys", "M", "M_by_h", "mean", "second",
            "counts", "innov")}
    required_shapes = {
        "M": (size, 4, 4), "M_by_h": (size, 3, 4, 4), "mean": (size, 4),
        "second": (size, 4, 4), "counts": (size, 9),
        "innov": (size, 9, 4, 4), "episode_index": (size,),
        "seed_keys": (size, 5)}
    for key, shape in required_shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {path.name} {key}: expected {shape}")
    ids = np.arange(size)
    seeds = data["seed_keys"]
    if not (np.array_equal(data["episode_index"], ids)
            and np.array_equal(seeds[:, 4], ids)
            and len(np.unique(seeds[:, :4], axis=0)) == 1):
        raise ValueError(f"Episode identity mismatch in {path.name}")
    if not (seeds[0, 0] == 20261105 and (data["counts"] >= 0).all()
            and (data["counts"].sum(axis=1) == 2048).all()):
        raise ValueError(f"Seed date or regime counts mismatch in {path.name}")
    if not np.allclose(data["M"], data["M_by_h"].mean(axis=1), rtol=0, atol=1e-12):
        raise ValueError(f"Unequal horizon aggregation in {path.name}")
    return data


@lru_cache(maxsize=None)
def critical(level, degrees_of_freedom):
    return tcrit((1 + level) / 2, degrees_of_freedom)


def interval(episodes, level):
    observations = np.asarray(episodes, dtype=float)
    n = len(observations)
    mean = float(observations.mean())
    se = float(observations.std(ddof=1) / np.sqrt(n))
    radius = critical(level, n - 1) * se
    return {"mean": mean, "se": se, "level": level,
            "lower": float(mean - radius), "upper": float(mean + radius)}


def covariance(data, n):
    mean = data["mean"][:n].mean(axis=0)
    return data["second"][:n].mean(axis=0) - np.outer(mean, mean)


def selection(axis_scores):
    axis_scores = np.asarray(axis_scores)
    indices = np.argsort(axis_scores, kind="stable")
    separation = axis_scores[indices[2]] - axis_scores[indices[1]]
    tie = abs(separation) <= 64 * np.finfo(float).eps * max(1., max(abs(axis_scores)))
    return None if tie else sorted((indices[:2] + 1).tolist())


def gaps(axis_scores):
    return {name: axis_scores[..., b] - axis_scores[..., a]
            for name, (a, b) in zip(PAIR_NAMES, PAIRS)}


def candidate_risk(Q, variances, validation):
    # Per-transition difference of oracle Gaussian log losses. Account for the
    # Jacobian of the normalized shear as well as its diagonal projected head.
    projected_var = np.einsum("ij,uj,ij->ui", Q, variances, Q)
    log_det = np.linalg.slogdet(Q)[1]
    regime_kl = .5 * np.log(projected_var / variances).sum(axis=1) - log_det
    precision_difference = np.stack([
        Q.T @ np.diag(1 / vq) @ Q - np.diag(1 / vs)
        for vq, vs in zip(projected_var, variances)])
    quadratic = np.einsum("nuij,uij->nu", validation["innov"], precision_difference)
    per_episode = (validation["counts"] @ regime_kl + .5 * quadratic.sum(axis=1)) / 2048
    return float(regime_kl.mean()), per_episode


def axis_fidelity(Q, state_cov):
    y_var = scores(Q, state_cov)
    matched_cov = np.diag(Q @ state_cov)
    r2 = matched_cov ** 2 / (y_var * np.diag(state_cov))
    if np.min(r2) < -1e-12 or np.max(r2) > 1 + 1e-12:
        raise ValueError(f"Invalid matched R2 {r2}")
    return np.clip(r2, 0, 1)


def prediction_error(pred, observed):
    error = np.asarray(pred, dtype=float) - np.asarray(observed, dtype=float)
    return {"pred_minus_observed": error.tolist(), "absolute": np.abs(error).tolist()}


def summarize_candidate(phi, name, Q, model, calibration_cov, validation, prediction):
    matrix = np.asarray(model["M"], dtype=float)
    theoretical = scores(Q, matrix)
    actual = scores(Q, validation["M"])
    population_gap = float(theoretical[1] - theoretical[0])
    risk, realized_risk = candidate_risk(Q, np.asarray(model["D"]), validation)
    eta = np.linalg.norm(Q - np.eye(4), axis=1)
    beta = 4 * np.sqrt(np.diag(matrix)) * eta + 4 * eta ** 2
    norm_lowers = {
        label: float(matrix[b, b] - matrix[a, a] - beta[a] - beta[b])
        for label, (a, b) in zip(PAIR_NAMES, PAIRS)}
    pop_r2 = axis_fidelity(Q, np.eye(4))
    pop_set = selection(theoretical)
    population = {
        "risk_F": risk, "axis_L2_error": eta.tolist(), "axis_R2": pop_r2.tolist(),
        "scores": theoretical.tolist(), "gap": population_gap,
        "top2": pop_set, "target_cross_gaps": {k: float(v) for k, v in gaps(theoretical).items()},
        "A": bool(np.min(pop_r2) >= .95), "S": pop_set == TARGET,
        "J": bool(np.min(pop_r2) >= .95 and pop_set == TARGET),
        "matching_at_endpoint_ambiguous": name in ("r+45", "r-45"),
        "norm_lower_gaps": norm_lowers,
        "norm_decision": "certify" if min(norm_lowers.values()) > 0 else "abstain"}
    cal_variance = scores(Q, calibration_cov)
    if np.min(cal_variance) <= 0:
        raise ValueError(f"Nonpositive calibration variance for {phi}/{name}")
    heldout = {}
    for n in PREFIXES:
        episode_scores = actual[:n]
        means = episode_scores.mean(axis=0)
        empirical_set = selection(means)
        state_cov = covariance(validation, n)
        r2 = axis_fidelity(Q, state_cov)
        error_matrix = Q - np.eye(4)
        empirical_eta = np.sqrt(np.maximum(0, scores(error_matrix, validation["second"][:n].mean(axis=0))))
        gap_intervals = {}
        for label, observations in gaps(episode_scores).items():
            level = .95
            if phi == 80 and name == "block+9":
                level = .9875
            elif phi == 80 and name in ("r+9", "r-9") and label == "S2-S1":
                level = .975
            gap_intervals[label] = interval(observations, level)
        empirical_scaled = means / cal_variance
        gap = gap_intervals["S2-S1"]["mean"]
        ablated = prediction.get("ablated_gap")
        heldout[str(n)] = {
            "scores": means.tolist(), "top2": empirical_set,
            "axis_R2": r2.tolist(), "axis_L2_error_population_gauge": empirical_eta.tolist(),
            "A": bool(np.min(r2) >= .95), "S": empirical_set == TARGET,
            "J": bool(np.min(r2) >= .95 and empirical_set == TARGET),
            "gap": gap, "target_cross_gaps": gap_intervals,
            "risk_estimate_descriptive": interval(realized_risk[:n], .95),
            "calibration_scaled_scores": empirical_scaled.tolist(),
            "calibration_scaled_gap": float(empirical_scaled[1] - empirical_scaled[0]),
            "calibration_scaled_top2": selection(empirical_scaled),
            "direction_error": prediction_error(prediction["gap"], gap),
            "direction_axis_score_error": prediction_error(prediction["scores"], means),
            "ablation_gap_error": None if ablated is None else prediction_error(ablated, gap),
            "frozen_scaled_score_error": prediction_error(prediction["scaled_scores"], empirical_scaled),
            "oracle_gap_error": prediction_error(population_gap, gap)}
    return {"population": population, "frozen_calibration_prediction": prediction,
            "validation_prefixes": heldout}, actual[:, 1] - actual[:, 0]


def readout(row, n):
    values = row["validation_prefixes"][str(n)]
    return {key: values[key] for key in
            ("gap", "scores", "top2", "A", "S", "J", "axis_R2", "target_cross_gaps",
             "calibration_scaled_top2", "calibration_scaled_gap")}


def transitions(points):
    # Preserve all recrossings and treat zero separately. For a noisy grid
    # there need not be one flip or an exact value for its location.
    changes = []
    for (a, x), (b, y) in zip(points, points[1:]):
        if np.sign(x) != np.sign(y):
            changes.append([a, b])
    return {"grid_magnitude_and_gap": points, "sign_change_brackets_deg": changes,
            "already_nonpositive_at_zero": points[0][1] <= 0,
            "first_nonpositive_magnitude": next((a for a, x in points if x <= 0), None)}


def angle_boundary(rows, phi):
    magnitudes = sorted({abs(int(name[1:])) for name in rows if name.startswith("r")})
    curves = {}
    for n in PREFIXES:
        negative, both = [], []
        for angle in magnitudes:
            neg = rows[f"r{-angle:+d}"]["validation_prefixes"][str(n)]["gap"]
            pos = rows[f"r{angle:+d}"]["validation_prefixes"][str(n)]["gap"]
            negative.append([angle, neg])
            both.append([angle, min(neg, pos)])
        curves[str(n)] = {"negative_branch": transitions(negative),
                          "worst_of_two_observed_signs": transitions(both)}
    return {"population_boundary_deg": 45 - phi / 2,
            "prefixes": curves, "phi0_population_endpoint_is_tie": phi == 0}


def direction_outcome(gap_interval, positive):
    if (gap_interval["lower"] > 0 if positive else gap_interval["upper"] < 0):
        return "expected_direction_resolved"
    if (gap_interval["upper"] < 0 if positive else gap_interval["lower"] > 0):
        return "opposite_direction_resolved"
    return "unresolved"


def main():
    lock_path = RUN / "calibration_lock.json"
    model_path = RUN / "generator.json"
    lock = json.loads(lock_path.read_text())
    if lock["gate_passed"] is not True:
        raise RuntimeError("Calibration gate stopped this run")
    models = json.loads(model_path.read_text())["models"]
    report = {
        "calibration_lock_sha256": digest(lock_path), "generator_sha256": digest(model_path),
        "target_one_based": TARGET, "prefixes": PREFIXES, "independent_unit": "episode",
        "method": "Pooled state covariance for affine R². Signed gap equals S2-S1. J is the joint empirical point readout, with interval criteria reported separately. Each grid prefix reuses its first episodes.",
        "models": {}, "E3": {}}
    seed_set = set()
    for phi in PHIS:
        model = models[str(phi)]
        cal = load_split("cal", phi, 64)
        val = load_split("val", phi, 256)
        for data in (cal, val):
            keys = set(map(tuple, data["seed_keys"].tolist()))
            if seed_set.intersection(keys):
                raise ValueError("Duplicate seed key across configurations/splits")
            seed_set.update(keys)
        cal_cov = covariance(cal, 64)
        rows, all_series = {}, {}
        for name, Q in candidates(phi):
            pred = lock["predictions"][str(phi)][name]
            rows[name], all_series[name] = summarize_candidate(phi, name, Q, model, cal_cov, val, pred)
        report["models"][str(phi)] = {
            "cal_sha256": digest(RUN / f"cal_{phi}.npz"),
            "val_sha256": digest(RUN / f"val_{phi}.npz"),
            "cal_state_covariance": cal_cov.tolist(), "validation_state_covariance": covariance(val, 256).tolist(),
            "validation_regime_counts": val["counts"].sum(axis=0).tolist(),
            "candidates": rows}
        report["E3"][str(phi)] = angle_boundary(rows, phi)
        if phi == 80:
            minus = readout(rows["r-9"], 256)
            plus = readout(rows["r+9"], 256)
            negative_check = direction_outcome(minus["target_cross_gaps"]["S2-S1"], False)
            positive_check = direction_outcome(plus["target_cross_gaps"]["S2-S1"], True)
            paired = all_series["r+9"] - all_series["r-9"]
            even = (all_series["r+9"] + all_series["r-9"]) / 2
            resolved = negative_check == positive_check == "expected_direction_resolved"
            report["E1"] = {
                "minus": minus, "plus": plus, "negative_check": negative_check,
                "positive_check": positive_check, "primary_reversal_resolved": resolved,
                "full_sets_match_predictions": minus["top2"] == [2, 3] and plus["top2"] == TARGET,
                "paired_difference_plus_minus": interval(paired, .95),
                "odd_part": interval(paired / 2, .95), "even_part": interval(even, .95)}
            block = readout(rows["block+9"], 256)
            report["E2"] = {
                "block": block,
                "block_preservation_resolved": all(v["lower"] > 0 for v in block["target_cross_gaps"].values()),
                "weakest_lower": min(v["lower"] for v in block["target_cross_gaps"].values())}
            report["shear"] = {name: readout(rows[name], 256)
                               for name in ("shear-0.22", "shear+0.22")}
    report["distinct_seed_keys"] = len(seed_set)
    output = RUN / "analysis.json"
    stage = output.with_suffix(".tmp")
    stage.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    stage.replace(output)
    print(json.dumps({
        "E1_resolved": report["E1"]["primary_reversal_resolved"],
        "minus": report["E1"]["minus"]["target_cross_gaps"]["S2-S1"],
        "plus": report["E1"]["plus"]["target_cross_gaps"]["S2-S1"],
        "sets": [report["E1"][s]["top2"] for s in ("minus", "plus")],
        "E2_resolved": report["E2"]["block_preservation_resolved"],
        "E2_weakest_lower": report["E2"]["weakest_lower"]}, indent=2))


if __name__ == "__main__":
    main()
