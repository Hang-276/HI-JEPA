"""Fixed-design CPU verification for exact and perturbed Gaussian AR(1) representations."""

import itertools
import json
from pathlib import Path

import numpy as np


LAGS = (4, 16, 32)
RATES = {
    "wide": (0.9000, 0.8750, 0.8550, 0.8200),
    "crossing": (0.9000, 0.8750, 0.8550, 0.7600),
    "narrow": (0.9000, 0.8655, 0.8645, 0.8200),
}
HADAMARD = np.array(
    [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
) / 2
WINDOW = 192
PANEL_COUNT = 24
EVAL_COUNT = 256
PERMUTATIONS = list(itertools.permutations(range(4)))


def trajectories(rng, count, rho):
    noise = rng.standard_normal((WINDOW, count, 4))
    states = np.empty_like(noise)
    states[0] = noise[0]
    innovation_scale = np.sqrt(1 - rho**2)
    for t in range(1, WINDOW):
        states[t] = rho * states[t - 1] + innovation_scale * noise[t]
    return states


def lag_moments(states):
    moments = []
    for lag in LAGS:
        increments = states[lag:] - states[:-lag]
        moments.append(np.einsum("tbi,tbj->bij", increments, increments) / len(increments))
    return np.mean(moments, axis=0)


def factor_scores(moments, rows):
    return np.einsum("ij,bjk,ik->bi", rows, moments, rows)


def top2(scores):
    return np.sort(np.argsort(scores, axis=1, kind="stable")[:, :2], axis=1)


def mistakes(scores, target):
    return np.any(top2(scores) != target, axis=1)


def rotation(theta):
    cosine, sine = np.cos(theta), np.sin(theta)
    rows = np.eye(4)
    rows[1:3, 1:3] = [[cosine, -sine], [sine, cosine]]
    return rows


def estimated_axes(calibration_states):
    rows = calibration_states @ HADAMARD.T
    previous = rows[:-1].reshape(-1, 4)
    following = rows[1:].reshape(-1, 4)
    cross = previous.T @ following / len(previous)
    _, vectors = np.linalg.eigh((cross + cross.T) / 2)
    alignment = HADAMARD.T @ vectors
    names = max(
        PERMUTATIONS,
        key=lambda permutation: np.abs(alignment[np.arange(4), permutation]).sum(),
    )
    ordered = vectors[:, list(names)]
    signs = np.where(np.diag(HADAMARD.T @ ordered) >= 0, 1, -1)
    return (ordered * signs).T @ HADAMARD


def spectral_summary(rho):
    scores = 2 * (1 - np.mean(rho[None, :] ** np.array(LAGS)[:, None], axis=0))
    rotated = HADAMARD**2 @ scores
    np.testing.assert_allclose(rotated, np.full(4, scores.mean()), atol=1e-15)
    modes = (
        (float(np.prod(rho**powers)), powers)
        for powers in itertools.product(range(4), repeat=4)
        if 1 <= sum(powers) <= 3
    )
    best = sorted(modes, reverse=True)[:4]
    assert best[-1][0] > rho.max()**4
    linear_alignment = 2 * (4 - rho.sum())
    whitened_alignment = 2 * (4 - sum(value for value, _ in best))
    np.testing.assert_allclose(
        2 * (4 - np.trace(HADAMARD @ np.diag(rho) @ HADAMARD.T)),
        linear_alignment,
        atol=1e-15,
    )
    return {
        "rho": rho.tolist(),
        "spectral_gap": float(rho.min() - rho.max() ** 2),
        "physical_population_scores": scores.tolist(),
        "physical_population_gap": float(scores[2] - scores[1]),
        "hadamard_population_scores": rotated.tolist(),
        "linear_alignment": float(linear_alignment),
        "optimal_whitened_alignment": float(whitened_alignment),
        "top_four_whitened_modes": [
            {"eigenvalue": value, "degrees": list(powers)} for value, powers in best
        ],
    }


def panel(rho, seed):
    calibration_rng, reference_rng, eval_rng = map(np.random.default_rng, seed.spawn(3))
    calibration = trajectories(calibration_rng, 256, rho)
    reference = reference_rng.standard_normal((512, 4))
    moment = lag_moments(trajectories(eval_rng, EVAL_COUNT, rho))
    truth = np.diagonal(moment, axis1=1, axis2=2)
    true_gap = np.diff(np.sort(truth, axis=1), axis=1)[:, 1]
    target = top2(truth)
    values = {"window_gap_at_most_0.02": true_gap <= 0.02}

    angle = np.deg2rad(8)
    signed_choices = []
    for name, sign in [("plus_8deg", 1), ("minus_8deg", -1)]:
        scores = factor_scores(moment, rotation(sign * angle))
        predicted_gap = (
            np.cos(2 * angle) * (truth[:, 2] - truth[:, 1])
            + 2 * sign * np.sin(2 * angle) * moment[:, 1, 2]
        )
        np.testing.assert_allclose(scores[:, 2] - scores[:, 1], predicted_gap, atol=2e-15)
        values[name] = mistakes(scores, target)
        signed_choices.append(top2(scores))
    values["plus_minus_disagreement"] = np.any(signed_choices[0] != signed_choices[1], axis=1)
    assert not np.any(mistakes(factor_scores(moment, np.eye(4)), target))

    variants = {
        "exact_axes": np.eye(4),
        "cal_8": estimated_axes(calibration[:, :8]),
        "cal_256": estimated_axes(calibration),
    }
    for label, axes in variants.items():
        numerators = factor_scores(moment, axes)
        values[f"{label}_popref"] = mistakes(numerators, target)
        for count in (32, 512):
            denominators = np.var(reference[:count] @ axes.T, axis=0, ddof=1)
            values[f"{label}_ref_{count}"] = mistakes(numerators / denominators, target)
    return values


def summarize(panels, resample):
    result = {}
    near = np.stack([panel["window_gap_at_most_0.02"] for panel in panels])
    for label in panels[0]:
        values = np.stack([panel[label] for panel in panels])
        counts = values.sum(axis=1)
        fractions = counts / values.shape[1]
        result[label] = {
            "count": int(counts.sum()),
            "total": int(values.size),
            "fraction": float(fractions.mean()),
            "panel_bootstrap_95": np.quantile(fractions[resample].mean(axis=1), [0.025, 0.975]).tolist(),
        }
        if label in {"plus_8deg", "minus_8deg", "plus_minus_disagreement"}:
            near_wrong = (near & values).sum(axis=1)
            near_total = near.sum(axis=1)
            result[label]["near_boundary"] = {
                "count": int(near_wrong.sum()),
                "total": int(near_total.sum()),
                "fraction": float(near_wrong.sum() / near_total.sum()),
                "panel_bootstrap_95": np.quantile(
                    near_wrong[resample].sum(axis=1) / near_total[resample].sum(axis=1),
                    [0.025, 0.975],
                ).tolist(),
            }
    return result


def main():
    seed = np.random.SeedSequence(20261112)
    setting_seeds = seed.spawn(len(RATES))
    result = {
        "seed": seed.entropy,
        "panels_per_setting": PANEL_COUNT,
        "evaluation_trajectories_per_panel": EVAL_COUNT,
        "window_length": WINDOW,
        "lags": list(LAGS),
        "rotation_one_to_one_r2_min": float(np.cos(np.deg2rad(8)) ** 2),
        "settings": {},
    }
    for (label, rates), setting_seed in zip(RATES.items(), setting_seeds):
        rho = np.array(rates)
        streams = setting_seed.spawn(PANEL_COUNT + 1)
        panels = [panel(rho, panel_seed) for panel_seed in streams[:-1]]
        resample = np.random.default_rng(streams[-1]).integers(0, PANEL_COUNT, size=(4096, PANEL_COUNT))
        result["settings"][label] = {
            **spectral_summary(rho),
            "window_results": summarize(panels, resample),
        }

    wide, crossing, narrow = (result["settings"][key] for key in RATES)
    assert wide["physical_population_gap"] == crossing["physical_population_gap"]
    assert wide["spectral_gap"] == narrow["spectral_gap"]
    assert crossing["optimal_whitened_alignment"] < crossing["linear_alignment"]
    np.testing.assert_allclose(wide["optimal_whitened_alignment"], wide["linear_alignment"])
    np.testing.assert_allclose(narrow["optimal_whitened_alignment"], narrow["linear_alignment"])
    Path(__file__).with_name("SUMMARY.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
