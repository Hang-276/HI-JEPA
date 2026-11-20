"""Post-reveal diagnostics for frozen point predictions in the AR simulator."""

import json
import time

import numpy as np
import torch

from pipeline import (
    HERE,
    MASTER,
    TRAINING,
    anchor_response_samples,
    assign_names,
    bank_definitions,
    digest,
    dynamic_moments,
    embed,
    fit_lag_axes,
    make_mlp_encoder,
    panel_paths,
    scores,
    test_paths,
)


NEW_STATIONARY_REFERENCES = (2048, 32768)
LARGE_CALIBRATION_TRAJECTORIES = 1024


def random_stream(role, bank):
    return np.random.default_rng(np.random.SeedSequence([MASTER, role, bank]))


def top_two(values):
    return np.sort(np.argsort(values, axis=1, kind="stable")[:, :2], axis=1)


def metrics(target, baseline, candidate):
    expected = top_two(target)
    baseline_top = top_two(baseline)
    candidate_top = top_two(candidate)
    baseline_ok = (expected == baseline_top).all(axis=1)
    candidate_ok = (expected == candidate_top).all(axis=1)
    return {
        "n": len(expected),
        "correct": int(candidate_ok.sum()),
        "changed_from_nominal": int(np.any(baseline_top != candidate_top, axis=1).sum()),
        "fixed_nominal_error": int((~baseline_ok & candidate_ok).sum()),
        "new_error": int((baseline_ok & ~candidate_ok).sum()),
    }


def evaluate_bank(model, rates, bank, test_ids, config):
    length = config["trajectory_length"]
    cal = embed(model, panel_paths(rates, "calibration", bank, config["cal_trajectories"], length))
    ref = embed(model, panel_paths(rates, "reference", bank, config["ref_trajectories"], length))
    anchors = anchor_response_samples(model, bank, config).mean(axis=0)
    test = embed(model, test_paths(rates, test_ids, length))
    innovation = random_stream(201, bank).normal(size=(LARGE_CALIBRATION_TRAJECTORIES, length, 4))
    enlarged_cal = innovation.copy()
    innovation_scale = np.sqrt(1 - rates**2)
    for t in range(1, length):
        enlarged_cal[:, t, :] = rates * enlarged_cal[:, t - 1, :] + innovation_scale * innovation[:, t, :]
    cal_large = embed(model, enlarged_cal)
    iid_reference = random_stream(202, bank).normal(size=(max(NEW_STATIONARY_REFERENCES), 4))
    iid_ref_embedded = embed(model, iid_reference)
    results = {}
    names = {}
    for calibration_key, calibration in (("original_cal", cal), ("cal1024", cal_large)):
        basis, eigenvalues, _ = fit_lag_axes(calibration)
        permutation, _ = assign_names(anchors @ basis)
        names[calibration_key] = {
            "anchor_permutation": permutation.tolist(),
            "anchor_vs_rate_order_agree": bool(np.array_equal(permutation, np.arange(4))),
            "lag_eigenvalues": eigenvalues.tolist(),
        }
        moments = dynamic_moments(test, config["lags"])
        numerator = np.einsum("dj,nde,ej->nj", basis, moments, basis)
        variance = {"original_ref": (ref @ basis).var(axis=(0, 1))}
        variance.update(
            {f"iid_ref{n}": (iid_ref_embedded[:n] @ basis).var(axis=0) for n in NEW_STATIONARY_REFERENCES}
        )
        for reference_key, denominator in variance.items():
            with_rate_order = numerator / denominator
            results[f"{calibration_key}_{reference_key}_rate_order"] = with_rate_order
            results[f"{calibration_key}_{reference_key}_anchor"] = with_rate_order[:, permutation]
    nominal = scores(test @ basis, ref @ basis, config["lags"])
    if not np.allclose(nominal, results["cal1024_original_ref_rate_order"], rtol=0, atol=1e-12):
        raise ValueError("dynamic moments disagree with direct score")
    return results, names


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    config = json.loads((TRAINING / "protocol.json").read_text())
    receipt = json.loads((HERE / "pipeline_forecast_receipt.json").read_text())
    forecast_path = HERE / "forecasts.npz"
    reveal_path = HERE / "reveal.npz"
    assert digest(forecast_path) == receipt["forecasts_sha256"]
    assert digest(HERE / "pipeline.py") == receipt["pipeline_sha256"]
    with np.load(forecast_path, allow_pickle=False) as forecast:
        assigned = forecast["test_bank"]
        nominal = forecast["test_scores"]
        bank_cases = forecast["bank_case"]
        case_names = forecast["case_names"].tolist()
    with np.load(reveal_path, allow_pickle=False) as reveal:
        assert reveal["forecasts_sha256"].item() == receipt["forecasts_sha256"]
        assert np.array_equal(assigned, reveal["test_bank"])
        target = reveal["test_physical_scores"]
    bank_results = []
    test_candidates = {}
    previous = None
    for bank, (_, case, seed, _, rates) in enumerate(bank_definitions(config)):
        checkpoint = TRAINING / "runs" / f"{case}_{seed}.pt"
        if checkpoint != previous:
            assert digest(checkpoint) == receipt["checkpoint_sha256"][checkpoint.name]
            model = make_mlp_encoder(4, config["hidden"], config["layers"], device="cpu").eval()
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            previous = checkpoint
        rows = np.flatnonzero(assigned == bank)
        values, names = evaluate_bank(model, rates, bank, rows, config)
        assert np.allclose(values["original_cal_original_ref_anchor"], nominal[rows], rtol=0, atol=1e-10)
        assert np.array_equal(top_two(values["original_cal_original_ref_anchor"]), top_two(nominal[rows]))
        bank_results.append({
            "bank": bank,
            "names": names,
            "metrics": {key: metrics(target[rows], nominal[rows], estimate) for key, estimate in values.items()},
        })
        for key, estimate in values.items():
            if key not in test_candidates:
                test_candidates[key] = np.empty_like(nominal)
            test_candidates[key][rows] = estimate
    case_results = {}
    for case, name in enumerate(case_names):
        rows = bank_cases[assigned] == case
        case_results[name] = {key: metrics(target[rows], nominal[rows], est[rows]) for key, est in test_candidates.items()}
    results = {
        "status": "post-reveal diagnostic only",
        "forecast_sha256": receipt["forecasts_sha256"],
        "reveal_sha256": digest(reveal_path),
        "n": len(nominal),
        "original_nominal_replay": "max absolute tolerance 1e-10 and exact agreement on every predicted set",
        "stationary_iid_reference_sizes": NEW_STATIONARY_REFERENCES,
        "unlabelled_calibration_trajectories": LARGE_CALIBRATION_TRAJECTORIES,
        "original_anchor_bases_fixed": config["named_anchor_bases"],
        "overall": {key: metrics(target, nominal, estimate) for key, estimate in test_candidates.items()},
        "cases": case_results,
        "banks": bank_results,
        "seconds": time.monotonic() - started,
    }
    (HERE / "point_localization.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
