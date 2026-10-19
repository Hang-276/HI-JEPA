"""Leakage-resistant evaluation for low-dimensional HI-JEPA mechanisms.

All decoders/probes fit train only; signed/permutation matching fits calibration
only. Test-time subspace angles are invariant error metrics, not axis matching.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .diagnostics import state_jacobian, theory_diagnostics, trajectory_diagnostics, target_spectrum_diagnostics


def _array(value):
    return np.asarray(value, dtype=np.float64)


def _mse(prediction, target):
    prediction, target = _array(prediction), _array(target)
    mse = float(np.mean((prediction-target)**2))
    denominator = float(np.mean((target-target.mean(axis=0, keepdims=True))**2))
    return {"mse": mse, "normalized_mse": mse/denominator if denominator>1e-15 else None}


def fit_ridge_probe(x, y, ridge=1e-5, max_samples=10000):
    """Deterministic shared sample budget; intercept is not regularized."""
    x, y = _array(x), _array(y)
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    if len(x) != len(y) or not len(x):
        raise ValueError("probe needs nonempty paired train samples")
    if max_samples is not None and max_samples > 0 and len(x)>max_samples:
        indices = np.linspace(0, len(x)-1, int(max_samples), dtype=np.int64)
        x, y = x[indices], y[indices]
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    centered_x, centered_y = x-mean_x, y-mean_y
    gram = centered_x.T @ centered_x / len(x)
    cross = centered_x.T @ centered_y / len(x)
    weights = np.linalg.lstsq(gram+float(ridge)*np.eye(x.shape[-1]), cross, rcond=1e-12)[0]
    return {"weights": weights.tolist(), "intercept": (mean_y-mean_x@weights).tolist(),
            "n_fit_samples": int(len(x)), "ridge": float(ridge), "fit_split": "train", "max_samples": max_samples}


def apply_ridge_probe(x, probe):
    return _array(x)@_array(probe["weights"])+_array(probe["intercept"])


def fit_signed_permutation(pred_cal, true_cal):
    """Fit signs/permutation ONLY, with no scaling, rotation, or translation.

    Inputs must already have their scale fixed from train. At m<=4 enumeration
    is exact and inexpensive. Returned order maps predicted axes to truth axes.
    """
    predicted, truth = _array(pred_cal), _array(true_cal)
    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError("matching arrays must have equal [sample,axis] shape")
    m = predicted.shape[1]
    if m>8:
        raise ValueError("exact permutation matching supports m<=8; low-dimensional proposal uses m<=4")
    best = None
    for permutation in itertools.permutations(range(m)):
        chosen = predicted[:, permutation]
        signs = np.where(np.sum(chosen*truth, axis=0)<0, -1., 1.)
        error = float(np.mean((chosen*signs-truth)**2))
        if best is None or error<best[0]:
            best = (error, list(permutation), signs.tolist())
    return {"permutation": best[1], "signs": best[2], "calibration_mse": best[0],
            "fit_split": "calibration", "allowed_transform": "signed_permutation_only"}


def apply_signed_permutation(values, match):
    return _array(values)[..., match["permutation"]]*_array(match["signs"])


def _orthonormal_columns(matrix):
    matrix = _array(matrix)
    if not matrix.size:
        return np.empty((matrix.shape[0], 0))
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 1)
    return u[:, singular>tolerance]


def subspace_metrics(a, b, expected_dimension=None):
    """Compare column spans; rank deficiency is explicitly penalized."""
    qa, qb = _orthonormal_columns(a), _orthonormal_columns(b)
    expected = int(expected_dimension if expected_dimension is not None else _array(b).shape[1])
    cosines = np.linalg.svd(qa.T@qb, compute_uv=False).clip(0, 1)
    padded = np.zeros(expected)
    padded[:min(expected, len(cosines))] = cosines[:expected]
    angles = np.arccos(padded)
    return {"expected_dimension": expected, "predicted_rank": int(qa.shape[1]), "target_rank": int(qb.shape[1]),
            "principal_angles_radians": angles.tolist(), "mean_sin_squared": float(np.mean(1-padded**2)),
            "max_sin_angle": float(np.sqrt(max(0, 1-float(padded.min())**2))),
            "axis_recovery_claim": False}


def _batch_representations(model, observations, batch_size=4096):
    x = _array(observations)
    shape = x.shape[:-1]
    flattened = x.reshape(-1, x.shape[-1])
    parameter = next(model.parameters())
    fine, coarse = [], []
    with torch.no_grad():
        for start in range(0, len(flattened), batch_size):
            part = torch.as_tensor(flattened[start:start+batch_size], device=parameter.device, dtype=parameter.dtype)
            fine.append(model.encode(part).detach().cpu().double().numpy())
            coarse.append(model.coarse(part).detach().cpu().double().numpy())
    z, c = np.concatenate(fine), np.concatenate(coarse)
    return z.reshape(*shape, z.shape[-1]), c.reshape(*shape, c.shape[-1])


def _clusters(selection, m):
    groups = selection.get("clusters") or [[i] for i in range(m)]
    groups = [[int(index) for index in group] for group in groups]
    if sorted(index for group in groups for index in group) != list(range(m)):
        raise ValueError("selection clusters must partition the selected coordinates")
    return groups


def _scale_groups(selection, m):
    """Scale by actual real/oscillatory modes, never by a near-repeated cluster."""
    groups, start = [], 0
    for block in selection.get("blocks", []):
        size = 2 if block.get("kind") == "rotation" else 1
        groups.append(list(range(start, start+size)))
        start += size
    if start != m:
        raise ValueError("selection blocks must describe all selected mode coordinates")
    return groups


def _train_mode_scales(values, groups):
    flat = _array(values).reshape(-1, values.shape[-1])
    scales = np.empty(flat.shape[1])
    for group in groups:
        # Each real mode has its own scalar; a rotation block shares one scalar.
        scales[group] = max(float(np.sqrt(np.mean(flat[:, group]**2))), 1e-12)
    return scales


def _fit_cluster_linear_map(predicted, truth):
    """General invertible cluster map is allowed; never count it as axis recovery."""
    mapping = np.linalg.lstsq(_array(predicted), _array(truth), rcond=1e-10)[0]
    singular = np.linalg.svd(mapping, compute_uv=False)
    full_rank = bool(np.linalg.matrix_rank(mapping) == mapping.shape[0])
    condition = float(singular.max()/singular.min()) if full_rank and singular.min()>0 else None
    return {"matrix": mapping.tolist(), "condition_number": condition, "invertible_at_numeric_tolerance": full_rank,
            "singular_values": singular.tolist(), "fit_split": "calibration", "axis_recovery_claim": False}


def _restricted_cluster_match(predicted_cal, truth_cal, groups):
    """Do not split an oscillation/near-repeated cluster during axis matching."""
    m = predicted_cal.shape[-1]
    best = None
    for assignment in itertools.permutations(range(len(groups))):
        if any(len(groups[source]) != len(groups[target]) for target, source in enumerate(assignment)):
            continue
        local_orders = [list(itertools.permutations(groups[source])) for source in assignment]
        for choices in itertools.product(*local_orders):
            order = list(range(m))
            for target_group, source_order in zip(groups, choices):
                for target, source in zip(target_group, source_order):
                    order[target] = source
            chosen = predicted_cal[:, order]
            signs = np.where(np.sum(chosen*truth_cal, axis=0)<0, -1., 1.)
            error = float(np.mean((chosen*signs-truth_cal)**2))
            if best is None or error<best[0]:
                best = (error, order, signs.tolist())
    return {"permutation": best[1], "signs": best[2], "calibration_mse": best[0],
            "fit_split": "calibration", "allowed_transform": "cluster_preserving_signed_permutation_only"}


def _forecast(model, z, c, fine_probe, coarse_probe, target, horizon, has_fine=True):
    parameter = next(model.parameters())
    horizon = min(int(horizon), z.shape[1]-1)
    if horizon<1:
        return {"available": False, "reason": "trajectory too short"}
    starts = z.shape[1]-horizon
    zflat = z[:, :starts].reshape(-1, z.shape[-1])
    cflat = c[:, :starts].reshape(-1, c.shape[-1])
    records = []
    batch = 4096
    for q in range(1, horizon+1):
        fine_predictions, coarse_predictions = [], []
        with torch.no_grad():
            for start in range(0, len(zflat), batch):
                fine = torch.as_tensor(zflat[start:start+batch], device=parameter.device, dtype=parameter.dtype) if has_fine else None
                coarse = torch.as_tensor(cflat[start:start+batch], device=parameter.device, dtype=parameter.dtype)
                if has_fine:
                    fine = model.predict_fine(fine)
                    fine_predictions.append(fine.cpu().double().numpy())
                coarse = model.predict_coarse(coarse, 1)
                coarse_predictions.append(coarse.cpu().double().numpy())
        if has_fine:
            zflat = np.concatenate(fine_predictions)
        cflat = np.concatenate(coarse_predictions)
        actual = target[:, q:q+starts].reshape(-1, target.shape[-1])
        fine_score = _mse(apply_ridge_probe(zflat, fine_probe), actual) if has_fine else None
        coarse_score = _mse(apply_ridge_probe(cflat, coarse_probe), actual)
        records.append({"horizon": q, "fine": fine_score, "coarse": coarse_score})
    return {"available": True, "target_space": "common_clean_original_observation_space", "n_start_points": int(len(zflat)),
            "decoder_fit_split": "train", "by_horizon": records,
            "fine_available": has_fine,
            "fine_unavailable_reason": None if has_fine else "direct model has no fine predictor",
            "fine_mean_mse": float(np.mean([row["fine"]["mse"] for row in records])) if has_fine else None,
            "coarse_mean_mse": float(np.mean([row["coarse"]["mse"] for row in records]))}


def evaluate_model(model, system, datasets, selection, config, output_dir=None):
    """Fit on train/calibration and return a JSON-serializable test evaluation."""
    previous_mode = model.training
    model.eval()
    try:
        return _evaluate_model(model, system, datasets, selection, config, output_dir)
    finally:
        model.train(previous_mode)


def _evaluate_model(model, system, datasets, selection, config, output_dir):
    options = config.get("evaluation", {})
    ridge, limit = float(options.get("ridge", 1e-5)), int(options.get("probe_max_samples", 10000))
    encodings = {name: _batch_representations(model, datasets[name]["observations"]) for name in ("train", "calibration", "test")}
    truth = {name: _array(system.true_modes(datasets[name]["states"], selection)) for name in encodings}
    m = encodings["train"][1].shape[-1]
    if m<1 or truth["train"].shape[-1] != m:
        raise ValueError("selected truth and learned high-level representation must have identical positive dimension")
    groups = _clusters(selection, m)
    scaling_groups = _scale_groups(selection, m)
    scales_c = _train_mode_scales(encodings["train"][1], scaling_groups)
    scales_t = _train_mode_scales(truth["train"], scaling_groups)
    normalized_c = {name: value[1]/scales_c for name, value in encodings.items()}
    normalized_t = {name: value/scales_t for name, value in truth.items()}
    match = _restricted_cluster_match(normalized_c["calibration"].reshape(-1, m), normalized_t["calibration"].reshape(-1, m), groups)
    matched_test = apply_signed_permutation(normalized_c["test"], match).reshape(-1, m)
    true_test = normalized_t["test"].reshape(-1, m)
    cluster_records = []
    learned_b = state_jacobian(model, system)
    step = 1e-5
    points = np.eye(system.dimension)*step
    true_b = ((_array(system.true_modes(points, selection))-_array(system.true_modes(-points, selection)))/(2*step)).T
    learned_b = learned_b[match["permutation"]]/scales_c[match["permutation"], None]*_array(match["signs"])[:, None]
    true_b = true_b/scales_t[:, None]
    axis_scores = []
    for number, group in enumerate(groups):
        function_score = subspace_metrics(matched_test[:, group], true_test[:, group], len(group))
        local_score = subspace_metrics(learned_b[group].T, true_b[group].T, len(group))
        pcal = apply_signed_permutation(normalized_c["calibration"], match).reshape(-1, m)[:, group]
        tcal = normalized_t["calibration"].reshape(-1, m)[:, group]
        # Orthogonal cluster matching is legal only for cluster reconstruction.
        u, _, vt = np.linalg.svd(pcal.T@tcal, full_matrices=False)
        rotation = u@vt
        frozen_cluster_recovery = _mse(matched_test[:, group]@rotation, true_test[:, group])
        cluster_linear_map = _fit_cluster_linear_map(pcal, tcal)
        linear_recovery = _mse(matched_test[:, group]@_array(cluster_linear_map["matrix"]), true_test[:, group])
        record = {"cluster": number, "coordinates": group, "dimension": len(group),
                  "functionspace": function_score, "local_state_subspace": local_score,
                  "calibration_frozen_orthogonal_function_recovery": frozen_cluster_recovery,
                  "orthogonal_function_recovery_role": "auxiliary diagnostic; general cluster equivalence can be nonorthogonal",
                  "calibration_frozen_linear_function_recovery": linear_recovery,
                  "calibration_frozen_linear_map": cluster_linear_map,
                  "orthogonal_match": rotation.tolist(), "orthogonal_match_fit_split": "calibration",
                  "orthogonal_error_counts_as_axis_recovery": False}
        if len(group)==1:
            score = _mse(matched_test[:, group], true_test[:, group])
            axis_scores.append(score["mse"])
            record["axis"] = score
        else:
            record["signed_permutation_axis_diagnostic"] = _mse(matched_test[:, group], true_test[:, group])
            record["axis_recovery_claim"] = False
        cluster_records.append(record)
    train_z, train_c = encodings["train"]
    test_z, test_c = encodings["test"]
    train_states, test_states = _array(datasets["train"]["states"]), _array(datasets["test"]["states"])
    train_raw = _array(datasets["train"].get("clean_observations", datasets["train"]["observations"]))
    test_raw = _array(datasets["test"].get("clean_observations", datasets["test"]["observations"]))
    fine_probe = fit_ridge_probe(train_z, train_raw, ridge, limit)
    coarse_probe = fit_ridge_probe(train_c, train_raw, ridge, limit)
    state_probe = fit_ridge_probe(train_z, train_states, ridge, limit)
    coarse_state_probe = fit_ridge_probe(train_c, train_states, ridge, limit)
    target_state_probe = fit_ridge_probe(truth["train"], train_states, ridge, limit)
    detail_train = train_states-apply_ridge_probe(truth["train"], target_state_probe)
    detail_test = test_states-apply_ridge_probe(truth["test"], target_state_probe)
    fine_detail_probe = fit_ridge_probe(train_z, detail_train, ridge, limit)
    coarse_detail_probe = fit_ridge_probe(train_c, detail_train, ridge, limit)
    has_fine = config.get("model", {}).get("variant") != "direct"
    forecast = _forecast(model, test_z, test_c, fine_probe, coarse_probe, test_raw, options.get("horizon", 10), has_fine)
    parameter = next(model.parameters())
    x = torch.as_tensor(datasets["test"]["observations"][:, :-1].reshape(-1, system.dimension), device=parameter.device, dtype=parameter.dtype)
    with torch.no_grad():
        c = model.coarse(x)
        coarse_next = model.predict_coarse(c, 1).detach().cpu().double().numpy()
        fine_next = model.predict_fine(model.encode(x)).detach().cpu().double().numpy() if has_fine else None
    coarse_residual = _mse(coarse_next, test_c[:, 1:].reshape(-1, m))
    fine_residual = _mse(fine_next, test_z[:, 1:].reshape(-1, test_z.shape[-1])) if has_fine else {"available": False, "reason": "direct model has no fine predictor", "mse": None, "normalized_mse": None}
    counts = model.parameter_counts() if hasattr(model, "parameter_counts") else {}
    result = {
        "stage": config.get("stage", "unknown"), "variant": config.get("model", {}).get("variant", "unknown"),
        "seed": config.get("training", {}).get("seed"),
        "evidence": "local_integration_engineering_only" if config.get("stage")=="local_integration" else "held_out_synthetic_mechanism_experiment",
        "protocol": {"probe_fit_split": "train", "matching_fit_split": "calibration", "score_split": "test",
                     "scale_fit_split": "train", "scale_rule": "RMS about zero, real modes individually, rotation blocks jointly; no cross-mode whitening",
                     "axis_transform": "cluster-preserving signed permutation", "test_used_for_fit": False,
                     "probe_max_samples": limit, "ridge": ridge, "probe_type": "linear_ridge_with_intercept"},
        "axis": {"isolated_axis_count": len(axis_scores), "isolated_axis_mean_mse": float(np.mean(axis_scores)) if axis_scores else None,
                 "note": "multi-dimensional clusters never count as successful axis recovery"},
        "clusters": cluster_records,
        "cluster_mean_functionspace_sin_squared": float(np.mean([row["functionspace"]["mean_sin_squared"] for row in cluster_records])),
        "frozen_matching": match,
        "train_scales": {"prediction": scales_c.tolist(), "truth": scales_t.tolist()},
        "fine_retention": {"fine_state": _mse(apply_ridge_probe(test_z, state_probe), test_states),
                           "coarse_state": _mse(apply_ridge_probe(test_c, coarse_state_probe), test_states),
                           "fine_detail": _mse(apply_ridge_probe(test_z, fine_detail_probe), detail_test),
                           "coarse_detail": _mse(apply_ridge_probe(test_c, coarse_detail_probe), detail_test),
                           "detail_definition": "state residual after train-fitted linear probe from true selected modes; evaluation labels only",
                           "fine_predictor_available": has_fine,
                           "direct_hidden_tap_note": None if has_fine else "z is a hidden tap of the direct E/G model; no fine predictor is trained",
                           "limitation": "linear probe measures decodable detail, not total mutual information"},
        "original_space_forecast": forecast,
        "coarse_one_step_residual": coarse_residual,
        "fine_one_step_residual": fine_residual,
        "collapse": {"fine": trajectory_diagnostics(test_z), "coarse": trajectory_diagnostics(test_c)},
        "theory_diagnostics": theory_diagnostics(model, system, selection, config),
        "target_spectrum_diagnostics": target_spectrum_diagnostics(model, selection, config),
        "selection": {"status": selection.get("status"), "m": selection.get("m"), "valid_for_training": selection.get("valid_for_training"),
                      "certificate_scope": "local spectral selection, not learned recovery"},
        "parameter_counts": counts,
        "probes": {"fine_original_space": fine_probe, "coarse_original_space": coarse_probe,
                   "fine_state": state_probe, "coarse_state": coarse_state_probe,
                   "target_state": target_state_probe, "fine_detail": fine_detail_probe, "coarse_detail": coarse_detail_probe},
    }
    # Recovery and prediction thresholds are frozen config values, never tuned on test.
    coarse_metric = result["cluster_mean_functionspace_sin_squared"]
    fine_metric = forecast.get("fine_mean_mse")
    result["common_thresholds"] = {"coarse_metric": "cluster_mean_functionspace_sin_squared", "coarse_value": coarse_metric,
        "subspace_threshold": float(options.get("subspace_threshold", .1)),
        "fine_metric": "original_space_forecast.fine_mean_mse", "fine_value": fine_metric,
        "fine_forecast_threshold": float(options.get("fine_forecast_threshold", .1)),
        "threshold_config_keys": ["evaluation.subspace_threshold", "evaluation.fine_forecast_threshold"],
        "both_met": bool(coarse_metric<=options.get("subspace_threshold", .1) and fine_metric is not None and fine_metric<=options.get("fine_forecast_threshold", .1))}
    result["common_thresholds"]["use_for_training_or_step_selection"] = False
    result["common_thresholds"]["note"] = "final test description only; transfer step selection uses the separate frozen calibration monitor"
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination/"seed_outputs.npz",
                            calibration=normalized_c["calibration"], test=normalized_c["test"])
        (destination/"evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


def evaluate_transfer_thresholds(model, system, datasets, selection, config, frozen=None):
    """Freeze at target adaptation step zero, then score calibration-monitor only.

    Target train determines fixed RMS units at step zero. Calibration trajectories
    are split in order into disjoint fit/monitor halves. The fit half determines
    the signed-permutation/cluster alignment and a fixed raw-space fine decoder.
    Neither truth matching nor probes are re-fitted on later calls. Both variants
    receive identical data, permissions and sample budgets. No test access occurs.
    """
    previous_mode = model.training
    model.eval()
    try:
        if config.get("model", {}).get("variant") == "direct":
            return {"available": False, "coarse_error": None, "fine_error": None, "reached": False,
                    "reason": "direct model has no fine predictor and is not a full/no_fine transfer arm", "frozen": frozen}
        options = config.get("evaluation", {})
        transfer_options = config.get("transfer", {})
        calibration = datasets["calibration"]
        n = len(calibration["observations"])
        if n<2:
            raise ValueError("transfer monitor needs at least two independent calibration trajectories")
        if frozen is None:
            boundary = n//2
            train_z, train_c = _batch_representations(model, datasets["train"]["observations"])
            train_truth = _array(system.true_modes(datasets["train"]["states"], selection))
            groups = _clusters(selection, train_c.shape[-1])
            scaling_groups = _scale_groups(selection, train_c.shape[-1])
            pred_scale = _train_mode_scales(train_c, scaling_groups)
            truth_scale = _train_mode_scales(train_truth, scaling_groups)
            fit_z, fit_c = _batch_representations(model, calibration["observations"][:boundary])
            fit_truth = _array(system.true_modes(calibration["states"][:boundary], selection))
            normalized_c = (fit_c/pred_scale).reshape(-1, train_c.shape[-1])
            normalized_t = (fit_truth/truth_scale).reshape(-1, train_c.shape[-1])
            match = _restricted_cluster_match(normalized_c, normalized_t, groups)
            matched = apply_signed_permutation(normalized_c, match)
            rotations, linear_maps = [], []
            for group in groups:
                if len(group)==1:
                    rotation = np.ones((1, 1))
                else:
                    u, _, vt = np.linalg.svd(matched[:, group].T@normalized_t[:, group], full_matrices=False)
                    rotation = u@vt
                rotations.append(rotation.tolist())
                if len(group)==1:
                    linear_maps.append({"matrix": [[1.]], "condition_number": 1., "invertible_at_numeric_tolerance": True,
                                        "fit_split": "none; singleton permits sign/permutation only"})
                else:
                    linear_maps.append(_fit_cluster_linear_map(matched[:, group], normalized_t[:, group]))
            fit_raw = _array(calibration.get("clean_observations", calibration["observations"])[:boundary])
            probe = fit_ridge_probe(fit_z, fit_raw, float(options.get("ridge", 1e-5)), int(options.get("probe_max_samples", 10000)))
            probe["fit_split"] = "target_calibration_fit_at_adaptation_step_zero"
            frozen = {"schema_version": 1, "frozen_at": "target_adaptation_step_zero", "fit_trajectory_indices": list(range(boundary)),
                      "monitor_trajectory_indices": list(range(boundary, n)), "calibration_trajectory_count": n,
                      "scale_fit_split": "target_train_at_adaptation_step_zero", "prediction_scales": pred_scale.tolist(),
                      "truth_scales": truth_scale.tolist(), "groups": groups, "signed_permutation": match,
                      "scale_groups": scaling_groups, "cluster_rotations": rotations, "cluster_linear_maps": linear_maps,
                      "fine_original_space_probe": probe,
                      "horizon": int(options.get("horizon", 10)), "coarse_threshold": float(transfer_options.get("coarse_threshold", .25)),
                      "fine_threshold": float(transfer_options.get("fine_threshold", .25)),
                      "threshold_config_keys": ["transfer.coarse_threshold", "transfer.fine_threshold"],
                      "protocol": "same target train/calibration access and probe budget for full/no_fine; never refit after step zero"}
        if frozen["calibration_trajectory_count"] != n:
            raise ValueError("frozen transfer payload belongs to a different calibration trajectory count")
        indices = frozen["monitor_trajectory_indices"]
        observations = _array(calibration["observations"])[indices]
        z, c = _batch_representations(model, observations)
        truth = _array(system.true_modes(_array(calibration["states"])[indices], selection))
        pred = apply_signed_permutation(c/_array(frozen["prediction_scales"]), frozen["signed_permutation"])
        true = truth/_array(frozen["truth_scales"])
        squared_error = []
        per_cluster = []
        maps_invertible = True
        for group, mapping in zip(frozen["groups"], frozen["cluster_linear_maps"]):
            aligned = pred[..., group]@_array(mapping["matrix"])
            maps_invertible = maps_invertible and mapping["invertible_at_numeric_tolerance"]
            error = float(np.mean((aligned-true[..., group])**2))
            squared_error.extend([error]*len(group))
            per_cluster.append({"coordinates": group, "frozen_alignment_mse": error, "axis_claim": len(group)==1,
                                "condition_number": mapping["condition_number"], "map_invertible": mapping["invertible_at_numeric_tolerance"]})
        coarse_error = float(np.mean(squared_error))
        raw = _array(calibration.get("clean_observations", calibration["observations"]))[indices]
        # Coarse forecast is irrelevant here; using the same fixed probe as a
        # placeholder keeps rollout code shared without fitting an extra decoder.
        dummy_coarse = {"weights": np.zeros((c.shape[-1], raw.shape[-1])).tolist(), "intercept": np.zeros(raw.shape[-1]).tolist()}
        rollout = _forecast(model, z, c, frozen["fine_original_space_probe"], dummy_coarse, raw, frozen["horizon"])
        fine_error = rollout.get("fine_mean_mse")
        return {"available": True, "coarse_error": coarse_error, "fine_error": fine_error,
                "reached": bool(maps_invertible and coarse_error<=frozen["coarse_threshold"] and fine_error is not None and fine_error<=frozen["fine_threshold"]),
                "coarse_metric": "train-scaled monitor MSE: frozen signed/permutation for axes, frozen invertible linear maps for clusters",
                "fine_metric": "calibration-monitor original-space fine rollout mean MSE under fixed step-zero decoder",
                "score_split": "target_calibration_monitor", "monitor_trajectories": len(indices), "clusters": per_cluster,
                "test_accessed": False, "refitted_after_step_zero": False, "frozen": frozen}
    finally:
        model.train(previous_mode)


def cross_seed_consistency(calibration_outputs, test_outputs, clusters=None):
    """Pairwise cal-fitted consistency; no arbitrary test-time axis rotation."""
    if len(calibration_outputs) != len(test_outputs) or not calibration_outputs:
        raise ValueError("paired nonempty calibration/test output lists are required")
    m = _array(calibration_outputs[0]).shape[-1]
    groups = clusters or [[i] for i in range(m)]
    reference_cal = _array(calibration_outputs[0]).reshape(-1, m)
    reference_test = _array(test_outputs[0]).reshape(-1, m)
    records = []
    for index in range(1, len(test_outputs)):
        cal = _array(calibration_outputs[index]).reshape(-1, m)
        test = _array(test_outputs[index]).reshape(-1, m)
        match = _restricted_cluster_match(cal, reference_cal, groups)
        aligned_cal, aligned_test = apply_signed_permutation(cal, match), apply_signed_permutation(test, match)
        cluster_errors = []
        for group in groups:
            u, _, vt = np.linalg.svd(aligned_cal[:, group].T@reference_cal[:, group], full_matrices=False)
            rotation = u@vt if len(group)>1 else np.ones((1, 1))
            orthogonal_error = _mse(aligned_test[:, group]@rotation, reference_test[:, group])["mse"]
            if len(group)>1:
                linear = _fit_cluster_linear_map(aligned_cal[:, group], reference_cal[:, group])
                linear_error = _mse(aligned_test[:, group]@_array(linear["matrix"]), reference_test[:, group])["mse"]
            else:
                linear = {"matrix": [[1.]], "condition_number": 1., "invertible_at_numeric_tolerance": True,
                          "fit_split": "none; sign/permutation already fitted on calibration"}
                linear_error = orthogonal_error
            cluster_errors.append({"coordinates": list(group), "mse": linear_error,
                                   "calibration_frozen_general_linear_mse": linear_error,
                                   "general_linear_match": linear, "orthogonal_mse_auxiliary": orthogonal_error,
                                   "axis_claim": len(group)==1})
        records.append({"reference_seed_index": 0, "seed_index": index, "matching": match, "clusters": cluster_errors})
    return {"pairs": records, "fit_split": "calibration", "score_split": "test", "scales": "caller must supply train-fixed units",
            "test_used_for_alignment": False, "multi_axis_rotations_count_as_axis_recovery": False}
