"""Score fresh, same-law AR/FHN windows with registered frozen LeJEPA encoders."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEGACY = ROOT / "work/hcf_stable_id_20261027"
BASELINE = ROOT / "outputs/HCF_STABLE_ID_RUN_20261027_V1/lejepa"
sys.path[:0] = [str(LEGACY / "v1"), str(LEGACY / "confirmation_v1")]

import data_metrics as data
import data_owner_engineering as owner
from lejepa_adapter import LeJEPAAdapter

UNITS = ("ar_caseA", "fhn_caseA_eps008")
LAGS = (4, 16, 32)
CAL_TAG = 202611121
REFERENCE_TAGS = tuple(range(202611130, 202611138))
EVALUATION_TAGS = tuple(range(202611140, 202611148))
SEEDS = (0, 1, 2)
PREDICTION = HERE / "FORECAST.json"
FORECAST_ARRAYS = HERE / "FORECAST.npz"
SCORES = HERE / "SUMMARY.json"
torch.set_num_threads(1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checked(receipt):
    path = Path(receipt["path"])
    if sha(path) != receipt["sha256"]:
        raise ValueError(f"File identity changed: {path}")
    return path


def historical_models(unit, train, cal):
    result = {}
    entries = read(BASELINE / "REPRESENTATION_MODELS_LOCK.json")
    manifest = read(BASELINE / "REPRESENTATION_COMPLETED.json")
    checked(manifest["models_lock"])
    for seed in SEEDS:
        folder = BASELINE / "formal/lambda_001" / unit / f"seed{seed}"
        finished = next(r for r in entries["records"] if Path(r["path"]) == folder / "COMPLETED.json")
        record = read(checked(finished))
        evidence = next(r for r in manifest["features"] if Path(r["path"]) == folder / "FEATURES_RECEIPT.json")
        features = read(checked(evidence))
        checkpoint = checked(features["checkpoint"])
        if features["checkpoint"] not in record["artifacts"] or record["seed"] != seed:
            raise ValueError("Inconsistent checkpoint lineage")
        job = read(checked(record["job"]))
        for path in (Path(LeJEPAAdapter.__module__.replace(".", "/") + ".py").name,
                     "vendor/lejepa_identifiability/experiments/lejepa_id/models.py"):
            source = LEGACY / "v1" / path
            if sha(source) != record["source_sha256"][str(source)]:
                raise ValueError("Historical encoder source mismatch")
        adapter = LeJEPAAdapter(job["config"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        adapter.load_state_dict(state["adapter"])
        adapter.eval()
        with np.load(checked(features["features"]), allow_pickle=False) as archive:
            errors = {}
            for label, observed in (("train", train), ("cal", cal)):
                actual = adapter.encode_sequences(observed)
                expected = archive[f"{label}_z"]
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
                errors[label] = float(np.max(np.abs(actual - expected)))
        result[f"lejepa{seed}"] = (adapter.encode_sequences,
                                  dict(checkpoint=features["checkpoint"], selected_step=record["selected_step"],
                                       historical_feature_maxabs=errors))
    return result


def generator(unit):
    spec = data.unit_spec(unit)
    _, manifest_path, manifest, _ = data._paths(spec)
    env, transform = owner._math()
    if spec["benchmark"] == "ar":
        namespace = dict(np=np, data_seed=manifest["data_seed"], D=4, A=2,
                         L=manifest["transitions"], burn=manifest["burnin_steps"],
                         mode=manifest["mode"], alpha=np.asarray(manifest["alpha"]),
                         B=np.asarray(manifest["action_weights"]))
        owner._extract(owner.AR_SOURCE, ("simulate",), namespace, parent="generate_case")
        simulate = namespace["simulate"]
        mix = manifest["observation_config_for_generator_only"]
    else:
        config = env.Config(**manifest["config"])

        def guard(cfg, tag, count, capability):
            valid_tags = (101, 211, CAL_TAG, *REFERENCE_TAGS, *EVALUATION_TAGS)
            if cfg != config or tag not in valid_tags or not (0 < count <= 128) or capability is not None:
                raise ValueError("Unregistered local trajectory call")

        namespace = dict(np=np, copy=copy, env=env, require=owner.require,
                         ah=owner.array_sha, _draw_guard=guard)
        owner._extract(owner.FHN_SOURCE, ("_draw", "_physical"), namespace)

        def simulate(count, tag):
            return namespace["_physical"](config, tag, count)[:3]

        mix = manifest["observation_mixing"]

    def draw(count, tag):
        physical, actions, regimes = simulate(count, tag)
        observed = np.ascontiguousarray(transform(physical, mix), dtype=np.float32)
        truth = np.ascontiguousarray(physical, dtype=np.float32)
        return dict(x=observed, actions=actions, regimes=regimes), truth

    def public_draw(count, tag):
        public, _ = draw(count, tag)
        return public["x"], receipt(count, tag, public)

    def receipt(count, tag, public):
        return dict(count=count, rng_tag=tag, arrays={k: owner.identity(v) for k, v in public.items()})

    return draw, public_draw, receipt, dict(manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
                                             physical_seed=spec["data_seed"], mix=mix)


def fit_readouts(features, targets):
    x, y = features.reshape(-1, 4).astype(np.float64), targets.reshape(-1, 4).astype(np.float64)
    x_centered, y_centered = x - x.mean(0), y - y.mean(0)
    dense, _, rank, _ = np.linalg.lstsq(x_centered, y_centered, rcond=None)
    if rank != 4:
        raise ValueError("Calibration design rank deficient")
    denom = np.linalg.norm(x_centered, axis=0)[:, None] * np.linalg.norm(y_centered, axis=0)[None, :]
    corr = (x_centered.T @ y_centered) / denom
    choices = list(itertools.permutations(range(4)))
    ordered = sorted(choices, key=lambda p: sum(abs(corr[p[j], j]) for j in range(4)), reverse=True)
    matching = ordered[0]
    affine = np.zeros((4, 4))
    for j, k in enumerate(matching):
        affine[k, j] = (x_centered[:, k] @ y_centered[:, j]) / (x_centered[:, k] @ x_centered[:, k])
    return {kind: (beta, y.mean(0) - x.mean(0) @ beta)
            for kind, beta in (("dense", dense), ("matched", affine))}, list(matching)


def numerators(values):
    values = values.astype(np.float64)
    return np.mean([(values[:, lag:] - values[:, :-lag]).__pow__(2).mean(axis=1)
                    for lag in LAGS], axis=0)


def select(scores):
    return np.sort(np.argsort(scores, axis=1, kind="stable")[:, :2], axis=1)


def forecast():
    if PREDICTION.exists() or FORECAST_ARRAYS.exists() or SCORES.exists():
        raise FileExistsError("Existing forecast or score")
    started = time.perf_counter()
    arrays = {}
    report = {}
    for unit in UNITS:
        draw, public_draw, get_receipt, origin = generator(unit)
        historical = data.load_unit(unit)
        for split, tag in (("train", 101), ("cal", 211)):
            archived = historical[split]
            regenerated, _ = draw(len(archived["x"]), tag)
            for field in data.FIELDS:
                np.testing.assert_array_equal(regenerated[field], archived[field])
        models = historical_models(unit, historical["train_x"], historical["cal_x"])
        models["raw"] = (np.asarray, {"checkpoint": None})
        calibration, targets = draw(96, CAL_TAG)
        fit = {}
        for name, (encode, origin_model) in models.items():
            fit[name], matching = fit_readouts(encode(calibration["x"]), targets)
            origin_model.update(matched_coordinates_by_physical_axis=matching,
                                readouts={key: dict(beta=b.tolist(), intercept=c.tolist())
                                          for key, (b, c) in fit[name].items()})
        panels = []
        for panel, (ref_tag, eval_tag) in enumerate(zip(REFERENCE_TAGS, EVALUATION_TAGS, strict=True)):
            reference, ref_receipt = public_draw(64, ref_tag)
            evaluation, eval_receipt = public_draw(64, eval_tag)
            panels.append(dict(reference=ref_receipt, evaluation=eval_receipt))
            for name, (encode, _) in models.items():
                encoded_ref, encoded_eval = encode(reference), encode(evaluation)
                for kind, (beta, intercept) in fit[name].items():
                    key = f"{unit}/{panel}/{name}/{kind}"
                    evaluation_hat = encoded_eval @ beta + intercept
                    predicted_v = (encoded_ref @ beta + intercept).var(axis=(0, 1), ddof=0)
                    if np.any(predicted_v <= 0):
                        raise ValueError("Reference variance degenerate")
                    predicted_n = numerators(evaluation_hat)
                    arrays[f"{key}/evaluation_hat"] = evaluation_hat
                    arrays[f"{key}/numerator"] = predicted_n
                    arrays[f"{key}/variance"] = predicted_v
                    arrays[f"{key}/selected"] = select(predicted_n / predicted_v)
        report[unit] = dict(generator=origin, historical_data_parity="TRAIN and CAL observations/actions/regimes exact",
                            calibration=get_receipt(96, CAL_TAG, calibration), panels=panels,
                            models={name: origin_model for name, (_, origin_model) in models.items()})
    with FORECAST_ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write(PREDICTION, dict(protocol_sha256=sha(HERE / "PROTOCOL.md"), code_sha256=sha(__file__),
                           original_math_sha256=owner.source_hashes(), forecast_arrays_sha256=sha(FORECAST_ARRAYS),
                           seconds=time.perf_counter() - started, torch_version=torch.__version__,
                           numpy_version=np.__version__, device="cpu", optimizer_updates=0,
                           old_test_final_numeric_arrays_read=False, new_evaluation_truth_returned_to_predictor=False,
                           units=report))


def measurements(predicted_n, predicted_v, oracle_n, oracle_v, gap_info):
    true_selected, runner_up, fastest_in_set, true_gap, near = gap_info
    values = dict()
    for key, score in (("estimated_both", predicted_n / predicted_v),
                       ("physical_n_only", oracle_n / predicted_v),
                       ("physical_v_only", predicted_n / oracle_v),
                       ("physical_both", oracle_n / oracle_v)):
        errors = np.any(select(score) != true_selected, axis=1)
        rows = np.arange(len(score))
        gap = score[rows, runner_up] - score[rows, fastest_in_set]
        values[key] = dict(windows=len(errors), wrong=int(errors.sum()), near_n=int(near.sum()),
                           near_wrong=int(errors[near].sum()), signed_gap_mae=float(np.abs(gap - true_gap).mean()))
    return values


def score():
    started = time.perf_counter()
    forecast_receipt = read(PREDICTION)
    for path, expected in ((FORECAST_ARRAYS, forecast_receipt["forecast_arrays_sha256"]),
                           (HERE / "PROTOCOL.md", forecast_receipt["protocol_sha256"]),
                           (Path(__file__), forecast_receipt["code_sha256"])):
        if sha(path) != expected:
            raise ValueError(f"Forecast precommit mismatch: {path}")
    if owner.source_hashes() != forecast_receipt["original_math_sha256"]:
        raise ValueError("Simulation source changed")
    output = {}
    with np.load(FORECAST_ARRAYS, allow_pickle=False) as predictions:
        for unit, locked in forecast_receipt["units"].items():
            draw, _, get_receipt, origin = generator(unit)
            if locked["generator"] != origin:
                raise ValueError("Physics origin changed")
            rows = {f"{model}/{kind}": [] for model in locked["models"] for kind in ("dense", "matched")}
            errors = {key: [] for key in rows}
            truth_panels = []
            for i, panel in enumerate(locked["panels"]):
                replay = []
                for role in ("reference", "evaluation"):
                    saved = panel[role]
                    public, physical = draw(saved["count"], saved["rng_tag"])
                    if get_receipt(saved["count"], saved["rng_tag"], public) != saved:
                        raise ValueError("Unsealed evaluation trajectory")
                    replay.append(physical.astype(np.float64))
                reference, truth = replay
                truth_panels.append(truth)
                oracle_n, oracle_v = numerators(truth), reference.var(axis=(0, 1), ddof=0)
                oracle_scores = oracle_n / oracle_v
                ranks = np.argsort(oracle_scores, axis=1, kind="stable")
                sorted_scores = np.take_along_axis(oracle_scores, ranks, axis=1)
                true_gap = sorted_scores[:, 2] - sorted_scores[:, 1]
                gap_info = select(oracle_scores), ranks[:, 2], ranks[:, 1], true_gap, true_gap <= .05
                for label, panel_rows in rows.items():
                    key = f"{unit}/{i}/{label}"
                    predicted_n, predicted_v = predictions[f"{key}/numerator"], predictions[f"{key}/variance"]
                    np.testing.assert_array_equal(select(predicted_n / predicted_v), predictions[f"{key}/selected"])
                    panel_rows.append(measurements(predicted_n, predicted_v, oracle_n, oracle_v, gap_info))
                    errors[label].append((predictions[f"{key}/evaluation_hat"] - truth) ** 2)
            target = np.concatenate(truth_panels, axis=0)
            target_v = target.var(axis=(0, 1), ddof=0)
            unit_summary = {}
            for label, panel_rows in rows.items():
                aggregates = {}
                for variant in panel_rows[0]:
                    variant_rows = [row[variant] for row in panel_rows]
                    aggregates[variant] = {key: sum(r[key] for r in variant_rows)
                                           for key in ("windows", "wrong", "near_n", "near_wrong")}
                    aggregates[variant]["signed_gap_mae"] = np.mean([r["signed_gap_mae"] for r in variant_rows]).item()
                r2 = 1 - np.concatenate(errors[label], axis=0).mean(axis=(0, 1)) / target_v
                unit_summary[label] = dict(r2_per_axis=r2.tolist(), r2_average=r2.mean().item(),
                                           r2_min=r2.min().item(), aggregate=aggregates, panels=panel_rows)
            rng = np.random.default_rng(20261112)
            index = rng.integers(0, len(truth_panels), size=(10000, len(truth_panels)))
            paired = {}
            for label, panel_rows in rows.items():
                family, kind = label.split("/")
                if family == "raw":
                    continue
                comparison = rows[f"raw/{kind}"]
                pair_delta = np.asarray([(a["estimated_both"]["wrong"] - b["estimated_both"]["wrong"]) / 64
                                         for a, b in zip(panel_rows, comparison, strict=True)])
                paired[label] = dict(mean=pair_delta.mean().item(),
                                     panel_bootstrap_95=np.quantile(pair_delta[index].mean(axis=1), [.025, .975]).tolist())
            output[unit] = dict(models=unit_summary, wrong_rate_difference_vs_raw=paired)
    write(SCORES, dict(forecast_sha256=sha(PREDICTION), seconds=time.perf_counter() - started,
                       oracle_target="Independent panel empirical physical reference variance, lag 4/16/32",
                       result=output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("forecast", "score"))
    args = parser.parse_args()
    {"forecast": forecast, "score": score}[args.phase]()
