"""Replay existing draws to separate reference and axis stability, before reveal."""

import json

import numpy as np
import torch

from pipeline import (
    HERE,
    LOW_QUANTILE,
    PERMUTATIONS,
    TRAINING,
    anchor_response_samples,
    bank_definitions,
    bootstrap_weights,
    candidate_scores,
    digest,
    embed,
    lag_transforms,
    make_mlp_encoder,
    mean_and_cov,
    named_permutations,
    panel_paths,
    stream,
    test_paths,
)


def summarize_margins(nominal, candidates):
    indices = np.argsort(nominal, axis=1, kind="stable")[:, :2]
    inside = np.zeros(nominal.shape, dtype=bool)
    np.put_along_axis(inside, indices, True, axis=1)
    margins = (
        np.where(~inside, candidates[1:], np.inf).min(axis=2)
        - np.where(inside, candidates[1:], -np.inf).max(axis=2)
    )
    return np.quantile(margins, LOW_QUANTILE, axis=0), (margins > 0).mean(axis=0)


def analysis_components(model, config, rates, bank):
    length = config["trajectory_length"]
    cal = embed(model, panel_paths(rates, "calibration", bank, config["cal_trajectories"], length))
    ref = embed(model, panel_paths(rates, "reference", bank, config["ref_trajectories"], length))
    anchors = anchor_response_samples(model, bank, config)
    random = stream("bootstrap", bank)
    transforms, _, _ = lag_transforms(cal, bootstrap_weights(random, len(cal)))
    responses = np.einsum("bn,nij->bij", bootstrap_weights(random, len(anchors)), anchors)
    names, normalized = named_permutations(responses @ transforms)
    transforms = np.take_along_axis(transforms, names[:, None, :], axis=2)
    _, reference_cov = mean_and_cov(ref, bootstrap_weights(random, len(ref)))
    original_transforms = np.repeat(transforms[:1], len(transforms), axis=0)
    original_ref = np.repeat(reference_cov[:1], len(reference_cov), axis=0)

    variants = {}
    for name, basis, covariance in (
        ("joint", transforms, reference_cov),
        ("reference_only", original_transforms, reference_cov),
        ("calibration_anchors_only", transforms, original_ref),
    ):
        variances = np.einsum("bdj,bde,bej->bj", basis, covariance, basis)
        variants[name] = (basis, variances)

    assignment_totals = np.sort(normalized[0, np.arange(4)[None, :], PERMUTATIONS].sum(axis=1))
    mapping = {
        "same_full_mapping_fraction": float(np.all(names[1:] == names[0], axis=1).mean()),
        "same_axis_mapping_fraction": np.mean(names[1:] == names[0], axis=0).tolist(),
        "assignment_objective_gap": float(assignment_totals[-1] - assignment_totals[-2]),
    }
    return variants, mapping


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((TRAINING / "protocol.json").read_text())
    path = HERE / "forecasts.npz"
    receipt = json.loads((HERE / "pipeline_forecast_receipt.json").read_text())
    assert digest(path) == receipt["forecasts_sha256"]
    assert digest(HERE / "pipeline.py") == receipt["pipeline_sha256"]
    with np.load(path, allow_pickle=False) as forecasts:
        locked = {key: forecasts[key] for key in forecasts.files}
    results = {
        f"{split}_{variant}_{key}": np.empty(len(locked[f"{split}_bank"]))
        for split in ("dev", "test")
        for variant in ("joint", "reference_only", "calibration_anchors_only")
        for key in ("q10", "match_fraction")
    }
    mapping_diagnostics = []
    previous = None
    for bank, (_, case, seed, _, rates) in enumerate(bank_definitions(config)):
        checkpoint = TRAINING / "runs" / f"{case}_{seed}.pt"
        if checkpoint != previous:
            assert digest(checkpoint) == receipt["checkpoint_sha256"][checkpoint.name]
            model = make_mlp_encoder(4, config["hidden"], config["layers"], device="cpu").eval()
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            previous = checkpoint
        variants, mapping = analysis_components(model, config, rates, bank)
        mapping_diagnostics.append({"bank": bank, **mapping})
        for split in ("dev", "test"):
            rows = np.flatnonzero(locked[f"{split}_bank"] == bank)
            if not len(rows):
                continue
            length = config["trajectory_length"]
            paths = (
                panel_paths(rates, "development", bank, len(rows), length)
                if split == "dev"
                else test_paths(rates, rows, length)
            )
            embedded = embed(model, paths)
            nominal = locked[f"{split}_scores"][rows]
            for variant, (basis, variance) in variants.items():
                values = candidate_scores(embedded, basis, variance, config["lags"])
                assert np.allclose(values[0], nominal, rtol=0, atol=1e-12)
                quantile, fraction = summarize_margins(nominal, values)
                results[f"{split}_{variant}_q10"][rows] = quantile
                results[f"{split}_{variant}_match_fraction"][rows] = fraction
                if variant == "joint":
                    assert np.allclose(quantile, locked[f"{split}_uncertainty"][rows], rtol=0, atol=1e-12)
    np.savez_compressed(HERE / "pipeline_diagnostics.npz", **results, forecasts_sha256=digest(path))
    (HERE / "pipeline_bank_mapping.json").write_text(
        json.dumps(mapping_diagnostics, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
