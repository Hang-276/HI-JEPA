"""Re-estimate frozen state moments and split archived reference accounting."""

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parents[1]
REPO = REPORTS.parent
OLD = REPORTS / "lejepa_dynamic_boundary_20261113"
LOAD_PATH = REPORTS / "lejepa_natural_direction_repair_20261113/loaders.py"

spec = importlib.util.spec_from_file_location("a1_joint_reference_loaders", LOAD_PATH)
loaders = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loaders)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def iid(master, role, size):
    return np.random.default_rng(np.random.SeedSequence([master, role])).normal(size=(size, 4))


def infer(readout, physical, chunk):
    return np.concatenate([readout(physical[start:start + chunk]) for start in range(0, len(physical), chunk)])


def empirical_moments(z, v):
    residual = v - z
    centered_z = z - z.mean(axis=0)
    centered_residual = residual - residual.mean(axis=0)
    c_samples = centered_z * centered_residual
    squared_samples = centered_residual**2
    c = c_samples.mean(axis=0)
    residual_variance = squared_samples.mean(axis=0)
    visible_variance = v.var(axis=0)
    np.testing.assert_allclose(visible_variance, z.var(axis=0) + 2*c + residual_variance, atol=1e-12, rtol=0)
    return {
        "covariance_z_e": c.tolist(),
        "residual_variance": residual_variance.tolist(),
        "physical_sample_variance": z.var(axis=0).tolist(),
        "direct_visible_variance": visible_variance.tolist(),
        "structural_visible_variance": (1 + 2*c + residual_variance).tolist(),
        "state_mse": (residual**2).mean(axis=0).tolist(),
    }, np.column_stack([c_samples, squared_samples])


def state_result(z, v, panel_size, n_original_ref):
    pooled, moments = empirical_moments(z, v)
    influence_cov = np.cov(moments, rowvar=False, ddof=1) / len(z)
    complete = np.column_stack([np.eye(4)*2, np.eye(4)])
    parameter_z = 1.959963984540054
    c_sd, e_sd = np.sqrt(np.diag(influence_cov)).reshape(2, 4)
    pop_sd = np.sqrt(np.diag(complete @ influence_cov @ complete.T))
    def with_interval(mean, sd):
        return np.stack([np.asarray(mean) - parameter_z*sd, np.asarray(mean) + parameter_z*sd], axis=1).tolist()
    pooled.update({
        "covariance_z_e_interval95": with_interval(pooled["covariance_z_e"], c_sd),
        "residual_variance_interval95": with_interval(pooled["residual_variance"], e_sd),
        "structural_visible_variance_interval95": with_interval(pooled["structural_visible_variance"], pop_sd),
        "estimated_original_iid_ref_variance_se": np.std((v-v.mean(axis=0))**2, axis=0, ddof=1).tolist(),
        "influence_covariance_of_pooled_c_then_v": influence_cov.tolist(),
    })
    pooled["estimated_original_iid_ref_variance_se"] = (
        np.asarray(pooled["estimated_original_iid_ref_variance_se"])/np.sqrt(n_original_ref)
    ).tolist()
    panels = [empirical_moments(z[i:i+panel_size], v[i:i+panel_size])[0] for i in range(0, len(z), panel_size)]
    return {"pooled": pooled, "panels": panels}


def diagnostic_cell(zs, ref_z, cal_z, vis, ref_u, cal_u, normalizer, s, new_n, round_ref_n):
    offset, scale = (np.asarray(normalizer[key]) for key in ("offset", "scale"))
    visible, ref_v, cal_v = ((x-offset)/scale for x in (vis, ref_u, cal_u))
    result = state_result(zs, visible, new_n, round_ref_n)
    result["original_reference"] = empirical_moments(ref_z, ref_v)[0]
    result["original_calibration"] = empirical_moments(cal_z, cal_v)[0]
    np.testing.assert_allclose(s, ref_v.var(axis=0), atol=2e-12, rtol=1e-10)
    fit = np.linalg.lstsq(np.column_stack([np.ones(len(cal_z)), cal_z]), cal_u, rcond=None)[0]
    np.testing.assert_allclose(fit[0], offset, atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(fit[1:].diagonal(), scale, atol=2e-12, rtol=1e-10)
    return result


def ref_components(s, moments):
    c = np.asarray(moments["covariance_z_e"])
    v = np.asarray(moments["residual_variance"])
    delta = s - (1 + 2*c + v)
    return np.stack((2*c, v, delta))


def cell_arrays(parts, prefix):
    return {
        field: parts[prefix + field]
        for field in ("physical_scores", "cross_over_reference", "energy_over_reference", "reference_accounting")
    }


def window_diagnostic(parts, config, stats, s):
    window = config["window"]
    key = f"{window['stage']}__mlp__{window['process']}__{window['model']}__"
    row = window["row"]
    values = {field: array[row] for field, array in cell_arrays(parts, key).items()}
    d = values["physical_scores"]
    fast, slow = window["fast_axis"] - 1, window["slow_axis"] - 1
    ref = ref_components(s, stats["pooled"])
    per_axis = -d[None, :] / s[None, :] * ref
    per_axis_empirical = -d[None,:] / s[None,:] * np.stack([
        np.asarray(stats["original_reference"]["physical_sample_variance"]) - 1,
        2*np.asarray(stats["original_reference"]["covariance_z_e"]),
        stats["original_reference"]["residual_variance"],
    ])
    for terms in (per_axis, per_axis_empirical):
        np.testing.assert_allclose(terms.sum(axis=0), values["reference_accounting"], atol=1e-12, rtol=0)
    axis_coeff = np.zeros(4)
    axis_coeff[fast], axis_coeff[slow] = -d[fast]/s[fast], d[slow]/s[slow]
    cov = np.asarray(stats["pooled"]["influence_covariance_of_pooled_c_then_v"])
    matrices = np.array([
        np.concatenate([2*axis_coeff, np.zeros(4)]),
        np.concatenate([np.zeros(4), axis_coeff]),
        np.concatenate([-2*axis_coeff, -axis_coeff]),
    ])
    intervals = {}
    for i, field in enumerate(("2c", "v", "delta_ref")):
        value = float(per_axis[i, fast] - per_axis[i, slow])
        se = float(np.sqrt(matrices[i] @ cov @ matrices[i]))
        intervals[field] = {
            "gap": value, "integration_se": se,
            "interval95": [value - 1.959963984540054*se, value + 1.959963984540054*se],
            "per_axis": per_axis[i].tolist(),
        }
    return {
        "window": window,
        "archived_terms_by_axis": {field: x.tolist() for field, x in values.items()},
        "archived_gaps": {field: float(x[fast] - x[slow]) for field, x in values.items()},
        "estimated_population_ref_split": intervals,
        "same_empirical_ref_gap_split": dict(zip(
            ("physical_sample_variance_minus_one", "empirical_2c", "empirical_v"),
            (per_axis_empirical[:, fast] - per_axis_empirical[:, slow]).tolist(),
        )),
    }


def sums(array):
    d = array["physical_scores"]
    ordering = np.argsort(d, axis=1, kind="stable")
    fields = {key: value for key, value in array.items() if key != "physical_scores"}
    result = {}
    for name, term in fields.items():
        ordered = np.take_along_axis(term, ordering, axis=1)
        gaps = ordered[:, None, 2:] - ordered[:, :2, None]
        result[name] = {
            "all_axis_square_sum": float(np.square(term).sum()),
            "cross_boundary_gap_square_sum": float(np.square(gaps).sum()),
        }
    return result


def run():
    start = time.monotonic()
    config = read(HERE / "protocol.json")
    frozen_protocol = read(OLD / "protocol.json")
    digest_receipt = read(OLD / "forecast/RECEIPT.json")
    digest_summary = read(OLD / "SUMMARY.json")
    assert digest(OLD / "SCORE_COMPONENTS.npz") == digest_summary["components_sha256"]
    assert digest(OLD / "forecast/RECEIPT.json") == digest_summary["forecast_receipt_sha256"]
    assert digest(OLD / "REVEAL.npz") == digest_summary["label_sha256"]
    _, identity, cells = loaders.load_panel()
    cells = [cell for cell in cells if cell["layer"] == "mlp"]
    names = list(dict.fromkeys(cell["name"] for cell in cells))
    assert len(names) == 14 and len(cells) == 28
    provenance = loaders.sources_manifest()
    new_z = np.concatenate([
        iid(master, config["new_iid_role"], config["new_states_per_panel"])
        for master in config["new_masters"]
    ])
    rounds = frozen_protocol["rounds"]
    cal_z = {stage: iid(master, config["original_calibration_role"], frozen_protocol["n_state_cal"]) for stage, master in rounds.items()}
    ref_z = {stage: iid(master, config["original_reference_role"], frozen_protocol["n_reference"]) for stage, master in rounds.items()}
    models, archives = {}, {}
    for name in names:
        cell = next(cell for cell in cells if cell["name"] == name)
        readout, _ = loaders.load_readout(cell, identity)
        visible = infer(readout, new_z, config["forward_chunk"])
        models[name] = {}
        for stage in rounds:
            stage_meta = {}
            for matching in (cell for cell in cells if cell["name"] == name):
                key = f'mlp__{matching["process"]}__{name}'
                path = OLD / "forecast" / stage / f"{key}.json"
                assert digest(path) == digest_receipt["files"][str(path.relative_to(OLD))]
                stage_meta[matching["process"]] = read(path)
                archives[(stage, matching["process"], name)] = stage_meta[matching["process"]]
            same = list(stage_meta.values())
            assert same[0]["normalizer"] == same[1]["normalizer"] and same[0]["reference_variance"] == same[1]["reference_variance"]
            normalizer, s = same[0]["normalizer"], np.asarray(same[0]["reference_variance"])
            models[name][stage] = diagnostic_cell(
                new_z, ref_z[stage], cal_z[stage], visible,
                infer(readout, ref_z[stage], config["forward_chunk"]),
                infer(readout, cal_z[stage], config["forward_chunk"]),
                normalizer, s, config["new_states_per_panel"], frozen_protocol["n_reference"],
            )
        print(f"verified moments: {name}", flush=True)
    window = config["window"]
    s = np.asarray(archives[(window["stage"], window["process"], window["model"])]["reference_variance"])
    with np.load(OLD / "SCORE_COMPONENTS.npz", allow_pickle=False) as components:
        example = window_diagnostic(components, config, models[window["model"]][window["stage"]], s)
        aggregate = {}
        for stage in rounds:
            tallies = []
            for process, name in ((cell["process"], cell["name"]) for cell in cells):
                s = np.asarray(archives[(stage, process, name)]["reference_variance"])
                arrays = cell_arrays(components, f"{stage}__mlp__{process}__{name}__")
                d = arrays["physical_scores"]
                ref = ref_components(s, models[name][stage]["pooled"])
                for i, label in enumerate(("ref_from_2c", "ref_from_v", "ref_from_delta")):
                    arrays[label] = -d/s*ref[i]
                np.testing.assert_allclose(
                    arrays["ref_from_2c"] + arrays["ref_from_v"] + arrays["ref_from_delta"],
                    arrays["reference_accounting"], atol=1e-12, rtol=0,
                )
                tallies.append(sums(arrays))
            count = len(cells) * frozen_protocol["n_test"] * 4
            aggregate[stage] = {
                term: {
                    key.replace("_square_sum", "_rms"): float(np.sqrt(sum(t[term][key] for t in tallies) / count))
                    for key in tallies[0][term]
                }
                for term in tallies[0]
            }
    dependencies = [HERE / name for name in ("run.py", "protocol.json", "PROTOCOL.md")]
    dependencies += [OLD / name for name in ("protocol.json", "EXAMPLES.json", "SCORE_COMPONENTS.npz", "REVEAL.npz", "SUMMARY.json", "forecast/RECEIPT.json")]
    provenance.update({str(path.relative_to(REPO)): digest(path) for path in dependencies})
    assert all(digest(REPO / name) == expected for name, expected in provenance.items())
    output = {
        "status": "post-reveal diagnostic; independently sampled state moments; no new TEST evaluation",
        "config": config,
        "sources": provenance,
        "moments": models,
        "archived_window_diagnostic": example,
        "all_archived_windows_descriptive": aggregate,
        "elapsed_seconds": time.monotonic() - start,
    }
    with (HERE / "SUMMARY.json").open("x") as result:
        json.dump(output, result, indent=2, allow_nan=False)
        result.write("\n")


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    run()
