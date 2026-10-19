"""Dispatch canonical experiment subsets without changing scientific protocols."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def frozen_json(path: Path, value):
    if path.exists() and read_json(path) != value:
        raise RuntimeError(f"Existing launch protocol differs: {path}. Choose a new output directory.")
    write_json(path, value)


def cli(*arguments, log: Path | None = None):
    command = [sys.executable, str(REPO_ROOT / "run.py"), *map(str, arguments)]
    if log is None:
        subprocess.run(command, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(command, check=True, stdout=stream)


def select_matrix(experiment: str, profile: str):
    source = REPO_ROOT / "configs" / f"matrix_{profile}.json"
    canonical = read_json(source)
    by_name = {job["name"]: job for job in canonical["jobs"]}
    requested = [job["name"] for job in canonical["jobs"]
                 if experiment == "all" or job["name"].startswith(experiment + "_")]
    if not requested:
        raise RuntimeError(f"No canonical jobs match {experiment}.")
    selected = set(requested)
    reasons = {}

    def include(name: str, reason: str):
        if name not in by_name:
            raise RuntimeError(f"Canonical matrix lacks required dependency {name}.")
        selected.add(name)
        reasons.setdefault(name, []).append(reason)

    if profile == "formal":
        include("e1_baseline", "predeclared formal anti-collapse acceptance")
    while True:
        previous = set(selected)
        for name in list(selected):
            source_name = by_name[name].get("reuse_from")
            if source_name:
                include(source_name, f"result reused by {name}")
        if selected == previous:
            break
    reasons = {name: list(dict.fromkeys(items)) for name, items in reasons.items()}
    jobs = [copy.deepcopy(job) for job in canonical["jobs"] if job["name"] in selected]
    seen = set()
    for job in jobs:
        dependency = job.get("reuse_from")
        if dependency and dependency not in seen:
            raise RuntimeError(f"Canonical order must put {dependency} before {job['name']}.")
        seen.add(job["name"])
    matrix = copy.deepcopy(canonical)
    matrix["base"] = "base.json"
    matrix["jobs"] = jobs
    selection = {
        "experiment": experiment, "profile": profile,
        "stage": "formal" if profile == "formal" else "local_integration",
        "device": "cuda" if profile == "formal" else "cpu",
        "canonical_matrix": str(source.relative_to(REPO_ROOT)),
        "requested_jobs": requested,
        "dependency_jobs": [name for name in by_name if name in selected and name not in requested],
        "dependency_reasons": reasons,
        "selected_job_order": [job["name"] for job in jobs],
        "seeds": canonical.get("seeds", [0, 1, 2]),
        "result_reuse": [{"job": job["name"], "reuse_from": job["reuse_from"],
                          "independent_replication": False}
                         for job in jobs if job.get("reuse_from")],
        "resume": True, "cross_invocation_result_reuse": False, "protocol_changes": [],
    }
    return matrix, selection


def formal_acceptance(configs: list[dict], output: Path):
    """Complete existing acceptance seeds before dispatching other formal jobs.

    Individual output directories match the main matrix. Its --resume reuses
    these canonical results, without adding an acceptance training experiment.
    """
    baseline = [cfg for cfg in configs if cfg["name"].startswith("e1_baseline_seed")]
    required = baseline[0]["anticollapse"]["acceptance_seeds"]
    selected = {cfg["training"]["seed"]: cfg for cfg in baseline}
    if len(required) != len(set(required)) or any(seed not in selected for seed in required):
        raise RuntimeError("Canonical formal matrix does not cover the declared acceptance seeds.")
    records = []
    for seed in required:
        cfg = selected[seed]
        config_path = output / "launch" / f"{cfg['name']}.json"
        frozen_json(config_path, cfg)
        print(f"[formal acceptance] {cfg['name']} (canonical budget; resume enabled)", flush=True)
        try:
            cli("run", "--config", config_path, "--output", output / cfg["name"], "--resume")
            result = read_json(output / cfg["name"] / "result.json")
            record = {"name": cfg["name"], "seed": seed, "status": result.get("status"),
                      "anticollapse": result.get("anticollapse")}
            records.append(record)
        except (subprocess.CalledProcessError, OSError, ValueError) as error:
            records.append({"name": cfg["name"], "seed": seed, "status": "failed", "error": str(error)})
        acceptance = {"passed": len(records) == len(required) and all(
            item.get("status") == "completed" and bool((item.get("anticollapse") or {}).get("passed", False))
            for item in records), "seeds": required, "records": records,
            "scope": "canonical baseline runs; no additional experiments or threshold changes"}
        write_json(output / "prelaunch_anticollapse_acceptance.json", acceptance)
        if records[-1].get("status") != "completed":
            raise RuntimeError("Formal baseline acceptance could not complete; no other formal tasks were launched.")
    if not acceptance["passed"]:
        raise RuntimeError("The existing formal anti-collapse gate did not pass; no other formal tasks were launched.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("e1", "e2", "e3", "all"))
    parser.add_argument("profile", choices=("formal", "local"), nargs="?", default="formal")
    parser.add_argument("output", nargs="?", help="Results directory; relative paths use your current directory")
    parser.add_argument("--manifest-only", action="store_true", help="Write manifests without device checks or training")
    args = parser.parse_args(argv)
    # Parse --help before loading scientific dependencies.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from hijepa.config import expand_matrix, load_config
    default_name = args.profile if args.experiment == "all" else f"{args.profile}_{args.experiment}"
    output = Path(args.output).expanduser().resolve() if args.output else REPO_ROOT / "runs" / default_name
    matrix, selection = select_matrix(args.experiment, args.profile)
    output.mkdir(parents=True, exist_ok=True)
    subset_path = output / "launch" / "matrix.json"
    canonical_source = REPO_ROOT / selection["canonical_matrix"]
    canonical_base = canonical_source.parent / read_json(canonical_source)["base"]
    frozen_json(output / "launch" / "base.json", load_config(canonical_base))
    frozen_json(subset_path, matrix)
    configs = expand_matrix(subset_path)
    selection["run_count"] = len(configs)
    if args.profile == "formal":
        baseline = next(cfg for cfg in configs if cfg["name"].startswith("e1_baseline_seed"))
        selection["formal_acceptance"] = {
            "baseline_job": "e1_baseline", "seeds": baseline["anticollapse"]["acceptance_seeds"],
            "canonical_thresholds": baseline["anticollapse"], "runs_before_other_jobs": True,
            "main_matrix_resumes_same_result_directories": True,
        }
    frozen_json(output / "launch" / "selection.json", selection)
    cli("manifest", "--config", subset_path, "--output", output / "planned_manifest.json",
        log=output / "launch" / "manifest_command.log")
    manifest = read_json(output / "planned_manifest.json")
    manifest["launch"] = selection
    write_json(output / "planned_manifest.json", manifest)
    print(json.dumps({"experiment": args.experiment, "profile": args.profile,
                      "run_count": len(configs), "requested_jobs": len(selection["requested_jobs"]),
                      "dependency_jobs": selection["dependency_jobs"], "output": str(output),
                      "manifest_only": args.manifest_only}, ensure_ascii=False, indent=2), flush=True)
    if args.manifest_only:
        return
    if args.profile == "formal":
        if not sys.platform.startswith("linux"):
            raise RuntimeError("Formal experiments require Linux + CUDA. Use local or --manifest-only on this machine.")
        parity_payload = Path(os.environ.get("HIJEPA_PARITY_BUNDLE", str(REPO_ROOT / "validation" / "parity_bundle" / "bundle.pt"))).expanduser().resolve()
        if not parity_payload.is_file():
            raise FileNotFoundError(f"Parity bundle not found: {parity_payload}. Set HIJEPA_PARITY_BUNDLE to the migrated bundle.pt.")
        cli("doctor", "--device", "cuda", "--output", output / "doctor_cuda.json")
        cli("parity-check", "--bundle", parity_payload, "--device", "cuda", "--output", output / "parity_cuda.json")
        formal_acceptance(configs, output)
    if args.experiment in ("e2", "all"):
        cli("audit-pseudosolutions", "--output", output / "pseudo_solution_audit.json")
    # Local always dispatches the existing CPU integration profile.
    cli("matrix", "--config", subset_path, "--output", output, "--resume")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"HI-JEPA launcher: {error}", file=sys.stderr)
        raise SystemExit(1)
