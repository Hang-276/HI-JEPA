#!/usr/bin/env python3
"""Capture and verify read-only SHA-256 baseline for 2026-11-17 local closeout."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/closeout_20261117"

PROTECTED_DIRS = [
    "reports",
    "work/hcf_stable_id_20261027",
    "archive",
    "src",
    "configs",
    "tests",
    "results/theory_checks_20261116",
    "results/exploratory_20261116",
    "results/fourroom_hjepa",
    "results/visual_sprites_lejepa",
    "results/low_dim_ar1",
    "results/legacy_fine",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def enumerate_protected(include_existing_docs_scripts: bool = True, pinned_files: list[str] | None = None) -> list[dict]:
    records: dict[str, Path] = {}
    for folder in PROTECTED_DIRS:
        base = ROOT / folder
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.is_symlink():
                records[str(p.relative_to(ROOT))] = p
    if pinned_files is not None:
        for rel in pinned_files:
            p = ROOT / rel
            if p.is_file() and not p.is_symlink():
                records[rel] = p
    elif include_existing_docs_scripts:
        for folder in ("docs", "scripts"):
            base = ROOT / folder
            for p in sorted(base.rglob("*")):
                rel = str(p.relative_to(ROOT))
                if rel.startswith("scripts/closeout_20261117/") or rel.endswith("_20261117.md"):
                    continue
                if p.is_file() and not p.is_symlink():
                    records[rel] = p
    return [
        {"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for rel, path in sorted(records.items())
    ]


def snapshot(baseline_path: Path) -> None:
    t0 = time.perf_counter()
    files = enumerate_protected(include_existing_docs_scripts=True)
    payload = {
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protected_dirs": PROTECTED_DIRS,
        "file_count": len(files),
        "total_bytes": sum(r["bytes"] for r in files),
        "scan_seconds": time.perf_counter() - t0,
        "files": files,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    baseline_path.write_text(text)
    (OUT / "protected_before.json").write_text(text)
    print(json.dumps({k: v for k, v in payload.items() if k != "files"}, ensure_ascii=False))


def verify(baseline_path: Path) -> None:
    t0 = time.perf_counter()
    before_bytes = baseline_path.read_bytes()
    before = json.loads(before_bytes)
    prev = {r["path"]: r for r in before["files"]}
    pinned_extra = [p for p in prev if p.startswith("docs/") or p.startswith("scripts/")]
    curr_list = enumerate_protected(include_existing_docs_scripts=False, pinned_files=pinned_extra)
    curr = {r["path"]: r for r in curr_list}

    changed = sorted(k for k in (prev.keys() & curr.keys()) if prev[k] != curr[k])
    missing = sorted(prev.keys() - curr.keys())
    added = sorted(curr.keys() - prev.keys())
    scan_seconds = time.perf_counter() - t0

    if changed or missing or added:
        raise RuntimeError(f"Integrity violation! changed={changed[:5]}, missing={missing[:5]}, added={added[:5]}")

    py_checked = []
    for p in sorted((ROOT / "scripts/closeout_20261117").glob("*.py")):
        ast.parse(p.read_text(), filename=str(p), feature_version=(3, 11))
        py_checked.append(str(p.relative_to(ROOT)))

    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t_repo = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-B", "scripts/check_repository.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    repo_res = json.loads(proc.stdout)
    if repo_res.get("failures"):
        raise RuntimeError(f"check_repository.py reported failures: {repo_res['failures']}")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "PASSED",
        "verified_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "baseline_sha256": hashlib.sha256(before_bytes).hexdigest(),
        "protected_files": len(curr_list),
        "protected_bytes": sum(r["bytes"] for r in curr_list),
        "changed": changed,
        "missing": missing,
        "added": added,
        "hash_scan_seconds": scan_seconds,
        "python311_syntax_checked": py_checked,
        "repository_check": {
            "seconds": time.perf_counter() - t_repo,
            "result": repo_res,
        },
    }
    (OUT / "INTEGRITY_CLOSEOUT_20261117.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("snapshot", "verify"))
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/hijepa_closeout_20261117_before.json"))
    args = parser.parse_args()
    if args.mode == "snapshot":
        snapshot(args.baseline)
    else:
        verify(args.baseline)


if __name__ == "__main__":
    main()
