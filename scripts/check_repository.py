"""Read-only checks. Never import or execute experiment modules."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    counts = {"python": 0, "shell": 0, "json": 0, "links": 0}
    failures = []
    for directory in ("src", "scripts", "tests", "configs"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                if path.suffix == ".py":
                    ast.parse(path.read_text(), filename=str(path))
                    counts["python"] += 1
                elif path.suffix == ".sh":
                    subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
                    counts["shell"] += 1
                elif path.suffix == ".json":
                    json.loads(path.read_text())
                    counts["json"] += 1
            except (SyntaxError, ValueError, OSError, subprocess.CalledProcessError) as error:
                failures.append({"path": str(path.relative_to(ROOT)), "error": str(error)})
    for directory in ("configs", "results", "experiments", "docs"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_symlink():
                counts["links"] += 1
                if not path.exists():
                    failures.append({"path": str(path.relative_to(ROOT)), "error": "broken navigation link"})
    for relative in ("reports", "work/hcf_stable_id_20261027"):
        path = ROOT / relative
        if not path.is_dir() or path.is_symlink():
            failures.append({"path": relative, "error": "sealed physical path must remain in place"})
    old = ROOT / "archive/releases/hjepa_slow_selection_20261115"
    equal = {}
    for before, after in ((old / "metrics.py", ROOT / "src/slowrepr/metrics.py"),
                          (old / "config.json", ROOT / "configs/fourroom_hjepa/default.json")):
        if before.exists():
            same = before.read_bytes() == after.read_bytes()
            equal[str(after.relative_to(ROOT))] = same
            if not same:
                failures.append({"path": str(after.relative_to(ROOT)), "error": "changed from archived delivery"})
    # Check frozen source bytes without importing a historical module.
    index = ROOT / "docs/audit/20261116/SEALED_SOURCE_CHECKS.json"
    checked = {}
    if index.exists():
        for record in json.loads(index.read_text())["records"]:
            receipt_path = ROOT / record["receipt"]
            receipt = json.loads(receipt_path.read_text())
            # These three forecast implementations seal paths relative to HERE;
            # panel/local/B source locks explicitly use the repository root.
            local_names = {"lejepa_a2_readout_20261115", "lejepa_a3_gap_20261115",
                           "lejepa_a4_20261115"}
            base = receipt_path.parent if receipt_path.parent.name in local_names else ROOT
            for relative, expected in receipt.get("sources", {}).items():
                path = (base / relative).resolve()
                key = str(path.relative_to(ROOT))
                if key in checked:
                    if checked[key] != expected:
                        failures.append({"path": key, "error": "conflicting sealed digests"})
                    continue
                if not path.is_file():
                    failures.append({"path": key, "error": "sealed input absent; restore original artifacts"})
                    continue
                with path.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                checked[key] = expected
                if actual != expected:
                    failures.append({"path": key, "error": "sealed source digest mismatch"})
    print(json.dumps({"checks": counts, "sealed_sources": len(checked),
                      "unchanged_science_files": equal, "failures": failures,
                      "model_or_metric_execution": False}, indent=2, ensure_ascii=False))
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
