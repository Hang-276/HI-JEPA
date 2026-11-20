#!/usr/bin/env python3
"""Copy the sealed A4 training inputs to a fresh external working directory.

Standard library only. This program never imports research modules or launches
training, feature extraction, evaluation, or a subprocess.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
PANEL = PurePosixPath("reports/lejepa_a4_20261115/panel")
RECEIPT = ROOT / PANEL / "RECEIPT.json"
RECEIPT_SHA256 = "6a19ccf92e28ca1dd9dee1a05b18e9ea572adc0d0d1bddb52e2255c7522a50c8"
PANEL_INPUTS = {str(PANEL / name) for name in ("run.py", "protocol.json", "PROTOCOL.md")}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(destination):
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination must not already exist, including a symbolic link")
    lexical = destination.absolute()
    if lexical.is_relative_to(ROOT) or any(
        parent.exists() and parent.samefile(ROOT) for parent in lexical.parents
    ):
        raise ValueError("Destination must be outside this research repository")
    destination = destination.resolve()
    if destination.is_relative_to(ROOT):
        raise ValueError("Destination must be outside this research repository")
    receipt_bytes = RECEIPT.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != RECEIPT_SHA256:
        raise ValueError("A4 receipt differs from the audited sealed receipt")

    sources = json.loads(receipt_bytes)["sources"]
    if len(sources) != 13 or not PANEL_INPUTS.issubset(sources):
        raise ValueError("Unexpected A4 source manifest; inspect it before preparing a copy")
    for name, expected in sources.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in {"reports", "work"}:
            raise ValueError(f"Unsafe source path: {name}")
        if relative.is_relative_to(PANEL) and name not in PANEL_INPUTS:
            raise ValueError(f"Refusing to copy an existing A4 output: {name}")
        source = ROOT / relative
        if not source.resolve().is_relative_to(ROOT):
            raise ValueError(f"Source resolves outside this repository: {name}")
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Missing or changed sealed input: {name}")

    # Validate every input before creating anything. Preserve the original
    # reports/work layout because the frozen code uses path names in its locks.
    destination.mkdir(parents=True, exist_ok=False)
    for name, expected in sources.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        if sha256(target) != expected:
            raise ValueError(f"Copied input failed SHA256 verification: {name}")

    manifest = {
        "status": "PREPARED_ONLY_NOT_EXECUTED",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": str(ROOT),
        "source_receipt": str(PANEL / "RECEIPT.json"),
        "source_receipt_sha256": RECEIPT_SHA256,
        "files_sha256": sources,
        "extra_import_dependencies": [],
        "scope": "Fresh A4 encoder/readout panel training inputs; no A4 checkpoints or outputs copied",
        "validation": "File hashes and static dependency review only; runtime has not been tested by this preparer",
    }
    with (destination / "PREPARED.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"Prepared {len(sources)} verified input files at {destination}")
    print("No model or experiment has been run.")
    print("After separately authorizing training and installing NumPy/PyTorch, run:")
    print(f"cd {shlex.quote(str(destination))}")
    print(f"{shlex.quote(sys.executable)} {PANEL}/run.py > {PANEL}/run.log 2>&1")
    print("This trains and audits the A4 panel only; it does not run the slow-selection TEST comparison.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True,
                        help="A new directory outside the repository; existing directories are refused")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required by the sealed A4 code and this preparer")
    try:
        prepare(args.destination.expanduser())
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Preparation failed: {error}\nNo model was run. If copying had begun, keep the incomplete destination for inspection.\n")


if __name__ == "__main__":
    main()
