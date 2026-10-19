"""Portable source-checkout entry point; no editable installation required."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from hijepa.cli import main

if __name__ == "__main__":
    main()
