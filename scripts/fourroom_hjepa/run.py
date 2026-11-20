"""Frozen visual H-JEPA: physical-factor recovery and selective slow-axis choice."""

import argparse
from pathlib import Path
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Unique result folder name")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Engineering check only; never scientific evidence")
    parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from slowrepr.fourroom_hjepa import main

    main()
