"""Verify the sealed analysis adapter and invoke it on the replication archive."""

from pathlib import Path
import sys


ANALYSIS = Path(__file__).resolve().parents[2] / "analysis"
sys.path.insert(0, str(ANALYSIS))
from score import digest
from replication_score import run


if __name__ == "__main__":
    current = digest(ANALYSIS / "replication_score.py")
    expected = (ANALYSIS / "REPLICATION_ANALYSIS_SHA256SUMS").read_text().split()[0]
    if current != expected:
        raise ValueError("Frozen replication adapter changed.")
    run()
