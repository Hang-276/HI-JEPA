"""Reuse the frozen CPU Fine adapter for the two replication slots."""

from dataclasses import replace
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from types import ModuleType


SLOTS = ("ar_rich_fresh2", "fhn_fresh_eps05")
BASE_ADAPTER = Path(__file__).resolve().parents[1] / "fine_independent_validation_20261109/fine_forward.py"
BASE_SHA256 = "27033a86d79f138a32de5f2711912b86be7d758f54b7b5151ec0ae5331cf464a"


@lru_cache(maxsize=1)
def _adapter():
    source = BASE_ADAPTER.read_bytes()
    if hashlib.sha256(source).hexdigest() != BASE_SHA256:
        raise RuntimeError("Archived Fine adapter differs from the primary study lock")
    name = "_hcf_replication_fine_adapter"
    module = ModuleType(name)
    module.__file__ = str(BASE_ADAPTER)
    sys.modules[name] = module
    exec(compile(source, str(BASE_ADAPTER), "exec"), module.__dict__)
    # The independently loaded namespace keeps the primary study unchanged.
    module.SLOTS = frozenset(SLOTS)
    return module


def load_fine(slot, rep):
    """Restore the CAL-selected frozen Fine for replication slot and seed."""
    fine = _adapter().load_fine(slot, rep)
    provenance = {
        **fine.provenance,
        "base_adapter": {"path": str(BASE_ADAPTER), "sha256": BASE_SHA256},
    }
    return replace(fine, provenance=provenance)


def smoke(slot, rep, fine=None):
    """Verify one existing TRAIN trajectory against archived Fine features."""
    loaded = load_fine(slot, rep) if fine is None else fine
    return _adapter().smoke(slot, rep, fine=loaded)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", choices=SLOTS, default=SLOTS[0])
    parser.add_argument("--rep", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    print(json.dumps(smoke(args.slot, args.rep), indent=2, allow_nan=False))
