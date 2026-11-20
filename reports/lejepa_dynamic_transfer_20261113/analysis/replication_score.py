"""Run the unchanged analysis on an independent TEST and REF forecast archive."""

from pathlib import Path

import score as original
import residual_diagnostic as residual


HERE = Path(__file__).resolve().parent
REPLICATION = original.EXP / "replication"
OUT = HERE / "replication"


def run():
    config, training, history = original.inputs()
    replication_config = original.read(REPLICATION / "protocol.json")
    if replication_config["master"] == config["master"]:
        raise ValueError("Replication requires a separate data stream.")
    lock = original.read(HERE / "VAL_MATCH_LOCK.json")
    expected = original.dependencies()
    if any(lock[key] != value for key, value in expected.items()):
        raise ValueError("Original analysis lock changed.")
    writer = original.write
    OUT.mkdir(exist_ok=True)

    def write_replication(path, result):
        writer(OUT / path.name, result)

    original.inputs = lambda: (config, training, history)
    original.dependencies = lambda: expected
    original.EXP = REPLICATION
    original.write = write_replication
    residual.EXP = REPLICATION
    residual.inputs = original.inputs
    residual.write = write_replication
    original.score()
    residual.run()


if __name__ == "__main__":
    run()
