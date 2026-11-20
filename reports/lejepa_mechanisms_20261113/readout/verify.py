"""Verify preserved readout comparisons and replay scoring from saved parameters."""

import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("readout_run", Path(__file__).with_name("run.py"))
readout_run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = readout_run
spec.loader.exec_module(readout_run)

from readout_run import (
    HERE,
    MODES,
    RULES,
    TRAINING,
    datasets,
    digest,
    inputs,
    make_mlp_encoder,
    mlp,
    predict_mlp,
    predict_ridge,
    task_metrics,
)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    archive = HERE / "runs"
    receipt = json.loads((archive / "results.json").read_text())
    config = receipt["config"]
    assert receipt["protocol_sha256"] == digest(HERE / "protocol.json")
    assert receipt["source_sha256"] == digest(HERE / "run.py")
    parent_config = json.loads((TRAINING / "protocol.json").read_text())
    cases = parent_config["cases"]
    frozen = [row for row in receipt["runs"].values() if row["mode"] == MODES[0]]
    assert len(frozen) == 16
    assert len(receipt["runs"]) == 22
    all_sets = {
        case: datasets(config, np.asarray(rates)) for case, rates in cases.items()
    }
    outputs_checked = 0
    for name, panel in receipt["runs"].items():
        assert json.loads((archive / f"{name}.json").read_text()) == panel
        weights = torch.load(archive / f"{name}_readouts.pt", weights_only=False)
        encoder = None
        if panel["mode"] == "frozen":
            checkpoint = TRAINING / "runs" / f"{name}.pt"
            assert digest(checkpoint) == panel["encoder_sha256"]
            encoder = make_mlp_encoder(4, parent_config["hidden"], parent_config["layers"], device="cpu").eval()
            encoder.load_state_dict(torch.load(checkpoint, weights_only=True))

        def encode(z):
            h = inputs(panel["mode"], z, encoder).reshape(-1, 4)
            return (h - weights["input_mean"]) / weights["input_std"]

        ref = encode(all_sets[panel["case"]]["reference"])
        with np.load(archive / f"{name}.npz", allow_pickle=False) as windows:
            for rule in RULES:
                if rule == "mlp":
                    decoder = mlp(4, config["mlp_width"]).eval()
                    decoder.load_state_dict(weights[rule])
                    predict = lambda x: predict_mlp(x, decoder)
                else:
                    predict = lambda x: predict_ridge(x, weights[rule], rule == "cubic")
                predicted_ref = predict(ref)
                evaluation = panel["readouts"][rule]["evaluation"]
                assert len(evaluation) == 2
                for eval_case, row in evaluation.items():
                    physical = all_sets[eval_case]["test"]
                    predicted = predict(encode(physical)).reshape(physical.shape)
                    stats, arrays = task_metrics(predicted, predicted_ref, physical, config["lags"])
                    assert stats == row["task"], (name, rule, eval_case)
                    for key, values in arrays.items():
                        assert np.array_equal(values, windows[f"{eval_case}_{rule}_{key}"])
                    outputs_checked += len(physical)
    print(f"Replayed {outputs_checked} saved window-by-rule predictions from all 22 fitted panels.")


if __name__ == "__main__":
    main()
