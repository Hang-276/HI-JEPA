"""Describe equal-coverage errors within the preselected high-R² checkpoints."""

import json
from pathlib import Path

import numpy as np

from score import top2


HERE = Path(__file__).resolve().parent


def main():
    pairs = json.loads((HERE / "R2_MATCH_LOCK.json").read_text())["pairs"]
    paired_seeds = [pair["seed"] for pair in pairs if pair["included"]]
    with (
        np.load(HERE / "forecasts.npz", allow_pickle=False) as forecast,
        np.load(HERE / "reveal.npz", allow_pickle=False) as truth,
    ):
        np.testing.assert_array_equal(forecast["test_bank"], truth["test_bank"])
        bank = forecast["test_bank"]
        case_index = forecast["bank_case"][bank]
        seed_index = forecast["bank_model_seed"][bank]
        score = forecast["test_scores"]
        ordered = np.sort(score, axis=1)
        gap = ordered[:, 2] - ordered[:, 1]
        joint = forecast["test_uncertainty"]
        wrong = np.any(top2(score) != top2(truth["test_physical_scores"]), axis=1)
        groups = []
        for case in ("separated_wide", "separated_narrow"):
            case_mask = case_index == forecast["case_names"].tolist().index(case)
            for seed in paired_seeds:
                indices = np.flatnonzero(case_mask & (seed_index == seed))
                accepted = len(indices) // 5
                errors = {}
                for name, ranks in (("gap_only", gap), ("joint", joint)):
                    selected = indices[np.argsort(-ranks[indices], kind="stable")[:accepted]]
                    errors[name] = int(wrong[selected].sum())
                groups.append(
                    {
                        "case": case,
                        "seed": seed,
                        "windows": len(indices),
                        "accepted_per_rule": accepted,
                        "errors": errors,
                    }
                )
    output = {"scope": "Descriptive top 20% within each frozen checkpoint", "groups": groups}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
