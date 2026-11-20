"""Test selective window ranking with one unobserved Gaussian AR factor."""

import json
from pathlib import Path

import numpy as np


RATES = (0.90, 0.88, 0.85, 0.80)
LAGS = (4, 16, 32)
LENGTHS = (64, 192, 512, 1024)
CALIBRATION_SIZE = 32768
EVALUATION_SIZE = 8192
CALIBRATION_FAILURE = 0.01
ANSWER_RISK = 0.05
SEED = 804137
BATCH_SIZE = 512


def trajectory_scores(generator, rates, length, count):
    rates = np.array(rates)
    innovation_scale = np.sqrt(1 - rates**2)
    scores = np.empty((count, rates.size))
    for start in range(0, count, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, count)
        paths = generator.standard_normal((length, stop - start, rates.size))
        for step in range(1, length):
            paths[step] = rates * paths[step - 1] + innovation_scale * paths[step]
        scores[start:stop] = np.mean(
            [np.mean((paths[lag:] - paths[:-lag]) ** 2, axis=0) for lag in LAGS],
            axis=0,
        )
    return scores


def evaluate_length(length, stream):
    calibration_stream, visible_stream, hidden_stream = stream.spawn(3)
    hidden_calibration = np.sort(
        trajectory_scores(
            np.random.default_rng(calibration_stream),
            RATES[-1:],
            length,
            CALIBRATION_SIZE,
        )[:, 0]
    )
    visible = trajectory_scores(
        np.random.default_rng(visible_stream), RATES[:-1], length, EVALUATION_SIZE
    )
    second = np.sort(visible, axis=1)[:, 1]
    probability = np.searchsorted(hidden_calibration, second, side="right")
    probability = probability / CALIBRATION_SIZE
    cdf_error = np.sqrt(np.log(2 / CALIBRATION_FAILURE) / (2 * CALIBRATION_SIZE))
    exclude = probability + cdf_error <= ANSWER_RISK
    include = probability - cdf_error >= 1 - ANSWER_RISK
    accept = include | exclude
    full_prediction = probability >= 0.5

    hidden = trajectory_scores(
        np.random.default_rng(hidden_stream), RATES[-1:], length, EVALUATION_SIZE
    )[:, 0]
    hidden_enters = hidden < second
    wrong_when_accepted = (exclude & hidden_enters) | (include & ~hidden_enters)
    return {
        "length": length,
        "evaluation_windows": EVALUATION_SIZE,
        "hidden_enters": int(np.count_nonzero(hidden_enters)),
        "forced_decision_errors": int(np.count_nonzero(full_prediction != hidden_enters)),
        "accepted": int(np.count_nonzero(accept)),
        "include_answers": int(np.count_nonzero(include)),
        "exclude_answers": int(np.count_nonzero(exclude)),
        "incorrect_answers": int(np.count_nonzero(wrong_when_accepted)),
    }


def main():
    streams = np.random.SeedSequence(SEED).spawn(len(LENGTHS))
    results = {
        "rates": RATES,
        "lags": LAGS,
        "calibration_size_per_length": CALIBRATION_SIZE,
        "calibration_failure_per_length": CALIBRATION_FAILURE,
        "conditional_answer_risk_given_calibration_event": ANSWER_RISK,
        "rows": [
            evaluate_length(length, stream) for length, stream in zip(LENGTHS, streams)
        ],
    }
    destination = Path(__file__).with_name("MISSING_AXIS_RESULTS.json")
    with destination.open("x") as output:
        json.dump(results, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
