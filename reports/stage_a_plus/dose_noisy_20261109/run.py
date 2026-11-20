"""Independent CPU verification of the unknown-period phase certificate."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from statistics import NormalDist

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = '../dose_window_20261109/run.py'
SPEC = importlib.util.spec_from_file_location('dose_window', HERE / PARENT)
window = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(window)

PILOT, FORMAL, BOOT = 2026285101, 2026285102, 2026285103
PERIODS, BUDGETS = (.16, .144), (64, 256, 1024, 4096)
GROUPS, SIZE, LENGTH = 32, 128, 193
SIGMA, ALPHA, EPS = .02, .01, 1e-12
Z = NormalDist().inv_cdf(1 - ALPHA / 2)
RULES = ('phase', 'cauchy', 'known_period')
SOURCES = ('run.py', 'PROTOCOL.md', 'THEORY.md', PARENT,
           '../dose_window_20261109/THEORY.md')


def digest(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def array_hash(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(name, record):
    with (HERE / name).open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read(name):
    return json.loads((HERE / name).read_text())


def calibration(root, group, period):
    rng = np.random.default_rng(np.random.SeedSequence([root, 0, group]))
    x, noise = rng.standard_normal((2, max(BUDGETS)))
    dose = 1 + np.arange(len(x)) % 2
    y = x + dose * period + SIGMA * noise
    return np.column_stack([x, dose, y])


def ci(data):
    x, dose, y = data.T
    information = dose @ dose
    estimate = dose @ (y - x) / information
    width = Z * SIGMA / np.sqrt(information)
    return estimate - width, estimate + width


def bound_cross_products(values, weights, a, c):
    """Bounds k*C over all periods, then thresholds, without choosing a lift."""
    low = np.empty(len(values))
    high = np.empty(len(values))
    # Chunk only evaluation windows to bound peak memory, preserving all rows.
    for start in range(0, len(values), 8):
        end = min(start + 8, len(values))
        x = values[start:end]
        difference = x[:, None, :] - x[:, :, None]
        quotient_a = difference / a
        quotient_c = difference / c
        floor_a = np.floor(quotient_a)
        floor_c = np.floor(quotient_c)
        # Allow roundoff at a quotient close to a discontinuity.
        fixed = ((floor_a == floor_c)
                 & (np.abs(quotient_a - np.rint(quotient_a)) > EPS)
                 & (np.abs(quotient_c - np.rint(quotient_c)) > EPS))
        residue_a = difference - a * floor_a
        residue_c = difference - c * floor_c
        r_min = np.where(fixed, np.minimum(residue_a, residue_c) - EPS, 0.)
        r_max = np.where(fixed, np.maximum(residue_a, residue_c) + EPS, c)
        r_max[difference == 0.] = 0.
        b = -weights[start:end, None, :]
        lo = np.where(b >= 0, r_min, r_max) * b
        hi = np.where(b >= 0, r_max, r_min) * b
        low[start:end] = np.minimum(lo.sum(axis=-1).min(axis=-1), 0.)
        high[start:end] = np.maximum(hi.sum(axis=-1).max(axis=-1), 0.)
    return low, high


def known(values, weights, d, period):
    lo, hi = window.cross_extrema(values, weights, period)
    return window.envelope(d, lo, hi, period)


def finite_bands(values, weights, d, a, c):
    if not 0 < a <= c < 2:
        # 0 and a deliberately oversized finite sentinel imply abstention.
        unknown = (np.zeros_like(d), np.full_like(d, np.finfo(float).max))
        return unknown, unknown
    lo, hi = bound_cross_products(values, weights, a, c)
    phase = window.envelope(d, lo / c, hi / c, c)
    cauchy = window.envelope(d, -np.sqrt(d), np.sqrt(d), c)
    intersection = (np.maximum(phase[0], cauchy[0]), np.minimum(phase[1], cauchy[1]))
    assert np.all(intersection[0] <= intersection[1])
    return intersection, cauchy


def evaluate_geometry(x):
    scores = window.scores(x)
    values = x[:, :, 0]
    d = scores[:, 0]
    cutoff = np.sort(scores[:, 1:], axis=1)[:, 1]
    return values, window.cross_weights(values), d, cutoff


def pilot():
    x = window.trajectory(PILOT, LENGTH, 0, 8)
    values, weights, d, _ = evaluate_geometry(x)
    for period in PERIODS:
        np.testing.assert_allclose(
            finite_bands(values, weights, d, period, period)[0],
            known(values, weights, d, period), atol=1e-10, rtol=0)
        data = calibration(PILOT, 0, period)
        a, c = ci(data[:64])
        lo, hi = bound_cross_products(values, weights, a, c)
        for k in np.linspace(a, c, 81):
            exact_lo, exact_hi = window.cross_extrema(values, weights, k)
            assert np.all(lo <= k * exact_lo + EPS)
            assert np.all(hi >= k * exact_hi - EPS)
        for j in (1, 2):
            physical = window.warp(values, period) * (.16 / period)
            shifted = window.warp(values + j * period, period) * (.16 / period)
            np.testing.assert_allclose(shifted - physical, j * .16, rtol=0, atol=1e-12)
        shifted_data = data.copy()
        shifted_data[:, 2] += .003 * shifted_data[:, 1]
        np.testing.assert_allclose(np.array(ci(shifted_data)) - ci(data), .003,
                                   rtol=0, atol=1e-12)
    return dict(windows=len(x), periods_per_band=81, endpoint_parity=True,
                internal_bounds=True, matched_doses=True, calibration_equivariance=True)


def forecast():
    public, output = {}, {}
    for group in range(GROUPS):
        x = window.trajectory(FORMAL, LENGTH, group, SIZE)
        values, weights, d, cutoff = evaluate_geometry(x)
        for slot, period in enumerate(PERIODS):
            key = f'{group}_{slot}'
            data = calibration(FORMAL, group, period)
            public[key] = dict(x=array_hash(x), calibration=array_hash(data))
            bands, intervals = [], []
            oracle = known(values, weights, d, period)
            for n in BUDGETS:
                a, c = ci(data[:n])
                intervals.append([a, c])
                finite = finite_bands(values, weights, d, a, c)
                bands.append(np.stack([*finite, oracle], axis=0))
            bounds = np.stack(bands, axis=0)
            answers = np.where(bounds[:, :, 1] < cutoff, 1,
                               np.where(bounds[:, :, 0] > cutoff, -1, 0))
            assert np.all((answers[:, 1] == 0) | (answers[:, 0] == answers[:, 1]))
            output[key + '_bounds'] = bounds
            output[key + '_answers'] = answers
            output[key + '_ci'] = np.asarray(intervals)
    with (HERE / 'FORECAST.npz').open('xb') as stream:
        np.savez_compressed(stream, **output)
    return public


def score(public):
    totals = np.zeros((len(PERIODS), GROUPS, len(BUDGETS), len(RULES), 5), dtype=int)
    misses = np.zeros((len(PERIODS), GROUPS, len(BUDGETS)), dtype=bool)
    exclusions = np.zeros((len(PERIODS), GROUPS), dtype=int)
    with np.load(HERE / 'FORECAST.npz', allow_pickle=False) as packet:
        for group in range(GROUPS):
            x = window.trajectory(FORMAL, LENGTH, group, SIZE)
            values, _, d, cutoff = evaluate_geometry(x)
            for slot, period in enumerate(PERIODS):
                key = f'{group}_{slot}'
                data = calibration(FORMAL, group, period)
                assert public[key] == dict(x=array_hash(x), calibration=array_hash(data))
                interval = packet[key + '_ci']
                misses[slot, group] = (interval[:, 0] > period) | (interval[:, 1] < period)
                bounds = packet[key + '_bounds']
                answers = packet[key + '_answers']
                warped = window.warp(values, period) / math.sqrt(window.warp_variance(period))
                sine = window.scores(warped[..., None])[:, 0]
                assert np.all((d != cutoff) & (sine != cutoff))
                exclusions[slot, group] = (d > cutoff).sum()
                for world, truth in enumerate((d, sine)):
                    truth_answer = np.where(truth < cutoff, 1, -1)
                    totals[slot, group, :, :, world + 1] = (
                        (answers != 0) & (answers != truth_answer)).sum(axis=-1)
                    totals[slot, group, :, :, world + 3] = (
                        (bounds[:, :, 0] > truth) | (bounds[:, :, 1] < truth)).sum(axis=-1)
                totals[slot, group, :, :, 0] = (answers != 0).sum(axis=-1)
    valid = totals[..., 1:] * (~misses)[..., None, None]
    good = not np.any(valid) and not np.any(totals[:, :, :, 2, 1:])
    sample = np.random.default_rng(BOOT).integers(GROUPS, size=(4096, GROUPS))
    runs = []
    n_eval = GROUPS * SIZE
    for slot, period in enumerate(PERIODS):
        budgets = []
        for budget_index, n in enumerate(BUDGETS):
            counts = totals[slot, :, budget_index]
            total = counts.sum(axis=0)
            draws = counts[sample].sum(axis=1)
            difference = (draws[:, 0, 0] - draws[:, 1, 0]) / n_eval
            loss = (draws[:, 2, 0] - draws[:, 0, 0]) / n_eval
            budgets.append(dict(n=n, confidence_miss_panels=int(misses[slot, :, budget_index].sum()),
                                phase_gain=float((total[0, 0] - total[1, 0]) / n_eval),
                                phase_gain_ci=np.quantile(difference, [.025, .975]).tolist(),
                                oracle_gap=float((total[2, 0] - total[0, 0]) / n_eval),
                                oracle_gap_ci=np.quantile(loss, [.025, .975]).tolist(),
                                rules={rule: dict(accepted=int(row[0]), wrong=row[1:3].tolist(),
                                                  outside=row[3:5].tolist())
                                       for rule, row in zip(RULES, total, strict=True)}))
        runs.append(dict(period=period, windows=n_eval, groups=GROUPS,
                         id_axis1_excluded=int(exclusions[slot].sum()), budgets=budgets))
    main = runs[0]['budgets'][-1]
    return dict(runs=runs, gates=dict(phase_gain=main['phase_gain'] >= .05,
                                     oracle_gap=main['oracle_gap'] <= .10,
                                     confidence_event_sound=good))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'pilot', 'forecast', 'score'))
    phase = parser.parse_args().phase
    sources = {name: digest(name) for name in SOURCES}
    if phase == 'lock':
        save('LOCK.json', dict(utc=utc(), sources=sources, pilot=PILOT,
                               formal=FORMAL, bootstrap=BOOT))
        return
    assert read('LOCK.json')['sources'] == sources
    lock = digest('LOCK.json')
    if phase == 'pilot':
        save('PILOT.json', dict(utc=utc(), lock=lock, **pilot()))
        return
    assert read('PILOT.json')['lock'] == lock
    if phase == 'forecast':
        public = forecast()
        save('FORECAST.json', dict(utc=utc(), lock=lock, pilot=digest('PILOT.json'),
                                  public=public, predictions=digest('FORECAST.npz')))
        return
    receipt = read('FORECAST.json')
    assert receipt['lock'] == lock and receipt['pilot'] == digest('PILOT.json')
    assert receipt['predictions'] == digest('FORECAST.npz')
    save('SUMMARY.json', dict(utc=utc(), forecast=digest('FORECAST.json'),
                              **score(receipt['public'])))


if __name__ == '__main__':
    main()
