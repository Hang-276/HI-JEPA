"""Exact finite-window selection from a shared uncertain orientation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT, FORMAL = 2026248001, 2026248002
LO, HI = np.radians([-5., 9.])
MID = (LO + HI) / 2
PSI, STRENGTH = (11.25, 30.), (0., .12, .8)
RHO, SIGMA = .97, math.sqrt(1 - .97 ** 2)
N, T, CAL, LAGS = 4096, 192, 8192, (4, 16, 32)
PLANES = ((0, 3), (1, 2))
BITS = 1 << np.arange(4)
ALPHA, TOL = .02, 1e-12


def digest(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(name, data):
    with (HERE / name).open('x') as file:
        json.dump(data, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write('\n')


def read(name):
    return json.loads((HERE / name).read_text())


def streams(root, dynamic, cell, role):
    return np.random.default_rng(np.random.SeedSequence([root, dynamic, cell, role]))


def rotate(angle, planes):
    angle = np.asarray(angle)
    r = np.broadcast_to(np.eye(4), angle.shape + (4, 4)).copy()
    c, s = np.cos(angle), np.sin(angle)
    for i, j in planes:
        r[..., i, i] = r[..., j, j] = c
        r[..., i, j], r[..., j, i] = s, -s
    return r


def drift(index):
    w = rotate(np.radians(43.), ((1, 2),))
    return RHO * w @ rotate(np.radians(PSI[index]), ((0, 1),)) @ w.T


def true_angles(root, dynamic, cell, n):
    return streams(root, dynamic, cell, 0).uniform(LO, HI, size=n)


def observable_data(root, dynamic, cell, n, angles=None):
    theta = true_angles(root, dynamic, cell, n) if angles is None else angles
    rng = streams(root, dynamic, cell, 1)
    tau = STRENGTH[cell]
    reference_variance = 1 + (2 * tau ** 2 + tau) / 13
    labels = rng.integers(0, 13, size=(T, n), dtype=np.int8)
    mean, sd = np.zeros((13, 4)), np.ones((13, 4))
    mean[1:9:2], mean[2:9:2] = tau * np.eye(4), -tau * np.eye(4)
    sd[9 + np.arange(4), np.arange(4)] = math.sqrt(1 + tau)
    innovations = rng.standard_normal((T, n, 4)) * sd[labels] + mean[labels]
    if tau != 0:
        innovations = np.einsum('tbi,bij->tbj', innovations, rotate(theta, PLANES))
    x = np.empty((T + 1, n, 4))
    x[0] = math.sqrt(reference_variance) * rng.standard_normal((n, 4))
    a = drift(dynamic)
    for t in range(T):
        x[t + 1] = x[t] @ a.T + SIGMA * innovations[t]

    # This is the exact joint law of count and aligned residual sum from CAL.
    # No hidden angle is returned to either predictor.
    cal = streams(root, dynamic, cell, 2)
    counts = cal.binomial(CAL, 8 / 13, size=n)
    unit = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    total = counts[:, None] * tau * unit + np.sqrt(counts[:, None]) * cal.standard_normal((n, 2))
    return x, labels, counts, total


def aligned_sum(noise, labels):
    total = np.zeros((noise.shape[1], 2))
    for a, b in PLANES:
        first = (labels == 2 * a + 1).astype(float) - (labels == 2 * a + 2)
        second = (labels == 2 * b + 1).astype(float) - (labels == 2 * b + 2)
        total[:, 0] += (first * noise[..., a] + second * noise[..., b]).sum(0)
        total[:, 1] += (first * noise[..., b] - second * noise[..., a]).sum(0)
    return ((labels >= 1) & (labels <= 8)).sum(0), total


def interval(counts, totals, ambiguous):
    count = np.maximum(counts, 1)
    v = totals / count[:, None]
    radius = np.sqrt(-2 * np.log(ALPHA) / count)
    length = np.linalg.norm(v, axis=1)
    width = np.arcsin(np.minimum(1., radius / np.maximum(length, 1e-300)))
    phase = MID + (np.arctan2(v[:, 1], v[:, 0]) - MID + np.pi) % (2 * np.pi) - np.pi
    lo, hi = np.maximum(LO, phase - width), np.minimum(HI, phase + width)
    full = ambiguous | (counts == 0) | (length <= radius) | (lo > hi)
    return np.where(full, LO, lo), np.where(full, HI, hi)


def moment(x, tau):
    result = 0.
    for h in LAGS:
        dx = x[h:] - x[:-h]
        result = result + np.einsum('tbi,tbj->bij', dx, dx) / len(dx)
    return result / (len(LAGS) * (1 + (2 * tau ** 2 + tau) / 13))


def coefficients(m):
    coeff = np.empty((len(m), 4, 3))
    for a, b in PLANES:
        avg, diff = (m[:, a, a] + m[:, b, b]) / 2, (m[:, a, a] - m[:, b, b]) / 2
        coeff[:, a] = np.stack([avg, diff, m[:, a, b]], axis=-1)
        coeff[:, b] = np.stack([avg, -diff, -m[:, a, b]], axis=-1)
    return coeff


def evaluate(coeff, theta):
    a, b, c = np.moveaxis(coeff, -1, 0)
    shape = (len(theta),) + (1,) * (a.ndim - 1)
    return a + b * np.cos(2 * theta.reshape(shape)) + c * np.sin(2 * theta.reshape(shape))


def minimum(coeff, lo, hi):
    a, b, c = np.moveaxis(coeff, -1, 0)
    shape = (len(lo),) + (1,) * (a.ndim - 1)
    low, high = 2 * lo.reshape(shape), 2 * hi.reshape(shape)
    phase = np.arctan2(c, b) + np.pi
    stationary = phase + 2 * np.pi * np.ceil((low - phase) / (2 * np.pi))
    endpoints = np.minimum(evaluate(coeff, lo), evaluate(coeff, hi))
    return np.where(stationary <= high, a - np.hypot(b, c), endpoints)


def selection(scores):
    indices = np.argsort(scores, axis=1, kind='stable')
    ordered = np.take_along_axis(scores, indices, axis=1)
    return np.where(ordered[:, 2] > ordered[:, 1], BITS[indices[:, :2]].sum(1), 0)


def decisions(m, lo, hi):
    coeff = coefficients(m)
    center = selection(evaluate(coeff, (lo + hi) / 2))
    chosen = (center[:, None] & BITS) > 0
    cross = chosen[:, :, None] & ~chosen[:, None, :]
    gap = coeff[:, None] - coeff[:, :, None]
    joint_low = minimum(gap, lo, hi)
    axis_min = minimum(coeff, lo, hi)
    axis_max = -minimum(-coeff, lo, hi)
    separate_low = axis_min[:, None] - axis_max[:, :, None]
    joint = np.where(np.all(~cross | (joint_low > TOL), axis=(1, 2)), center, 0)
    separate = np.where(np.all(~cross | (separate_low > TOL), axis=(1, 2)), center, 0)
    assert np.all((separate == 0) | (separate == joint))
    return joint, separate


def direct_score(m, theta):
    r = rotate(theta, PLANES)
    return np.diagonal(r @ m @ r.transpose(0, 2, 1), axis1=1, axis2=2)


def pilot():
    n = 64
    maximum = 0.
    for dyn in range(2):
        x, u, c, y = observable_data(PILOT, dyn, 0, n, np.full(n, LO))
        other = observable_data(PILOT, dyn, 0, n, np.full(n, HI))
        for left, right in zip((x, u, c, y), other, strict=True):
            np.testing.assert_array_equal(left, right)
        m = moment(x, 0.)
        co = coefficients(m)
        angles = true_angles(PILOT, dyn, 0, n)
        difference = np.max(abs(evaluate(co, angles) - direct_score(m, angles)))
        maximum = max(maximum, float(difference))
        assert difference < 1e-12
        lo = np.full(n, LO)
        hi = np.full(n, HI)
        bound = minimum(co, lo, hi)
        for t in np.linspace(LO, HI, 65):
            assert np.all(direct_score(m, np.full(n, t)) >= bound - TOL)
        decisions(m, lo, hi)
        theta = np.radians(3.)
        labels = np.arange(1, 9).reshape(8, 1)
        mu = np.repeat(np.eye(4), 2, axis=0) * np.tile([1., -1.], 4)[:, None]
        noise = (mu @ rotate(theta, PLANES))[:, None]
        count, total = aligned_sum(noise, labels)
        np.testing.assert_allclose(total[0] / count[0], [np.cos(theta), np.sin(theta)], atol=1e-14)
    return dict(exact_null_equal=True, direct_score_max_error=maximum,
                trig_extrema_grid_checked=True, signed_innovation_identity=True,
                panels_per_dynamics=n)


def forecast():
    arrays = {}
    for d in range(2):
        for j, tau in enumerate(STRENGTH):
            x, u, c, total = observable_data(FORMAL, d, j, N)
            m = moment(x, tau)
            count, signed = aligned_sum((x[1:] - x[:-1] @ drift(d).T) / SIGMA, u)
            key = f'{d}_{j}'
            arrays[f'{key}_m'] = m
            for budget, counts, sums in ((0, count, signed), (CAL, count + c, signed + total)):
                lo, hi = interval(counts, sums, tau == 0)
                joint, separate = decisions(m, lo, hi)
                arrays[f'{key}_{budget}'] = np.stack([lo, hi, joint, separate], axis=1)
    with (HERE / 'FORECAST.npz').open('xb') as file:
        np.savez_compressed(file, **arrays)


def summary():
    # All bounds below treat an entire evaluation path and its calibration as one unit.
    radius = math.sqrt(math.log(32 / .05) / (2 * N))
    cells = []
    with np.load(HERE / 'FORECAST.npz', allow_pickle=False) as f:
        for d, psi in enumerate(PSI):
            for j, tau in enumerate(STRENGTH):
                key = f'{d}_{j}'
                m = f[f'{key}_m']
                theta = true_angles(FORMAL, d, j, N)
                truth = selection(direct_score(m, theta))
                assert np.all(truth > 0)
                endpoint = selection(direct_score(m, np.full(N, LO))) != selection(direct_score(m, np.full(N, HI)))
                predictions = []
                outputs = []
                for k in (0, CAL):
                    lo, hi, joint, separate = f[f'{key}_{k}'].T
                    covered = (lo <= theta) & (theta <= hi)
                    result = dict(calibration=k, angle_median_width_deg=float(np.median(np.degrees(hi - lo))),
                                  angle_missed=int((~covered).sum()))
                    for label, pred in (('joint', joint), ('separate', separate)):
                        active = pred != 0
                        wrong = active & (pred != truth)
                        assert not np.any(wrong & covered)
                        result[label] = dict(accepted=int(active.sum()), wrong=int(wrong.sum()),
                                             coverage=float(active.mean()),
                                             wrong_upper=float(wrong.mean() + radius))
                    outputs.append(joint)
                    predictions.append(result)
                paired_gain = float((outputs[1] != 0).mean() - (outputs[0] != 0).mean())
                null = bool(np.array_equal(*outputs) and all(r['joint']['wrong'] == 0 for r in predictions)
                            and predictions[0]['joint']['coverage'] >= .10 and endpoint.mean() >= .10)
                strong = paired_gain >= .10 and paired_gain > 2 * radius
                cells.append(dict(psi_deg=psi, tau=tau, n=N, endpoint_disagreement=int(endpoint.sum()),
                                  predictions=predictions, paired_gain=paired_gain,
                                  gain_lower=paired_gain - 2 * radius,
                                  null_gate=null if tau == 0 else None,
                                  strong_gate=strong if tau == .8 else None))
    return dict(cells=cells, risk_gate=all(p['joint']['wrong_upper'] < .05 for c in cells for p in c['predictions']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('lock', 'pilot', 'forecast', 'score'))
    phase = p.parse_args().phase
    sources = {name: digest(name) for name in ('run.py', 'THEORY.md', 'PROTOCOL.md',
                                              '../phase2_boundary_20261107/BOUNDARY.md')}
    if phase == 'lock':
        write('LOCK.json', dict(utc=utc(), sources=sources, pilot=PILOT, formal=FORMAL))
        return
    assert read('LOCK.json')['sources'] == sources
    lock = digest('LOCK.json')
    if phase == 'pilot':
        write('PILOT.json', dict(utc=utc(), lock=lock, **pilot()))
    elif phase == 'forecast':
        assert read('PILOT.json')['lock'] == lock
        forecast()
        write('FORECAST_RECEIPT.json', dict(utc=utc(), lock=lock, pilot=digest('PILOT.json'),
                                            forecast=digest('FORECAST.npz')))
    else:
        receipt = read('FORECAST_RECEIPT.json')
        assert receipt['lock'] == lock and receipt['forecast'] == digest('FORECAST.npz')
        write('SUMMARY.json', dict(utc=utc(), receipt=digest('FORECAST_RECEIPT.json'), **summary()))


if __name__ == '__main__':
    main()
