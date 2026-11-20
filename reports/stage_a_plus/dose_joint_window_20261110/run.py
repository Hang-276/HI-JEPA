"""Frozen local CPU check for joint window length and commensurate dose."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.signal import lfilter
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
PRIOR = '../dose_window_20261109'
SOURCES = ('PROTOCOL.md', 'THEORY.md', 'run.py', f'{PRIOR}/run.py',
           f'{PRIOR}/THEORY.md', f'{PRIOR}/SHARP_CONSTANTS.md')
LENGTHS = (257, 1025, 4097, 16385)
PERIODS = (.35, .35 / math.sqrt(2))
LAGS = (4, 16, 32)
NAMES = ('tie', 'fixed_gap', 'interior_gap')
METRICS = ('witnessed', 'safe', 'unresolved', 'intended_competitor')
GROUPS, SIZE, PILOT_SIZE, BOOTSTRAPS = 16, 64, 8, 4096
PILOT_SEED, FORMAL_SEED, BOOT_SEED = 2026335001, 2026335002, 2026335003
RHO_1, RHO_FIXED, RHO_3, RHO_4 = .995, .99, .999, .90
WIDTH, RAMP_WEIGHT = .02, .98
UNIFORM_VARIANCE = (RAMP_WEIGHT * (1 - WIDTH))**2 / 12
FLOAT_GUARD = 1e-12

spec = importlib.util.spec_from_file_location('frozen_phase_outer', HERE / PRIOR / 'run.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def save(name, **data):
    with (HERE / name).open('x') as file:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(), **data), file,
                  ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')


def check_new(name):
    if (HERE / name).exists():
        raise FileExistsError(HERE / name)


def verify_lock():
    locked = read('LOCK.json')
    if locked['sources'] != {name: sha(HERE / name) for name in SOURCES}:
        raise ValueError('A frozen source differs from LOCK.json')
    return sha(HERE / 'LOCK.json')


def population_score(rho):
    return 2 * (1 - sum(rho**lag for lag in LAGS) / len(LAGS))


def warp(x, ell):
    u = np.remainder(x / ell, 1.)
    ramp = np.clip((u - (1 - WIDTH) / 2) / WIDTH, 0., 1.)
    return x + (RAMP_WEIGHT * ell) * (ramp - u)


def variance_bounds(ell):
    # Odd lift residual has zero Gaussian mean. Fourier wrapping bounds the
    # density deviation and the Gaussian linear covariance, as in the old proof.
    q = math.exp(-2 * math.pi**2 / ell**2)
    phase_deviation = 2 * q / (1 - q)
    middle = 1 + ell**2 * UNIFORM_VARIANCE
    error = phase_deviation * (2 + ell**2 * UNIFORM_VARIANCE)
    return middle, np.nextafter(middle - error, -np.inf), np.nextafter(middle + error, np.inf), error


def population_change(ell):
    r = RHO_1**np.asarray(LAGS)
    n = np.arange(1, 65)
    wave = 2 * np.pi * n / ell
    coefficients = RAMP_WEIGHT * np.sinc(n * WIDTH) / (np.pi * n)
    covariance = ((np.exp(-(1 - r[:, None]) * wave**2) -
                   np.exp(-(1 + r[:, None]) * wave**2)) @ coefficients**2) / 2
    # For these two periods the n!=m Fourier terms are smaller than 1e-60,
    # and the omitted diagonal terms smaller than this as well. The input
    # stationary Gaussian variance is bounded separately for window scoring.
    return float(2 * ell**2 * (r.mean() * UNIFORM_VARIANCE - covariance.mean()) /
                 variance_bounds(ell)[0])


def long_run_variance(rho):
    k = np.arange(-4096, 4097)
    cov_square = sum((rho**np.abs(k + b - a) - rho**np.abs(k - a)
                      - rho**np.abs(k + b) + rho**np.abs(k))**2
                     for a in LAGS for b in LAGS)
    return float(2 * cov_square.sum() / len(LAGS)**2)


def analytic_design():
    base = population_score(RHO_1)
    slope = (RHO_1**np.asarray(LAGS)).mean() / 6
    out = []
    for ell in PERIODS:
        shift = population_change(ell)
        interior = brentq(lambda rho: population_score(rho) - base - shift / 2, RHO_FIXED, RHO_1)
        rows = {}
        for label, rho in zip(NAMES, (RHO_1, RHO_FIXED, interior), strict=True):
            gap = population_score(rho) - base
            paired_sd = math.sqrt(long_run_variance(RHO_1) + long_run_variance(rho))
            se = paired_sd / np.sqrt(LENGTHS)
            rows[label] = dict(rho=rho, population_gap=gap, sqrt_T_gap_sd=paired_sd,
                               normal_witness_benchmark=(ndtr(gap / se) - ndtr((gap - shift) / se)).tolist())
        out.append(dict(period=ell, population_shift=shift, full_class_sharp_coefficient=slope,
                        reference_variance=variance_bounds(ell)[0],
                        reference_wrapping_error_bound=variance_bounds(ell)[3], competitors=rows))
    return out


def trajectory(seed, group, length, count, design):
    rng = np.random.default_rng(np.random.SeedSequence([seed, group]))
    innovations = rng.standard_normal((4, count, length))

    def ar(axis, rho):
        source = innovations[axis]
        tail, _ = lfilter([math.sqrt(1 - rho**2)], [1, -rho], source[:, 1:], axis=1,
                          zi=rho * source[:, :1])
        return np.concatenate([source[:, :1], tail], axis=1)

    rhos = (RHO_1, RHO_FIXED, *(row['competitors']['interior_gap']['rho'] for row in design))
    return (ar(0, RHO_1), [ar(1, rho) for rho in rhos], ar(2, RHO_3), ar(3, RHO_4),
            hashlib.sha256(innovations.tobytes()).hexdigest())


def path_scores(x, lengths):
    """Equal lag weights and each lag's actual number of endpoint pairs."""
    values = np.zeros((len(lengths), len(x)))
    for lag in LAGS:
        difference = x[:, lag:] - x[:, :-lag]
        prefix = np.cumsum(difference**2, axis=1)
        values += prefix[:, np.asarray(lengths) - lag - 1].T / (np.asarray(lengths) - lag)[:, None]
    return values / len(LAGS)


def top2(scores):
    return np.sort(np.argsort(scores, kind='stable', axis=1)[:, :2], axis=1)


def run_panel(seed, group, lengths, count, design, engineering=False):
    x, competitors, slow, fast, innovation_hash = trajectory(seed, group, max(lengths), count, design)
    base_score = path_scores(x, lengths)
    competitor_scores = [path_scores(v, lengths) for v in competitors]
    slow_score, fast_score = (path_scores(v, lengths) for v in (slow, fast))
    weights = [prior.cross_weights(x[:, :length]) for length in lengths]
    counts = np.zeros((len(lengths), len(PERIODS), len(NAMES), len(METRICS)), dtype=int)
    maximum_band_violation = 0.
    parity = dict(two_dose=0., lag_score=0., phase_extrema=0.)
    for i, ell in enumerate(PERIODS):
        warped = warp(x, ell)
        numerator = path_scores(warped, lengths)
        variance, min_var, max_var, _ = variance_bounds(ell)
        candidate = numerator / variance
        score_lo, score_hi = numerator / max_var, numerator / min_var
        if engineering:
            for dose in (ell, 2 * ell):
                parity['two_dose'] = max(parity['two_dose'], float(np.abs(warp(x + dose, ell) - warped - dose).max()))
        for j, length in enumerate(lengths):
            path = x[:, :length]
            lo, hi = prior.cross_extrema(path, weights[j], ell)
            lo, hi = prior.envelope(base_score[j], lo, hi, ell)
            maximum_band_violation = max(maximum_band_violation, float(np.max(
                np.stack([lo - base_score[j], base_score[j] - hi, lo - score_lo[j], score_hi[j] - hi]))))
            if engineering:
                old_scores = prior.scores(path[..., None])[:, 0]
                parity['lag_score'] = max(parity['lag_score'], float(np.abs(old_scores - base_score[j]).max()))
                u = np.remainder(path[0] / ell, 1.)
                w = weights[j][0]
                candidates = (u[:, None] >= np.r_[0., np.sort(u), 1.])
                direct = w @ candidates - w @ u
                a, b = prior.cross_extrema(path[:1], weights[j][:1], ell)
                parity['phase_extrema'] = max(parity['phase_extrema'],
                    abs(float(a[0]) - float(direct.min())), abs(float(b[0]) - float(direct.max())))
            for k, index in enumerate((0, 1, 2 + i)):
                others = np.stack([competitor_scores[index][j], slow_score[j], fast_score[j]], axis=1)
                cutoff = np.sort(others, axis=1)[:, 1]
                # Evaluate the exact named top-2 for both compatible explanations.
                point_base = top2(np.column_stack([base_score[j], others]))
                point_warp = top2(np.column_stack([candidate[j], others]))
                witness = np.any(point_base != point_warp, axis=1)
                # Neither truth readout is allowed to sit in a numerical tie at the cutoff.
                assert (np.abs(base_score[j] - cutoff) > FLOAT_GUARD).all()
                assert ((score_hi[j] < cutoff - FLOAT_GUARD) |
                        (score_lo[j] > cutoff + FLOAT_GUARD)).all()
                np.testing.assert_array_equal(witness, (base_score[j] < cutoff) != (candidate[j] < cutoff))
                safe = (hi < cutoff) | (lo > cutoff)
                if np.any(safe & witness):
                    raise AssertionError('A certified slow set disagrees with a compatible interpretation')
                unresolved = ~(safe | witness)
                counts[j, i, k] = (witness.sum(), safe.sum(), unresolved.sum(), (cutoff == others[:, 0]).sum())
    assert maximum_band_violation <= FLOAT_GUARD
    assert (counts[..., :3].sum(axis=-1) == count).all()
    return counts, innovation_hash, dict(phase_outer_max_violation=maximum_band_violation, **parity)


def pilot(lock_sha):
    check_new('PILOT.json')
    _, innovation_hash, checks = run_panel(PILOT_SEED, 0, LENGTHS[:1], PILOT_SIZE,
                                           analytic_design(), engineering=True)
    variance_errors = []
    for ell in PERIODS:
        knots = ell * (np.arange(math.ceil(10 / ell))[:, None] + np.array([0., .49, .51]))
        knots = knots[(knots > 0.) & (knots < 10.)]
        integral, _ = quad(lambda x: float(warp(x, ell))**2 * math.exp(-x*x / 2),
                           0., 10., points=knots, epsabs=2e-13, epsrel=0., limit=512)
        direct = 2 * integral / math.sqrt(2 * math.pi)
        variance_errors.append(abs(direct - variance_bounds(ell)[0]))
    checks['reference_variance'] = max(variance_errors)
    passed = all(value <= FLOAT_GUARD for value in checks.values())
    save('PILOT.json', lock_sha=lock_sha, passed=passed, trajectories=PILOT_SIZE,
         length=LENGTHS[0], innovation_hash=innovation_hash, checks=checks)
    if not passed:
        raise RuntimeError('Frozen engineering gates failed')


def estimate(value, draws):
    sampled = value[draws].mean(axis=1)
    return dict(estimate=float(value.mean()), ci95=np.quantile(sampled, [.025, .975]).tolist())


def formal(lock_sha):
    check_new('SUMMARY.json')
    pilot_record = read('PILOT.json')
    assert pilot_record['lock_sha'] == lock_sha and pilot_record['passed']
    design = analytic_design()
    records = [run_panel(FORMAL_SEED, i, LENGTHS, SIZE, design) for i in range(GROUPS)]
    counts = np.asarray([row[0] for row in records])
    hashes = [row[1] for row in records]
    assert len(set(hashes + [pilot_record['innovation_hash']])) == GROUPS + 1
    rates = counts / SIZE
    draws = np.random.default_rng(BOOT_SEED).integers(GROUPS, size=(BOOTSTRAPS, GROUPS))
    results = []
    for t, length in enumerate(LENGTHS):
        for e, ell in enumerate(PERIODS):
            for c, label in enumerate(NAMES):
                stats = {metric: estimate(rates[:, t, e, c, m], draws) |
                         {'count': int(counts[:, t, e, c, m].sum())} for m, metric in enumerate(METRICS)}
                results.append(dict(length=length, period=ell, condition=label, **stats))
    hit = rates[..., 0]
    contrasts = (
        ('tied_coarse_long_minus_short', hit[:, -1, 0, 0] - hit[:, 0, 0, 0], .15),
        ('tied_coarse_long_minus_fixed_gap_long', hit[:, -1, 0, 0] - hit[:, -1, 0, 1], .25),
        ('tied_long_coarse_minus_fine', hit[:, -1, 0, 0] - hit[:, -1, 1, 0], .05),
        ('interior_coarse_long_minus_short', hit[:, -1, 0, 2] - hit[:, 0, 0, 2], .25),
    )
    gates = {}
    for name, values, threshold in contrasts:
        result = estimate(values, draws)
        gates[name] = result | {'minimum_lower95': threshold, 'passed': result['ci95'][0] > threshold}
    save('SUMMARY.json', lock_sha=lock_sha, pilot_sha=sha(HERE / 'PILOT.json'),
         trajectories=GROUPS * SIZE, independent_panels=GROUPS, bootstrap_resamples=BOOTSTRAPS,
         design=design, results=results, gates=gates, all_gates=all(x['passed'] for x in gates.values()),
         group_counts=counts.tolist(), group_axes=['panel', 'length', 'period', 'condition', 'metric'],
         metric_order=list(METRICS), condition_order=list(NAMES), lengths=list(LENGTHS), periods=list(PERIODS),
         innovation_hashes=hashes, phase_outer_max_violation=max(r[2]['phase_outer_max_violation'] for r in records),
         certified_disagreements=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('lock', 'pilot', 'formal'))
    phase = parser.parse_args().phase
    if phase == 'lock':
        save('LOCK.json', sources={name: sha(HERE / name) for name in SOURCES},
             design=analytic_design(), seeds=dict(pilot=PILOT_SEED, formal=FORMAL_SEED, bootstrap=BOOT_SEED),
             lengths=list(LENGTHS), pilot_size=PILOT_SIZE, panels=GROUPS, trajectories_per_panel=SIZE)
    else:
        {'pilot': pilot, 'formal': formal}[phase](verify_lock())


if __name__ == '__main__':
    main()
