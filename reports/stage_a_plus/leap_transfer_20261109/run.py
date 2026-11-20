"""Compare native LEAP slow-set errors against same-information calibrators."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
EXT = STAGE / 'fine_replication_extension_20261109'
EQ = STAGE / 'fine_affine_control_20261109'
sys.path[:0] = [str(EXT), str(EQ)]

import equivariant as eq
import extra_fresh_data as generator
from extra_fine_forward import load_fine, smoke as fine_smoke
import leap_forward

SLOT, PANEL_COUNT, N, REF, START = 'ar_rich_fresh2', 64, 128, 256, 11
REFERENCE = 2026262000
ENGINEERING = (2026262001, 2026262002, 2026262003)
PANELS = tuple((2026263000 + 2 * i, 2026263001 + 2 * i) for i in range(PANEL_COUNT))
LAGS, DRAWS, ALARMS, RESAMPLES = (4, 16, 32), 32, 6, 4096
MATCH = (STAGE / 'fine_independent_validation_20261109/run.py')
MATCH_NAME = STAGE / 'step6_fine_20261105/forecast_lock.json'
VIEWS = ('leap0', 'leap1', 'leap2', 'fine0', 'fine1', 'fine2')
WARN = ('native', 'magnitude', 'signed')
ACCEPT = ('joint', 'box')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(name, record):
    with (HERE / name).open('x') as file:
        json.dump(record, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write('\n')


def fine_permutation(seed):
    rows = read(MATCH_NAME)['parents']
    selected = next(x for x in rows if x['slot'] == SLOT and x['rep'] == seed)
    assert selected['assignment_objective_lead'] > 0
    return selected['perm']


def models():
    result = {}
    for seed in range(3):
        model = leap_forward.load(seed)
        result[f'leap{seed}'] = model, model.permutation, model.provenance
        model = load_fine(SLOT, seed)
        result[f'fine{seed}'] = model, fine_permutation(seed), model.provenance
    return result


def sources(restored):
    files = (Path(__file__), HERE / 'PROTOCOL.md', HERE / 'leap_forward.py',
             EXT / 'extra_fine_forward.py', EQ / 'equivariant.py', MATCH_NAME, MATCH)
    digests = generator.source_hashes() | {str(p): sha(p) for p in files}

    def walk(item):
        if isinstance(item, dict):
            if 'path' in item and 'sha256' in item:
                path = item['path']
                assert sha(path) == item['sha256']
                digests[path] = item['sha256']
            for v in item.values():
                walk(v)
        elif isinstance(item, (tuple, list)):
            for v in item:
                walk(v)

    for _, _, provenance in restored.values():
        walk(provenance)
    return digests


def stream_check():
    old = set()

    def walk(item):
        if isinstance(item, dict):
            if type(item.get('trajectory_seed')) is int:
                old.add(item['trajectory_seed'])
            for v in item.values():
                walk(v)
        elif isinstance(item, list):
            for v in item:
                walk(v)

    paths = list(STAGE.glob('fine_*_20261109/FORECAST.json'))
    for p in paths:
        walk(read(p))
    seeds = (REFERENCE, *ENGINEERING, *(s for pair in PANELS for s in pair))
    fresh = [generator._fresh_seed(SLOT, s) for s in seeds]
    assert len(fresh) == len(set(fresh)) and not set(fresh) & old
    return dict(prior=len(old), checked_files=len(paths), fresh=len(fresh), overlap=[])


def encode(view, model, observations):
    x = observations['x']
    values = model.encode(x)
    if view.startswith('fine'):
        values = values[:, START:]
    assert values.shape == (len(x), x.shape[1] - START, 4)
    return values.astype(np.float64)


def score_matrices(y):
    result = np.zeros((len(y), 4, 4))
    for lag in LAGS:
        dx = y[:, lag:] - y[:, :-lag]
        result += np.einsum('eti,etj->eij', dx, dx) / dx.shape[1]
    return result / len(LAGS)


def covariance(y):
    values = y.reshape(-1, 4)
    centered = values - values.mean(axis=0)
    return centered.T @ centered / len(centered)


def scores(moment, rows):
    return np.einsum('ai,eij,aj->ea', rows, moment, rows)


def top(values):
    order = np.argsort(values, axis=1, kind='stable')
    ordered = np.take_along_axis(values, order, axis=1)
    selected = (1 << order[:, :2]).sum(1)
    return np.where(ordered[:, 2] > ordered[:, 1], selected, 0)


def margin(values, native):
    select = (native[:, None] & (1 << np.arange(4))) != 0
    lower = np.min(np.where(select, np.inf, values), axis=-1)
    upper = np.max(np.where(select, values, -np.inf), axis=-1)
    return lower - upper


def alarms(value, native):
    order = np.argsort(np.where(native > 0, value, np.inf), kind='stable')
    flagged = np.zeros(len(native), dtype=bool)
    flagged[order[:ALARMS]] = True
    return flagged


def unit_rows(rows, cov):
    variance = np.einsum('ai,ij,aj->a', rows, cov, rows)
    assert np.all(variance > 0)
    return rows / np.sqrt(variance)[:, None]


def unmix(moment, cov):
    values, vectors = np.linalg.eigh(moment[0])
    if values[0] <= 0:
        return None
    whiten = (vectors / np.sqrt(values)) @ vectors.T
    contrast = whiten @ (moment[1:] - moment[0]) @ whiten
    values, vectors = np.linalg.eigh((contrast + contrast.swapaxes(-1, -2)) * .5)
    tolerance = eq.RANK_TOLERANCE * abs(values).max(axis=1)
    if np.any(values[:, -1] <= 0) or np.any(values[:, -1] - values[:, -2] <= tolerance):
        return None
    axes = vectors[:, :, -1].T
    if np.linalg.cond(axes) > eq.MAX_CONDITION:
        return None
    return unit_rows(np.linalg.solve(axes, whiten), cov)


def calibrate(y, public, cov, weights, parity=False):
    fit = np.arange(len(y)) % 4 == 0
    a, u = public['actions'][:, START:], public['regimes'][:, START:]
    drift = eq.fit_drift(y[fit], a[fit], u[fit])
    if drift is None:
        return None, [], None
    residual = np.diff(y[~fit], axis=1) - eq._design(y[~fit, :-1], a[~fit]) @ drift
    sums, counts = [], []
    for code in eq.REGIMES:
        select = u[~fit] == code
        sums.append(np.einsum('eti,etj->eij', residual, residual * select[..., None]))
        counts.append(select.sum(1))
    s, c = np.stack(sums, axis=1), np.stack(counts, axis=1)

    def evaluate(w):
        n = w @ c
        if np.any(n == 0):
            return None
        moments = (w @ s.reshape(len(s), -1)).reshape(len(eq.REGIMES), 4, 4) / n[:, None, None]
        return unmix(moments, cov)

    pooled = evaluate(np.ones(len(s)))
    error = None
    if parity:
        archived = eq.rows(y, a, u, drift, cov, ~fit)
        assert (archived is None) == (pooled is None)
        if pooled is not None:
            sign = np.where(np.einsum('ai,ij,aj->a', pooled, cov, archived) < 0, -1., 1.)
            error = float(np.max(abs(pooled * sign[:, None] - archived)))
            assert error < 1e-10
    return pooled, [candidate for w in weights if (candidate := evaluate(w)) is not None], error


def forecast_view(view, state, reference_cov, cal, eval_, weights, parity=False):
    model, perm, _ = state
    yc, ye = encode(view, model, cal), encode(view, model, eval_)
    matrix = score_matrices(ye)
    rows = unit_rows(np.eye(4)[perm, :], reference_cov)
    native_score = scores(matrix, rows)
    native = top(native_score)
    m0 = margin(native_score, native)
    anchor, candidates, drift_error = calibrate(yc, cal, reference_cov, weights, parity)
    correction = native
    bounds = dict(native=m0, magnitude=m0, signed=m0)
    accept = dict(joint=np.zeros(len(native), dtype=bool), box=np.zeros(len(native), dtype=bool))
    score_error = None
    if anchor is not None:
        same_sign = np.where(np.einsum('ai,ij,aj->a', anchor, reference_cov, rows) < 0, -1., 1.)
        delta = anchor * same_sign[:, None] - rows
        energy = np.maximum(scores(matrix, delta), 0.)
        cross = np.einsum('ai,eij,aj->ea', rows, matrix, delta)
        signed = native_score + 2 * cross + energy
        score_error = float(np.max(abs(signed - scores(matrix, anchor))))
        assert score_error < 1e-10
        error_size = 2 * np.sqrt(np.maximum(native_score, 0.) * energy) + energy
        select = (native[:, None] & (1 << np.arange(4))) != 0
        bounds['magnitude'] = margin(native_score + np.where(select, error_size, -error_size), native)
        bounds['signed'] = margin(signed, native)
        correction = top(signed)
        if len(candidates) >= 31:
            matrices = np.stack([anchor, *candidates])
            values = np.einsum('kai,eij,kaj->kea', matrices, matrix, matrices)
            joint = margin(values, native).min(0)
            box = margin(np.where(select, values.max(0), values.min(0)), native)
            assert np.all(box <= joint + 1e-12)
            accept = dict(joint=(native > 0) & (joint > 0), box=(native > 0) & (box > 0))
    return dict(native=native, corrected=correction, alarms={m: alarms(v, native) for m, v in bounds.items()},
                accept=accept, available=anchor is not None, candidates=len(candidates),
                direct_rayleigh_error=score_error, drift_parity_error=drift_error)


def draw_weights(index):
    rng = np.random.default_rng(2026265000 + index)
    return np.stack([np.bincount(x, minlength=96) for x in rng.integers(96, size=(DRAWS, 96))])


def do_verify(restored):
    start = time.monotonic()
    verification = [leap_forward.smoke(i, restored[f'leap{i}'][0]) for i in range(3)]
    verification += [fine_smoke(SLOT, i, restored[f'fine{i}'][0]) for i in range(3)]
    ref, _ = generator.fresh_observations(SLOT, 16, ENGINEERING[0])
    cal, _ = generator.fresh_observations(SLOT, N, ENGINEERING[1])
    evaluate, _ = generator.fresh_observations(SLOT, 1, ENGINEERING[2])
    tested = {}
    for v, state in restored.items():
        c = covariance(encode(v, state[0], ref))
        p = forecast_view(v, state, c, cal, evaluate, draw_weights(PANEL_COUNT), parity=True)
        tested[v] = {k: p[k] for k in ('available', 'candidates', 'direct_rayleigh_error', 'drift_parity_error')}
        assert p['available'] and p['candidates'] >= 31
    return dict(engineering=tested, historical=verification, seconds=time.monotonic() - start)


def forecasts(restored):
    start = time.monotonic()
    ref, receipt = generator.fresh_observations(SLOT, REF, REFERENCE)
    cov = {v: covariance(encode(v, state[0], ref)) for v, state in restored.items()}
    arrays, panels = {}, []
    for i, (cs, es) in enumerate(PANELS):
        c, cr = generator.fresh_observations(SLOT, N, cs)
        e, er = generator.fresh_observations(SLOT, N, es)
        weights = draw_weights(i)
        recorded = {}
        for v, state in restored.items():
            p = forecast_view(v, state, cov[v], c, e, weights)
            arrays[f'{i}_{v}'] = np.stack([p['native'], p['corrected'],
                                          *(p['alarms'][m] for m in WARN),
                                          *(p['accept'][m] for m in ACCEPT)], axis=0)
            assert all(x.sum() == ALARMS for x in p['alarms'].values())
            recorded[v] = {k: p[k] for k in ('available', 'candidates', 'direct_rayleigh_error')}
        panels.append(dict(index=i, calibration=cr, evaluation=er, models=recorded))
        if (i + 1) % 8 == 0:
            print(f'{i + 1}/{PANEL_COUNT} forecast panels', flush=True)
    with (HERE / 'FORECAST.npz').open('xb') as file:
        np.savez_compressed(file, **arrays)
    return dict(reference=receipt, panels=panels, seconds=time.monotonic() - start)


def measure(packet, truth, near):
    native, corrected, *flags = packet
    strict = truth > 0
    ok = native == truth
    eligible = strict & (native > 0)
    bad, good = eligible & ~ok, eligible & ok

    def count(outcome, target):
        return np.array([(outcome & target).sum(), target.sum()], dtype=np.int64)

    return dict(
        native_near=count(ok, near), corrected_near=count(corrected == truth, near),
        native_all=count(ok, strict), corrected_all=count(corrected == truth, strict),
        **{f'{k}_hit': count(f > 0, bad) for k, f in zip(WARN, flags[:3], strict=True)},
        **{f'{k}_fpr': count(f > 0, good) for k, f in zip(WARN, flags[:3], strict=True)},
        **{f'{k}_accepted': count(f > 0, strict) for k, f in zip(ACCEPT, flags[3:], strict=True)},
        **{f'{k}_right': count((f > 0) & ok, strict) for k, f in zip(ACCEPT, flags[3:], strict=True)},
        **{f'{k}_wrong': count((f > 0) & ~ok, strict) for k, f in zip(ACCEPT, flags[3:], strict=True)},
    ), bad


def summarize(record):
    reference = generator.replay_physical(record['reference'])[:, START:].astype(np.float64)
    scale = reference.std(axis=(0, 1))
    measurements = {v: [] for v in VIEWS}
    union, near_windows = 0, 0
    with np.load(HERE / 'FORECAST.npz', allow_pickle=False) as frozen:
        for p in record['panels']:
            state = generator.replay_physical(p['evaluation'])[:, START:].astype(np.float64) / scale
            slow = np.diagonal(score_matrices(state), axis1=1, axis2=2)
            truth = top(slow)
            ordered = np.sort(slow, axis=1)
            near = (truth > 0) & ((ordered[:, 2] - ordered[:, 1]) <= .05)
            native_errors = []
            for v in VIEWS:
                statistics, bad = measure(frozen[f"{p['index']}_{v}"], truth, near)
                measurements[v].append(statistics)
                if v.startswith('leap'):
                    native_errors.append(bad)
            union += np.any(native_errors, axis=0).sum()
            near_windows += near.sum()
    rng = np.random.default_rng(2026266100)
    weights = np.stack([np.bincount(row, minlength=PANEL_COUNT)
                        for row in rng.integers(PANEL_COUNT, size=(RESAMPLES, PANEL_COUNT))])
    metrics = tuple(measurements[VIEWS[0]][0])

    def calculate(selected):
        point, samples = {}, {}
        for k in metrics:
            rows = np.sum([[r[k] for r in measurements[v]] for v in selected], axis=0)
            count, total = rows.sum(0)
            sim = weights @ rows
            samples[k] = np.divide(sim[:, 0], sim[:, 1], out=np.full(len(sim), np.nan), where=sim[:, 1] > 0)
            point[k] = dict(count=int(count), total=int(total), rate=float(count / total) if total else None)
        contrasts = {}
        for a, b in (('corrected_near', 'native_near'), ('corrected_all', 'native_all'),
                     ('signed_hit', 'native_hit'), ('signed_hit', 'magnitude_hit'),
                     ('joint_right', 'box_right'), ('joint_wrong', 'box_wrong')):
            d = (samples[a] - samples[b])
            valid = d[np.isfinite(d)]
            q = [float(v) for v in np.quantile(valid, [.05 / 3, .975])] if len(valid) else [None, None]
            ap, bp = point[a]['rate'], point[b]['rate']
            contrasts[f'{a}_vs_{b}'] = dict(gain=ap - bp if ap is not None and bp is not None else None,
                                            lower=q[0], upper=q[1])
        upper = {}
        for m in WARN:
            v = samples[f'{m}_fpr']
            valid = v[np.isfinite(v)]
            upper[m] = float(np.quantile(valid, 1 - .05 / 3)) if len(valid) else None
        return dict(metrics=point, contrasts=contrasts, false_alarm_upper=upper)

    outputs = {v: calculate([v]) for v in VIEWS}
    outputs['LEAP'] = calculate(VIEWS[:3])
    outputs['Fine'] = calculate(VIEWS[3:])
    leap = outputs['LEAP']
    support = bool(near_windows >= 256 and union >= 120 and all(
        outputs[v]['metrics']['signed_hit']['total'] >= 30 for v in VIEWS[:3]))
    main = leap['contrasts']
    valid_fpr = all(x is not None and x < .05 for x in leap['false_alarm_upper'].values())
    warning = all(x['gain'] is not None and x['gain'] >= .1 and x['lower'] is not None and x['lower'] > 0
                  for x in (main['signed_hit_vs_native_hit'], main['signed_hit_vs_magnitude_hit']))
    point = main['corrected_near_vs_native_near']
    return dict(support=support, distinct_near=int(near_windows), distinct_native_LEAP_errors=int(union),
                warning_gate=support and warning and valid_fpr,
                point_gate=bool(support and point['gain'] is not None and point['gain'] >= .03
                                and point['lower'] > 0 and main['corrected_all_vs_native_all']['gain'] > -.01),
                unavailable={v: sum(not p['models'][v]['available'] for p in record['panels']) for v in VIEWS},
                fewer_than_31={v: sum(p['models'][v]['candidates'] < 31 for p in record['panels']) for v in VIEWS},
                results=outputs, panel_statistics=measurements)


def pack(value):
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [pack(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('lock', 'verify', 'forecast', 'score'))
    phase = p.parse_args().phase
    lock_path, verification, forecast_path = (HERE / n for n in ('LOCK.json', 'ENGINEERING.json', 'FORECAST.json'))
    if phase == 'lock':
        restored = models()
        write('LOCK.json', pack(dict(utc=now(), sources=sources(restored), roots=stream_check(),
                                     permutations={v: m[1] for v, m in restored.items()})))
        return
    lock = read(lock_path)
    lock_sha = sha(lock_path)
    assert all(sha(p) == d for p, d in lock['sources'].items())
    if phase == 'verify':
        write('ENGINEERING.json', pack(dict(utc=now(), lock=lock_sha, **do_verify(models()))))
        return
    assert read(verification)['lock'] == lock_sha
    if phase == 'forecast':
        record = forecasts(models())
        write('FORECAST.json', pack(dict(utc=now(), lock=lock_sha, verification=sha(verification),
                                        predictions=sha(HERE / 'FORECAST.npz'), **record)))
        return
    frozen = read(forecast_path)
    assert frozen['lock'] == lock_sha and frozen['verification'] == sha(verification)
    assert frozen['predictions'] == sha(HERE / 'FORECAST.npz')
    write('SUMMARY.json', pack(dict(utc=now(), lock=lock_sha, forecast=sha(forecast_path), **summarize(frozen))))


if __name__ == '__main__':
    main()
