"""Stage 3: falsify scalar coordinates with paired named shifts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

from data_access import fresh_pairs, open_views, pilot_pairs, source_hashes
from metrics import comparison, summarize

HERE = Path(__file__).resolve().parent
PANELS, BOOT = 16, 2026289301
DOSES = np.array([.035, .035 * np.sqrt(2), -.0175])
INPUTS = ('fine', 'raw')
SOURCES = ('run.py', 'data_access.py', 'metrics.py', 'PROTOCOL.md')


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def save(name, data):
    with (HERE / name).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read(name):
    return json.loads((HERE / name).read_text())


def check_doses(pairs):
    np.testing.assert_allclose(pairs['fit']['shifts'],
                               (DOSES[:2, None, None] * np.eye(4)[None]).reshape(8, 4),
                               atol=1e-15, rtol=0)
    np.testing.assert_allclose(pairs['check']['shifts'], DOSES[2]*np.eye(4), atol=1e-15, rtol=0)


def response(encode, pairs, order):
    matrices, duplicates = [], []
    for role in ('fit', 'check'):
        baseline, branches = pairs[role]['baseline'], pairs[role]['branches']
        assert baseline.shape == (24, 8, 4)
        assert branches.shape == ((8 if role == 'fit' else 4), 24, 8, 4)
        values = encode(np.concatenate([baseline[None], baseline[None], branches], axis=0))
        values = np.asarray(values, dtype=np.float64)[..., order]
        # Duplicate is the same observation. The separate encoding also tests batching parity.
        separately = np.asarray(encode(baseline), dtype=np.float64)[..., order]
        duplicates.append(float(max(np.abs(values[0] - values[1]).max(),
                                    np.abs(values[0] - separately).max())))
        matrices.append(values[2:] - values[0])
    positive, negative = matrices
    return np.concatenate([positive.reshape(2, 4, 24, 8, 4), negative[None]], axis=0), max(duplicates)


def readout(changes, covariance):
    primary = changes[0]
    target = np.broadcast_to(DOSES[0]*np.eye(4)[:, None, None], primary.shape)
    x = primary.reshape(-1, 4)
    scale = np.sqrt((x**2).mean(axis=0))
    if np.any(scale <= 0):
        return dict(available=False, condition=None)
    beta, _, rank, singular = np.linalg.lstsq(x/scale, target.reshape(-1, 4), rcond=None)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
    if rank < 4 or condition is None or condition >= 1e6:
        return dict(available=False, condition=condition)
    beta /= scale[:, None]
    norm = np.sqrt(np.diag(beta.T @ np.asarray(covariance) @ beta))
    assert np.isfinite(norm).all() and np.all(norm > 0)
    return dict(available=True, condition=condition, beta=beta.tolist(), reference_sd=norm.tolist())


def pilot():
    views, perm, provenance = open_views()
    pairs, receipt = pilot_pairs()
    check_doses(pairs)
    controls, duplicate, parity = {}, {}, {}
    for name in INPUTS:
        order = perm if name == 'fine' else np.arange(4)
        values, duplicate[name] = response(views[name], pairs, order)
        scale_name = 'native' if name == 'fine' else 'raw'
        controls[name] = readout(values, provenance['train_covariance'][scale_name])
        parity[name] = list(values.shape)
    zero = np.zeros((2, 3, 4, 1, 1, 4))
    for i, dose in enumerate(DOSES):
        zero[:, i, :, 0, 0] = dose * np.diag([-2., .6, 4., -1.])
    ideal = summarize(zero, BOOT, mapped=False)
    assert np.max(np.abs(ideal['best_permutation_off_target']['estimate'])) <= 1e-14
    return dict(duplicate=duplicate, shapes=parity, controls=controls, perm=perm.tolist(),
                provenance=provenance, receipt=receipt,
                engineering_pass=bool(max(duplicate.values()) <= 1e-4))


def forecast(pilot_record):
    views, perm, provenance = open_views()
    assert np.array_equal(perm, pilot_record['perm']) and provenance == pilot_record['provenance']
    recorded = {name: [] for name in INPUTS}
    duplicates = {name: [] for name in INPUTS}
    receipts = []
    for group in range(PANELS):
        pairs, receipt = fresh_pairs(group)
        check_doses(pairs)
        receipts.append(receipt)
        for name in INPUTS:
            order = perm if name == 'fine' else np.arange(4)
            value, duplicate = response(views[name], pairs, order)
            recorded[name].append(value)
            duplicates[name].append(duplicate)
        if (group+1) % 4 == 0:
            print(f'calibration panels {group+1}/{PANELS}', flush=True)
    with (HERE / 'RESPONSES.npz').open('xb') as stream:
        np.savez_compressed(stream, **{name: np.stack(values) for name, values in recorded.items()})
    return dict(panels=receipts, duplicates=duplicates, responses=digest('RESPONSES.npz'))


def score(pilot_record, frozen):
    assert digest('RESPONSES.npz') == frozen['responses']
    with np.load(HERE / 'RESPONSES.npz', allow_pickle=False) as packet:
        changes = {name: packet[name] for name in INPUTS}
    assert all(a.shape == (PANELS, 3, 4, 24, 8, 4) for a in changes.values())
    metrics, calibrated = {}, {}
    for name in INPUTS:
        metrics[name] = summarize(changes[name], BOOT, mapped=False)
        control = pilot_record['controls'][name]
        if not control['available']:
            continue
        calibrated[name] = changes[name] @ np.asarray(control['beta'])
        physical = summarize(calibrated[name], BOOT, mapped=True)
        standardized = calibrated[name] / np.asarray(control['reference_sd'])
        metrics[name+'_affine'] = summarize(standardized, BOOT, mapped=False)
        metrics[name+'_affine']['relative_rmse'] = physical['relative_rmse']
    compare = comparison(calibrated['fine'], calibrated['raw'], BOOT) if len(calibrated) == 2 else None
    best = metrics['fine']['best_permutation_off_target']
    duplicate = {k: max(v) for k, v in frozen['duplicates'].items()}
    gate = all(best['ci95'][dose][0] > .05 for dose in (0, 2)) and duplicate['fine'] <= 1e-4
    return dict(panels=PANELS, trajectories=PANELS*48, contexts_per_trajectory=8,
                doses=DOSES.tolist(), metrics=metrics, relative_rmse_fine_minus_raw=compare,
                duplicate_max=duplicate, gates=dict(scalar_falsified=bool(gate),
                                                   duplicate_sound=bool(duplicate['fine'] <= 1e-4)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'pilot', 'forecast', 'score'))
    phase = parser.parse_args().phase
    sources = source_hashes() | {name: digest(name) for name in SOURCES}
    if phase == 'lock':
        save('LOCK.json', dict(utc=utc(), sources=sources, groups=PANELS, bootstrap=BOOT,
                               trajectory_seed_base=2026289100, query_seed_base=2026289200))
        return
    assert read('LOCK.json')['sources'] == sources
    lock = digest('LOCK.json')
    if phase == 'pilot':
        save('PILOT.json', dict(utc=utc(), lock=lock, **pilot()))
        return
    check = read('PILOT.json')
    assert check['lock'] == lock and check['engineering_pass']
    if phase == 'forecast':
        save('FORECAST.json', dict(utc=utc(), lock=lock, pilot=digest('PILOT.json'), **forecast(check)))
        return
    frozen = read('FORECAST.json')
    assert frozen['lock'] == lock and frozen['pilot'] == digest('PILOT.json')
    save('SUMMARY.json', dict(utc=utc(), forecast=digest('FORECAST.json'), **score(check, frozen)))


if __name__ == '__main__':
    main()
