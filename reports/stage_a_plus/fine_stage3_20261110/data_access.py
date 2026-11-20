"""Open frozen TRAIN-normalized encoders and calibration-only FHN queries."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ACTIVE = HERE.parent / 'fine_active_shift_20261109'
sys.path.insert(0, str(ACTIVE))
spec = importlib.util.spec_from_file_location('_stage3_frozen_active', ACTIVE / 'run.py')
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)
sys.path.remove(str(ACTIVE))

SLOT = 'fhn_fresh_eps05'
REP = 0
TARGET_EPS = .08
GROUPS = 16
CALIBRATION_SEED = 2026289100
QUERY_SEED = 2026289200
TIMES = ac.TIMES
PILOT_FORECAST = ACTIVE / 'FORECAST.json'
ACTIVE_LOCK = ACTIVE / 'LOCK.json'


def checkpoint_reference():
    sources = ac.base.read(ACTIVE_LOCK)['sources']
    suffix = f'/representation/OursFine4/{SLOT}/seed{REP}/model/Fine_latest.pt'
    path, digest = next((p, h) for p, h in sources.items() if p.endswith(suffix))
    if ac.base.sha(Path(path)) != digest:
        raise ValueError('Frozen Fine checkpoint differs from the original active study')
    return {'path': path, 'sha256': digest}


def open_views():
    """Return native Fine and raw callables, the historical permutation, and provenance."""
    checkpoint = checkpoint_reference()
    _, historical, _, _ = ac.src.reference.historical(SLOT)
    views = {name: historical[name][1] for name in ('fine', 'raw')}
    perm = np.asarray(ac.base.historical_match(SLOT, REP)['perm'], dtype=np.intp)
    if not np.array_equal(np.sort(perm), np.arange(4)):
        raise ValueError('Invalid historical axis permutation')
    train_covariance = {
        'raw': ac.bias.all_cov(historical['raw'][0]).tolist(),
        'native': ac.bias.all_cov(historical['fine'][0][..., perm]).tolist(),
    }
    provenance = dict(slot=SLOT, replica=REP, target_epsilon=TARGET_EPS,
                      source_epsilon=ac.src.SOURCE_EPS[SLOT], checkpoint=checkpoint,
                      permutation_source=str(ac.base.HISTORICAL_MATCH),
                      normalization='archived Fine G0 and raw public TRAIN moments',
                      train_covariance=train_covariance)
    return views, perm, provenance


def pilot_pairs():
    """Replay only the old panel's calibration arrays and match its active receipt."""
    forecast = ac.base.read(PILOT_FORECAST)
    if forecast['lock_sha'] != ac.base.sha(ACTIVE_LOCK):
        raise ValueError('Pilot forecast is no longer linked to its original lock')
    row = forecast['slots'][SLOT]['panels'][0]
    calibration, expected = row['calibration'], row['active']
    cs, _, ss = ac.seeds(SLOT)['panels'][0]
    if (row['index'] != 0 or calibration['role'] != 'calibration'
            or calibration['slot'] != SLOT or calibration['epsilon'] != TARGET_EPS
            or calibration['count'] != 96 or calibration['seed'] != cs
            or expected['seed'] != ss or expected['contexts'] != list(TIMES)):
        raise ValueError('Unexpected historical pilot calibration identity')
    fold = np.arange(calibration['count']) % 4
    pairs, active = ac.observed_pairs(calibration, fold == 0, fold == 1, TIMES, ss)
    if active['hashes'] != expected['hashes']:
        raise ValueError('Historical active calibration array hash mismatch')
    return pairs, dict(calibration=calibration, active=active, pilot_hash_match=True)


def fresh_pairs(group):
    """Generate public shifts for a fresh 48-trajectory calibration panel."""
    if not isinstance(group, (int, np.integer)) or not 0 <= group < GROUPS:
        raise ValueError('Unknown Stage 3 calibration group')
    group = int(group)
    _, calibration = ac.src.shift_data.fresh_observations(
        SLOT, TARGET_EPS, 48, CALIBRATION_SEED + group, 'calibration')
    fit = np.arange(48) < 24
    pairs, active = ac.observed_pairs(calibration, fit, ~fit, TIMES, QUERY_SEED + group)
    return pairs, dict(calibration=calibration, active=active, group=group)


def source_hashes():
    """Return absolute-path hashes for reused sources and calibration replay."""
    checkpoint = checkpoint_reference()
    metadata = Path(checkpoint['path']).with_name('Fine.json')
    paths = (Path(__file__).resolve(), PILOT_FORECAST, ACTIVE_LOCK, metadata)
    original = ac.sources() | {str(p): ac.base.sha(p) for p in paths}
    return {str((ac.base.ROOT / path).resolve()): digest for path, digest in original.items()}
