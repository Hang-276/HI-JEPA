"""Sealed, single-threaded regime-calibration pilot on existing Fine caches."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parent / 'phase1_cache_20261107'
STEP6 = HERE.parent / 'step6_fine_20261105'
sys.path.insert(0, str(P1))
from run_phase1 import BASE, ROOT, SLOTS, as_json, dynamic, member_head, model_dir, order, phys_file, train_calibrator
import summarize_phase1 as aggregation

PANEL = ('ar_rich_fresh1', 'fhn_fresh_eps008')
OLD_METHODS = ('b_fixed', 'b_current', 'direction')
METHODS = OLD_METHODS + ('pooled', 'regime', 'mismatched')
OWNER = BASE / 'data' / 'DATA_PREPARED.json'
SOURCES = (HERE / 'PROTOCOL.md', Path(__file__).resolve(), HERE / 'INVENTORY.json',
           P1 / 'run_phase1.py', P1 / 'summarize_phase1.py', P1 / 'SHA256SUMS',
           STEP6 / 'transfer.py', STEP6 / 'summarize.py', STEP6 / 'SHA256SUMS', OWNER)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    with (HERE / name).open('x') as out:
        json.dump(value, out, default=as_json, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        out.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def saved(path):
    manifest = dict((name, digest) for digest, name in
                    (line.split(maxsplit=1) for line in (path.parent / 'SHA256SUMS').read_text().splitlines()))
    assert sha(path) == manifest[path.name]
    return read(path)


def close(value, expected):
    np.testing.assert_allclose(value, expected, atol=1e-12, rtol=0)


def owner_identity(x):
    digest = hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode())
    digest.update(x.tobytes())
    return dict(dtype=str(x.dtype), shape=list(x.shape), sha256=digest.hexdigest())


def load_regimes(owner, split):
    slot = owner['slot_id']
    file = BASE / 'data' / 'TRAIN_CAL' / (slot + '.npz')
    assert owner['observations'] == dict(path=str(file), sha256=sha(file))
    with np.load(file, allow_pickle=False) as archive:
        assert set(archive.files) == {s+'__'+k for s in ('train', 'cal') for k in ('x', 'actions', 'regimes')}
        r = archive[split+'__regimes']
    expected = owner['diagnostics'][split]['arrays']
    assert owner_identity(r) == expected['regimes']
    assert r.dtype == np.int64 and ((r >= 0) & (r <= 8)).all()
    return r


def inventory():
    records = []
    owner = read(OWNER)
    assert owner['stage'] == 'TRAIN_CAL'
    for row in owner['records']:
        slot = row['slot_id']
        assert slot in SLOTS
        obs = BASE / 'data' / 'TRAIN_CAL' / (slot + '.npz')
        heads = {}
        with zipfile.ZipFile(obs) as archive:
            for split in ('train', 'cal'):
                regime = load_regimes(row, split)
                counts = np.bincount(regime.ravel(), minlength=9)
                support = [(regime == u).any(1).sum() for u in range(9)]
                for field in ('x', 'actions', 'regimes'):
                    key = f'{split}__{field}'
                    with archive.open(key + '.npy') as member:
                        head = member_head(member)
                    identity = row['diagnostics'][split]['arrays'][field]
                    assert not head['fortran_order'] and list(head['shape']) == identity['shape']
                    assert str(np.dtype(head['descr'])) == identity['dtype']
                n, t = regime.shape
                assert row['diagnostics'][split]['arrays']['x']['shape'] == [n, t+1, 4]
                assert row['diagnostics'][split]['arrays']['actions']['shape'] == [n, t, 2]
                heads[split] = dict(state_shape=[n, t+1, 4], action_shape=[n, t, 2], regime_shape=[n, t],
                                    counts=counts, trajectories_with_regime=support)
        features = []
        for rep in range(3):
            folder = model_dir(slot, rep)
            comp = read(folder / 'COMPLETED.json')
            assert (comp['slot_id'], comp['logical_rep'], comp['valid_t_start']) == (slot, rep, 0)
            path = folder / 'FEATURES.npz'
            assert comp['features'] == dict(path=str(path), sha256=sha(path))
            job_path = BASE / 'execution' / 'registration' / 'representation' / 'OursFine4' / slot / f'seed{rep}' / 'JOB.json'
            assert comp['job'] == dict(path=str(job_path), sha256=sha(job_path))
            job = read(job_path)
            assert job['data_prepared'] == dict(path=str(OWNER), sha256=sha(OWNER))
            assert job['data_manifest']['source'] == row['observations']
            assert job['data_identity'] == {s: row['diagnostics'][s]['arrays'] for s in heads}
            with zipfile.ZipFile(path) as cache:
                for split in heads:
                    with cache.open(split+'.npy') as member:
                        head = member_head(member)
                    assert list(head['shape']) == heads[split]['state_shape']
                    assert str(np.dtype(head['descr'])) == 'float32' and not head['fortran_order']
            features.append(dict(rep=rep, features_sha256=comp['features']['sha256'], valid_t_start=0))
        records.append(dict(slot=slot, observations=row['observations'], split=heads, features=features))
    assert {r['slot'] for r in records} == set(SLOTS)
    write('INVENTORY.json', dict(utc=utc(), slots=records, owner_sha256=sha(OWNER),
                                 array_payloads_read=['train__regimes', 'cal__regimes']))


def allowed_array(slot, path, key, expected):
    allowed = {(phys_file(slot), 'train__physical')}
    allowed.update((model_dir(slot, i) / 'FEATURES.npz', name)
                   for i in range(3) for name in ('train', 'cal'))
    assert (path, key) in allowed
    with np.load(path, allow_pickle=False) as archive:
        x = archive[key]
    assert hashlib.sha256(x.tobytes()).hexdigest() == expected[(str(path.relative_to(ROOT)), key)]
    assert x.ndim == 3 and x.shape[-1] == 4 and x.dtype == np.float32 and np.isfinite(x).all()
    return x.astype(np.float64)


def incoming(regime):
    return np.pad(regime, ((0, 0), (1, 0)), constant_values=-1)


def conditional(y, z, labels, b, c):
    maps = []
    for u in range(9):
        mask = labels == u
        frames = int(mask.sum())
        trajectories = int(mask.any(1).sum())
        if trajectories < 32 or frames < 128:
            return None
        a, d = y[mask], z[mask]
        coef, _, rank, _ = np.linalg.lstsq(a - a.mean(0), d - d.mean(0), rcond=1e-12)
        if rank != 4:
            return None
        bb = coef.T
        cc = d.mean(0) - bb @ a.mean(0)
        maps.append(dict(regime=u, B=.75*bb+.25*b, intercept=.75*cc+.25*c,
                         frames=frames, independent_train_trajectories=trajectories))
    return maps


def mapped(y, labels, b, c, maps=()):
    value = y @ b.T + c
    if labels is not None:
        assert labels.shape == y.shape[:2] and (labels[:, 0] == -1).all()
        for item in maps:
            mask = labels == item['regime']
            value[mask] = y[mask] @ item['B'].T + item['intercept']
        close(value[:, 0], (y @ b.T + c)[:, 0])
    return value


def forecast(slot, only_one):
    start = time.monotonic()
    frozen = saved(P1 / f'freeze_{slot}.json')
    expected = {(r['path'], r['member']): r['full_member_sha256'] for r in frozen['access']}
    olds = {(p['slot'], p['rep']): p for p in saved(STEP6 / 'forecast_lock.json')['parents']}
    obs = next(r for r in read(OWNER)['records'] if r['slot_id'] == slot)
    rtrain, rcal = (incoming(load_regimes(obs, split)) for split in ('train', 'cal'))
    assert (rtrain[:, 1:] >= 0).all() and (rcal[:, 1:] >= 0).all()
    z = allowed_array(slot, phys_file(slot), 'train__physical', expected)
    rows, calibrators = [], []
    for parent in frozen['parents'][:1] if only_one else frozen['parents']:
        rep = parent['rep']
        old = olds[(slot, rep)]
        path = model_dir(slot, rep) / 'FEATURES.npz'
        yy, yc = (allowed_array(slot, path, split, expected) for split in ('train', 'cal'))
        y, cal = [((v-old['train_mu_y'])/old['train_sd_y'])[..., old['perm']]*old['sign'] for v in (yy, yc)]
        zz = (z-old['train_mu_z'])/old['train_sd_z']
        assert y.shape == zz.shape and y.shape[:2] == rtrain.shape and cal.shape[:2] == rcal.shape
        params = train_calibrator(y, zz)
        assert params['rank'] == 4 and params['available']
        for key in ('B', 'intercept', 'T', 'train_residual_cross'):
            close(params[key], parent['cal'][key])
        b, c = params['B'], params['intercept']
        physical_train = dynamic(zz).mean(0)
        ep = slice(0, 1) if only_one else slice(None)
        cache = {}
        values = {}
        for method, labels in (('pooled', (None, None)), ('regime', (rtrain, rcal)),
                               ('mismatched', (np.roll(rtrain, 1, axis=0), np.roll(rcal, 1, axis=0)))):
            train_u, cal_u = labels
            maps = () if train_u is None else conditional(y, zz, train_u, b, c)
            if maps is None:
                cache[method] = dict(available=False)
                values[method] = [None] * len(cal[ep])
                continue
            core = dynamic(mapped(y, train_u, b, c, maps)).mean(0)
            correction = physical_train-core
            close(core+correction, physical_train)
            estimate = dynamic(mapped(cal[ep], None if cal_u is None else cal_u[ep], b, c, maps))
            values[method] = (estimate + correction).mean(1)
            cache[method] = dict(available=True, maps=maps, unconditional_train_correction=correction)
            if method == 'pooled':
                close(correction, params['T']+params['train_residual_cross'])
        fine = dynamic(cal[ep]).mean(1)
        for j, old_row in enumerate(parent['episodes'][ep]):
            p = old_row['predictions']
            close(fine[j], p['identity'])
            assert order(fine[j]) == old_row['fine_top2'] and j == old_row['episode']
            close(values['pooled'][j], np.array(p['b_current'])+params['train_residual_cross'].mean(0))
            rows.append(dict(slot=slot, rep=rep, episode=j, train_r2=parent['train_r2'], y=fine[j],
                             predictions={**{m:p[m] for m in OLD_METHODS},
                                          **{m:v[j] for m,v in values.items()}}))
        calibrators.append(dict(rep=rep, B=b, intercept=c, methods=cache))
    return dict(slot=slot, rows=rows, calibrators=calibrators, utc=utc(),
                elapsed_seconds=time.monotonic()-start, phase1_freeze_sha256=sha(P1 / f'freeze_{slot}.json'),
                lock_sha256=sha(HERE / 'METHOD_LOCK.json'))


def reveal():
    locks = {slot: read(HERE / f'PREDICT_{slot}.json') for slot in PANEL}
    combined = []
    for slot, pred in locks.items():
        assert pred['lock_sha256'] == sha(HERE / 'METHOD_LOCK.json')
        original = saved(P1 / f'reveal_{slot}.json')
        assert original['forecast_sha'] == pred['phase1_freeze_sha256']
        truth = {(x['rep'],x['episode']): x for x in original['rows']}
        assert len(pred['rows']) == len(truth)
        for row in pred['rows']:
            old = truth[(row['rep'], row['episode'])]
            close(old['train_r2'], row['train_r2'])
            close(old['y'], row['y'])
            for m in OLD_METHODS:
                close(old['predictions'][m], row['predictions'][m])
            combined.append({**old, 'predictions': row['predictions']})
    assert len(combined) == 168
    aggregation.METHODS = METHODS
    aggregation.GATE_METHODS = ('pooled', 'regime', 'mismatched')
    metrics = aggregation.load_metrics()
    aggregates = aggregation.metric_bundle(combined, metrics)
    pairs = aggregation.pair_records(combined, metrics)
    matched = aggregation.matched_bundle(pairs, PANEL, metrics)
    common = all(aggregation.available(row, method) for row in combined for method in METHODS)
    rank_checks = {}
    for tier, stats in matched.items():
        comp = stats['overall']['common_all_methods']['methods']
        a, b, c = (comp[k] for k in ('regime', 'pooled', 'mismatched'))
        weight = 'equal_slot_episode_pair_concordance'
        each = 'per_slot_episode_weighted'
        rank_checks[tier] = bool(common and all(
            a[each].get(slot, -1) > b[each].get(slot, 1) for slot in PANEL)
            and a[weight] > c[weight])
    actual = aggregates['micro']['regime']['correctness_confusion']
    baseline = aggregates['micro']['pooled']['correctness_confusion']
    warning = bool(common and actual['TN'] > baseline['TN'] and actual['FN'] <= baseline['FN'])
    checks = dict(full_prediction_coverage=common, paired=rank_checks, warning=warning)
    write('SUMMARY.json', dict(utc=utc(), checks=checks, passed=all(rank_checks.values()) and warning,
                              metrics=aggregates, matched=matched, paired_records=pairs,
                              lock_sha256=sha(HERE / 'METHOD_LOCK.json'),
                              prediction_sha256={s:sha(HERE / f'PREDICT_{s}.json') for s in PANEL},
                              physical_score_sources={s:sha(P1 / f'reveal_{s}.json') for s in PANEL}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('inventory', 'lock', 'pilot', 'predict', 'reveal'))
    phase = parser.parse_args().phase
    if phase == 'inventory':
        inventory()
        return
    source_hash = {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}
    if phase == 'lock':
        write('METHOD_LOCK.json', dict(utc=utc(), sources=source_hash, panel=PANEL,
                                       methods=METHODS, model_training_runs=0, cpu_threads=1))
        return
    assert read(HERE / 'METHOD_LOCK.json')['sources'] == source_hash
    if phase == 'pilot':
        result = forecast(PANEL[0], True)
        assert len(result['rows']) == 1 and all(v is not None for v in result['rows'][0]['predictions'].values())
        write('PILOT.json', dict(**result, passed=True))
    elif phase == 'predict':
        pilot = read(HERE / 'PILOT.json')
        assert pilot['passed'] and pilot['lock_sha256'] == sha(HERE / 'METHOD_LOCK.json')
        for slot in PANEL:
            write(f'PREDICT_{slot}.json', forecast(slot, False))
    else:
        reveal()


if __name__ == '__main__':
    main()
