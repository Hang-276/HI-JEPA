"""Run the unchanged small-panel protocol on two additional frozen instances."""
import argparse
import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLICATION = HERE.parent / 'fine_calibration_replication_20261109'
spec = importlib.util.spec_from_file_location('replication_extension_protocol', REPLICATION / 'run.py')
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

import platform
import sys
from extra_fine_forward import load_fine, smoke
from extra_fresh_data import fresh_observations, replay_physical, source_hashes, verify_historical_train

SLOTS = {'ar_rich_fresh2': (128, 129, (1, 2)), 'fhn_fresh_eps05': (96, 193, (1, 2))}
PRIMARY = 'fhn_fresh_eps05'
rep.bias.SLOTS = tuple(SLOTS)
rep.base.old.SLOTS = SLOTS
rep.load_fine = load_fine
rep.fresh_observations = fresh_observations
rep.replay_physical = replay_physical
rep.HERE = HERE


def schedule(slot):
    s = tuple(SLOTS).index(slot)
    return [(2026113000 + 1000 * s + 2 * k, 2026113001 + 1000 * s + 2 * k)
            for k in range(rep.PANEL_COUNT)]


rep.schedule = schedule


def sources():
    paths = (Path(__file__), HERE / 'PROTOCOL.md', HERE / 'ENGINEERING.json',
             HERE / 'extra_fine_forward.py', HERE / 'extra_fresh_data.py',
             REPLICATION / 'run.py', REPLICATION / 'LOCK.json', REPLICATION / 'SUMMARY.json')
    refs = rep.base.current_sources() | source_hashes() | {str(p): rep.base.sha(p) for p in paths}
    for row in rep.base.read(HERE / 'ENGINEERING.json')['fine']:
        for ref in (row['completion'], row['job'], row['checkpoint'], row['g0'], *row['sources']):
            refs[ref['path']] = rep.base.sha(Path(ref['path']))
            assert refs[ref['path']] == ref['sha256']
    return refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('verify', 'lock', 'forecast', 'score'))
    phase = parser.parse_args().phase
    lock_file = HERE / 'LOCK.json'
    forecast_file = HERE / 'FORECAST.json'
    if phase == 'verify':
        generation = [verify_historical_train(slot) for slot in SLOTS]
        fine = [smoke(slot, seed) for slot in SLOTS for seed in range(3)]
        rep.write('ENGINEERING.json', dict(utc=rep.base.now(), architecture=platform.machine(),
                  python=sys.version.split()[0], generation=generation, fine=fine))
        return
    if phase == 'lock':
        rep.write('LOCK.json', dict(utc=rep.base.now(), sources=sources(), slots=SLOTS, primary=PRIMARY,
                  seeds={s: schedule(s) for s in SLOTS}, panels=rep.PANEL_COUNT, evaluation=rep.EVAL_N))
        return
    lock_sha = rep.base.sha(lock_file)
    assert rep.base.read(lock_file)['sources'] == sources()
    if phase == 'forecast':
        slots = {s: rep.forecast_slot(s) for s in SLOTS}
        rep.write('FORECAST.json', dict(utc=rep.base.now(), lock_sha=lock_sha, slots=slots))
        return
    forecast = rep.base.read(forecast_file)
    assert forecast['lock_sha'] == lock_sha
    slots = {s: rep.score_slot(s, forecast['slots'][s]) for s in SLOTS}
    rep.write('SUMMARY.json', dict(utc=rep.base.now(), lock_sha=lock_sha,
              forecast_sha=rep.base.sha(forecast_file), slots=slots))


if __name__ == '__main__':
    main()
