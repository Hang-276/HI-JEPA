"""Post-hoc same-query replay; no evaluation windows or new alarm performance."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
spec = importlib.util.spec_from_file_location('frozen_query_budget_for_scale_audit', PARENT / 'run.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(name, record):
    with (HERE / name).open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


def freeze():
    original_lock = prior.verify()
    source = {str(HERE / name): sha(HERE / name) for name in ('PROTOCOL.md', 'run.py', 'capture_worker.py', 'SELF_CHECK.json')}
    source.update({str(PARENT / name): sha(PARENT / name) for name in ('LOCK.json', 'FORECAST.json', 'SUMMARY.json')})
    save('LOCK.json', dict(utc=datetime.now(timezone.utc).isoformat(), original_lock_sha=original_lock,
                          sources=source, post_hoc=True, new_samples=0, new_query_information=0))


def verify():
    lock = read(HERE / 'LOCK.json')
    assert prior.verify() == lock['original_lock_sha']
    for path, digest in lock['sources'].items():
        assert sha(path) == digest, path
    return sha(HERE / 'LOCK.json')


def solve(z, jacobian, anchors):
    with tempfile.TemporaryDirectory(prefix='fine_rejected_scale_') as directory:
        directory = Path(directory)
        np.savez(directory / 'input.npz', z=z, anchor_jacobian=jacobian, anchor_ids=anchors)
        subprocess.run([prior.GRAPH_PYTHON, str(HERE / 'capture_worker.py'), '--input',
                        str(directory / 'input.npz'), '--output', str(directory / 'output.json')],
                       check=True, capture_output=True, text=True)
        return read(directory / 'output.json')


def replay(lock_sha):
    start = time.perf_counter()
    forecast = read(PARENT / 'FORECAST.json')
    old = prior.old
    views, permutation, provenance = old.prior.open_views()
    pilot = read(old.PILOT)
    assert permutation.tolist() == pilot['perm'] and provenance == pilot['provenance']
    betas = {rep: np.asarray(pilot['controls'][rep]['beta']) for rep in ('fine', 'raw')}

    def fine_encode(values):
        return np.asarray(views['fine'](values), dtype=np.float64)[..., permutation]

    def raw_encode(values):
        return np.asarray(views['raw'](values), dtype=np.float64)

    encoders = dict(fine=fine_encode, raw=raw_encode)
    rows = []
    for original in forecast['panels']:
        reference = original['reference']
        receipt = reference['receipt']
        public, replay_receipt = old.public_panel(receipt['count'], receipt['seed'])
        assert replay_receipt == receipt
        ordering = np.random.default_rng(reference['anchor_seed']).permutation(reference['node_count']).astype(np.int64)
        assert hashlib.sha256(ordering.tobytes()).hexdigest() == reference['permutation_sha256']
        anchors = ordering[:64]
        plus, minus, query = prior.node_branches(public, receipt, anchors, old.DOSE)
        assert query == reference['queries']['64']
        records = {}
        for rep in ('fine', 'raw'):
            features = encoders[rep](public['x'])
            z = features @ betas[rep]
            np.testing.assert_array_equal(features.std((0, 1)), reference['native_sd'][rep])
            np.testing.assert_array_equal(z.std((0, 1)), reference['affine_sd'][rep])
            plus_y, minus_y = encoders[rep](np.stack([plus, minus]))
            jacobian = ((plus_y - minus_y) @ betas[rep]) / (2 * old.DOSE)
            record = solve(z, jacobian, anchors)
            assert record['diagnostics'] == reference['graphs'][rep]['64'], (original['index'], rep)
            assert record['returned_scale'] == reference['scales'][rep]['64']
            records[rep] = record
        rows.append(dict(index=original['index'], receipt=receipt, query=query,
                         replay_exact=True, captures=records))
        if (original['index'] + 1) % 4 == 0:
            print(f"Verified original reference/query/capture {original['index'] + 1}/16", flush=True)
    save('REPLAY.json', dict(lock_sha=lock_sha, elapsed_seconds=time.perf_counter() - start,
                             original_forecast_sha=sha(PARENT / 'FORECAST.json'), labels_read=False, panels=rows))


def score(lock_sha):
    replayed = read(HERE / 'REPLAY.json')
    assert replayed['lock_sha'] == lock_sha
    forecast = read(PARENT / 'FORECAST.json')
    original_summary = read(PARENT / 'SUMMARY.json')
    rows = []
    for row, oldrow, scored in zip(replayed['panels'], forecast['panels'], original_summary['panels'], strict=True):
        assert row['index'] == oldrow['index'] == scored['index']
        physical = prior.old.shift.replay_physical(row['receipt']).astype(np.float64)
        truth_sd = physical.std((0, 1))
        np.testing.assert_array_equal(truth_sd, scored['physical_reference_sd'])
        scales = {f'{rep}_64': row['captures'][rep]['solution_scale'] for rep in ('fine', 'raw')}
        scales.update({f'{rep}_all': oldrow['reference']['scales'][rep]['all'] for rep in ('fine', 'raw')})
        errors = {key: np.asarray(value) / truth_sd - 1 for key, value in scales.items()}
        rows.append(dict(index=row['index'], physical_reference_sd=truth_sd.tolist(),
                         scale=scales, relative_error={key: value.tolist() for key, value in errors.items()},
                         mean_absolute_relative_error={key: float(np.abs(value).mean()) for key, value in errors.items()},
                         max_absolute_relative_error={key: float(np.abs(value).max()) for key, value in errors.items()},
                         edge_relative_rms={rep: row['captures'][rep]['diagnostics']['relative_edge_rms'] for rep in ('fine', 'raw')},
                         originally_accepted={rep: row['captures'][rep]['diagnostics']['available'] for rep in ('fine', 'raw')}))
    aggregate = {}
    for key in ('fine_64', 'raw_64', 'fine_all', 'raw_all'):
        error = np.asarray([row['relative_error'][key] for row in rows])
        aggregate[key] = dict(mean_absolute_relative_error=float(np.abs(error).mean()),
                              max_absolute_relative_error=float(np.abs(error).max()),
                              per_axis_mean_absolute_relative_error=np.abs(error).mean(0).tolist(),
                              per_panel_mean_absolute_relative_error=np.abs(error).mean(1).tolist())
    save('RESULT.json', dict(lock_sha=lock_sha, replay_sha=sha(HERE / 'REPLAY.json'), post_hoc=True,
                             main_decision_unchanged=original_summary['decision'], new_alarm_performance=False,
                             aggregate=aggregate, panels=rows))
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('lock', 'replay', 'score'))
    phase = parser.parse_args().phase
    if phase == 'lock':
        freeze()
    else:
        {'replay': replay, 'score': score}[phase](verify())
