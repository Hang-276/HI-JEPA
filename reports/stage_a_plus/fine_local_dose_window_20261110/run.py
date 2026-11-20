"""Freeze, forecast and independently score the prospective Fine test."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

from dose_score import evaluate, lag_scores
from query_data import STAGE3, endpoint_branches, prior, public_panel, shift

HERE = Path(__file__).resolve().parent
PILOT = STAGE3 / 'PILOT.json'
PILOT_SHA = '1556767c68d23052a7a321691db20a129c6ed3df320ab901b5339a563abcb6a0'
RULES = ('native', 'affine', 'magnitude', 'signed')
REFERENCE_SEED = 2026331100
PANEL_SEED = 2026331200
PANELS, N, SCREEN, ALARMS = 16, 128, 24, 6
DOSE = .0175
BOOTSTRAP_SEED = 2026331301


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((HERE / name).read_text())


def save(name, data):
    with (HERE / name).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def sources():
    local = [HERE / name for name in ('PROTOCOL.md', 'run.py', 'query_data.py', 'dose_score.py')]
    previous = [PILOT, STAGE3 / 'LOCK.json', STAGE3 / 'run.py']
    return prior.source_hashes() | {str(path): sha(path) for path in local + previous}


def verify_lock():
    frozen = read('LOCK.json')
    for path, digest in frozen['sources'].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f'Source differs from lock: {path}')
    return sha(HERE / 'LOCK.json')


def fit_reference(count, seed):
    assert sha(PILOT) == PILOT_SHA
    pilot = json.loads(PILOT.read_text())
    views, perm, provenance = prior.open_views()
    assert perm.tolist() == pilot['perm'] and provenance == pilot['provenance']
    control = pilot['controls']['fine']
    assert control['available'] and pilot['engineering_pass']
    beta = np.asarray(control['beta'])

    def encode(values):
        return np.asarray(views['fine'](values), dtype=np.float64)[..., perm]

    public, receipt = public_panel(count, seed)
    y = encode(public['x'])
    scale = y.std((0, 1))
    affine_scale = (y @ beta).std((0, 1))
    assert np.isfinite(scale).all() and (scale > 0).all() and (affine_scale > 0).all()
    return encode, scale, beta, affine_scale, receipt


def evaluate_public(index, seed, count, screen, alarms, reference, engineering=False):
    encode, scale, beta, affine_scale, _ = reference
    public, receipt = public_panel(count, seed)
    y = encode(public['x'])
    native = lag_scores(y / scale)
    old = prior.ac.base
    sets = old.point_sets(native)
    strict = (sets >= 0).all(1)
    safe_sets = np.where(strict[:, None], sets, [0, 1])
    gaps = np.where(strict, old.margin(native, safe_sets), 0.)
    selected = np.argsort(gaps, kind='stable')[:screen]
    plus, minus, audit = endpoint_branches(public, receipt, selected, DOSE)
    branch_values = encode(np.stack([plus, minus]))
    local = evaluate(y[selected], *branch_values, scale, beta, affine_scale, DOSE)
    np.testing.assert_array_equal(local['native_set'], sets[selected])
    np.testing.assert_allclose(local['native_gap'], gaps[selected], rtol=0, atol=1e-10)
    scores = {key: local[f'{key}_gap'] for key in RULES}
    orders = {name: np.lexsort((selected, values))[:alarms] for name, values in scores.items()}
    flagged = {name: selected[rows].tolist() for name, rows in orders.items()}
    assert flagged['native'] == np.argsort(gaps, kind='stable')[:alarms].tolist()
    if engineering:
        historical = np.diagonal(old.score_matrix(y / scale), axis1=1, axis2=2)
        duplicated = encode(np.stack([public['x'][selected], public['x'][selected]]))
        audit.update(score_parity=float(np.max(np.abs(native - historical))),
                     duplicate_error=float(np.max(np.abs(duplicated - y[selected]))))
    audit.update(safe_count=int(local['safe'].sum()),
                 max_condition=float(local['max_condition'].max()))
    return dict(index=index, evaluation=receipt, candidates=selected.tolist(),
                native_set=sets.tolist(), gaps={name: values.tolist() for name, values in scores.items()},
                flags=flagged, audit=audit)


def smoke(lock_sha):
    start = time.perf_counter()
    reference = fit_reference(8, 2026331000)
    panel = evaluate_public(-1, 2026331001, 8, 2, 1, reference, engineering=True)
    audit = panel['audit']
    elapsed = time.perf_counter() - start
    passed = (audit['public_parity'] and audit['score_parity'] <= 1e-10
              and audit['duplicate_error'] <= 1e-4 and audit['safe_count'] == 2 and elapsed <= 120.)
    save('ENGINEERING.json', dict(utc=utc(), lock_sha=lock_sha, elapsed_seconds=elapsed,
                                  passed=passed, reference=reference[-1], panel=panel))
    print(f'engineering passed={passed}, seconds={elapsed:.2f}', flush=True)


def forecast(lock_sha):
    trial = read('ENGINEERING.json')
    assert trial['passed'] and trial['lock_sha'] == lock_sha
    reference = fit_reference(256, REFERENCE_SEED)
    panels = []
    for index in range(PANELS):
        panels.append(evaluate_public(index, PANEL_SEED + index, N, SCREEN, ALARMS, reference))
        if (index + 1) % 4 == 0:
            print(f'finished {index + 1}/{PANELS} observable panels', flush=True)
    save('FORECAST.json', dict(utc=utc(), lock_sha=lock_sha, reference=reference[-1], panels=panels))


def score_panel(row, physical_sd):
    old = prior.ac.base
    physical = shift.replay_physical(row['evaluation']).astype(np.float64)
    true_scores = lag_scores(physical / physical_sd)
    truth = old.point_sets(true_scores)
    native = np.asarray(row['native_set'])
    valid = (truth >= 0).all(1) & (native >= 0).all(1)
    wrong = valid & np.any(truth != native, axis=1)
    selected = np.asarray(row['candidates'])
    flags = {key: np.asarray(row['flags'][key]) for key in RULES}
    assert all(len(np.unique(ids)) == ALARMS and np.isin(ids, selected).all() for ids in flags.values())
    true_gap = old.margin(true_scores[selected], np.where((native[selected] >= 0).all(1)[:, None],
                                                         native[selected], [0, 1]))
    oracle_order = np.lexsort((selected, np.where(valid[selected], true_gap, np.inf)))
    oracle = selected[oracle_order[:ALARMS]]
    hits = {key: ids[wrong[ids]] for key, ids in (flags | {'oracle': oracle}).items()}
    exchange = {key: dict(signed_only=int(len(np.setdiff1d(hits['signed'], hits[key]))),
                          baseline_only=int(len(np.setdiff1d(hits[key], hits['signed']))))
                for key in RULES if key != 'signed'}
    gaps = {key: np.asarray(row['gaps'][key]) for key in RULES}
    keep = valid[selected]
    return dict(index=row['index'], wrong_total=int(wrong.sum()), wrong_queried=int(wrong[selected].sum()),
                wrong_ids=np.flatnonzero(wrong).tolist(), invalid=int((~valid).sum()),
                safe_count=row['audit']['safe_count'], hit={key: len(ids) for key, ids in hits.items()},
                hit_ids={key: ids.tolist() for key, ids in hits.items()}, exchange=exchange,
                truth_gap=true_gap.tolist(), valid_count=int(keep.sum()),
                abs_gap_error={key: float(np.abs(values - true_gap)[keep].sum()) for key, values in gaps.items()},
                direction_agreement=int(((np.sign(gaps['signed'] - gaps['affine']) ==
                                          np.sign(true_gap - gaps['affine'])) & keep).sum()))


def score(lock_sha):
    source = read('FORECAST.json')
    assert source['lock_sha'] == lock_sha and len(source['panels']) == PANELS
    physical_ref = shift.replay_physical(source['reference']).astype(np.float64)
    physical_sd = physical_ref.std((0, 1))
    panels = [score_panel(row, physical_sd) for row in source['panels']]
    support = np.asarray([row['wrong_queried'] for row in panels])
    total = sum(row['wrong_total'] for row in panels)
    bootstrap = np.random.default_rng(BOOTSTRAP_SEED).integers(PANELS, size=(4096, PANELS))
    denominator = support[bootstrap].sum(1)
    ok = denominator > 0
    hits = {key: np.asarray([row['hit'][key] for row in panels]) for key in (*RULES, 'oracle')}
    contrasts = {}
    for key in RULES:
        if key == 'signed':
            continue
        difference = hits['signed'] - hits[key]
        change = float(difference.sum() / support.sum()) if support.sum() else None
        ci = (np.quantile(difference[bootstrap].sum(1)[ok] / denominator[ok], [.025, .975]).tolist()
              if ok.any() else [None, None])
        contrasts[key] = dict(delta_queried_recall=change, ci95=ci,
                              exchange={direction: sum(row['exchange'][key][direction] for row in panels)
                                        for direction in ('signed_only', 'baseline_only')},
                              passed=change is not None and change >= .05 and ci[0] > 0.)
    usable = sum(row['safe_count'] for row in panels)
    enough = int(support.sum()) >= 40 and int((support > 0).sum()) >= 8
    decision = ('engineering_insufficient' if usable < .95 * PANELS * SCREEN else
                'insufficient_wrong_windows' if not enough else
                'passed' if all(value['passed'] for value in contrasts.values()) else 'no_separation')
    valid = sum(row['valid_count'] for row in panels)
    summary = dict(utc=utc(), lock_sha=lock_sha, forecast_sha=sha(HERE / 'FORECAST.json'),
                   decision=decision, wrong_total=int(total), wrong_queried=int(support.sum()),
                   panels_with_wrong_queried=int((support > 0).sum()), queried=PANELS * SCREEN,
                   safe_count=usable, scored=N * PANELS, alarm_count_per_rule=ALARMS * PANELS,
                   invalid=sum(row['invalid'] for row in panels), physical_reference_sd=physical_sd.tolist(),
                   hits={key: dict(count=int(h.sum()),
                                   queried_recall=float(h.sum() / support.sum()) if support.sum() else None,
                                   full_recall=float(h.sum() / total) if total else None) for key, h in hits.items()},
                   contrasts=contrasts, post_reveal=dict(valid_queried=valid,
                       gap_mae={key: sum(row['abs_gap_error'][key] for row in panels) / valid if valid else None
                                for key in RULES},
                       direction_agreement=sum(row['direction_agreement'] for row in panels) / valid if valid else None),
                   panels=panels)
    save('SUMMARY.json', summary)
    print(json.dumps({key: summary[key] for key in ('decision', 'wrong_total', 'wrong_queried', 'hits', 'contrasts')},
                     indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('lock', 'smoke', 'forecast', 'score'))
    phase = parser.parse_args().phase
    if phase == 'lock':
        assert sha(PILOT) == PILOT_SHA
        save('LOCK.json', dict(utc=utc(), sources=sources(), checkpoint=prior.checkpoint_reference()))
    else:
        {'smoke': smoke, 'forecast': forecast, 'score': score}[phase](verify_lock())


if __name__ == '__main__':
    main()
