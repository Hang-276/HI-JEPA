"""Frozen TRAIN anchor calibration, independent forecast, and separate reveal."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OBSERVABILITY = HERE.parent / 'fine_boundary_observability_20261108'
HISTORICAL_MATCH = HERE.parent / 'step6_fine_20261105/forecast_lock.json'
OLD_RUN = importlib.util.spec_from_file_location('observable_train', OBSERVABILITY / 'run.py')
old = importlib.util.module_from_spec(OLD_RUN)
OLD_RUN.loader.exec_module(old)

N = 1024
SEEDS = dict(zip(old.SLOTS, (2026110901, 2026110902)))
ALPHA = .05 / 6
NEAR = (.05, .02, .005)
PAIRS = tuple(itertools.combinations(range(4), 2))
SOURCE_PATHS = (HERE / 'PROTOCOL.md', Path(__file__), HERE / 'fresh_data.py',
                HERE / 'fine_forward.py', OBSERVABILITY / 'LOCK.json',
                OBSERVABILITY / 'run.py', HISTORICAL_MATCH, old.OWNER, HERE / 'ENGINEERING.json')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    destination = HERE / name
    temporary = destination.with_suffix('.part')
    if destination.exists():
        raise FileExistsError(destination)
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'),
                  allow_nan=False, default=lambda x: x.tolist())
        stream.write('\n')
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink()


def current_sources():
    from fresh_data import source_hashes
    sources = {str(p.relative_to(ROOT)): sha(p) for p in SOURCE_PATHS} | source_hashes()
    for check in read(HERE / 'ENGINEERING.json')['fine']:
        for ref in (check['completion'], check['job'], check['checkpoint'], check['g0'], *check['sources']):
            path = Path(ref['path'])
            sources[str(path)] = sha(path)
            assert sources[str(path)] == ref['sha256']
    return sources


def historical_match(slot, rep):
    info = next(r for r in read(HISTORICAL_MATCH)['parents']
                if r['slot'] == slot and r['rep'] == rep)
    assert info['assignment_objective_lead'] > 0
    return info


def inputs(slot):
    n, t, _ = old.SLOTS[slot]
    a, u, owner = old.context_arrays(slot, n, t)
    with np.load(old.BASE / 'data/TRAIN_CAL' / f'{slot}.npz', allow_pickle=False) as archive:
        x = archive['train__x']
    assert old.identity(x) == owner['diagnostics']['train']['arrays']['x']
    assert x.shape == (n, t, 4)
    return x.astype(np.float64), a, u, owner


def scale_fit(x):
    mean = x.mean((0, 1))
    sd = x.std((0, 1), ddof=0)
    assert (sd > 0).all()
    return mean, sd


def score_matrix(y):
    return old.dynamic_matrices(y)[0]


def point_sets(scores):
    ordered = np.argsort(scores, axis=1, kind='stable')
    sets = np.sort(ordered[:, :2], axis=1)
    values = np.take_along_axis(scores, ordered, axis=1)
    strict = values[:, 2] > values[:, 1]
    sets[~strict] = -1
    return sets


def margin(scores, sets):
    mask = (np.arange(4) == sets[:, :, None]).any(1)
    return np.min(np.where(mask, np.inf, scores), axis=-1) - np.max(
        np.where(mask, scores, -np.inf), axis=-1)


def fit_directions(train, a, u, groups, weights):
    ref, *direction_groups = groups
    err = old.residual(train, a, u, ref)
    sums, counts = old.episode_stats(err[..., None] * err[..., None, :], u)
    index = np.concatenate(direction_groups)
    pool = old.axes(sums[index].sum(0), counts[index].sum(0))
    fold = [old.axes(sums[g].sum(0), counts[g].sum(0)) for g in direction_groups]
    boot = old.axes(np.einsum('be,ecij->bcij', weights, sums[index]), weights @ counts[index])
    _, cov = old.dynamic_matrices(train)
    return pool, fold, boot, cov


def candidate(train, y, a, u, groups, weights):
    pool, folds, boot, cov = fit_directions(train, a, u, groups, weights)
    mm = score_matrix(y)
    count = int(boot['good'].sum())
    info = dict(pooled_condition=float(pool['condition']) if np.isfinite(pool['condition']) else None,
                pooled_dominance=pool['dominance'],
                fold_usable=[bool(f['good']) for f in folds],
                valid_bootstrap=count)
    if not pool['good']:
        return dict(info=info, scores=None, points=np.full((len(y), 2), -1),
                    accepted=np.zeros(len(y), dtype=bool), min_margin_q025=None)
    scores = old.slowness(pool['columns'][None], mm, cov)[0]
    points = point_sets(scores)
    strict = (points >= 0).all(1)
    eval_sets = np.where(strict[:, None], points, [0, 1])
    same = np.zeros(len(y), dtype=bool)
    if all(f['good'] for f in folds):
        f_scores = old.slowness(np.stack([f['columns'] for f in folds]), mm, cov)
        same = (margin(f_scores, eval_sets) > 0).all(0)
    q = None
    if count:
        bs = old.slowness(boot['columns'][boot['good']], mm, cov)
        q = np.quantile(margin(bs, eval_sets), .025, axis=0)
    accepted = (strict & same & (q > 0) if q is not None and count / old.NBOOT >= .95
                else np.zeros(len(y), dtype=bool))
    return dict(info=info, scores=scores, points=points, accepted=accepted, min_margin_q025=q)


def forecast_slot(slot):
    from fresh_data import fresh_observations
    from fine_forward import load_fine

    n, t, pair = old.SLOTS[slot]
    train, a, u, owner = inputs(slot)
    public, receipt = fresh_observations(slot, N, SEEDS[slot])
    x = public['x']
    assert x.shape == (N, t, 4)
    groups = [np.arange(i, n, 4) for i in range(4)]
    old_seed = read(OBSERVABILITY / 'LOCK.json')['plan'][slot]['seed']
    _, weights = old.make_schedule(groups[1:], old_seed)
    mu_x, sd_x = scale_fit(train)
    raw = candidate((train - mu_x) / sd_x, (x - mu_x) / sd_x, a, u, groups, weights)
    fine = []
    for rep in range(3):
        train_y, identity = old.fine_train(slot, rep, n, t, owner)
        info = historical_match(slot, rep)
        assert identity['dtype'] == 'float32'
        frozen = load_fine(slot, rep)
        y0 = frozen.encode(x)
        np.testing.assert_allclose(frozen.mean, info['train_mu_y'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(frozen.std, info['train_sd_y'], atol=1e-12, rtol=0)
        assert y0.shape == x.shape and np.isfinite(y0).all()
        y = (y0.astype(np.float64) - info['train_mu_y']) / info['train_sd_y']
        native_scores = np.diagonal(score_matrix(y), axis1=1, axis2=2)[:, info['perm']]
        native = point_sets(native_scores)
        rule = candidate(train_y, y, a, u, groups, weights)
        rule['warns'] = rule['accepted'] & (rule['points'] != native).any(1) & (native >= 0).all(1)
        raw_warns = raw['accepted'] & (raw['points'] != native).any(1) & (native >= 0).all(1)
        fine.append(dict(rep=rep, feature_identity=old.identity(y0), provenance=frozen.provenance,
                         native_scores=native_scores,
                         native=native, anchor=rule, raw_warns=raw_warns))
    return dict(pair=pair, receipt=receipt, raw=raw, fine=fine)


def exact_binomial(a, b):
    n = a + b
    return sum(math.comb(n, i) for i in range(a, n + 1)) / (2 ** n)


def paired(ours, ref, subset):
    n = int(subset.sum())
    a = int((ours & ~ref & subset).sum())
    b = int((ref & ~ours & subset).sum())
    return dict(n=n, win=a, loss=b, gain=(a - b) / n if n else None,
                p_one_sided=exact_binomial(a, b))


def fraction(mask, subset):
    count, n = int((mask & subset).sum()), int(subset.sum())
    return dict(count=count, n=n, rate=count / n if n else None)


def cp_upper(k, n, alpha=.05):
    if n == 0 or k == n:
        return 1.
    log_coeff = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                 for i in range(k + 1)]
    lo, hi = k / n, 1.
    for _ in range(48):
        p = (lo + hi) / 2
        terms = (c + i * math.log(p) + (n - i) * math.log1p(-p)
                 for i, c in enumerate(log_coeff))
        if math.fsum(math.exp(v) for v in terms) > alpha:
            lo = p
        else:
            hi = p
    return hi


def evaluate_method(data, truth, subsets):
    points = np.asarray(data['points'])
    accepted = np.asarray(data['accepted'])
    ok = (points == truth).all(1) & (truth >= 0).all(1)
    out = {}
    for key, mask in subsets.items():
        error = fraction(~ok & accepted, mask)
        out[key] = dict(point_hit=fraction(ok, mask), covered=fraction(accepted, mask),
                        wrong_and_decide=error, conditional_error=fraction(~ok, mask & accepted),
                        unconditional_upper_95=cp_upper(error['count'], error['n']))
    return ok, out


def physical_gaps(score, pair):
    a, b = pair
    return score[:, b] - score[:, a]


def interval_mae(prediction, target, mask):
    if prediction is None or not mask.any():
        return None
    estimate = np.asarray(prediction)
    gap_errors = [abs(physical_gaps(estimate, pair) - physical_gaps(target, pair))
                  for pair in PAIRS]
    return float(np.stack(gap_errors)[:, mask].mean())


def warning_result(warn, true_wrong, eligible):
    correct = eligible & ~true_wrong
    false_alarm = fraction(warn, correct)
    return dict(detection=fraction(warn, eligible & true_wrong), false_alarm=false_alarm,
                false_alarm_upper_95=cp_upper(false_alarm['count'], false_alarm['n']))


def score_slot(slot, data):
    from fresh_data import replay_physical

    pair = data['pair']
    z = replay_physical(data['receipt']).astype(np.float64)
    sd_z = np.asarray(historical_match(slot, 0)['train_sd_z'])
    assert all(np.array_equal(sd_z, historical_match(slot, r)['train_sd_z']) for r in (1, 2))
    target_score = np.diagonal(score_matrix(z / sd_z), axis1=1, axis2=2)
    truth = point_sets(target_score)
    all_windows = (truth >= 0).all(1)
    a, b = pair
    straddle = (truth == a).any(1) ^ (truth == b).any(1)
    gap = abs(physical_gaps(target_score, pair))
    edges = np.sort(target_score, axis=1)
    subsets = {'all': all_windows}
    for width in NEAR:
        subsets[f'near_{width}'] = all_windows & straddle & (gap <= width)
        subsets[f'any_edge_{width}'] = all_windows & ((edges[:, 2] - edges[:, 1]) <= width)
    raw_ok, raw = evaluate_method(data['raw'], truth, subsets)
    models = []
    for record in data['fine']:
        native = np.asarray(record['native'])
        native_defined = (native >= 0).all(1)
        native_ok = (native == truth).all(1) & all_windows
        anchor_ok, anchor = evaluate_method(record['anchor'], truth, subsets)
        eligible = all_windows & native_defined
        wrong = eligible & ~native_ok
        warn = np.asarray(record['anchor']['warns'])
        raw_warn = np.asarray(record['raw_warns'])
        near = subsets['near_0.05']
        point_comparisons = {}
        for name, reference in (('raw', raw_ok), ('native', native_ok)):
            main = paired(anchor_ok, reference, near)
            whole = paired(anchor_ok, reference, all_windows)
            main['passed'] = (main['n'] >= 40 and main['gain'] >= .05
                              and main['p_one_sided'] <= ALPHA and whole['gain'] >= -.01)
            point_comparisons[name] = dict(near=main, all=whole,
                                           narrow={name: paired(anchor_ok, reference, mask)
                                                   for name, mask in subsets.items() if name != 'all'})
        wa = warning_result(warn, wrong, eligible)
        wr = warning_result(raw_warn, wrong, eligible)
        warning_pair = paired(warn, raw_warn, wrong)
        warning_pair['passed'] = (warning_pair['n'] >= 20
                                  and max(wa['false_alarm_upper_95'], wr['false_alarm_upper_95']) <= .05
                                  and warning_pair['gain'] >= .10
                                  and warning_pair['p_one_sided'] <= ALPHA)
        models.append(dict(rep=record['rep'], native={k: fraction(native_ok, v) for k, v in subsets.items()},
                           anchor=anchor, comparisons=point_comparisons, anchor_warning=wa,
                           raw_warning=wr, warning_comparison=warning_pair, native_ties=int((~native_defined).sum()),
                           interval_mae=dict(anchor=interval_mae(record['anchor']['scores'], target_score, all_windows),
                                             raw=interval_mae(data['raw']['scores'], target_score, all_windows),
                                             native=interval_mae(record['native_scores'], target_score, all_windows)),
                           wrong_episode_indexes=np.flatnonzero(wrong),
                           recovered_wrong_indexes=np.flatnonzero(wrong & anchor_ok),
                           anchor_robust_wrong_indexes=np.flatnonzero(
                               np.asarray(record['anchor']['accepted']) & all_windows & ~anchor_ok)))
    primary, *replications = models
    same_direction = {}
    for name in ('raw', 'native'):
        same_direction[name] = any((rep['comparisons'][name]['near']['gain'] or 0) > 0 for rep in replications)
    same_direction['warning'] = any((rep['warning_comparison']['gain'] or 0) > 0 for rep in replications)
    gates = {name: primary['comparisons'][name]['near']['passed'] and same_direction[name]
             for name in ('raw', 'native')}
    gates['warning'] = primary['warning_comparison']['passed'] and same_direction['warning']
    return dict(pair=pair, n=len(z), truth_ties=int((~all_windows).sum()),
                subset_counts={k: int(v.sum()) for k, v in subsets.items()},
                raw=raw, raw_fit=data['raw']['info'], models=models, gates=gates,
                anchor_fits=[f['anchor']['info'] for f in data['fine']],
                truth_scores=target_score)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('phase', choices=('lock', 'forecast', 'score'))
    phase = cli.parse_args().phase
    if phase == 'lock':
        write('LOCK.json', dict(utc=now(), sources=current_sources(), seeds=SEEDS, n_per_slot=N,
                                near=NEAR, multiplicity_threshold=ALPHA,
                                evaluation='new independent trajectories from the same frozen instances'))
        return
    lock_file = HERE / 'LOCK.json'
    lock = read(lock_file)
    assert lock['sources'] == current_sources() and lock['seeds'] == SEEDS and lock['n_per_slot'] == N
    if phase == 'forecast':
        results = {slot: forecast_slot(slot) for slot in old.SLOTS}
        write('FORECAST.json', dict(utc=now(), lock_sha256=sha(lock_file), slots=results))
        return
    forecast = HERE / 'FORECAST.json'
    prediction = read(forecast)
    assert prediction['lock_sha256'] == sha(lock_file)
    results = {slot: score_slot(slot, prediction['slots'][slot]) for slot in old.SLOTS}
    write('SUMMARY.json', dict(utc=now(), lock_sha256=sha(lock_file),
                               forecast_sha256=sha(forecast), slots=results))


if __name__ == '__main__':
    main()
