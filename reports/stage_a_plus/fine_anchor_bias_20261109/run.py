"""Frozen comparison of sampling noise, moment choice, and affine readouts."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'fine_independent_validation_20261109'
sys.path.insert(0, str(PREVIOUS))
import importlib.util

spec = importlib.util.spec_from_file_location('independent_fine', PREVIOUS / 'run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

import json
import os
import numpy as np

from fine_forward import load_fine
from fresh_data import fresh_observations, replay_physical

SLOTS = tuple(base.old.SLOTS)
CAL_N, EVAL_N = 4096, 2048
TIMES = (8, 24, 40, 56, 72, 88, 104, 120)
SEEDS = {slot: dict(cal=2026110911 + i, evaluation=2026110921 + i, paired=2026110931 + i)
         for i, slot in enumerate(SLOTS)}
FORENSIC = ('native', 'passive_full', 'paired_p', 'labeled_state', 'labeled_increment')


def write(name, obj):
    target = HERE / name
    if target.exists():
        raise FileExistsError(target)
    tmp = target.with_suffix('.part')
    with tmp.open('x') as stream:
        json.dump(obj, stream, default=lambda x: x.tolist(), separators=(',', ':'),
                  ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    try:
        os.link(tmp, target)
    finally:
        tmp.unlink()


def sources():
    from paired_innovations import source_hashes
    return base.current_sources() | source_hashes() | {
        str(p): base.sha(p) for p in (Path(__file__), HERE / 'PROTOCOL.md',
                                      PREVIOUS / 'LOCK.json', PREVIOUS / 'SUMMARY.json')}


def arm_views(slot, train_x, x_cal, x_eval):
    n, t, _ = base.old.SLOTS[slot]
    _, _, owner = base.old.context_arrays(slot, n, t)
    mx, sx = base.scale_fit(train_x)
    yield 'raw', (train_x - mx) / sx, (x_cal - mx) / sx, (x_eval - mx) / sx, None
    for rep in range(3):
        train, _ = base.old.fine_train(slot, rep, n, t, owner)
        encoder = load_fine(slot, rep)
        cal = (encoder.encode(x_cal) - encoder.mean) / encoder.std
        evaluation = (encoder.encode(x_eval) - encoder.mean) / encoder.std
        yield f'fine{rep}', train, cal, evaluation, encoder


def design(y, a):
    return np.concatenate((np.ones(y.shape[:-1] + (1,)), y, y ** 2, y ** 3, a), axis=-1)


def fit_drift(train, a, u):
    group = np.arange(0, len(train), 4)
    f = design(train[group, :-1], a[group])
    mask = np.isin(u[group], base.old.CODES)
    target = np.diff(train[group], axis=1)
    coefficient, _, rank, _ = np.linalg.lstsq(f[mask], target[mask], rcond=1e-12)
    assert rank == f.shape[-1]
    return coefficient


def innovation(y, a, coefficient):
    return np.diff(y, axis=1) - design(y[:, :-1], a) @ coefficient


def stats(r, u):
    return base.old.episode_stats(r[..., None] * r[..., None, :], u)


def pooled(parts, mask):
    sums, counts = parts
    return base.old.axes(sums[mask].sum(0), counts[mask].sum(0))


def unit_rows(w, covariance):
    length = np.einsum('ai,ij,aj->a', w, covariance, w)
    assert (length > 0).all()
    return w / np.sqrt(length)[:, None]


def project(mm, rows):
    return np.einsum('ai,eij,aj->ea', rows, mm, rows)


def from_columns(data, covariance, mm):
    info = base.old.export_axes(data)
    if not data['good']:
        return dict(rows=None, scores=None, fit=info)
    rows = unit_rows(np.linalg.inv(data['columns']), covariance)
    return dict(rows=rows, scores=project(mm, rows), fit=info)


def all_cov(y):
    y = y.reshape(-1, 4)
    center = y - y.mean(0)
    return center.T @ center / len(center)


def calc_passive(train, cal, evaluation, train_a, train_u, public):
    n = len(train)
    cov = all_cov(train)
    mm = base.score_matrix(evaluation)
    coefficient = fit_drift(train, train_a, train_u)
    r = innovation(train, train_a, coefficient)
    parity = float(abs(r - base.old.residual(train, train_a, train_u, np.arange(0, n, 4))).max())
    assert parity < 1e-12
    historic = pooled(stats(r, train_u), np.arange(n) % 4 != 0)
    r = innovation(cal, public['actions'], coefficient)
    indexes = np.arange(len(cal))
    keep = indexes % 4 != 0
    parts = stats(r, public['regimes'])
    sparse_parts = stats(r[:, TIMES], public['regimes'][:, TIMES])
    axes = {'historical': historic,
            'passive_n': pooled(parts, keep & (indexes < n)),
            'passive_4n': pooled(parts, keep & (indexes < 4 * n)),
            'passive_full': pooled(parts, keep),
            'passive_sparse': pooled(sparse_parts, keep)}
    result = {name: from_columns(value, cov, mm) for name, value in axes.items()}
    by_block = [pooled(parts, keep & (indexes // (CAL_N // 4) == i)) for i in range(4)]
    result['block_angles'] = angles([item['columns'] for item in by_block])
    result['drift_parity_max'] = parity
    return result


def angles(columns):
    agreement = np.stack([abs((first * second).sum(0)).clip(0, 1)
                          for i, first in enumerate(columns) for second in columns[i + 1:]])
    return np.degrees(np.arccos(agreement.min(0)))


def passive(slot):
    cal, cal_receipt = fresh_observations(slot, CAL_N, SEEDS[slot]['cal'])
    evaluate, eval_receipt = fresh_observations(slot, EVAL_N, SEEDS[slot]['evaluation'])
    train_x, a, u, _ = base.inputs(slot)
    result = {}
    for name, train, yc, ye, encoder in arm_views(slot, train_x, cal['x'], evaluate['x']):
        result[name] = calc_passive(train, yc, ye, a, u, cal)
        if encoder is not None:
            info = base.historical_match(slot, int(name[-1]))
            rows = np.eye(4)[info['perm']] * np.asarray(info['sign'])[:, None]
            result[name]['native'] = dict(rows=rows, scores=project(base.score_matrix(ye), rows))
    return dict(cal=cal_receipt, evaluation=eval_receipt, arms=result)


def check_replay(slot, receipt, role):
    expected = CAL_N if role == 'cal' else EVAL_N
    public, current = fresh_observations(slot, expected, SEEDS[slot][role])
    assert receipt == current
    return public


def paired_moments(encode, paired, coefficient):
    current = encode(paired['current'])
    y0 = encode(paired['baseline'])
    drift = design(current, paired['actions']) @ coefficient
    r0 = y0 - current - drift
    r0 = r0.reshape(-1, 4)
    m0 = r0.T @ r0 / len(r0)
    d, p = [], []
    for arm in paired['spikes']:
        y1 = encode(arm)
        r1 = (y1 - current - drift).reshape(-1, 4)
        delta = (y1 - y0).reshape(-1, 4)
        d.append(r1.T @ r1 / len(r1) - m0)
        p.append(delta.T @ delta / len(delta))
    return d, p


def contrast_axes(matrices):
    values = np.stack([np.zeros((4, 4)), *matrices])
    return base.old.axes(values, np.ones(5))


def labeled_rows(y, z):
    yflat, zflat = y.reshape(-1, 4), z.reshape(-1, 4)
    yc, zc = yflat - yflat.mean(0), zflat - zflat.mean(0)
    state = np.linalg.lstsq(yc, zc, rcond=1e-12)[0].T
    xx, xz = [], []
    for h in base.old.LAGS:
        dy = (y[:, h:] - y[:, :-h]).reshape(-1, 4)
        dz = (z[:, h:] - z[:, :-h]).reshape(-1, 4)
        xx.append(dy.T @ dy / len(dy))
        xz.append(dy.T @ dz / len(dy))
    increment = np.linalg.solve(np.mean(xx, axis=0), np.mean(xz, axis=0)).T
    return state, increment


def sign_match(rows, y, z):
    yc = y.reshape(-1, 4)
    zc = z.reshape(-1, 4)
    corr = ((yc - yc.mean(0)) @ rows.T * (zc - zc.mean(0))).mean(0)
    assert (corr != 0).all()
    return np.sign(corr)


def controlled(slot, previous):
    from paired_innovations import paired_observations

    cal = check_replay(slot, previous['cal'], 'cal')
    evaluate = check_replay(slot, previous['evaluation'], 'evaluation')
    keep = np.arange(CAL_N) % 4 != 0
    sd = np.asarray(base.historical_match(slot, 0)['train_sd_z'])
    z = replay_physical(previous['cal']).astype(np.float64)[keep] / sd
    paired, audit = paired_observations(previous['cal'], SEEDS[slot]['paired'], TIMES)
    paired = {key: value[:, keep] if key == 'spikes' else value[keep] for key, value in paired.items()}
    x, a, u, _ = base.inputs(slot)
    mu_x, sd_x = base.scale_fit(x)
    arms = {}
    for name, train, yc, ye, encoder in arm_views(slot, x, cal['x'], evaluate['x']):
        if encoder is None:
            def encode(values):
                return (values - mu_x) / sd_x
        else:
            def encode(values):
                return (encoder.encode(values) - encoder.mean) / encoder.std
        cov = all_cov(train)
        mm = base.score_matrix(ye)
        coeff = fit_drift(train, a, u)
        d, p = paired_moments(encode, paired, coeff)
        candidate_d, candidate_p = contrast_axes(d), contrast_axes(p)
        state, increment = labeled_rows(yc[keep], z)
        arms[name] = {'paired_d': from_columns(candidate_d, cov, mm),
                      'paired_p': from_columns(candidate_p, cov, mm)}
        for label, rows in (('labeled_state', state), ('labeled_increment', increment)):
            rows = unit_rows(rows, cov)
            arms[name][label] = dict(rows=rows, scores=project(mm, rows))
        passive_sparse = np.asarray(previous['arms'][name]['passive_sparse']['fit']['columns'])
        passive_full = np.asarray(previous['arms'][name]['passive_full']['fit']['columns'])
        arms[name]['max_angle_deg'] = {
            label: angles(columns) for label, columns in (
                ('full_to_sparse', [passive_full, passive_sparse]),
                ('sparse_to_d', [passive_sparse, candidate_d['columns']]),
                ('d_to_p', [candidate_d['columns'], candidate_p['columns']]))}
        if name == 'fine0':
            for label in FORENSIC:
                forecast = arms[name].get(label, previous['arms'][name].get(label))
                if forecast is not None and forecast['rows'] is not None:
                    arms[name].setdefault('forensic_signs', {})[label] = sign_match(
                        np.asarray(forecast['rows']), yc[keep], z)
    return dict(arms=arms, paired_receipt=audit)


def decomposition(y, z, rows, signs, correct, edge):
    q = (y @ rows.T) * signs
    cross, energy = [], []
    for h in base.old.LAGS:
        dq = q[:, h:] - q[:, :-h]
        de = z[:, h:] - z[:, :-h] - dq
        cross.append((2 * dq * de).mean(1))
        energy.append((de ** 2).mean(1))
    cross, energy = np.mean(cross, axis=0), np.mean(energy, axis=0)
    target_scores = np.diagonal(base.score_matrix(z), axis1=1, axis2=2)
    estimate_scores = np.diagonal(base.score_matrix(q), axis1=1, axis2=2)
    discrepancy = float(np.max(abs(target_scores - estimate_scores - cross - energy)))
    assert discrepancy < 1e-12
    wrong = ~correct
    values = dict(max_reconstruction_error=discrepancy, wrong=int(wrong.sum()), near_wrong=int((wrong & edge).sum()))
    for label, mask in (('all', np.ones(len(y), bool)), ('wrong', wrong), ('near_wrong', wrong & edge)):
        if not mask.any():
            values[label] = None
            continue
        c = np.stack([base.physical_gaps(cross, pair) for pair in base.PAIRS])
        e = np.stack([base.physical_gaps(energy, pair) for pair in base.PAIRS])
        values[label] = dict(cross_mae=float(abs(c[:, mask]).mean()),
                             energy_mae=float(abs(e[:, mask]).mean()),
                             combined_mae=float(abs((c + e)[:, mask]).mean()))
    return values


def score(slot, p, c):
    receipt = p['evaluation']
    z = replay_physical(receipt).astype(np.float64) / base.historical_match(slot, 0)['train_sd_z']
    true = np.diagonal(base.score_matrix(z), axis1=1, axis2=2)
    target = base.point_sets(true)
    strict = (target >= 0).all(1)
    edge = np.sort(true, axis=1)
    near = strict & ((edge[:, 2] - edge[:, 1]) <= .05)
    a, b = base.old.SLOTS[slot][-1]
    cross = (target == a).any(1) ^ (target == b).any(1)
    specific = strict & cross & (abs(true[:, b] - true[:, a]) <= .05)
    masks = dict(all=strict, any_near=near, specified_near=specific)
    special = (1, 2) if slot.startswith('fhn') else (0, 2)
    arms = {}
    for name, passive_rows in p['arms'].items():
        group = {k: row for k, row in (passive_rows | c['arms'][name]).items()
                 if isinstance(row, dict) and 'scores' in row}
        methods, correctness = {}, {}
        for label, row in group.items():
            if row['scores'] is None:
                methods[label] = dict(available=False)
                continue
            predicted = np.asarray(row['scores'])
            points = base.point_sets(predicted)
            correct = (points == target).all(1) & strict
            correctness[label] = correct
            methods[label] = dict(available=True, **{
                k: dict(hit=base.fraction(correct, m),
                        gap_mae=base.interval_mae(predicted, true, m),
                        pair_bias=(float(np.mean((base.physical_gaps(predicted, special)
                                                  - base.physical_gaps(true, special))[m])) if m.any() else None))
                for k, m in masks.items()})
        paired = {f'{a}_vs_{b}': {k: base.paired(correctness[a], correctness[b], m)
                                   for k, m in masks.items()}
                  for a, b in (('passive_full', 'historical'), ('paired_d', 'passive_sparse'),
                               ('paired_p', 'paired_d'), ('labeled_state', 'passive_full'),
                               ('labeled_increment', 'passive_full'), ('passive_full', 'native'),
                               ('labeled_state', 'native'), ('labeled_increment', 'native'))
                  if a in correctness and b in correctness}
        arms[name] = dict(methods=methods, paired=paired, blocks=passive_rows['block_angles'],
                          moment_angles=c['arms'][name]['max_angle_deg'])
        if name == 'fine0':
            public = check_replay(slot, receipt, 'evaluation')
            encoder = load_fine(slot, 0)
            y = (encoder.encode(public['x']) - encoder.mean) / encoder.std
            arms[name]['residual'] = {
                label: decomposition(y, z, np.asarray(group[label]['rows']), signs, correctness[label], near)
                for label, signs in c['arms'][name].get('forensic_signs', {}).items()
            }
    return dict(support={k: int(mask.sum()) for k, mask in masks.items()}, pair=special,
                arms=arms, paired_receipt=c['paired_receipt'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'passive', 'controlled', 'score'))
    phase = parser.parse_args().phase
    lock = HERE / 'LOCK.json'
    passive_file = HERE / 'PASSIVE.json'
    controlled_file = HERE / 'CONTROLLED.json'
    if phase == 'lock':
        write('LOCK.json', dict(utc=base.now(), sources=sources(), seeds=SEEDS,
                                n_cal=CAL_N, n_evaluation=EVAL_N, times=TIMES))
        return
    assert base.read(lock)['sources'] == sources()
    lock_sha = base.sha(lock)
    if phase == 'passive':
        slots = {slot: passive(slot) for slot in SLOTS}
        write('PASSIVE.json', dict(utc=base.now(), lock_sha=lock_sha, slots=slots))
        return
    forecast = base.read(passive_file)
    assert forecast['lock_sha'] == lock_sha
    passive_sha = base.sha(passive_file)
    if phase == 'controlled':
        slots = {slot: controlled(slot, forecast['slots'][slot]) for slot in SLOTS}
        write('CONTROLLED.json', dict(utc=base.now(), lock_sha=lock_sha, passive_sha=passive_sha, slots=slots))
        return
    counterfactual = base.read(controlled_file)
    assert counterfactual['lock_sha'] == lock_sha and counterfactual['passive_sha'] == passive_sha
    slots = {slot: score(slot, forecast['slots'][slot], counterfactual['slots'][slot]) for slot in SLOTS}
    write('SUMMARY.json', dict(utc=base.now(), lock_sha=lock_sha, passive_sha=passive_sha,
                               controlled_sha=base.sha(controlled_file), slots=slots))


if __name__ == '__main__':
    main()
