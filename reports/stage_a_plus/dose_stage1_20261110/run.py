"""Stage 1: separate exact calibration constraints from joint score bounds."""

import argparse
import importlib.util
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

from knot_model import InfeasibleKnots, gaussian_variance, interpolate, make_knots
from linear_cert import LinearCertificate

HERE = Path(__file__).resolve().parent
PREVIOUS = '../dose_state_dependent_20261110/run.py'
SPEC = importlib.util.spec_from_file_location('dose_state', HERE / PREVIOUS)
old = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(old)

FORMAL, BOOT = 2026287102, 2026287103
WORLDS, SCENARIOS = ('tanh', 'sin'), ('noisy', 'exact')
GROUPS, SIZE, LENGTH = old.GROUPS, old.SIZE, old.LENGTH
RULES = ('A', 'B_graph', 'B', 'C')
SOURCES = ('PROTOCOL.md', 'METHOD.md', 'run.py', 'knot_model.py', 'linear_cert.py',
           PREVIOUS, '../dose_window_20261109/run.py')
EPS = 1e-8
GRID = old.GRID
P = old.REF_WEIGHT
PAIR_I, PAIR_J = np.triu_indices(GRID.size, 1)
PAIR_W = P[PAIR_I] * P[PAIR_J]


def reference_matrix():
    left, right = GRID[:-1], GRID[1:]
    first = old.pdf(left) - old.pdf(right)
    second = old.REF_MASS + left * old.pdf(left) - right * old.pdf(right)
    weight = ((left + right) * first - second - left * right * old.REF_MASS) / np.diff(GRID)**2
    assert np.all(weight >= 0)
    diff = np.diff(np.eye(GRID.size), axis=0)
    matrix = np.diag(P) - P[:, None] * P - diff.T @ (weight[:, None] * diff)
    return weight, matrix


REF_EDGE, REF_MATRIX = reference_matrix()
REF_PAD = 2 * old.MHI * old.REF_ERROR


def grid_ranges(model):
    upper = model.distance[np.ix_(model.grid_index, model.grid_index)] + EPS / 100
    return -upper.T, upper


def reference_affine(low, high, center):
    lo = np.maximum(low[PAIR_I, PAIR_J], 0.)
    hi = high[PAIR_I, PAIR_J]
    secant = PAIR_W * (lo + hi)
    upper_coef = (np.bincount(PAIR_J, secant, minlength=GRID.size)
                  - np.bincount(PAIR_I, secant, minlength=GRID.size))
    delta = np.diff(center)
    correction = 2 * REF_EDGE * delta
    upper_coef[:-1] += correction
    upper_coef[1:] -= correction
    upper_base = -PAIR_W @ (lo * hi) + REF_EDGE @ (delta**2) + REF_PAD + EPS
    lower_coef = 2 * REF_MATRIX @ center
    lower_base = -center @ REF_MATRIX @ center - REF_PAD - EPS
    return (lower_coef, lower_base), (upper_coef, upper_base)


def window_affine(values, low, high, center):
    coef = np.zeros((2, GRID.size))
    base = np.array([-EPS, EPS])
    for lag in old.window.LAGS:
        x = np.minimum(values[lag:], values[:-lag])
        y = np.maximum(values[lag:], values[:-lag])
        ix, tx, ex = old.interpolation(x)
        iy, ty, ey = old.interpolation(y)
        dx, dy = np.array([1-tx, tx]), np.array([1-ty, ty])
        jx, jy = np.array([ix, ix+1]), np.array([iy, iy+1])
        weights = dx[:, None, :] * dy[None, :, :]
        lo = (weights * low[jx[:, None, :], jy[None, :, :]]).sum(axis=(0, 1))
        hi = (weights * high[jx[:, None, :], jy[None, :, :]]).sum(axis=(0, 1))
        e = ex + ey
        lo = np.maximum(lo - e - EPS, old.MLO * (y-x) - EPS)
        hi = np.minimum(hi + e + EPS, old.MHI * (y-x) + EPS)
        lo = np.maximum(lo, 0.)
        g = (dy * center[jy]).sum(axis=0) - (dx * center[jx]).sum(axis=0)
        g = np.maximum(g, 0.)
        factor = 1 / (len(old.window.LAGS) * len(x))
        for i, slope in enumerate((2*g, lo+hi)):
            coef[i] += factor * (np.bincount(jy.ravel(), (dy*slope).ravel(), minlength=GRID.size)
                                 - np.bincount(jx.ravel(), (dx*slope).ravel(), minlength=GRID.size))
        base[0] += factor * np.sum(-g**2 - 2*g*e)
        base[1] += factor * np.sum(-lo*hi + (lo+hi)*e)
    return (coef[0], base[0]), (coef[1], base[1])


def expand(model, vector):
    full = np.zeros(len(model.nodes))
    full[model.grid_index] = vector
    return full


def signed_answer(lo, hi):
    return 1 if hi < -EPS else -1 if lo > EPS else 0


def verify_proposal(model, center, candidate):
    if candidate is None:
        return None
    # An LP tolerance can put an extreme proposal just outside the exact graph.
    for fraction in (0., 1e-6, 1e-4, .01):
        if fraction > 0 and center is None:
            break
        y = candidate.copy() if fraction == 0 else (1-fraction)*candidate + fraction*center
        y -= y[model.anchor]
        if model.validate(y):
            return y
    return None


def witness_sign(model, candidate, values, cutoff):
    h = interpolate(model.nodes, candidate, values)
    n = sum(np.mean((h[lag:]-h[:-lag])**2) for lag in old.window.LAGS) / len(old.window.LAGS)
    v = gaussian_variance(model.nodes, candidate)
    return n - cutoff*v


def initial_answer(geometry, lower, upper):
    _, d, cutoff, prior, _, endpoints = geometry
    answer, _, _, _ = old.predict(d, cutoff, prior, endpoints, lower, upper)
    return answer[-1]


def solve(geometry, lower, upper, noisy):
    values, d, cutoff, _, _, endpoints = geometry
    a = initial_answer(geometry, lower, upper)
    answer = np.repeat(a[None, :], len(RULES), axis=0)
    b_interval = np.zeros((len(a), 2))
    c_interval = np.zeros((len(a), 2))
    tested = np.zeros(len(a), dtype=bool)
    witness_ids, witness_values, witness_signs = [], [], []
    try:
        model = make_knots(GRID, old.DOSES, lower, upper)
    except InfeasibleKnots:
        return dict(answer=answer, tested=tested, affine_B=b_interval, affine_C=c_interval,
                    witness_ids=np.zeros(0, dtype=int), witness_values=np.zeros((0, 2, 0)),
                    witness_signs=np.zeros((0, 2)), graph_failure=True)
    low, high = grid_ranges(model)
    vl, vu = old.reference(low, high)
    nl, nu = old.numerator(low, high, endpoints)
    graph_only = old.classify((nl / vu, nu / vl), cutoff)
    answer[1] = old.combine(a, graph_only)
    answer[2:] = answer[1]
    unresolved = np.flatnonzero(answer[1] == 0)
    solver = LinearCertificate(model.nodes, model.left, model.right, model.cap + model.slack, model.anchor)
    center = (model.distance[model.anchor] - model.distance[:, model.anchor]) / 2
    center[model.anchor] = 0.
    exact_center = model.center
    if noisy and exact_center is None and len(unresolved):
        exact_center = verify_proposal(model, None, solver.feasible_center())
    if exact_center is not None:
        center = exact_center
    if len(unresolved):
        (vc_lo, vb_lo), (vc_hi, vb_hi) = reference_affine(low, high, center[model.grid_index])
        vc_lo, vc_hi = expand(model, vc_lo), expand(model, vc_hi)
        v_bound_lo = solver.minimize(vc_lo)[0] + vb_lo
        v_bound_hi = solver.maximize(vc_hi)[0] + vb_hi
        for i in unresolved:
            x, boundary = values[i], cutoff[i]
            (nc_lo, nb_lo), (nc_hi, nb_hi) = window_affine(x, low, high, center[model.grid_index])
            nc_lo, nc_hi = expand(model, nc_lo), expand(model, nc_hi)
            min_n = solver.minimize(nc_lo)[0] + nb_lo
            max_n = solver.maximize(nc_hi)[0] + nb_hi
            blo, bhi = min_n - boundary*v_bound_hi, max_n - boundary*v_bound_lo
            c_lo, lo_y = solver.minimize(nc_lo - boundary*vc_hi)
            c_hi, hi_y = solver.maximize(nc_hi - boundary*vc_lo)
            clo = c_lo + nb_lo - boundary*vb_hi
            chi = c_hi + nb_hi - boundary*vb_lo
            tested[i] = True
            b_interval[i] = blo, bhi
            c_interval[i] = clo, chi
            separate = signed_answer(blo, bhi)
            joint_low, joint_high = max(blo, clo), min(bhi, chi)
            assert joint_low <= joint_high + EPS
            combined = signed_answer(joint_low, joint_high)
            answer[2, i] = separate
            answer[3, i] = combined
            if noisy and combined == 0:
                witnesses = {}
                for proposal in (exact_center, lo_y, hi_y):
                    candidate = verify_proposal(model, exact_center, proposal)
                    if candidate is None:
                        continue
                    sign = witness_sign(model, candidate, x, boundary)
                    if abs(sign) > EPS:
                        witnesses[sign > 0] = (candidate, sign)
                if len(witnesses) == 2:
                    pair = (witnesses[False], witnesses[True])
                    witness_ids.append(i)
                    witness_values.append([p[0] for p in pair])
                    witness_signs.append([p[1] for p in pair])
    return dict(answer=answer, tested=tested, affine_B=b_interval, affine_C=c_interval,
                witness_ids=np.asarray(witness_ids, dtype=int),
                witness_values=np.asarray(witness_values).reshape(-1, 2, len(model.nodes)),
                witness_signs=np.asarray(witness_signs).reshape(-1, 2), graph_failure=False)


def verify_witnesses(result, geometry, lower, upper):
    if len(result['witness_ids']) == 0:
        return 0
    model = make_knots(GRID, old.DOSES, lower, upper)
    values, _, cutoff = geometry[:3]
    for i, candidates, signs in zip(result['witness_ids'], result['witness_values'],
                                    result['witness_signs'], strict=True):
        assert result['answer'][-1, i] == 0
        for j, y in enumerate(candidates):
            score = model.score(y, values[i])
            assert abs((score - cutoff[i])*model.variance(y) - signs[j]) <= EPS
        assert signs[0] < -EPS and signs[1] > EPS
    return len(result['witness_ids'])


def bounds_inputs(root, group, q, noisy):
    reads = q[:, :, None] + old.calibration_noise(root, group)
    lower, upper = old.confidence(reads, q, 20 if noisy else 0)
    return (lower, upper), old.array_hash(reads[:, :, :20])


def verify_scores(result, x, world, q, lower, upper):
    _, _, cutoff, _, _, _ = old.geometry(x)
    truth, v = old.truth(x[:, :, 0], world)
    assert np.all(truth != cutoff)
    answer = np.where(truth < cutoff, 1, -1)
    observed = result['answer']
    assert np.all((observed[:-1] == 0) | (observed[:-1] == observed[1:]))
    g = (truth - cutoff)*v
    checked = result['tested']
    wrong = ((observed != 0) & (observed != answer)).sum(axis=1)
    miss = bool(np.any((q < lower) | (q > upper)))
    gaps = []
    for key in ('affine_B', 'affine_C'):
        box = result[key]
        gaps.append(int(np.count_nonzero(checked & ((g < box[:, 0]-EPS) | (g > box[:, 1]+EPS)))))
    if not miss:
        assert not np.any(wrong) and not any(gaps) and not result['graph_failure']
    return (observed != 0).sum(axis=1), wrong, miss, gaps


def pilot():
    group = 0
    all_x = old.window.trajectory(old.FORMAL, LENGTH, group, SIZE)
    previous = HERE / '../dose_state_dependent_20261110'
    packet = old.read('FORECAST.json')
    assert packet['public']['g0_tanh']['x'] == old.array_hash(all_x)
    with np.load(previous / 'FORECAST.npz', allow_pickle=False) as recorded:
        a = recorded['g0_tanh_answer'][0, -1]
        ids = np.r_[np.flatnonzero(a != 0)[:8], np.flatnonzero(a == 0)[:8]]
        assert len(ids) == 16
        x = all_x[ids]
        geometry = old.geometry(x)
        elapsed, tested, witnesses = [], [], []
        for world in WORLDS:
            q = old.means(world)
            for noisy in (True, False):
                limits, input_hash = bounds_inputs(old.FORMAL, group, q, noisy)
                reads = q[:, :, None] + old.calibration_noise(old.FORMAL, group)
                assert packet['public'][f'g0_{world}']['calibration'] == old.array_hash(reads)
                assert input_hash == old.array_hash(reads[:, :, :20])
                start = time.perf_counter()
                result = solve(geometry, *limits, noisy)
                elapsed.append(time.perf_counter() - start)
                expected = recorded[f'g0_{world}_answer'][0 if noisy else 2, -1, ids]
                np.testing.assert_array_equal(result['answer'][0], expected)
                tested.append(int(result['tested'].sum()))
                witnesses.append(verify_witnesses(result, geometry, *limits))
                verify_scores(result, x, world, q, *limits)
    projected = sum(elapsed) * GROUPS * SIZE / len(ids) * 2
    return dict(indices=ids.tolist(), elapsed=elapsed, tested=tested, witnesses=witnesses,
                projected_seconds=projected, engineering_pass=bool(projected <= 7200))


def forecast():
    public, payload = {}, {}
    means = {world: old.means(world) for world in WORLDS}
    for group in range(GROUPS):
        x = old.window.trajectory(FORMAL, LENGTH, group, SIZE)
        geometry = old.geometry(x)
        for world in WORLDS:
            for scenario in SCENARIOS:
                noisy = scenario == 'noisy'
                key = f'g{group}_{world}_{scenario}'
                limits, input_hash = bounds_inputs(FORMAL, group, means[world], noisy)
                public[key] = dict(path=old.array_hash(x), calibration=input_hash,
                                   interval=old.array_hash(np.stack(limits)))
                result = solve(geometry, *limits, noisy)
                for name, arr in result.items():
                    payload[key + '_' + name] = arr
        if (group+1) % 6 == 0:
            print(f'forecast panels {group+1}/{GROUPS}', flush=True)
    with (HERE / 'FORECAST.npz').open('xb') as stream:
        np.savez_compressed(stream, **payload)
    return public


def score(public):
    counts = np.zeros((len(WORLDS), len(SCENARIOS), GROUPS, len(RULES), 2), dtype=int)
    audit = np.zeros((len(WORLDS), len(SCENARIOS), GROUPS, 6), dtype=int)
    fields = ('answer', 'tested', 'affine_B', 'affine_C', 'witness_ids',
              'witness_values', 'witness_signs', 'graph_failure')
    with np.load(HERE / 'FORECAST.npz', allow_pickle=False) as packet:
        for group in range(GROUPS):
            x = old.window.trajectory(FORMAL, LENGTH, group, SIZE)
            geometry = old.geometry(x)
            for j, world in enumerate(WORLDS):
                q = old.means(world)
                for k, scenario in enumerate(SCENARIOS):
                    key = f'g{group}_{world}_{scenario}'
                    limits, input_hash = bounds_inputs(FORMAL, group, q, scenario == 'noisy')
                    assert public[key] == dict(path=old.array_hash(x), calibration=input_hash,
                                               interval=old.array_hash(np.stack(limits)))
                    result = {name: packet[key + '_' + name] for name in fields}
                    witnesses = verify_witnesses(result, geometry, *limits)
                    accepted, wrong, missed, gaps = verify_scores(result, x, world, q, *limits)
                    counts[j, k, group, :, 0] = accepted
                    counts[j, k, group, :, 1] = wrong
                    audit[j, k, group] = missed, *gaps, witnesses, result['tested'].sum(), result['graph_failure']
    n = GROUPS*SIZE
    sample = np.random.default_rng(BOOT).integers(GROUPS, size=(4096, GROUPS))
    results = {}
    for j, world in enumerate(WORLDS):
        results[world] = {}
        for k, scenario in enumerate(SCENARIOS):
            total = counts[j, k].sum(axis=0)
            draws = counts[j, k][sample].sum(axis=1)[:, :, 0]
            stats = audit[j, k].sum(axis=0)
            remaining = n - total[-1, 0]
            results[world][scenario] = dict(
                rules={rule: dict(accepted=int(row[0]), wrong=int(row[1]),
                                  unconditional_wrong=float(row[1]/n),
                                  conditional_wrong=float(row[1]/row[0]) if row[0] else None)
                       for rule, row in zip(RULES, total, strict=True)},
                ci_miss_panels=int(stats[0]), outside_B=int(stats[1]), outside_C=int(stats[2]),
                witness_pairs=int(stats[3]), unresolved=int(remaining-stats[3]),
                witness_fraction=float(stats[3]/remaining) if remaining else None,
                solved_windows=int(stats[4]), graph_failures=int(stats[5]),
                gain_ci={name: np.quantile((draws[:, b]-draws[:, a])/n, [.025, .975]).tolist()
                         for name, a, b in (('B_minus_A', 0, 2), ('C_minus_B', 2, 3), ('C_minus_A', 0, 3))})
    gain = {world: (result['noisy']['rules']['C']['accepted'] -
                    result['noisy']['rules']['A']['accepted'])/n for world, result in results.items()}
    main = all(value >= .05 for value in gain.values())
    ambiguity = all((result['noisy']['witness_fraction'] or 0.) >= .5 for result in results.values())
    return dict(windows=n, panels=GROUPS, results=results, gains=gain,
                gates=dict(coverage=main, half_remaining_witnessed=ambiguity,
                           confidence_event_sound=True))


def digest(name):
    return old.hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        old.json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read(name):
    return old.json.loads((HERE / name).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'pilot', 'forecast', 'score'))
    phase = parser.parse_args().phase
    sources = {name: digest(name) for name in SOURCES}
    if phase == 'lock':
        save('LOCK.json', dict(utc=old.utc(), sources=sources, formal=FORMAL, bootstrap=BOOT))
        return
    assert read('LOCK.json')['sources'] == sources
    lock = digest('LOCK.json')
    if phase == 'pilot':
        save('PILOT.json', dict(utc=old.utc(), lock=lock, **pilot()))
        return
    check = read('PILOT.json')
    assert check['lock'] == lock and check['engineering_pass']
    if phase == 'forecast':
        public = forecast()
        save('FORECAST.json', dict(utc=old.utc(), lock=lock, pilot=digest('PILOT.json'),
                                  public=public, predictions=digest('FORECAST.npz')))
        return
    receipt = read('FORECAST.json')
    assert receipt['lock'] == lock and receipt['pilot'] == digest('PILOT.json')
    assert receipt['predictions'] == digest('FORECAST.npz')
    save('SUMMARY.json', dict(utc=old.utc(), forecast=digest('FORECAST.json'), **score(receipt['public'])))


if __name__ == '__main__':
    main()
