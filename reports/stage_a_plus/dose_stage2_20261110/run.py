"""Stage 2: fixed versus available origin layouts at a fixed query budget."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE1 = '../dose_stage1_20261110'
sys.path.insert(0, str((HERE / STAGE1).resolve()))
SPEC = importlib.util.spec_from_file_location('dose_stage1', HERE / STAGE1 / 'run.py')
stage1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage1)

from acquisition import ORIGIN_LAYOUTS, calibration_reads, layout_origins, response_means
from layout_constraints import InfeasibleKnots, exact_model, propagated_ranges

old = stage1.old
PILOT, FORMAL, BOOT = 2026288201, 2026288202, 2026288203
GROUPS, SIZE, LENGTH = old.GROUPS, old.SIZE, old.LENGTH
GRID, WORLDS, SCENARIOS = stage1.GRID, stage1.WORLDS, stage1.SCENARIOS
SOURCES = ('run.py', 'acquisition.py', 'layout_constraints.py', 'PROTOCOL.md',
           f'{STAGE1}/run.py', f'{STAGE1}/knot_model.py', f'{STAGE1}/linear_cert.py',
           '../dose_state_dependent_20261110/run.py', '../dose_window_20261109/run.py')
FIELDS = ('answer', 'tested', 'affine_B', 'affine_C', 'graph_failure',
          'numerator', 'variance')
DIFFERENCES = (('random_minus_grid', 1, 0), ('spread_minus_grid', 2, 0),
               ('spread_minus_random', 2, 1))


def solve(geometry, origins, lower, upper, noisy):
    """Reuse Stage 1 score envelopes on the unchanged evaluation/reference grid."""
    values, d, cutoff, prior, _, endpoints = geometry
    answer = np.repeat(prior[None, :], len(stage1.RULES), axis=0)
    b_interval = np.zeros((len(d), 2))
    c_interval = np.zeros((len(d), 2))
    tested = np.zeros(len(d), dtype=bool)
    try:
        model = exact_model(GRID, origins, old.DOSES, lower, upper)
    except InfeasibleKnots:
        return dict(answer=answer, tested=tested, affine_B=b_interval, affine_C=c_interval,
                    graph_failure=True, numerator=np.stack((old.MLO**2*d, old.MHI**2*d)),
                    variance=np.asarray((old.MLO**2, old.MHI**2)))

    a = propagated_ranges(GRID, origins, old.DOSES, lower, upper)
    if a is not None:
        avl, avu = old.reference(*a)
        anl, anu = old.numerator(*a, endpoints)
        answer[0] = old.combine(prior, old.classify((anl/avu, anu/avl), cutoff))

    low, high = stage1.grid_ranges(model)
    vl, vu = old.reference(low, high)
    nl, nu = old.numerator(low, high, endpoints)
    answer[1] = old.combine(answer[0], old.classify((nl/vu, nu/vl), cutoff))
    answer[2:] = answer[1]
    unresolved = np.flatnonzero(answer[1] == 0)
    if len(unresolved):
        solver = stage1.LinearCertificate(model.nodes, model.left, model.right,
                                          model.cap + model.slack, model.anchor)
        center = (model.distance[model.anchor] - model.distance[:, model.anchor]) / 2
        center[model.anchor] = 0.
        exact_center = model.center
        if noisy and exact_center is None:
            exact_center = stage1.verify_proposal(model, None, solver.feasible_center())
        if exact_center is not None:
            center = exact_center
        center = center[model.grid_index]
        (vc_lo, vb_lo), (vc_hi, vb_hi) = stage1.reference_affine(low, high, center)
        vc_lo, vc_hi = stage1.expand(model, vc_lo), stage1.expand(model, vc_hi)
        v_bound_lo = solver.minimize(vc_lo)[0] + vb_lo
        v_bound_hi = solver.maximize(vc_hi)[0] + vb_hi
        for i in unresolved:
            boundary = cutoff[i]
            (nc_lo, nb_lo), (nc_hi, nb_hi) = stage1.window_affine(values[i], low, high, center)
            nc_lo, nc_hi = stage1.expand(model, nc_lo), stage1.expand(model, nc_hi)
            min_n = solver.minimize(nc_lo)[0] + nb_lo
            max_n = solver.maximize(nc_hi)[0] + nb_hi
            blo, bhi = min_n - boundary*v_bound_hi, max_n - boundary*v_bound_lo
            clo = solver.minimize(nc_lo - boundary*vc_hi)[0] + nb_lo - boundary*vb_hi
            chi = solver.maximize(nc_hi - boundary*vc_lo)[0] + nb_hi - boundary*vb_lo
            tested[i] = True
            b_interval[i], c_interval[i] = (blo, bhi), (clo, chi)
            joint_low, joint_high = max(blo, clo), min(bhi, chi)
            assert joint_low <= joint_high + stage1.EPS
            answer[2, i] = stage1.signed_answer(blo, bhi)
            answer[3, i] = stage1.signed_answer(joint_low, joint_high)
    return dict(answer=answer, tested=tested, affine_B=b_interval, affine_C=c_interval,
                graph_failure=False, numerator=np.stack((nl, nu)), variance=np.asarray((vl, vu)))


def responses(root, group, origins, world):
    q = response_means(origins, old.DOSES, lambda x: old.physical(x, world), old.MLO, old.MHI)
    reads = calibration_reads(root, group, q, old.SIGMA)
    assert reads.shape == (len(old.DOSES), len(GRID), 20)
    return q, reads


def limits(q, reads, scenario):
    return old.confidence(reads, q, 20 if scenario == 'noisy' else 0)


def public_receipt(pool, origins, x, reads, bounds):
    return dict(pool=old.array_hash(pool), origins=old.array_hash(origins),
                path=old.array_hash(x), readings=old.array_hash(reads),
                interval=old.array_hash(np.stack(bounds)))


def verify(result, x, world, q, bounds):
    accepted, wrong, miss, signed_outside = stage1.verify_scores(result, x, world, q, *bounds)
    target, variance = old.truth(x[:, :, 0], world)
    numerator = target*variance
    lo, hi = result['numerator']
    outside_n = int(np.count_nonzero((numerator < lo-stage1.EPS) | (numerator > hi+stage1.EPS)))
    vlo, vhi = result['variance']
    outside_v = int(variance < vlo-stage1.EPS or variance > vhi+stage1.EPS)
    if not miss:
        assert outside_n == 0 and outside_v == 0
    return accepted, wrong, miss, signed_outside, outside_n, outside_v


def pilot():
    group = 0
    pool, layouts = layout_origins(PILOT, group, GRID)
    assert all(len(origins) == len(GRID) for origins in layouts.values())
    assert np.array_equal(layouts['grid'], GRID)
    np.testing.assert_array_equal(layouts['random'], np.sort(pool[:len(GRID)]))
    assert np.all(np.isin(layouts['spread'], pool))
    assert layouts['spread'][0] == pool.min() and layouts['spread'][-1] == pool.max()
    elapsed = 0.
    x = old.window.trajectory(PILOT, LENGTH, group, 16)
    geometry = old.geometry(x)
    fixed_checks = 0
    paired_noise = None
    for world in WORLDS:
        for layout, origins in layouts.items():
            q, reads = responses(PILOT, group, origins, world)
            errors = reads - q[..., None]
            if paired_noise is None:
                paired_noise = errors
            np.testing.assert_allclose(errors, paired_noise, atol=1e-14, rtol=0)
            displacement = old.physical(q, world) - old.physical(origins, world)
            np.testing.assert_allclose(displacement, np.broadcast_to(old.DOSES[:, None], q.shape),
                                       atol=1e-13, rtol=0)
            for scenario in SCENARIOS:
                interval = limits(q, reads, scenario)
                start = time.perf_counter()
                result = solve(geometry, origins, *interval, scenario == 'noisy')
                elapsed += time.perf_counter() - start
                verify(result, x, world, q, interval)
                if layout == 'grid':
                    np.testing.assert_array_equal(q, old.means(world))
                    original = stage1.solve(geometry, *interval, scenario == 'noisy')
                    np.testing.assert_array_equal(result['answer'], original['answer'])
                    np.testing.assert_allclose(result['affine_C'], original['affine_C'], atol=1e-8, rtol=0)
                    fixed_checks += 1
    projected = elapsed*GROUPS*SIZE/16*2
    return dict(windows=16, grid_parity_checks=fixed_checks, natural_layouts_checked=True,
                shared_noise_checked=True,
                elapsed=elapsed, projected_seconds=projected,
                engineering_pass=bool(projected <= 7200))


def forecast():
    public, payload = {}, {}
    for group in range(GROUPS):
        pool, layouts = layout_origins(FORMAL, group, GRID)
        x = old.window.trajectory(FORMAL, LENGTH, group, SIZE)
        geometry = old.geometry(x)
        for layout, origins in layouts.items():
            payload[f'g{group}_{layout}_origins'] = origins
            for world in WORLDS:
                q, reads = responses(FORMAL, group, origins, world)
                for scenario in SCENARIOS:
                    interval = limits(q, reads, scenario)
                    key = f'g{group}_{world}_{layout}_{scenario}'
                    public[key] = public_receipt(pool, origins, x, reads, interval)
                    result = solve(geometry, origins, *interval, scenario == 'noisy')
                    for name, value in result.items():
                        payload[key + '_' + name] = value
        if (group+1) % 4 == 0:
            print(f'forecast panels {group+1}/{GROUPS}', flush=True)
    with (HERE / 'FORECAST.npz').open('xb') as stream:
        np.savez_compressed(stream, **payload)
    return public


def gap_record(by_group, draws):
    base = GROUPS*SIZE
    return dict(difference=float(by_group.sum()/base),
                ci=np.quantile(by_group[draws].sum(axis=1)/base, [.025, .975]).tolist())


def score(public):
    shape = len(WORLDS), len(ORIGIN_LAYOUTS), len(SCENARIOS), GROUPS
    counts = np.zeros((*shape, len(stage1.RULES), 2), dtype=int)
    audit = np.zeros((*shape, 6), dtype=int)
    support = np.zeros((len(ORIGIN_LAYOUTS), GROUPS, 4))
    with np.load(HERE / 'FORECAST.npz', allow_pickle=False) as packet:
        for group in range(GROUPS):
            pool, layouts = layout_origins(FORMAL, group, GRID)
            x = old.window.trajectory(FORMAL, LENGTH, group, SIZE)
            for a, (layout, origins) in enumerate(layouts.items()):
                np.testing.assert_array_equal(origins, packet[f'g{group}_{layout}_origins'])
                support[a, group] = (origins[0], origins[-1], np.count_nonzero(abs(origins) >= 2.5),
                                     np.abs(GRID[:, None]-origins[None, :]).min(axis=1).max())
                for j, world in enumerate(WORLDS):
                    q, reads = responses(FORMAL, group, origins, world)
                    for k, scenario in enumerate(SCENARIOS):
                        interval = limits(q, reads, scenario)
                        key = f'g{group}_{world}_{layout}_{scenario}'
                        assert public[key] == public_receipt(pool, origins, x, reads, interval)
                        result = {name: packet[key + '_' + name] for name in FIELDS}
                        accepted, wrong, miss, signed, outside_n, outside_v = verify(
                            result, x, world, q, interval)
                        counts[j, a, k, group, :, 0] = accepted
                        counts[j, a, k, group, :, 1] = wrong
                        audit[j, a, k, group] = miss, *signed, outside_n, outside_v, result['graph_failure']
    n = GROUPS*SIZE
    draws = np.random.default_rng(BOOT).integers(GROUPS, size=(4096, GROUPS))
    results, contrasts = {}, {}
    for j, world in enumerate(WORLDS):
        results[world], contrasts[world] = {}, {}
        coverage = counts[j, :, :, :, -1, 0]
        for a, layout in enumerate(ORIGIN_LAYOUTS):
            cells = {}
            for k, scenario in enumerate(SCENARIOS):
                total = counts[j, a, k].sum(axis=0)
                checks = audit[j, a, k].sum(axis=0).tolist()
                cells[scenario] = dict(
                    rules={rule: dict(accepted=int(row[0]), wrong=int(row[1]),
                                      unconditional_wrong=float(row[1]/n),
                                      conditional_wrong=float(row[1]/row[0]) if row[0] else None)
                           for rule, row in zip(stage1.RULES, total, strict=True)},
                    audits=dict(zip(('ci_miss_panels', 'outside_B', 'outside_C', 'outside_numerator',
                                     'outside_variance_panels', 'graph_failure_panels'), checks, strict=True)))
            cells['exact_minus_noisy_C'] = gap_record(coverage[a, 1] - coverage[a, 0], draws)
            results[world][layout] = cells
        for k, scenario in enumerate(SCENARIOS):
            contrasts[world][scenario] = {
                name: gap_record(coverage[a, k] - coverage[b, k], draws)
                for name, a, b in DIFFERENCES
            }
        contrasts[world]['extra_noise_loss_relative_to_grid'] = {
            ORIGIN_LAYOUTS[a]: gap_record((coverage[a, 1]-coverage[a, 0]) -
                                          (coverage[0, 1]-coverage[0, 0]), draws)
            for a in (1, 2)
        }
    random_ok = all(contrasts[w]['noisy']['random_minus_grid']['ci'][0] > -.05 for w in WORLDS)
    spread_ok = all(contrasts[w]['noisy']['spread_minus_grid']['ci'][0] > -.05 for w in WORLDS)
    random_loss = all((cell := contrasts[w]['noisy']['random_minus_grid'])['difference'] <= -.05
                      and cell['ci'][1] < 0 for w in WORLDS)
    spread_gain = all((cell := contrasts[w]['noisy']['spread_minus_random'])['difference'] >= .05
                      and cell['ci'][0] > 0 for w in WORLDS)
    origins = {layout: dict(zip(('median_min', 'median_max', 'median_count_abs_ge_2p5',
                               'median_max_grid_distance'), np.median(support[a], axis=0).tolist(),
                              strict=True)) for a, layout in enumerate(ORIGIN_LAYOUTS)}
    return dict(windows=n, panels=GROUPS, results=results, contrasts=contrasts, origins=origins,
                gates=dict(random_noninferior=random_ok, spread_noninferior=spread_ok,
                           random_loss=random_loss, spread_rescue=spread_gain,
                           confidence_event_sound=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('lock', 'pilot', 'forecast', 'score'))
    phase = parser.parse_args().phase
    sources = {name: old.hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCES}
    if phase == 'lock':
        save('LOCK.json', dict(utc=old.utc(), sources=sources, pilot=PILOT, formal=FORMAL, bootstrap=BOOT))
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
    save('SUMMARY.json', dict(utc=old.utc(), forecast=digest('FORECAST.json'),
                             **score(receipt['public'])))


def digest(name):
    return old.hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def read(name):
    return old.json.loads((HERE / name).read_text())


def save(name, data):
    with (HERE / name).open('x') as stream:
        old.json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
