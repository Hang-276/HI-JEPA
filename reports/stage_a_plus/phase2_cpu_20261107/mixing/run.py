"""Frozen matched-mixing experiment; CPU, one OLS, no external data."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

HERE = Path(__file__).resolve().parent
LAGS = (4, 16, 32)
CFG = dict(rho=.97, phi_degrees=43., theta_degrees=9., mu=.5, transitions=2048,
           train_base_pairs=16, pilot_pairs=32, validation_pairs=128,
           seed_prefix=[20261107, 731], phase_seeds=dict(train=101, pilot=202, validation=303),
           ci_multiplier=2.2414, precision_fraction=.5, match_tolerance=1e-10)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value))


def save(name, value):
    path = HERE / name
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(value, default=serial, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def read(name):
    return json.loads((HERE / name).read_text())


def rotation(i, j, angle):
    q = np.eye(4)
    q[i, i] = q[j, j] = np.cos(angle)
    q[i, j], q[j, i] = np.sin(angle), -np.sin(angle)
    return q


def constants():
    a, b = np.cos(np.deg2rad(CFG['theta_degrees'])), np.sin(np.deg2rad(CFG['theta_degrees']))
    W = rotation(1, 2, np.deg2rad(CFG['phi_degrees']))
    A = CFG['rho'] * W @ rotation(0, 1, np.pi / 16) @ W.T
    J = np.zeros((4, 4)); J[1, 2] = 1.; J[2, 1] = -1.
    L = np.eye(4); L[1, 1] = L[2, 2] = a; L += CFG['mu'] * b * J
    beta = b * np.sqrt(1 - CFG['mu'] ** 2)
    M = np.array([2 * np.eye(4) - np.linalg.matrix_power(A, h) - np.linalg.matrix_power(A, h).T for h in LAGS])
    V = L @ M @ L.T + beta ** 2 * J @ M @ J.T
    return dict(A=A, J=J, L=L, beta=beta, Rplus=rotation(1, 2, np.deg2rad(CFG['theta_degrees'])),
                Rminus=rotation(1, 2, -np.deg2rad(CFG['theta_degrees'])), M=M, V=V)


K = constants()


def moments(y):
    return np.stack([np.einsum('nti,ntj->nij', y[:, h:] - y[:, :-h], y[:, h:] - y[:, :-h]) / (y.shape[1] - h) for h in LAGS], axis=1)


def scores(y):
    return np.diagonal(moments(y), axis1=-2, axis2=-1).mean(1)


def score_matrices(M, cal, diagonal=False):
    B, C, MT = [np.asarray(cal[k]) for k in ('B', 'C', 'MT')]
    if diagonal:
        M = M * np.eye(4) + MT * (1 - np.eye(4))
    return (np.einsum('ji,...hik,jk->...hj', B, M, B) + C).mean(-2)


def population(cal):
    Ms = dict(fixed=K['V'], plus=K['Rplus'] @ K['M'] @ K['Rplus'].T,
              minus=K['Rminus'] @ K['M'] @ K['Rminus'].T)
    values = {name: dict(original=np.diagonal(M, axis1=-2, axis2=-1).mean(0),
                         full=score_matrices(M, cal), diag_anchor=score_matrices(M, cal, True)) for name, M in Ms.items()}
    gap = lambda v: float(v[2] - v[1])
    return dict(physical=np.diagonal(K['M'], axis1=-2, axis2=-1).mean(0),
                predicted=values, gap={name: {method: gap(v) for method, v in val.items()} for name, val in values.items()},
                paired_effect={sign: gap(values[sign]['full']) - gap(values['fixed']['full']) for sign in ('plus', 'minus')})


def generate(phase, n):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(CFG['seed_prefix'] + [CFG['phase_seeds'][phase], 0])))
    x = rng.standard_normal((2, n, CFG['transitions'] + 1, 4))
    noise = np.sqrt(1 - CFG['rho'] ** 2)
    for t in range(1, x.shape[2]):
        x[:, :, t] = x[:, :, t - 1] @ K['A'].T + noise * x[:, :, t]
    signs = np.array([1] * (3 * n // 4) + [-1] * (n // 4))
    srng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(CFG['seed_prefix'] + [CFG['phase_seeds'][phase], 1])))
    srng.shuffle(signs)
    return x[0], x[1], signs


def train():
    z, n, _ = generate('train', CFG['train_base_pairs'])
    zs, fixed, variable = [], [], []
    for physical, noise in ((z, n), (n, z)):
        for context, noise_sign in zip((1, 1, 1, -1), (1, -1, 1, -1)):
            zs.append(physical)
            fixed.append(physical @ K['L'].T + noise_sign * K['beta'] * noise @ K['J'].T)
            variable.append(physical @ K['Rplus' if context == 1 else 'Rminus'].T)
    Z, F, V = map(np.concatenate, (zs, fixed, variable))
    z2, f2, v2 = (a.reshape(-1, 4) for a in (Z, F, V))
    mz, my = z2.mean(0), f2.mean(0)
    zc, fc, vc = z2 - mz, f2 - my, v2 - v2.mean(0)
    covz, covf, covv = (a.T @ a / len(a) for a in (zc, fc, vc))
    crossf, crossv = zc.T @ fc / len(fc), zc.T @ vc / len(vc)
    B = np.linalg.solve(covf, crossf.T).T
    intercept = mz - B @ my
    r2f = np.diag(crossf) ** 2 / (np.diag(covz) * np.diag(covf))
    r2v = np.diag(crossv) ** 2 / (np.diag(covz) * np.diag(covv))
    MF, MV, MZ = moments(F).mean(0), moments(V).mean(0), moments(Z).mean(0)
    EF, EV = Z - F @ B.T - intercept, Z - V @ B.T - intercept
    TF, TV = np.diagonal(moments(EF).mean(0), axis1=-2, axis2=-1), np.diagonal(moments(EV).mean(0), axis1=-2, axis2=-1)
    C = np.diagonal(MZ - B @ MF @ B.T, axis1=-2, axis2=-1)
    deltas = dict(Y_mean=np.max(abs(f2.mean(0) - v2.mean(0))), Y_covariance=np.max(abs(covf - covv)),
                  ZY_covariance=np.max(abs(crossf - crossv)), axis_r2=np.max(abs(r2f - r2v)),
                  increment_matrices=np.max(abs(MF - MV)), residual_increment_energy=np.max(abs(TF - TV)),
                  variable_normal_equation=np.max(abs(B @ covv - crossv)))
    assert max(deltas.values()) <= CFG['match_tolerance'], deltas
    return dict(B=B, intercept=intercept, C=C, MT=MF, train_residual_energy=TF,
                axis_r2=r2f, exact_matching_differences=deltas,
                independent_base_pairs=CFG['train_base_pairs'], independent_physical_paths=2 * CFG['train_base_pairs'],
                dependent_views_per_arm=len(Z))


def predict(phase, n, cal):
    Z, N, signs = generate(phase, n)
    F = Z @ K['L'].T + K['beta'] * N @ K['J'].T
    V = np.empty_like(Z)
    for sign, R in ((1, K['Rplus']), (-1, K['Rminus'])):
        V[signs == sign] = Z[signs == sign] @ R.T
    records = []
    predictions = {}
    for arm, Y in (('fixed', F), ('variable', V)):
        MY = moments(Y)
        predictions[arm] = dict(original=np.diagonal(MY, axis1=-2, axis2=-1).mean(1),
                                full=score_matrices(MY, cal), diag_anchor=score_matrices(MY, cal, True))
    recovered = np.empty_like(V)
    for sign, R in ((1, K['Rplus']), (-1, K['Rminus'])):
        recovered[signs == sign] = V[signs == sign] @ R
    predictions['variable']['known_context_inverse'] = scores(recovered)
    for i in range(n):
        records.append(dict(episode=i, sign=int(signs[i]), arms={a: {m: v[i] for m, v in p.items()} for a, p in predictions.items()}))
    save(f'{phase.upper()}_PREDICTIONS.json', dict(calibrator=cal, records=records, source_phase=phase,
                                               note='Predictions persisted before scoring physical Z; synthetic generator is known.'))
    truth = scores(Z)
    for i, row in enumerate(records): row['physical_scores'] = truth[i]
    return records


def top(v):
    ids = np.argsort(v)
    return None if v[ids[1]] == v[ids[2]] else tuple(sorted(map(int, ids[:2])))


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a): return None
    mean = float(a.mean()); se = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None
    radius = CFG['ci_multiplier'] * se if se is not None else None
    return dict(n=len(a), mean=mean, sd=float(a.std(ddof=1)) if len(a) > 1 else None,
                approximate_interval=None if radius is None else [mean - radius, mean + radius])


def evaluate(records):
    out = dict(n_independent_pairs=len(records), paired_contrasts={}, conditional_gap_errors={}, decisions={})
    gap = lambda v: float(v[2] - v[1])
    for sign, label in ((1, 'plus'), (-1, 'minus')):
        group = [r for r in records if r['sign'] == sign]
        out['paired_contrasts'][label] = stats([gap(r['arms']['variable']['full']) - gap(r['arms']['fixed']['full']) for r in group])
        out['conditional_gap_errors'][label] = {}
        for arm in ('fixed', 'variable'):
            out['conditional_gap_errors'][label][arm] = {}
            for method in group[0]['arms'][arm]:
                errors = [gap(r['arms'][arm][method]) - gap(r['physical_scores']) for r in group]
                out['conditional_gap_errors'][label][arm][method] = dict(**stats(errors), mae=float(np.mean(np.abs(errors))))
    for target in ('fixed_I_2_3', 'original_Y_window_top2'):
        out['decisions'][target] = {}
        for arm in ('fixed', 'variable'):
            out['decisions'][target][arm] = {}
            for method in records[0]['arms'][arm]:
                if method == 'original': continue
                counts = dict(correct_certified=0, correct_rejected=0, wrong_detected=0, wrong_certified=0, abstain=0)
                for row in records:
                    selected = (2, 3) if target == 'fixed_I_2_3' else top(row['arms'][arm]['original'])
                    physical, prediction = top(row['physical_scores']), top(row['arms'][arm][method])
                    if None in (selected, physical, prediction): counts['abstain'] += 1; continue
                    actual, forecast = selected == physical, selected == prediction
                    counts[('correct_certified' if forecast else 'correct_rejected') if actual else ('wrong_certified' if forecast else 'wrong_detected')] += 1
                out['decisions'][target][arm][method] = counts
    out['context_inverse_max_axis_error'] = max(float(np.max(abs(np.asarray(r['arms']['variable']['known_context_inverse']) - r['physical_scores']))) for r in records)
    return out


def verify_lock():
    lock = read('METHOD_LOCK.json')
    assert lock['script_sha256'] == sha(Path(__file__)) and lock['protocol_sha256'] == sha(HERE / 'PROTOCOL.md')
    return lock


def main(phase):
    started = time.perf_counter()
    if phase == 'freeze':
        B = K['L'].T
        cal = dict(B=B, C=np.diagonal(K['M'] - B @ K['V'] @ B.T, axis1=-2, axis2=-1), MT=K['V'])
        save('METHOD_LOCK.json', dict(utc=datetime.now(timezone.utc).isoformat(), config=CFG,
             script_sha256=sha(Path(__file__)), protocol_sha256=sha(HERE / 'PROTOCOL.md'),
             generator={k: v for k, v in K.items()}, population=population(cal),
             population_axis_r2=[1., float(np.cos(np.deg2rad(9.)) ** 2), float(np.cos(np.deg2rad(9.)) ** 2), 1.],
             population_residual_variance=[0., float(K['beta'] ** 2), float(K['beta'] ** 2), 0.]))
        return
    lock = verify_lock()
    if phase == 'pilot':
        cal = train()
        expected = population(cal)
        physical_gap = float(expected['physical'][2] - expected['physical'][1])
        analytic_pass = (expected['gap']['fixed']['full'] < 0 < expected['gap']['plus']['full'] and
                         expected['gap']['minus']['full'] < 0 and
                         abs(expected['gap']['fixed']['full'] - physical_gap) <= abs(physical_gap) / 4)
        if not analytic_pass:
            save('PILOT.json', dict(passed=False, reason='frozen TRAIN population signature failed', calibrator=cal, expected=expected))
            print('pilot_stopped_analytic'); return
        records = predict('pilot', CFG['pilot_pairs'], cal)
        evaluation = evaluate(records)
        checks = {}
        for label, count, sign in (('plus', 3 * CFG['validation_pairs'] // 4, 1), ('minus', CFG['validation_pairs'] // 4, -1)):
            s = evaluation['paired_contrasts'][label]
            half = CFG['ci_multiplier'] * s['sd'] / np.sqrt(count)
            limit = CFG['precision_fraction'] * abs(expected['paired_effect'][label])
            checks[label] = dict(projected_validation_halfwidth=half, limit=limit,
                                 correct_pilot_sign=sign * s['mean'] > 0, passed=half <= limit and sign * s['mean'] > 0)
        passed = all(c['passed'] for c in checks.values())
        out = dict(passed=passed, method_lock_sha256=sha(HERE / 'METHOD_LOCK.json'), calibrator=cal,
                   expected=expected, pilot_gate=checks, evaluation=evaluation,
                   elapsed_seconds=time.perf_counter() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save('PILOT.json', out)
        print(json.dumps(dict(passed=passed, gate=checks, matching=cal['exact_matching_differences'], seconds=out['elapsed_seconds']), default=serial))
    else:
        pilot = read('PILOT.json')
        assert pilot['passed'] and pilot['method_lock_sha256'] == sha(HERE / 'METHOD_LOCK.json')
        records = predict('validation', CFG['validation_pairs'], pilot['calibrator'])
        evaluation = evaluate(records)
        passed = (evaluation['paired_contrasts']['plus']['approximate_interval'][0] > 0 and
                  evaluation['paired_contrasts']['minus']['approximate_interval'][1] < 0)
        out = dict(method_lock_sha256=sha(HERE / 'METHOD_LOCK.json'), pilot_sha256=sha(HERE / 'PILOT.json'),
                   validation_predictions_sha256=sha(HERE / 'VALIDATION_PREDICTIONS.json'), mechanism_gate_passed=passed,
                   evaluation=evaluation, records=records, elapsed_seconds=time.perf_counter() - started,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   scope='Controlled paired mechanism test, not actual Fine validation or a finite-sample certificate')
        save('RESULTS.json', out)
        print(json.dumps(dict(passed=passed, contrasts=evaluation['paired_contrasts'], decisions=evaluation['decisions'],
                              inverse_error=evaluation['context_inverse_max_axis_error'], seconds=out['elapsed_seconds']), default=serial))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('freeze', 'pilot', 'validate'))
    main(parser.parse_args().phase)
