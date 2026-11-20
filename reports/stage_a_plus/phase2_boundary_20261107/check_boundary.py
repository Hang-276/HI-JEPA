"""Deterministic population checks for the known-drift Gaussian example."""

import json
import math
import os

for thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                   'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[thread_var] = '1'

import numpy as np


IDENTITY = np.eye(4)
LAGS = (4, 16, 32)
RHO = .97
PSI = math.pi / 16
PHI = math.radians(43)
DELTA = math.radians(9)
BETA = math.pi / 4 - PHI
SIGMA = math.sqrt(1 - RHO**2)
TAUS = (.05, .1, .25, .5, 1.)


def rotate(a, b, angle):
    result = IDENTITY.copy()
    c, s = math.cos(angle), math.sin(angle)
    result[np.ix_([a, b], [a, b])] = [[c, s], [-s, c]]
    return result


def conditional_contexts(tau):
    base = SIGMA**2 * IDENTITY
    contexts = [(np.zeros(4), base)]
    for axis in IDENTITY:
        contexts.extend(((SIGMA * tau * axis, base), (-SIGMA * tau * axis, base)))
    for axis in IDENTITY:
        contexts.append((np.zeros(4), base + SIGMA**2 * tau * np.outer(axis, axis)))
    return contexts


def observed_contexts(contexts, physical_map):
    return [(physical_map.T @ mean, physical_map.T @ cov @ physical_map)
            for mean, cov in contexts]


def gaussian_kl(p, q):
    mean_p, cov_p = p
    mean_q, cov_q = q
    d = mean_q - mean_p
    return .5 * (np.trace(np.linalg.solve(cov_q, cov_p))
                 + d @ np.linalg.solve(cov_q, d) - len(d)
                 + np.linalg.slogdet(cov_q)[1] - np.linalg.slogdet(cov_p)[1])


def score(physical_map, drift, state_cov):
    scales = np.diag(physical_map @ state_cov @ physical_map.T)
    result = []
    for lag in LAGS:
        cross = np.linalg.matrix_power(drift, lag) @ state_cov
        moment = 2 * state_cov - cross - cross.T
        result.append(np.diag(physical_map @ moment @ physical_map.T) / scales)
    return np.mean(result, axis=0)


def score_record(scores):
    order = np.argsort(scores)
    return dict(scores=scores.tolist(), top2=sorted(int(i) + 1 for i in order[:2]),
                boundary_gap=float(scores[order[2]] - scores[order[1]]))


def check():
    w = rotate(1, 2, PHI)
    drift = RHO * w @ rotate(0, 1, PSI) @ w.T
    p_map = rotate(1, 2, -DELTA)
    q_map = rotate(1, 2, DELTA)
    normal2 = np.array([-math.sin(BETA), math.cos(BETA)])
    normal3 = np.array([-math.cos(BETA), -math.sin(BETA)])
    s_fast = 2 * (1 - np.mean([RHO**h * math.cos(h * PSI) for h in LAGS]))
    s_slow = 2 * (1 - np.mean([RHO**h for h in LAGS]))

    def closed_scores(phi):
        c2, s2 = math.cos(phi)**2, math.sin(phi)**2
        return [s_fast, c2 * s_fast + s2 * s_slow,
                s2 * s_fast + c2 * s_slow, s_slow]

    records = []
    for tau in TAUS:
        contexts = conditional_contexts(tau)
        physical_drift = p_map @ drift @ p_map.T
        prev = np.array([.2, -.4, .7, -.1])
        # Natural-parameter rank is evaluated conditional on a fixed previous state.
        rank_matrix = []
        for mean, cov in contexts:
            var = np.diag(cov)
            rank_matrix.append(np.r_[(physical_drift @ prev + mean) / var, -.5 / var])
        rank_matrix = np.asarray(rank_matrix)
        rank = int(np.linalg.matrix_rank(rank_matrix[1:] - rank_matrix[0]))
        assert rank == 8

        p = observed_contexts(contexts, p_map)
        q = observed_contexts(contexts, q_map)
        lam = SIGMA**2 * (1 + (2*tau**2 + tau) / 13) / (1 - RHO**2)
        state = lam * IDENTITY
        for sign, model in ((-1, p), (1, q)):
            np.testing.assert_allclose(np.mean([m for m, _ in model], axis=0), 0,
                                       atol=1e-14)
            innovation = np.mean([c + np.outer(m, m) for m, c in model], axis=0)
            np.testing.assert_allclose(innovation, (1 - RHO**2) * state, atol=1e-14)
            # Contexts ±e2 and ±e3 give the same signed scalar observation.
            for context, normal in ((3, normal2), (4, -normal2),
                                    (5, normal3), (6, -normal3)):
                mean, cov = model[context]
                np.testing.assert_allclose(normal @ mean[1:3] / SIGMA,
                                           tau * math.sin(sign*DELTA - BETA), atol=1e-14)
                np.testing.assert_allclose(normal @ cov[1:3, 1:3] @ normal / SIGMA**2,
                                           1, atol=1e-14)

        p_scores, q_scores, fine_scores = (score(m, drift, state)
                                           for m in (p_map, q_map, IDENTITY))
        for scores, phi in ((p_scores, PHI-DELTA), (q_scores, PHI+DELTA),
                            (fine_scores, PHI)):
            np.testing.assert_allclose(scores, closed_scores(phi), atol=1e-14)
        assert score_record(p_scores)['top2'] == score_record(fine_scores)['top2'] == [3, 4]
        assert score_record(q_scores)['top2'] == [2, 4]
        kl = tau**2 / 13 * (4 * (1 - math.cos(2*DELTA))
                            + math.sin(2*DELTA)**2 / (1+tau))
        for left, right in ((p, q), (q, p)):
            np.testing.assert_allclose(np.mean([gaussian_kl(a, b) for a, b in zip(left, right)]),
                                       kl, atol=1e-14, rtol=1e-10)
        min_signal = tau * math.sin(DELTA - BETA)
        log_upper = math.log1p(4/13 * math.expm1(-.5 * min_signal**2))
        low_budget = math.floor(.125 / kl)
        high_budget = math.ceil(math.log(.05) / log_upper)
        assert math.sqrt(low_budget * kl / 2) <= .25
        assert math.exp(high_budget * log_upper) <= .05
        records.append(dict(tau=tau, natural_parameter_rank=rank, one_transition_kl=kl,
                            n_at_most_for_tv_upper_quarter=low_budget,
                            n_sufficient_for_sign_test_error_at_most_05=high_budget,
                            active_count_sufficient_for_95pct_ci_to_exclude_zero_with_95pct_probability=
                            math.floor(8 * math.log(40) / min_signal**2) + 1))

    equal_gap_checks = []
    tau = .1
    contexts = conditional_contexts(tau)
    for distance_deg in (1, 4, 7):
        d = math.radians(distance_deg)
        minus, plus = (rotate(1, 2, BETA + sign*d) for sign in (-1, 1))
        p = score_record(score(minus, drift, IDENTITY))
        q = score_record(score(plus, drift, IDENTITY))
        expected_gap = (s_fast - s_slow) * math.sin(2*d)
        assert p['top2'] == [3, 4] and q['top2'] == [2, 4]
        np.testing.assert_allclose([p['boundary_gap'], q['boundary_gap']],
                                   expected_gap, atol=1e-14)
        expected_kl = (tau * math.sin(d))**2 / 13 * (8 + 4*math.cos(d)**2 / (1+tau))
        cp, cq = (observed_contexts(contexts, transform) for transform in (minus, plus))
        np.testing.assert_allclose(np.mean([gaussian_kl(a, b) for a, b in zip(cp, cq)]),
                                   expected_kl, atol=1e-14, rtol=1e-9)
        equal_gap_checks.append(dict(distance_degrees=distance_deg, tau=tau,
                                     shared_physical_gap=expected_gap, kl=expected_kl))

    return dict(kind='exact-moment numerical check, no draws', lags=LAGS, rho=RHO,
                angle_degrees=dict(psi=math.degrees(PSI), observed=math.degrees(PHI),
                                   rotation=math.degrees(DELTA)),
                fine=score_record(np.array(closed_scores(PHI))),
                p=score_record(np.array(closed_scores(PHI-DELTA))),
                q=score_record(np.array(closed_scores(PHI+DELTA))),
                boundary_axis_r2=math.cos(DELTA)**2,
                minimum_transition_budgets_are_not_inferred=True, checks=records,
                equal_gap_checks=equal_gap_checks,
                generated_trajectories=0, model_forwards=0, training_runs=0)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2, allow_nan=False))
