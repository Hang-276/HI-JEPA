"""Deterministic identities and boundary cases; no physical draws or models."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def nearest_tie(g: float, beta: float, alpha: float) -> dict:
    assert g > 0
    disc = beta*beta-alpha*g
    if alpha == 0 and beta == 0:
        return dict(radius=None, roots=[], kind='limiting_tie_only', discriminant=disc)
    if alpha > 0 and disc < 0:
        return dict(radius=None, roots=[], kind='uniform_positive_gap', discriminant=disc)
    assert disc >= 0
    radius = g/(abs(beta)+math.sqrt(disc))
    if alpha == 0:
        roots = [-g/(2*beta)]
    else:
        roots = [(-beta-math.sqrt(disc))/alpha, (-beta+math.sqrt(disc))/alpha]
    assert abs(min(abs(t) for t in roots)-radius) < 1e-12
    assert all(abs(g+2*beta*t+alpha*t*t) < 1e-11 for t in roots)
    return dict(radius=radius, roots=roots,
                kind='tangential_tie' if disc == 0 else 'crossing_tie', discriminant=disc)


def main() -> dict:
    fixtures = []
    for label, g, beta, alpha in [
        ('concave_two_signs', 1., 0., -1.),
        ('concave_asymmetric', 1., 2., -1.),
        ('linear', 1., .5, 0.),
        ('constant_limiting', 1., 0., 0.),
        ('convex_no_tie', 1., 0., 1.),
        ('convex_tangent', 1., 1., 1.),
        ('convex_two_same_sign_roots', 1., 2., 1.),
    ]:
        fixtures.append(dict(label=label, g=g, beta=beta, alpha=alpha,
                             **nearest_tie(g, beta, alpha)))

    # Matrix identity with both target/non-target orientations; no data samples.
    sigma = np.array([[1., .2, -.1], [.2, 1., .15], [-.1, .15, 1.]])
    moment = np.array([[.7, .12, -.03], [.12, 1.4, .09], [-.03, .09, 2.]])
    assert np.linalg.eigvalsh(sigma).min() > 0
    assert np.linalg.eigvalsh(moment).min() > 0
    identity_errors = []
    for i, j, t in [(0, 1, -.4), (1, 0, .3), (2, 1, -1.2)]:
        transform = np.eye(3)
        transform[i, j] = t
        transformed_scores = np.diag(transform@moment@transform.T)/np.diag(transform@sigma@transform.T)
        direct = (moment[i, i]+2*t*moment[i, j]+t*t*moment[j, j])/(1+2*t*sigma[i, j]+t*t)
        identity_errors.append(float(abs(direct-transformed_scores[i])))
        assert np.max(abs(np.delete(transformed_scores, i)-np.delete(np.diag(moment), i))) < 1e-14

    eps = .01
    rates = [.1, math.sqrt(1-eps*eps), .9, .2]
    horizons = [4, 16, 32]
    scores = np.array([-2*sum(math.expm1(h*math.log(r)) for h in horizons)/3 for r in rates])
    assert list(np.argsort(scores)[:2]) == [1, 2]
    threshold_sq = (scores[0]-scores[2])/(scores[2]-scores[1])
    threshold = math.sqrt(threshold_sq)
    ratio = eps*eps/.99

    def risk(t: float) -> float:
        z = ratio*t*t
        return (7*math.log1p(z)+math.log1p(z/9)+math.log1p(9*z))/18

    tests = []
    for t in [-1., -threshold*1.001, -threshold*.999, 0., threshold*.999, threshold, threshold*1.001, 1.]:
        values = scores.copy()
        values[0] = (scores[0]+t*t*scores[1])/(1+t*t)
        crossing_gap = float(values[0]-values[2])
        tests.append(dict(t=float(t), risk=risk(t), scores=values.tolist(),
                          selected_zero_based=sorted(np.argsort(values)[:2].tolist()),
                          changing_target_boundary_gap=crossing_gap))
    assert abs(next(x['changing_target_boundary_gap'] for x in tests if x['t'] == threshold)) < 1e-13
    assert tests[0]['selected_zero_based'] == tests[-1]['selected_zero_based'] == [0, 1]
    assert risk(1.) > risk(threshold)
    assert list(np.argsort(scores)[:2]) == [1, 2]
    return dict(status='completed_deterministic_shear_formula_checks',
        new_training=0, PRNG_calls=0, new_physical_samples=0, model_forward=0,
        scalar_polynomial_fixtures=fixtures,
        matrix_identity_max_error=max(identity_errors),
        persistent_four_dimensional_analytic_example=dict(
            epsilon=eps, rates=rates, horizons=horizons, state_covariance='identity',
            original_scores=scores.tolist(), target_zero_based=[1, 2],
            original_physical_margin=float(scores[3]-scores[2]),
            shear_zero_based=dict(output_axis=0, source_axis=1),
            critical_t_squared=float(threshold_sq), critical_absolute_t=threshold,
            critical_profiled_risk=risk(threshold), risk_at_t1=risk(1.),
            checks=tests),
        limits='Analytic laws and fixed matrices, not arrays or scores from trained Fine models.')


if __name__ == '__main__':
    out = Path(__file__).with_name('SHEAR_BARRIER_CHECKS.json')
    if out.exists():
        raise SystemExit('Preserve completed deterministic output.')
    out.write_text(json.dumps(main(), ensure_ascii=False, indent=2) + '\n')
    print(out)
