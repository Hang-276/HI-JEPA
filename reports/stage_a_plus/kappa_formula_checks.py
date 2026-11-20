"""Deterministic mathematical fixtures; no physical data, training, or PRNG.

Checks closed-form Gaussian rotation identities and the proved global bound.
This is neither a model experiment nor evidence about trained Fine weights.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np


def rotation(d: int, i: int, j: int, theta: float) -> np.ndarray:
    q = np.eye(d)
    c, s = math.cos(theta), math.sin(theta)
    q[i, i] = q[j, j] = c
    q[i, j], q[j, i] = s, -s
    return q


def risk(q: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> float:
    return float(sum(
        p * .5 * (np.log(np.diag(q @ np.diag(v) @ q.T)).sum() - np.log(v).sum())
        for p, v in zip(weights, variances)
    ))


def quotient_distance_sq(q: np.ndarray) -> float:
    d = len(q)
    return float(2 * d - 2 * max(
        sum(abs(q[i, perm[i]]) for i in range(d))
        for perm in itertools.permutations(range(d))
    ))


def check() -> dict:
    current = []
    for name, tau in [('rich', 1.), ('weak', .01)]:
        r = 1 + 2 * tau
        a = (r * r - 1) ** 2 / (4 * r * r)
        kappa = 8 * a / 9
        barrier = math.log1p(a) / 9
        theta = math.pi / 8
        z = math.sin(2 * theta) ** 2
        f = math.log1p(a * z) / 9
        inv_z = math.expm1(9 * f) / a
        h = 1e-5
        hessian_fixture = 2 * math.log1p(a * math.sin(2 * h) ** 2) / (9 * h * h)
        assert abs(inv_z - z) < 1e-13
        assert barrier * z <= f + 1e-14
        assert f <= kappa * z / 8 + 1e-14
        assert abs(hessian_fixture / kappa - 1) < 1e-8
        current.append(dict(law=name, tau=tau, std_multiplier=r,
                            kappa_curvature=kappa, barrier_45_degrees=barrier,
                            fixed_formula_theta_radians=theta, profiled_risk=f,
                            inverse_z_error=abs(inv_z-z),
                            hessian_relative_error=abs(hessian_fixture/kappa-1)))

    variances = np.array([[1., 1., 1.], [1.3, .9, 1.1], [.8, 1.5, .95]])
    weights = np.array([.2, .3, .5])
    pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
    kappas = {f'{i}{j}': float(np.sum(weights *
              (variances[:, i]-variances[:, j])**2 /
              (variances[:, i]*variances[:, j]))) for i, j in pairs}
    kappa_min = min(kappas.values())
    ratio = float(variances.max() / variances.min())
    q = rotation(3, 0, 1, .31) @ rotation(3, 1, 2, -.23)
    f = risk(q, variances, weights)
    delta_sq = quotient_distance_sq(q)
    lower = kappa_min * delta_sq / (16 * ratio**2)
    assert f >= lower - 1e-14
    w = q**2
    mixing_mass = float((3 - np.sum(w*w))/2)
    assert delta_sq <= 4*mixing_mass + 1e-14

    return dict(
        activity='deterministic_formula_checks_only',
        new_physical_data=0, training=0, model_inference=0, PRNG_draws=0,
        historical_result_modified=False,
        current_nominal_law_formula_rows=current,
        fixed_three_dimensional_matrix_fixture=dict(
            variances=variances.tolist(), weights=weights.tolist(),
            kappas=kappas, global_variance_ratio=ratio,
            profiled_risk=f, quotient_distance_sq=delta_sq,
            global_bound_lower=lower, mixing_mass=mixing_mass,
            mathematical_inequalities_verified=True),
        limits='Matrix fixtures verify algebra only, not scientific validity of any trained representation.'
    )


if __name__ == '__main__':
    out = Path(__file__).with_name('KAPPA_FORMULA_CHECKS.json')
    if out.exists():
        raise SystemExit('Preserve existing output; inspect it rather than overwrite.')
    out.write_text(json.dumps(check(), ensure_ascii=False, indent=2) + '\n')
    print(out)
