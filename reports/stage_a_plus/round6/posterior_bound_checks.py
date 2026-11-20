"""Fixed arithmetic for the Round6 proof; no data, model, fit, or sampling."""
from fractions import Fraction as F
from itertools import product
from math import exp, log, sqrt, isclose
from pathlib import Path
import hashlib
import json

OUT = Path(__file__).resolve().parent
checks = []


def check(name, passed, **details):
    checks.append(dict(name=name, passed=bool(passed), **details))


def norm2(x):
    return sum(z*z for z in x)


def noise(v):
    s = sum(v)
    return s - norm2(v)/s if s else F(0)


def bound(n, m, bins, eps, risk_hat, noise_hat, mass_hat=1., kappa=0., delta=.05):
    k = m*bins
    d_r = d_n = delta/2
    t = sqrt(log(2*k/d_n)/(2*n))
    r = sqrt(2*log(1/d_r)/n)
    upper_r = min(2., risk_hat+r)
    lower_n = max(0., noise_hat-2*k*t)
    upper_mass = min(1., mass_hat+k*t)
    raw = max(0., upper_r-lower_n+kappa*upper_mass)/eps
    cap = upper_mass*log(1/eps)
    return dict(n=n, m=m, bins=bins, epsilon=eps, delta=delta,
                indicator_radius=t, risk_radius=r, noise_penalty=2*k*t,
                risk_upper=upper_r, noise_lower=lower_n,
                mass_upper=upper_mass, raw_KL_bound=raw,
                trivial_cap=cap, final_bound=min(raw, cap))


# Nonconstant q in both occupied cells; a fifth atom lies outside Omega.
atoms = [
    (F(1,10), 0, [F(9,10), F(1,10)], [F(4,5), F(1,5)]),
    (F(2,10), 0, [F(1,5), F(4,5)], [F(2,5), F(3,5)]),
    (F(2,10), 1, [F(3,5), F(2,5)], [F(1,2), F(1,2)]),
    (F(1,10), 1, [F(1,10), F(9,10)], [F(1,4), F(3,4)]),
    (F(4,10), None, [F(1), F(0)], [F(1,3), F(2,3)]),
]
risk = sum(w*(1-norm2(p)+norm2([p[i]-q[i] for i in range(2)]))
           for w,b,p,q in atoms if b is not None)
a2 = sum(w*norm2([p[i]-q[i] for i in range(2)])
         for w,b,p,q in atoms if b is not None)
total_noise = F(0)
variation = F(0)
for b in (0, 1, 2):  # includes an empty cell
    rows = [(w,p) for w,bb,p,q in atoms if bb == b]
    mu = sum(w for w,p in rows)
    if not mu:
        check('empty_cell_continuous_extension', noise([F(0),F(0)]) == 0)
        continue
    pb = [sum(w*p[i] for w,p in rows)/mu for i in range(2)]
    total_noise += mu*(1-norm2(pb))
    within = sum(w*norm2([p[i]-pb[i] for i in range(2)]) for w,p in rows)
    independent_pair = sum(w*v*norm2([p[i]-q[i] for i in range(2)])
                           for w,p in rows for v,q in rows)/(2*mu)
    variation += within
    check(f'conditional_pair_variance_identity_cell_{b}', within == independent_pair)
check('restricted_Brier_identity_exact_rational', a2 == risk-total_noise+variation,
      A2=str(a2), risk=str(risk), noise=str(total_noise), variation=str(variation))
check('Omega_mass_not_conditionally_renormalized',
      sum(w for w,b,p,q in atoms if b is not None) == F(3,5))

# Fixed simplex vectors include zero probabilities and boundary cases.
vectors = [(F(i,4),F(j,4),F(k,4)) for i,j,k in product(range(5), repeat=3)
           if i+j+k <= 4]
gradient_checks = 0
for v in vectors:
    if sum(v):
        p = [x/sum(v) for x in v]
        for pi in p:
            derivative = 1-2*pi+norm2(p)
            assert 0 <= derivative <= 2
            gradient_checks += 1
lipschitz_checks = 0
for v in vectors:
    for w in vectors:
        assert abs(noise(v)-noise(w)) <= 2*sum(abs(x-y) for x,y in zip(v,w))
        lipschitz_checks += 1
check('noise_gradient_and_L1_constant_including_origin', True,
      gradient_cases=gradient_checks, pair_cases=lipschitz_checks)

kl_count = 0
for p in ([0.,1.], [.1,.9], [.5,.5], [.9,.1], [1.,0.]):
    for q in ([.1,.9], [.3,.7], [.5,.5], [.9,.1]):
        kl = sum(pi*log(pi/qi) for pi,qi in zip(p,q) if pi)
        chi = sum((pi-qi)**2/qi for pi,qi in zip(p,q))
        eps_bound = sum((pi-qi)**2 for pi,qi in zip(p,q))/min(q)
        assert -1e-14 <= kl <= chi+1e-14 <= eps_bound+2e-14
        kl_count += 1
check('KL_chisquare_floor_direction_with_zero_true_probabilities', True, cases=kl_count)

counter_kl = .9*log(1.8)+.1*log(.2)
check('coarse_calibration_counterexample_needs_variation', counter_kl > .3,
      bin_calibration_error=0., risk=.5, bin_noise=.5, variation=.32,
      true_KL=counter_kl, scope='generic coarsening, not full-rank Fine counterexample')

small_n = []
for n in (24, 32):
    row = bound(n, 9, 1, 1/9, 0., 8/9)
    row['interpretation'] = 'Optimistic arithmetic only: risk zero, maximal noise; not actual audit data'
    assert row['noise_lower'] == 0.
    assert row['raw_KL_bound'] > row['trivial_cap']
    assert row['risk_radius'] > log(9)/9
    small_n.append(row)
check('24_and_32_even_optimistic_constructor_is_trivial', True,
      applies_to_all_eps_at_most_one_ninth=True,
      claim='this construction only, not an information-theoretic impossibility')

# A finite arithmetic witness that the bound itself is not always trivial.
# A hypothetical balanced 2-class sample under p=q=(.5,.5); no samples generated.
nonempty = bound(10**8, 2, 1, .5, .5, .5)
check('fixed_hypothetical_finite_n_nonempty_budget', nonempty['final_bound'] < .01,
      **nonempty, scope='scalar posterior theorem only; rank-zero q; not a Fine/ICA fixture')

phi = .5*log(.5)-.5+1
absorption = (1/9)*(1/9)*phi*exp(-4)
check('Round5_optimistic_absorption_scale', 3e-5 < absorption < 4e-5,
      threshold=absorption, alpha=.5, epsilon=1/9, pi0=1/9, tail_mass=0.,
      scope='constant budget comparison, not actual model parameters')

for n in (24, 32, 10**8):
    d_r = d_n = .025
    k = 9
    t = sqrt(log(2*k/d_n)/(2*n))
    r = sqrt(2*log(1/d_r)/n)
    assert isclose(2*k*exp(-2*n*t*t), d_n, rel_tol=1e-12)
    assert isclose(exp(-n*r*r/2), d_r, rel_tol=1e-12)
check('finite_sample_failure_probability_bookkeeping', True)

result = dict(
    date='2026-11-05', scope='fixed rational/analytic arithmetic; no scientific data or simulation',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    proof_sha256=hashlib.sha256((OUT/'OBSERVABLE_POSTERIOR_BOUND.md').read_bytes()).hexdigest(),
    checks=checks, check_count=len(checks), all_passed=all(c['passed'] for c in checks),
    optimistic_small_n=small_n, finite_nonempty_arithmetic_example=nonempty,
    absorption_scale=absorption,
    independent_review_of_new_finite_sample_proof=False,
    cost=dict(scientific_array_reads=0, model_execution=0, new_fits=0,
              PRNG_calls=0, physical_sampling=0, server_connections=0, GPU_jobs=0))
assert result['all_passed']
(OUT/'POSTERIOR_BOUND_CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(checks=len(checks), all_passed=result['all_passed'],
                     small_n=small_n, absorption=absorption),ensure_ascii=False,indent=2))
