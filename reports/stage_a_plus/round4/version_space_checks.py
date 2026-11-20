"""Fixed finite quantities only; no observations, models, sampling, or fitting."""
from pathlib import Path
import itertools
import json
import math

OUT = Path(__file__).resolve().parent

# Exercise all extremal uniform-deviation patterns, including a poor reference.
population_losses = [2.0, 2.005, 2.4]
epsilon = 0.02
coverage = []
for errors in itertools.product((-epsilon, epsilon), repeat=3):
    empirical = [loss + error for loss, error in zip(population_losses, errors)]
    for reference in range(3):
        threshold = empirical[reference] + 2 * epsilon
        retained = [j for j, loss in enumerate(empirical) if loss <= threshold + 1e-14]
        assert 0 in retained and reference in retained
        coverage.append({"errors": errors, "reference": reference, "retained": retained})

# Unit-variance abstract second moments. A normalized one-row shear gives an
# exact representation distance and score without any generated trajectories.
# M_h=M for each h; endpoint domination C0=C1=1 gives c_h=2.
physical_scores = [.2, .5, 1.2, 1.8]
fixtures = []
for t in (.01, .2, 1.0):
    a = 1 / math.sqrt(1 + t*t)
    b = t / math.sqrt(1 + t*t)
    distance_squared = (a-1)**2 + b*b
    eta = math.sqrt(distance_squared)
    fine_scores = physical_scores.copy()
    fine_scores[0] = a*a*physical_scores[0] + b*b*physical_scores[2]
    beta = [4*math.sqrt(score)*eta + 4*eta*eta for score in fine_scores]
    errors = [abs(f-p) for f, p in zip(fine_scores, physical_scores)]
    assert all(error <= budget + 1e-14 for error, budget in zip(errors, beta))
    chosen = sorted(range(4), key=fine_scores.__getitem__)[:2]
    others = [j for j in range(4) if j not in chosen]
    margin_after_budget = min(fine_scores[j]-fine_scores[i]-beta[i]-beta[j]
                              for i in chosen for j in others)
    if margin_after_budget > 0:
        assert set(chosen) == {0, 1}
    fixtures.append({"t": t, "eta": eta, "fine_scores": fine_scores,
                     "physical_scores": physical_scores, "score_discrepancies": errors,
                     "beta": beta, "chosen": chosen,
                     "minimum_pair_gap_after_budget": margin_after_budget,
                     "certificate_passes": margin_after_budget > 0})

assert fixtures[0]["certificate_passes"]
assert not fixtures[-1]["certificate_passes"]
result = {"scope": "analytic loss arrays and covariance moments, no sampled data",
          "coverage_corner_cases": coverage, "score_fixtures": fixtures,
          "checks": {"all_24_loss_coverage_cases": True,
                     "poor_reference_still_preserves_truth_candidate": True,
                     "normalized_score_discrepancy_bound": True,
                     "nonempty_small_diameter_certificate": True,
                     "large_diameter_certificate_abstains": True},
          "cost": {"model_forward": 0, "fitting": 0, "array_reads": 0,
                   "PRNG": 0, "new_physical_samples": 0, "server_GPU": 0}}
destination = OUT / "VERSION_SPACE_CHECKS.json"
if destination.exists():
    raise SystemExit("Preserve completed checks; refusing to overwrite.")
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result["checks"], ensure_ascii=False))
