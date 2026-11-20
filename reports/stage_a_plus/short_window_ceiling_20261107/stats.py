"""Analysis-only statistics for frozen short-window likelihood ratios.

No I/O, simulation, or fitting. Held-out rows have independent, equally likely
random true sides. Confidence allocation stays at all 42 planned bounds:
18 error upper bounds, 18 coverage lower bounds, and 6 full-information duals.
"""

from collections import defaultdict
from functools import lru_cache
import math


ERROR_BUDGET = .05
BOUND_COUNT = 42
CELLS = ("strong", "weak", "small_gap")


def _bayes_error(log_ratio):
    q = math.exp(-abs(log_ratio))
    return q / (1 + q)


def choose_threshold(log_ratios, target=.042):
    """Smallest whole-tie cutoff meeting design posterior unconditional risk.

    Exact-zero LLRs always abstain. If all nonzero scores qualify, return zero.
    If no nonempty group qualifies, return a finite cutoff above every score.
    Only likelihood ratios are used; no realized selection labels enter.
    """
    values = list(log_ratios)
    if not values or not 0 <= target < .5 or any(not math.isfinite(x) for x in values):
        raise ValueError("finite nonempty ratios and target in [0,.5) required")
    groups = defaultdict(int)
    for value in values:
        if value != 0:
            groups[abs(value)] += 1
    largest = max(map(abs, values))
    threshold = math.nextafter(largest, math.inf)
    count, error_sum = 0, 0.0
    for magnitude in sorted(groups, reverse=True):
        count += groups[magnitude]
        error_sum += groups[magnitude] * _bayes_error(magnitude)
        if error_sum / len(values) > target:
            break
        threshold = magnitude
    else:
        if count:
            threshold = 0.0
    if not math.isfinite(threshold):
        raise ValueError("cannot encode full abstention with a finite threshold")
    return threshold


@lru_cache(maxsize=None)
def _cp_upper(wrong, n, tail):
    """One-sided exact-binomial Clopper-Pearson upper endpoint (FP64)."""
    if wrong == n:
        return 1.0
    if wrong == 0:
        return -math.expm1(math.log(tail) / n)
    constants = [math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
                 for j in range(wrong+1)]
    low, high = wrong/n, 1.0
    log_tail = math.log(tail)
    for _ in range(80):
        mid = (low+high)/2
        if mid in (low, high):
            break
        lp, lq = math.log(mid), math.log1p(-mid)
        terms = [c+j*lp+(n-j)*lq for j, c in enumerate(constants)]
        peak = max(terms)
        log_cdf = peak + math.log(math.fsum(math.exp(v-peak) for v in terms))
        if log_cdf > log_tail:
            low = mid
        else:
            high = mid
    return high


def _decision(ll, threshold):
    return (1 if ll > 0 else -1) if ll != 0 and abs(ll) >= threshold else 0


def _posterior_wrong(ll_full, decision):
    if decision == 0:
        return 0.0
    error = _bayes_error(ll_full)
    return error if decision == (1 if ll_full >= 0 else -1) else 1-error


def _method(rows, decisions, tail, radius):
    n = len(rows)
    decided = sum(d != 0 for d in decisions)
    wrong = sum(d != 0 and d != r["side"] for r, d in zip(rows, decisions))
    posterior_sum = math.fsum(_posterior_wrong(r["ll_full"], d)
                             for r, d in zip(rows, decisions))
    return dict(n=n, decided=decided, abstained=n-decided, wrong=wrong,
                correct=decided-wrong, coverage=decided/n,
                correct_rate=(decided-wrong)/n,
                wrong_unconditional=wrong/n,
                wrong_conditional=wrong/decided if decided else None,
                wrong_unconditional_upper=_cp_upper(wrong, n, tail),
                coverage_lower=max(0.0, decided/n-radius),
                full_posterior_expected_wrong_unconditional=posterior_sum/n,
                full_posterior_expected_wrong_conditional=(posterior_sum/decided if decided else None))


def score_cells(rows, thresholds, alpha=.05):
    """Score held-out outcomes without choosing or adjusting thresholds."""
    if not rows or not 0 < alpha < 1:
        raise ValueError("nonempty rows and alpha in (0,1) required")
    grouped = defaultdict(list)
    for row in rows:
        if not isinstance(row["dynamics"], str) or not row["dynamics"] or row["cell"] not in CELLS:
            raise ValueError("invalid dynamics/cell")
        if row["side"] not in (-1, 1) or row["old"] not in (-1, 0, 1):
            raise ValueError("invalid truth or old decision")
        if not all(math.isfinite(row[k]) for k in ("ll_full", "ll_projection")):
            raise ValueError("likelihood ratios must be finite")
        grouped[row["dynamics"], row["cell"]].append(row)
    dynamics = {key[0] for key in grouped}
    if len(dynamics) > 2 or len(grouped) > 6:
        raise ValueError("fixed family permits at most two dynamics and six cells")
    tail = alpha / BOUND_COUNT
    results = defaultdict(dict)
    for (dynamic, cell), records in grouped.items():
        cutoff = thresholds[cell]
        for method in ("full", "projection"):
            if not math.isfinite(cutoff[method]) or cutoff[method] < 0:
                raise ValueError("thresholds must be finite and nonnegative")
        n = len(records)
        radius = math.sqrt(math.log(1/tail)/(2*n))
        decisions = {method: [_decision(r["ll_"+method], cutoff[method]) for r in records]
                     for method in ("full", "projection")}
        decisions["old"] = [r["old"] for r in records]
        methods = {m: _method(records, values, tail, radius) for m, values in decisions.items()}
        t = _bayes_error(cutoff["full"])
        if t > 0:
            hinge = math.fsum(max(0.0, 1-_bayes_error(r["ll_full"])/t) for r in records)/n
            budget_term = ERROR_BUDGET/t
            bound = min(1.0, budget_term+hinge+radius)
        else:
            hinge, budget_term, bound = None, None, 1.0
        results[dynamic][cell] = dict(n=n,
            side_counts={str(side): sum(r["side"] == side for r in records) for side in (-1, 1)},
            thresholds={m: cutoff[m] for m in ("full", "projection")}, methods=methods,
            hoeffding_radius=radius,
            full_posterior_bayes_error=math.fsum(_bayes_error(r["ll_full"]) for r in records)/n,
            full_dual=dict(error_budget=ERROR_BUDGET, design_t=t,
                heldout_hinge_mean=hinge, budget_term=budget_term,
                achievable_coverage_upper=bound,
                trivial_from_budget=(t <= ERROR_BUDGET),
                formula="min(1, error_budget/t + mean(max(0,1-e/t)) + Hoeffding)",
                scope="full observable Bayes experiment; t fixed using independent design data"))
    verdicts = {}
    for dynamic, cells in results.items():
        strong = cells.get("strong")
        checks = dict(strong_present=strong is not None,
            coverage=(strong is not None and strong["methods"]["full"]["coverage_lower"] >= .35),
            error=(strong is not None and strong["methods"]["full"]["wrong_unconditional_upper"] <= ERROR_BUDGET))
        verdicts[dynamic] = dict(checks=checks, passed=all(checks.values()))
    return dict(alpha=alpha, planned_bound_count=BOUND_COUNT,
        per_bound_failure_probability=tail,
        allocation=dict(error_upper=18, coverage_lower=18, full_dual_upper=6),
        confidence_unit="independent held-out draws with iid equal-prior true sides",
        cells=dict(results), verdicts=verdicts,
        passed_by_dynamics={name: v["passed"] for name, v in verdicts.items()})
