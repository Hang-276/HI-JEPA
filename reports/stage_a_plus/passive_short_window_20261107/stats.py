"""Frozen passive/known-distance scoring; standard library, no I/O or fitting.

Design and held-out realizations are independent, with iid equal-prior sides.
The family allocation includes both planned dynamics even at the first gate.
"""

from collections import defaultdict
from functools import lru_cache
import math


METHODS = ("oracle", "passive", "known_distance")
BOUND_COUNT = 18


def _posterior_wrong(oracle_llr, decision):
    if decision == 0:
        return 0.0
    q = math.exp(-abs(oracle_llr))
    e = q/(1+q)
    return e if decision == (1 if oracle_llr >= 0 else -1) else 1-e


def _sign(score):
    return 1 if score > 0 else -1 if score < 0 else 0


def design_cutoff(score, oracle, target=.038):
    """Choose a whole-tie cutoff using unconditional posterior design risk.

    Risk is divided by ALL design realizations, including abstentions. The
    realized true labels are never used. Zero scores always abstain. Return
    zero when every nonzero score qualifies, otherwise an observed cutoff;
    a finite nextafter cutoff above the maximum denotes complete abstention.
    """
    score, oracle = list(score), list(oracle)
    if not score or len(score) != len(oracle) or not 0 <= target <= 1:
        raise ValueError("nonempty equal-length arrays and target in [0,1] required")
    if any(not math.isfinite(v) for v in score+oracle):
        raise ValueError("LLRs must be finite")
    groups = defaultdict(list)
    for s, truth_llr in zip(score, oracle):
        if s != 0:
            groups[abs(s)].append(_posterior_wrong(truth_llr, _sign(s)))
    cutoff = math.nextafter(max(map(abs, score)), math.inf)
    accumulated = 0.0
    for magnitude in sorted(groups, reverse=True):
        accumulated += math.fsum(groups[magnitude])
        if accumulated/len(score) > target:
            break
        cutoff = magnitude
    else:
        cutoff = 0.0
    if not math.isfinite(cutoff):
        raise ValueError("finite all-abstain cutoff cannot be represented")
    return cutoff


@lru_cache(maxsize=None)
def _cp_upper(k, n, tail):
    """One-sided exact-binomial Clopper-Pearson bound, computed in FP64."""
    if k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(tail)/n)
    constants = [math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
                 for j in range(k+1)]
    low, high, log_tail = k/n, 1.0, math.log(tail)
    for _ in range(80):
        p = (low+high)/2
        if p in (low, high):
            break
        lp, lq = math.log(p), math.log1p(-p)
        terms = [c+j*lp+(n-j)*lq for j, c in enumerate(constants)]
        peak = max(terms)
        log_cdf = peak+math.log(math.fsum(math.exp(v-peak) for v in terms))
        if log_cdf > log_tail:
            low = p
        else:
            high = p
    return high


def summarize(rows, thresholds, family_alpha=.05):
    """Evaluate one dynamics with all 18 planned bounds in the allocation."""
    if not rows or not 0 < family_alpha < 1:
        raise ValueError("nonempty rows and family_alpha in (0,1) required")
    for method in METHODS:
        if not math.isfinite(thresholds[method]) or thresholds[method] < 0:
            raise ValueError("thresholds must be finite and nonnegative")
    for row in rows:
        if type(row["side"]) is not int or row["side"] not in (-1, 1):
            raise ValueError("side must be -1 or +1")
        if any(not math.isfinite(row[method]) for method in METHODS):
            raise ValueError("LLRs must be finite")
    n = len(rows)
    tail = family_alpha/BOUND_COUNT
    band = math.sqrt(math.log(1/tail)/(2*n))
    decisions, methods = {}, {}
    for method in METHODS:
        decision = [_sign(r[method]) if abs(r[method]) >= thresholds[method] else 0 for r in rows]
        decisions[method] = decision
        accepted = sum(d != 0 for d in decision)
        wrong = sum(d != 0 and d != r["side"] for r, d in zip(rows, decision))
        posterior_sum = math.fsum(_posterior_wrong(r["oracle"], d) for r, d in zip(rows, decision))
        methods[method] = dict(n=n, decided=accepted, accepted=accepted,
            abstained=n-accepted, wrong=wrong, correct=accepted-wrong,
            coverage=accepted/n, wrong_unconditional=wrong/n,
            wrong_conditional=wrong/accepted if accepted else None,
            wrong_unconditional_upper=_cp_upper(wrong, n, tail),
            coverage_lower=max(0.0, accepted/n-band),
            coverage_upper=min(1.0, accepted/n+band),
            posterior_expected_wrong_unconditional=posterior_sum/n,
            posterior_expected_wrong_conditional=posterior_sum/accepted if accepted else None,
            threshold=thresholds[method])
    states = ("correct", "wrong", "abstain")
    paired = {a+"__"+b: 0 for a in states for b in states}
    decision_pairs = {str(a)+"__"+str(b): 0 for a in (-1, 0, 1) for b in (-1, 0, 1)}
    for row, known, passive in zip(rows, decisions["known_distance"], decisions["passive"]):
        state = lambda d: "abstain" if d == 0 else "correct" if d == row["side"] else "wrong"
        paired[state(known)+"__"+state(passive)] += 1
        decision_pairs[str(known)+"__"+str(passive)] += 1
    passive, oracle = methods["passive"], methods["oracle"]
    checks = dict(passive_error_upper=passive["wrong_unconditional_upper"] <= .05,
        passive_coverage_lower=passive["coverage_lower"] >= .35,
        passive_relative_coverage=passive["coverage_lower"] >= .70*oracle["coverage_upper"],
        oracle_error_upper=oracle["wrong_unconditional_upper"] <= .05)
    return dict(n=n, methods=methods, checks=checks, passed=all(checks.values()),
        side_counts={str(s): sum(r["side"] == s for r in rows) for s in (-1, 1)},
        family_alpha=family_alpha, planned_bound_count=BOUND_COUNT,
        per_bound_failure_probability=tail, fixed_hoeffding_band=band,
        confidence_unit="independent equal-prior random-side held-out realization",
        paired_known_vs_passive=dict(order=["known_distance", "passive"],
            outcome_counts=paired, decision_counts=decision_pairs,
            passive_minus_known_coverage=passive["coverage"]-methods["known_distance"]["coverage"],
            decision_agreement=sum(a == b for a, b in zip(decisions["known_distance"], decisions["passive"]))/n))
