"""Standard-library scoring of frozen hidden-boundary validation records.

No file I/O, sampling, or fitting. The confidence family remains 24 rates at
an interim with only one dynamics. Shared calibrations are counted once for
boundary interval coverage and do not create extra independent observations.
"""

from collections import defaultdict
import math


CELLS = ("ref", "long", "weak")
METHODS = ("estimated", "oracle")
FAMILY_SIZE = 24


def _rate(count, total):
    return count / total if total else None


def _summary(values):
    values = sorted(values)
    if not values:
        return dict(n=0, min=None, mean=None, median=None, q95=None, max=None)

    def quantile(q):
        index = q * (len(values) - 1)
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (index - lower) * (values[upper] - values[lower])

    return dict(n=len(values), min=values[0], mean=math.fsum(values)/len(values),
                median=quantile(.5), q95=quantile(.95), max=values[-1])


def _binomial_upper(k, n, tail):
    """One-sided Clopper-Pearson bound, via exact binomial CDF inversion.

    'Exact' refers to binomial coverage rather than an asymptotic interval;
    sums and root finding use ordinary floating-point arithmetic.
    """
    if n == 0:
        return None
    if k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(tail) / n)
    constants = [math.lgamma(n + 1) - math.lgamma(j + 1)
                 - math.lgamma(n - j + 1) for j in range(k + 1)]
    log_tail = math.log(tail)

    def log_cdf(p):
        lp, lq = math.log(p), math.log1p(-p)
        terms = [constant + j*lp + (n-j)*lq
                 for j, constant in enumerate(constants)]
        peak = max(terms)
        return peak + math.log(math.fsum(math.exp(t-peak) for t in terms))

    lower, upper = k / n, 1.0
    for _ in range(80):
        middle = (lower + upper) / 2
        if middle in (lower, upper):
            break
        if log_cdf(middle) > log_tail:
            lower = middle
        else:
            upper = middle
    return upper


def _method(rows, method, tail=None):
    n = len(rows)
    decided = sum(row[method] != 0 for row in rows)
    wrong = sum(row[method] not in (0, row["side"]) for row in rows)
    return dict(n=n, decided=decided, abstained=n-decided, wrong=wrong,
                correct=decided-wrong, coverage=_rate(decided, n),
                wrong_unconditional=_rate(wrong, n),
                wrong_conditional=_rate(wrong, decided),
                wrong_unconditional_upper=_binomial_upper(wrong, n, tail) if tail is not None else None)


def _balanced_method(rows, method, tail):
    strata = {str(s):_method([r for r in rows if r['side']==s],method,tail) for s in (-1,1)}
    sides = list(strata.values())
    if not sides[0]['n'] or sides[0]['n']!=sides[1]['n']:
        raise ValueError("each cell must contain equally many independent repetitions per side")
    result = _method(rows,method)
    result['per_side'] = strata
    result['wrong_unconditional_upper'] = math.fsum(v['wrong_unconditional_upper'] for v in sides)/2
    return result


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate(row):
    if not isinstance(row["dynamics"], str) or not row["dynamics"]:
        raise ValueError("dynamics must be a nonempty string")
    if row["cell"] not in CELLS:
        raise ValueError("cell must be ref, long, or weak")
    if type(row["side"]) is not int or row["side"] not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    if type(row["rep"]) is not int or row["rep"] < 0:
        raise ValueError("rep must be a nonnegative integer")
    for method in METHODS:
        if type(row[method]) is not int or row[method] not in (-1, 0, 1):
            raise ValueError("invalid decision: " + method)
    if type(row["active"]) is not int or row["active"] < 0:
        raise ValueError("active must be a nonnegative integer")
    if not _finite(row["drift_penalty"]) or row["drift_penalty"] < 0:
        raise ValueError("drift_penalty must be finite and nonnegative")
    if not _finite(row["beta_true_deg"]):
        raise ValueError("beta_true_deg must be finite")
    if row["beta_hat_deg"] is not None and not _finite(row["beta_hat_deg"]):
        raise ValueError("beta_hat_deg must be finite or None")
    radius = row["beta_radius_deg"]
    if radius is not None and (not _finite(radius) or radius < 0 or row["beta_hat_deg"] is None):
        raise ValueError("a bounded interval requires a finite center and nonnegative radius")


def _calibration(rows):
    unique = {}
    fields = ("beta_true_deg", "beta_hat_deg", "beta_radius_deg")
    for row in rows:
        key = row["side"], row["rep"]
        value = tuple(row[field] for field in fields)
        if key in unique and unique[key] != value:
            raise ValueError("shared calibration differs between cells for " + str(key))
        unique[key] = value
    bounded, covered, bounded_covered = 0, 0, 0
    errors, radii, details = [], [], []
    for (side, rep), (truth, center, radius) in sorted(unique.items()):
        error = None if center is None else center-truth
        if error is not None:
            errors.append(error)
        finite = radius is not None
        covers = not finite or abs(error) <= radius
        covered += covers
        if finite:
            bounded += 1
            bounded_covered += covers
            radii.append(radius)
        details.append(dict(side=side, rep=rep, beta_true_deg=truth,
            beta_hat_deg=center, beta_radius_deg=radius,
            bounded=finite, covered=covers, signed_error_deg=error))
    n = len(unique)
    return dict(unique_calibrations=n, bounded=bounded, unbounded=n-bounded,
        bounded_fraction=_rate(bounded, n), covered=covered,
        coverage_with_unbounded_covering=_rate(covered, n),
        bounded_covered=bounded_covered,
        bounded_only_coverage=_rate(bounded_covered, bounded),
        signed_error_deg=_summary(errors), absolute_error_deg=_summary([abs(e) for e in errors]),
        radius_deg=_summary(radii), width_deg=_summary([2*r for r in radii]),
        records=details)


def summarize(rows, method_alpha=.05, family_alpha=.05):
    """Return cell scores, unique-calibration diagnostics, and frozen gates."""
    if not rows or not 0 < method_alpha < 1 or not 0 < family_alpha < 1:
        raise ValueError("nonempty rows and alpha values in (0,1) required")
    seen = set()
    groups = defaultdict(list)
    for row in rows:
        _validate(row)
        key = row["dynamics"], row["cell"], row["side"], row["rep"]
        if key in seen:
            raise ValueError("duplicate realization within cell: " + str(key))
        seen.add(key)
        groups[row["dynamics"]].append(row)
    if len(groups) > 2:
        raise ValueError("fixed 24-rate family permits at most two dynamics")
    tail = family_alpha / FAMILY_SIZE
    results = {}
    for dynamics, items in groups.items():
        by_cell = defaultdict(list)
        for row in items:
            by_cell[row["cell"]].append(row)
        cells = {}
        for name, records in by_cell.items():
            cells[name] = dict(n=len(records),
                side_counts={str(s): sum(r["side"] == s for r in records) for s in (-1, 1)},
                methods={m: _balanced_method(records, m, tail) for m in METHODS},
                drift_penalty=_summary([r["drift_penalty"] for r in records]),
                active=_summary([r["active"] for r in records]),
                oracle_estimated_agreement=sum(r["oracle"] == r["estimated"] for r in records)/len(records))
        calibration = _calibration(items)
        complete = set(cells) == set(CELLS)
        coverage = lambda name, method: cells[name]["methods"][method]["coverage"]
        gates = dict(complete_cells=complete,
            both_sides=all(all(c["side_counts"].values()) for c in cells.values()),
            reference_coverage=("ref" in cells and coverage("ref", "estimated") >= .40
                and coverage("ref", "estimated") >= .50*coverage("ref", "oracle")),
            long_coverage=("long" in cells and coverage("long", "estimated") >= .70
                and coverage("long", "estimated") >= .70*coverage("long", "oracle")),
            oracle_long_coverage=("long" in cells and coverage("long", "oracle") >= .90),
            weak_coverage=("weak" in cells and coverage("weak", "estimated") <= .15),
            wrong_rate_upper=all(m["wrong_unconditional_upper"] <= method_alpha
                for c in cells.values() for m in c["methods"].values()),
            beta_coverage=calibration["coverage_with_unbounded_covering"] >= 1-method_alpha,
            beta_bounded_fraction=calibration["bounded_fraction"] >= .99)
        results[dynamics] = dict(cells=cells, beta=calibration, gates=gates,
                                 passed=all(gates.values()))
    passed = {name: result["passed"] for name, result in results.items()}
    return dict(method_alpha=method_alpha, family_alpha=family_alpha,
        confidence_family_size=FAMILY_SIZE, per_rate_tail_probability=tail,
        wrong_bound="per-side exact Clopper-Pearson with Bonferroni, then equal-side average",
        beta_unit="unique (side,rep) per dynamics; unbounded intervals count as covering",
        per_dynamics=results, passed_by_dynamics=passed, passed=all(passed.values()),
        both_dynamics_present=len(groups) == 2)
