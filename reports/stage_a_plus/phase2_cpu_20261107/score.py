"""Summarize fixed CPU validation records; no sampling, fitting, or file I/O.

Decision -1 certifies the fixed population set I={3,4} (mask 12), +1
rejects it, and 0 abstains.  The two sides are independently seeded.
Window agreement is descriptive and is never used in the population gate.
"""

from functools import lru_cache
import math


FIXED_I = 12
NAMES = ("ref", "weak", "small_gap", "short", "long")


def _rate(count, total):
    return count / total if total else None


def _normal_tail(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


@lru_cache(maxsize=None)
def _expected(n, tau, d_deg, alpha):
    """Binomial mixture of exact Gaussian tails, up to FP64 arithmetic."""
    p = 4.0 / 13.0
    mu = abs(tau * math.sin(math.radians(d_deg)))
    logp, logq = math.log(p), math.log1p(-p)
    base = math.lgamma(n + 1)
    logs = [base - math.lgamma(m + 1) - math.lgamma(n - m + 1)
            + m * logp + (n - m) * logq for m in range(n + 1)]
    peak = max(logs)
    weights = [math.exp(value - peak) for value in logs]
    total = math.fsum(weights)
    correct_terms, wrong_terms = [], []
    threshold = math.sqrt(2.0 * math.log(2.0 / alpha))
    for m, weight in enumerate(weights[1:], 1):
        signal = mu * math.sqrt(m)
        correct_terms.append(weight * _normal_tail(threshold - signal))
        wrong_terms.append(weight * _normal_tail(threshold + signal))
    correct = math.fsum(correct_terms) / total
    wrong = math.fsum(wrong_terms) / total
    coverage = correct + wrong
    return dict(coverage=coverage, correct=correct,
                wrong_unconditional=wrong,
                wrong_conditional=_rate(wrong, coverage),
                no_active_probability=weights[0] / total,
                absolute_projection_mean=mu)


def _method(decisions, sides):
    accepted = sum(d != 0 for d in decisions)
    correct = sum(d == s for d, s in zip(decisions, sides))
    wrong = accepted - correct
    count = len(decisions)
    return dict(n=count, accepted=accepted, abstained=count - accepted,
                correct_count=correct, wrong_count=wrong,
                coverage=_rate(accepted, count), correct=_rate(correct, count),
                wrong_unconditional=_rate(wrong, count),
                wrong_conditional=_rate(wrong, accepted))


def _deviation(values):
    return dict(n=len(values), mean_absolute=(math.fsum(abs(v) for v in values)
                / len(values) if values else None),
                rmse=(math.sqrt(math.fsum(v*v for v in values) / len(values))
                      if values else None),
                max_absolute=max(map(abs, values), default=None))


def _validate_panel(panel, n):
    side = panel["side"]
    if side not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    size = len(panel["active"])
    if not size:
        raise ValueError("empty panel")
    for key in ("oracle", "ols", "oracle_mean", "ols_mean", "drift_rmse"):
        if len(panel[key]) != size:
            raise ValueError("panel arrays must have equal lengths: " + key)
    if any(type(m) is not int or not 0 <= m <= n for m in panel["active"]):
        raise ValueError("active must contain integer counts in [0,n]")
    for key in ("oracle", "ols"):
        if any(type(d) is not int or d not in (-1, 0, 1) for d in panel[key]):
            raise ValueError("invalid decision: " + key)
    for key in ("oracle_mean", "ols_mean"):
        if any(v is not None and not math.isfinite(v) for v in panel[key]):
            raise ValueError("nonfinite projection mean")
    if any(not math.isfinite(v) or v < 0 for v in panel["drift_rmse"]):
        raise ValueError("drift_rmse must be finite and nonnegative")
    for window in panel["windows"].values():
        for key in ("physical", "observed"):
            if len(window[key]) != size:
                raise ValueError("window mask length mismatch")
            if any(type(mask) is not int or not 0 <= mask <= 15
                   for mask in window[key]):
                raise ValueError("window masks must be four-bit integers")
    return size


def _cell(cell, alpha, comparison_count, familywise_alpha):
    n, tau, d_deg = cell["n"], cell["tau"], cell["d_deg"]
    if type(n) is not int or n < 1 or not math.isfinite(tau) or tau <= 0:
        raise ValueError("invalid transition budget or tau")
    if not math.isfinite(d_deg) or not 0 < d_deg < 90:
        raise ValueError("d_deg must be in (0,90)")
    sides, decisions = [], {"oracle": [], "ols": []}
    side_counts = {"-1": 0, "1": 0}
    side_stats, windows = {}, {}
    mean_diff, margin_diff, drift = [], [], []
    agreement = 0
    for panel in cell["panels"]:
        size = _validate_panel(panel, n)
        side = panel["side"]
        side_counts[str(side)] += size
        sides.extend([side] * size)
        for method in decisions:
            decisions[method].extend(panel[method])
        drift.extend(panel["drift_rmse"])
        agreement += sum(a == b for a, b in zip(panel["oracle"], panel["ols"]))
        for m, oracle, ols in zip(panel["active"], panel["oracle_mean"], panel["ols_mean"]):
            if m and oracle is not None and ols is not None:
                radius = math.sqrt(2 * math.log(2 / alpha) / m)
                mean_diff.append(ols - oracle)
                margin_diff.append((abs(ols) - radius) - (abs(oracle) - radius))
        population = 12 if side == -1 else 10
        for length, window in panel["windows"].items():
            w = windows.setdefault(str(length), dict(n=0, physical_population_count=0,
                observed_physical_count=0, observed_population_count=0,
                fixed_I_physical_count=0, fixed_I_population_count=0))
            w["n"] += size
            w["physical_population_count"] += sum(x == population for x in window["physical"])
            w["observed_population_count"] += sum(x == population for x in window["observed"])
            w["observed_physical_count"] += sum(x == y for x, y in zip(window["observed"], window["physical"]))
            w["fixed_I_physical_count"] += sum(x == FIXED_I for x in window["physical"])
            w["fixed_I_population_count"] += size if side == -1 else 0
    count = len(sides)
    if not count or not all(side_counts.values()):
        raise ValueError("every cell requires observations from both sides")
    for side in (-1, 1):
        indices = [i for i, value in enumerate(sides) if value == side]
        side_stats[str(side)] = dict(population_mask=12 if side == -1 else 10,
            fixed_I_correct=(side == -1),
            **{method: _method([values[i] for i in indices], [side]*len(indices))
               for method, values in decisions.items()})
    for window in windows.values():
        for key in ("physical_population", "observed_physical", "observed_population",
                    "fixed_I_physical", "fixed_I_population"):
            window[key + "_rate"] = window[key + "_count"] / window["n"]
    methods = {method: _method(values, sides) for method, values in decisions.items()}
    paired = {a + "__" + b: 0 for a in ("correct", "wrong", "abstain")
              for b in ("correct", "wrong", "abstain")}
    for side, oracle, ols in zip(sides, decisions["oracle"], decisions["ols"]):
        left = "abstain" if oracle == 0 else "correct" if oracle == side else "wrong"
        right = "abstain" if ols == 0 else "correct" if ols == side else "wrong"
        paired[left + "__" + right] += 1
    expected = dict(_expected(n, float(tau), float(d_deg), float(alpha)))
    band = math.sqrt(math.log(2 * comparison_count / familywise_alpha) / (2 * count))
    checks = {key: abs(methods["oracle"][key] - expected[key]) <= band
              for key in ("coverage", "wrong_unconditional")}
    return dict(dynamics=cell["dynamics"], name=cell["name"], transitions=n,
                tau=tau, d_deg=d_deg, samples=count, side_counts=side_counts,
                methods=methods, oracle_expected=expected, hoeffding_half_width=band,
                oracle_expected_checks=checks, per_side=side_stats,
                ols_oracle_decision_agreement=agreement/count,
                paired_fixed_I_judgment_counts=paired,
                ols_minus_oracle_mean=_deviation(mean_diff),
                ols_minus_oracle_acceptance_margin=_deviation(margin_diff),
                drift_rmse=_deviation(drift), windows=windows,
                fixed_I_mask=FIXED_I,
                fixed_I_correct_population_rate=side_counts["-1"]/count)


def summarize(payload):
    """Return descriptive scores and the prespecified oracle-only gates."""
    alpha = payload["alpha"]
    familywise_alpha = payload["familywise_alpha"]
    if not 0 < alpha < 1 or not 0 < familywise_alpha < 1:
        raise ValueError("alpha values must be in (0,1)")
    if not payload["cells"]:
        raise ValueError("no cells")
    planned = payload.get("planned_cells", len(payload["cells"]))
    if type(planned) is not int or planned < len(payload["cells"]):
        raise ValueError("planned_cells must be an integer covering all supplied cells")
    comparisons = 2 * planned
    cells = [_cell(c, alpha, comparisons, familywise_alpha) for c in payload["cells"]]
    grouped = {}
    for cell in cells:
        group = grouped.setdefault(cell["dynamics"], {})
        if cell["name"] in group:
            raise ValueError("duplicate cell name within dynamics")
        group[cell["name"]] = cell
    verdicts = {}
    for dynamics, group in grouped.items():
        complete = set(group) == set(NAMES)
        cover = {name: c["methods"]["oracle"]["coverage"] for name, c in group.items()}
        gaps = {name: cover["ref"]-cover[name] for name in ("weak", "small_gap", "short")
                if name in cover and "ref" in cover}
        gates = dict(expected_rates=all(all(c["oracle_expected_checks"].values()) for c in group.values()),
            complete_cells=complete,
            long_coverage=(cover.get("long", -1) >= .90),
            low_coverage=all(cover.get(name, 1) <= .20 for name in ("weak", "small_gap", "short")),
            reference_separation=(len(gaps) == 3 and all(gap >= .40 for gap in gaps.values())),
            unconditional_wrong=max(c["methods"]["oracle"]["wrong_unconditional"] for c in group.values()) <= .05)
        verdicts[dynamics] = dict(names=list(group), gates=gates,
            side_counts={side: sum(c["side_counts"][side] for c in group.values())
                         for side in ("-1", "1")},
            hoeffding_bands={name: c["hoeffding_half_width"] for name, c in group.items()},
            reference_minus_low_coverage=gaps, passed=all(gates.values()))
    return dict(alpha=alpha, familywise_alpha=familywise_alpha,
                planned_cells=planned, observed_cells=len(cells),
                all_planned_cells_present=(len(cells) == planned),
                simultaneous_rate_comparisons=comparisons,
                hoeffding_unit="independently seeded panel realization, pooled across sides",
                cells=cells, per_dynamics=verdicts,
                passed_by_dynamics={name: v["passed"] for name, v in verdicts.items()},
                passed=all(v["passed"] for v in verdicts.values()),
                window_scope="descriptive window agreement only; excluded from population gates",
                pairing_scope="same realization pairs oracle/OLS and window comparisons; sides are independent")
