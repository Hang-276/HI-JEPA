"""One locked, no-refit ablation using archived JSON only; standard library."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import statistics

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "phase1_cache_20261107"
SLOTS = ("ar_rich_fresh1", "ar_rich_fresh2", "fhn_fresh_eps008", "fhn_fresh_eps05")
OLD = ("identity", "size", "direction", "persistence", "b_diag", "b_fixed", "b_current")
CAND = "b_current_plus_train_cross"
FIXED = "b_fixed_plus_train_cross"
METHODS = OLD + (FIXED, CAND)
COMPARATORS = (FIXED, "identity", "size", "direction")
GATE = (CAND,) + COMPARATORS
INPUTS = tuple(f"{phase}_{slot}.json" for phase in ("freeze", "reveal") for slot in SLOTS)
AXES = range(4)
PAIRS = tuple(combinations(AXES, 2))


def mean(values):
    values = [x for x in values if x is not None]
    return statistics.fmean(values) if values else None


def ratio(a, b):
    return a / b if b else None


def finite(values, size=4):
    return isinstance(values, list) and len(values) == size and all(isinstance(x, (float, int)) and math.isfinite(x) for x in values)


def top(values):
    if values is None:
        return None
    order = sorted(AXES, key=lambda j: values[j])
    return tuple(sorted(order[:2])) if values[order[1]] < values[order[2]] else None


def margin(values, selected):
    if values is None or selected is None:
        return None
    return min(values[j] for j in AXES if j not in selected) - max(values[j] for j in selected)


def key(row):
    return row["slot"], row["episode"], row["rep"]


def scope(rows):
    return dict(rows=len(rows), trajectories=len({(r["slot"], r["episode"]) for r in rows}), slots=len({r["slot"] for r in rows}))


def dump(name, value):
    path = HERE / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    dump("METHOD_LOCK.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), script_sha256=digest(Path(__file__)),
        protocol_sha256=digest(HERE / "PROTOCOL.md"),
        input_sha256={name: digest(SOURCE / name) for name in INPUTS},
        new_methods=[FIXED, CAND], operation="Add unweighted three-lag mean of stored TRAIN residual_cross; coefficient exactly 1",
        evidence="post-hoc development; all CAL already exposed", fits=0, new_samples=0,
    ))


def metrics(rows, method):
    counts = Counter({k: 0 for k in ("wrong_detected", "wrong_missed", "false_positive", "correct_not_flagged", "abstain", "unavailable")})
    errors, top_hits, top_n = [], 0, 0
    for row in rows:
        pred = row["predictions"][method]
        if pred is None:
            counts["unavailable"] += 1
            continue
        y, z, p = top(row["y"]), top(row["z"]), top(pred)
        errors.append(mean(abs((pred[b] - pred[a]) - (row["z"][b] - row["z"][a])) for a, b in PAIRS))
        if z is not None and p is not None:
            top_n += 1
            top_hits += z == p
        if None in (y, z, p):
            counts["abstain"] += 1
        elif y == z:
            counts["correct_not_flagged" if y == p else "false_positive"] += 1
        else:
            counts["wrong_missed" if y == p else "wrong_detected"] += 1
    counts = dict(counts)
    return dict(**counts, n_rows=len(rows), gap_mae=mean(errors), physical_top2_correct=top_hits,
                physical_top2_evaluable=top_n, physical_top2_accuracy=ratio(top_hits, top_n),
                wrong_detection_rate=ratio(counts["wrong_detected"], counts["wrong_detected"] + counts["wrong_missed"]),
                false_positive_rate=ratio(counts["false_positive"], counts["false_positive"] + counts["correct_not_flagged"]))


def matched(rows):
    families = defaultdict(list)
    for row in rows:
        families[(row["slot"], row["episode"])].append(row)
    result = []
    for (slot, episode), family in sorted(families.items()):
        assert len(family) == 3 and len({r["rep"] for r in family}) == 3
        for left, right in combinations(sorted(family, key=lambda r: r["rep"]), 2):
            dt = max(abs(a - b) for a, b in zip(left["train_r2"], right["train_r2"]))
            de = max(abs(a - b) for a, b in zip(left["eval_r2"], right["eval_r2"]))
            if dt > .01:
                continue
            yl, yr, zl, zr = top(left["y"]), top(right["y"]), top(left["z"]), top(right["z"])
            assert zl == zr
            discord = None not in (yl, yr, zl) and (yl == zl) != (yr == zr)
            margins = {m: [margin(r["predictions"][m], top(r["y"])) for r in (left, right)] for m in METHODS}
            verdict = {m: None for m in METHODS}
            good = (0 if yl == zl else 1) if discord else None
            for m, values in margins.items():
                if discord and None not in values:
                    delta = values[good] - values[1 - good]
                    verdict[m] = "tie" if abs(delta) <= 1e-12 else "win" if delta > 0 else "loss"
            result.append(dict(slot=slot, episode=episode, reps=[left["rep"], right["rep"]], train_r2_maxdiff=dt,
                               eval_r2_maxdiff=de, strict=de <= .01, discordant=discord,
                               correct_rep=None if good is None else [left["rep"], right["rep"]][good],
                               margins=margins, verdict=verdict))
    return result


def ranking(pairs, methods=METHODS):
    discord = [p for p in pairs if p["discordant"] and all(p["verdict"][m] is not None for m in methods)]
    answer = dict(eligible_pairs=len(pairs), discordant_pairs=len(discord),
                  trajectories=len({(p["slot"], p["episode"]) for p in discord}),
                  slots=len({p["slot"] for p in discord}), methods={})
    for m in methods:
        tally = Counter({"win": 0, "tie": 0, "loss": 0})
        episodes = defaultdict(list)
        for p in discord:
            verdict = p["verdict"][m]
            tally[verdict] += 1
            episodes[(p["slot"], p["episode"])].append({"win": 1., "tie": .5, "loss": 0.}[verdict])
        slots = {s: mean(mean(v) for (slot, _), v in episodes.items() if slot == s) for s in SLOTS}
        answer["methods"][m] = dict(**tally, per_slot=slots, equal_slot_concordance=mean(slots.values()),
                                      per_trajectory=[dict(slot=s, episode=e, concordance=mean(v)) for (s, e), v in sorted(episodes.items())])
    return answer


def gates(rows, pairs):
    common = [r for r in rows if top(r["y"]) is not None and top(r["z"]) is not None and
              all(top(r["predictions"][m]) is not None for m in GATE)]
    ms = {m: metrics(common, m) for m in GATE}
    differences = {m: dict(extra_detection=ms[CAND]["wrong_detected"] - ms[m]["wrong_detected"],
                           extra_false_positive=ms[CAND]["false_positive"] - ms[m]["false_positive"]) for m in COMPARATORS}
    classification = bool(common) and all(v["extra_detection"] > 0 and v["extra_false_positive"] <= 0 for v in differences.values())
    rank = ranking(pairs, GATE)
    winning_slots = []
    for slot in SLOTS:
        values = {m: rank["methods"][m]["per_slot"][slot] for m in GATE}
        if all(v is not None for v in values.values()) and all(values[CAND] > values[m] for m in COMPARATORS):
            winning_slots.append(slot)
    rank_pass = len(winning_slots) >= 2
    return dict(common_classification_coverage=scope(common), classification_metrics=ms, comparisons=differences,
                classification_pass=classification, ranking=rank, winning_slots=winning_slots, ranking_pass=rank_pass,
                decision=("candidate_for_independent_confirmation" if classification and rank_pass else
                          "ranking_only_development" if rank_pass else "development_only_gate_not_passed"))


def residual_rows(rows, offsets, pairs):
    pair_failures = set()
    for pair in pairs:
        if pair["discordant"] and pair["verdict"][CAND] in ("loss", "tie"):
            pair_failures.update((pair["slot"], pair["episode"], rep) for rep in pair["reps"])
    results, max_error = [], 0.
    for row in rows:
        offset = offsets[(row["slot"], row["rep"])]
        if offset is None:
            continue
        diag = row["full_diagnostic"]
        cross, shift = diag["residual_cross"], diag["residual_energy_shift"]
        pred = row["predictions"][CAND]
        for j in AXES:
            error = abs((row["z"][j] - pred[j]) - ((cross[j] - offset[j]) + shift[j]))
            max_error = max(max_error, error)
            assert error <= 1e-10, (key(row), j, error)
        selected, physical, predicted = top(row["y"]), top(row["z"]), top(pred)
        reasons = []
        if selected is not None and physical is not None and selected != physical:
            reasons.append("original_fine_error")
        if None not in (selected, physical, predicted) and (selected == physical) != (selected == predicted):
            reasons.append("missed_wrong" if selected != physical else "false_positive")
        if key(row) in pair_failures:
            reasons.append("candidate_pair_loss_or_tie")
        if not reasons:
            continue
        comparisons = []
        for a in selected or ():
            for b in AXES:
                if b in selected:
                    continue
                comparisons.append(dict(selected=a, unselected=b,
                    true_gap=row["z"][b] - row["z"][a],
                    old_current_gap=row["predictions"]["b_current"][b] - row["predictions"]["b_current"][a],
                    corrected_current_gap=pred[b] - pred[a],
                    train_cross_gap=offset[b] - offset[a], actual_cross_gap=cross[b] - cross[a],
                    remaining_cross_drift=(cross[b] - offset[b]) - (cross[a] - offset[a]),
                    residual_energy_shift=shift[b] - shift[a]))
        results.append(dict(slot=row["slot"], episode=row["episode"], rep=row["rep"], reasons=reasons,
                            fine_set=selected, physical_set=physical, predicted_physical_set=predicted,
                            true_margin=margin(row["z"], selected), old_current_margin=margin(row["predictions"]["b_current"], selected),
                            corrected_current_margin=margin(pred, selected), train_cross_mean=offset, cross_boundary_gaps=comparisons))
    return dict(max_identity_error=max_error, failure_or_original_error_rows=results)


def score():
    lock = json.loads((HERE / "METHOD_LOCK.json").read_text())
    assert lock["script_sha256"] == digest(Path(__file__))
    assert lock["protocol_sha256"] == digest(HERE / "PROTOCOL.md")
    sources = {}
    for name in INPUTS:
        assert digest(SOURCE / name) == lock["input_sha256"][name]
        sources[name] = json.loads((SOURCE / name).read_text())
    offsets, rows = {}, []
    for slot in SLOTS:
        frozen, revealed = sources[f"freeze_{slot}.json"], sources[f"reveal_{slot}.json"]
        assert frozen["slot"] == revealed["slot"] == slot
        assert not frozen["pilot"] and not revealed["pilot"]
        assert revealed["forecast_sha"] == lock["input_sha256"][f"freeze_{slot}.json"]
        for parent in frozen["parents"]:
            cal = parent["cal"]
            if cal["available"]:
                u = cal["train_residual_cross"]
                assert len(u) == 3 and all(finite(v) for v in u)
                offsets[(slot, parent["rep"])] = [mean(u[h][j] for h in range(3)) for j in AXES]
            else:
                offsets[(slot, parent["rep"])] = None
        for row in revealed["rows"]:
            assert row["slot"] == slot
            u = offsets[(slot, row["rep"])]
            for base, new in (("b_current", CAND), ("b_fixed", FIXED)):
                original = row["predictions"][base]
                row["predictions"][new] = None if u is None or original is None else [original[j] + u[j] for j in AXES]
            rows.append(row)
    rows.sort(key=key)
    assert len({key(r) for r in rows}) == len(rows)
    pairs = matched(rows)
    per_slot = {s: {m: metrics([r for r in rows if r["slot"] == s], m) for m in METHODS} for s in SLOTS}
    out = dict(method_lock_sha256=digest(HERE / "METHOD_LOCK.json"), scored_utc=datetime.now(timezone.utc).isoformat(),
               evidence="post-hoc development only", scope=scope(rows), methods=list(METHODS),
               offsets=[dict(slot=s, rep=r, train_cross_mean=u) for (s, r), u in sorted(offsets.items())],
               micro={m: metrics(rows, m) for m in METHODS}, per_slot=per_slot,
               macro_gap_mae={m: mean(per_slot[s][m]["gap_mae"] for s in SLOTS) for m in METHODS},
               matched=dict(train=ranking(pairs), train_and_eval=ranking([p for p in pairs if p["strict"]])),
               gates=dict(train=gates(rows, pairs), train_and_eval=gates(rows, [p for p in pairs if p["strict"]])),
               matched_pairs=pairs, residual_diagnosis=residual_rows(rows, offsets, pairs),
               record_predictions=[dict(slot=r["slot"], rep=r["rep"], episode=r["episode"],
                                        actual_fine_correct=top(r["y"]) == top(r["z"]),
                                        corrected_fixed=r["predictions"][FIXED], corrected_current=r["predictions"][CAND]) for r in rows],
               resources=dict(source_json_files=len(INPUTS), new_samples=0, fits=0, model_forwards=0, array_reads=0, test_reads=0, gpu_jobs=0))
    dump("RESULTS.json", out)
    print(json.dumps(dict(scope=out["scope"], decision=out["gates"]["train"]["decision"],
                         methods={m: dict(detected=out["micro"][m]["wrong_detected"], false_positive=out["micro"][m]["false_positive"],
                                          mae=out["macro_gap_mae"][m], ranking=out["matched"]["train"]["methods"][m])
                                  for m in ("b_current", FIXED, CAND)}), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "score"))
    args = parser.parse_args()
    freeze() if args.phase == "freeze" else score()
