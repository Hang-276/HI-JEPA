"""Score locked predictions against existing CAL labels, using zero-based axes."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
AXES = range(4)
AXIS_PAIRS = list(combinations(AXES, 2))
FORECASTS = ('size', 'direction', 'identity', 'persistence')
ORACLE = ('same_support_energy', 'same_support_signed')
METHODS = FORECASTS + ORACLE
MATCH_LIMIT = .01
MARGIN_TIE = 1e-12


def gap(s, a, b):
    return s[b] - s[a]


def sign(x):
    return (x > 0) - (x < 0)


def avg(values):
    values = [x for x in values if x is not None]
    return statistics.fmean(values) if values else None


def divide(a, b):
    return a/b if b else None


def top(s):
    ids = sorted(AXES, key=lambda j: s[j])
    if s[ids[2]] <= s[ids[1]]:
        return None
    return tuple(sorted(ids[:2]))


def margin(s, selected):
    if selected is None:
        return None
    return min(s[i] for i in AXES if i not in selected) - max(s[i] for i in selected)


def vector(row, method):
    return row['predictions'][method] if method in FORECASTS else row[method]


def gridpair(a, b):
    return f'{b}-{a}'


def criteria(c):
    tpr = divide(c['TP'], c['TP'] + c['FN'])
    tnr = divide(c['TN'], c['TN'] + c['FP'])
    n = sum(c[k] for k in ('TP', 'TN', 'FP', 'FN'))
    return dict(correct_parent_recall=tpr, wrong_parent_detection=tnr,
                wrong_flag_false_alarm=divide(c['FN'], c['TP'] + c['FN']),
                correctness_accuracy=divide(c['TP']+c['TN'], n),
                balanced_accuracy=(tpr+tnr)/2 if None not in (tpr, tnr) else None,
                n_classified=n)


def method_stats(rows, method):
    counts = Counter(dict.fromkeys(('TP', 'TN', 'FP', 'FN', 'outcome_tie',
                                    'predicted_set_tie', 'unscored_correctness'), 0))
    abs_errors = {gridpair(a,b): [] for a,b in AXIS_PAIRS}
    direction_changes = Counter()
    top_correct = top_evaluable = 0
    for row in rows:
        p = vector(row, method)
        z, y = row['z'], row['y']
        z_set, y_set, p_set = top(z), top(y), top(p)
        if z_set is not None and p_set is not None:
            top_evaluable += 1
            top_correct += int(z_set == p_set)
        if y_set is None or z_set is None:
            counts['outcome_tie'] += 1
        if p_set is None:
            counts['predicted_set_tie'] += 1
        if None in (y_set, z_set, p_set):
            counts['unscored_correctness'] += 1
        else:
            actual, forecast = y_set == z_set, y_set == p_set
            key = ('TP' if forecast else 'FN') if actual else ('FP' if forecast else 'TN')
            counts[key] += 1
        for a,b in AXIS_PAIRS:
            predicted = gap(p,a,b)
            realized = gap(z,a,b)
            original = gap(y,a,b)
            abs_errors[gridpair(a,b)].append(abs(predicted-realized))
            if 0 in (predicted, realized, original):
                direction_changes['exact_tie'] += 1
            elif sign(predicted) != sign(original):
                key = 'corrected_original_order' if sign(predicted) == sign(realized) else 'spoiled_original_order'
                direction_changes[key] += 1
            else:
                direction_changes['retained_original_order'] += 1
    errors = {k:avg(v) for k,v in abs_errors.items()}
    return dict(method=method,n_rows=len(rows),mae_all_six_gaps=avg(errors.values()),mae_by_gap=errors,
                physical_top2_correct=top_correct,physical_top2_evaluable=top_evaluable,
                physical_top2_ties=len(rows)-top_evaluable,physical_top2_accuracy=divide(top_correct,top_evaluable),
                correctness_confusion=dict(counts),**criteria(counts),
                pair_orders_relative_to_fine=dict(direction_changes))


def row_states(rows):
    counts = Counter()
    episodes = defaultdict(set)
    for row in rows:
        y,z,t = top(row['y']),top(row['z']),tuple(sorted(row['target']))
        if z is None or y is None:
            state = 'fine_or_physical_tie'
        elif z == t:
            state = 'physical_equals_target__fine_correct' if y==z else 'physical_equals_target__fine_missed'
        elif y == z:
            state = 'physical_changed__fine_tracked_physical'
        elif y == t:
            state = 'physical_changed__fine_kept_target'
        else:
            state = 'physical_changed__fine_selected_third_set'
        counts[state] += 1
        episodes[(row['slot'],row['episode'])].add('physical_tie' if z is None else ('aligned' if z==t else 'mismatch'))
    episode_counts = Counter(next(iter(v)) if len(v)==1 else 'different_across_parents' for v in episodes.values())
    return dict(parent_episode_states=dict(counts),unique_physical_episode_target_status=dict(episode_counts),
                physical_target_mismatch_rows=sum(top(r['z']) is not None and top(r['z'])!=tuple(sorted(r['target'])) for r in rows))


def macro(slot_stats, method):
    group = [s[method] for s in slot_stats.values()]
    keys = ('mae_all_six_gaps','physical_top2_accuracy','correctness_accuracy',
            'correct_parent_recall','wrong_parent_detection','wrong_flag_false_alarm','balanced_accuracy')
    result = {key:avg(r[key] for r in group) for key in keys}
    result['n_slots_defined'] = {key:sum(r[key] is not None for r in group) for key in keys}
    result['mae_by_gap'] = {gridpair(a,b):avg(r['mae_by_gap'][gridpair(a,b)] for r in group) for a,b in AXIS_PAIRS}
    return result


def match_rows(rows):
    by_episode = defaultdict(list)
    for r in rows:
        by_episode[(r['slot'],r['episode'])].append(r)
    pairs = []
    for (slot,ep), family in sorted(by_episode.items()):
        if len({r['rep'] for r in family}) != len(family):
            raise ValueError(f'Duplicate parent for {slot} episode {ep}')
        for l,r in combinations(sorted(family,key=lambda item:item['rep']),2):
            train_dist=max(abs(a-b) for a,b in zip(l['train_r2'],r['train_r2']))
            eval_dist=max(abs(a-b) for a,b in zip(l['eval_r2'],r['eval_r2']))
            if train_dist>MATCH_LIMIT:
                continue
            yl,yr,zl,zr=top(l['y']),top(r['y']),top(l['z']),top(r['z'])
            discordant = None not in (yl,yr,zl,zr) and zl==zr and (yl==zl)!=(yr==zr)
            margins = {m:[margin(vector(l,m),yl),margin(vector(r,m),yr)] for m in METHODS}
            verdict = {}
            good = None
            if discordant:
                good = 0 if yl==zl else 1
                for m,(ml,mr) in margins.items():
                    dif = ml-mr if good==0 else mr-ml
                    verdict[m]='tie' if abs(dif)<=MARGIN_TIE else ('win' if dif>0 else 'loss')
            pairs.append(dict(slot=slot,episode=ep,rep=[l['rep'],r['rep']],train_r2_maxdiff=train_dist,
                eval_r2_maxdiff=eval_dist,strict_also_eval=eval_dist<=MATCH_LIMIT,
                train_r2=[l['train_r2'],r['train_r2']],eval_r2=[l['eval_r2'],r['eval_r2']],
                fine_sets=[yl,yr],physical_sets=[zl,zr],
                physical_set_differs_between_parents=zl!=zr,has_exact_top2_tie=None in (yl,yr,zl,zr),
                discordant=discordant,correct_rep=([l['rep'],r['rep']][good] if good is not None else None),
                predicted_physical_margins_for_fine_set=margins,concordance=verdict))
    return pairs


def comparison_stats(pairs):
    discord=[p for p in pairs if p['discordant']]
    def unique(values):
        return dict(slots=len({p['slot'] for p in values}),
                    episodes=len({(p['slot'],p['episode']) for p in values}),pairs=len(values))
    results={}
    for method in METHODS:
        units=defaultdict(list)
        tally=Counter({'win':0,'tie':0,'loss':0})
        for p in discord:
            key=p['concordance'][method]
            tally[key]+=1
            units[(p['slot'],p['episode'])].append({'win':1.,'tie':.5,'loss':0.}[key])
        by_slot=defaultdict(list)
        for (slot,_),values in units.items():
            by_slot[slot].append(avg(values))
        totals={s:avg(vals) for s,vals in by_slot.items()}
        results[method]=dict(**tally,micro_concordance=divide(tally['win']+.5*tally['tie'],len(discord)),
                             per_slot_episode_weighted=totals,
                             equal_slot_episode_pair_concordance=avg(totals.values()))
    return dict(all_eligible=unique(pairs),discordant=unique(discord),
                exact_tie_pairs=sum(p['has_exact_top2_tie'] for p in pairs),
                physical_set_disagreement_pairs=sum(p['physical_set_differs_between_parents'] for p in pairs),
                methods=results)


def posthoc_changes(rows):
    changed=[]
    max_axis_error=0.
    tally=Counter()
    for r in rows:
        energy,signed,z,y=r['same_support_energy'],r['same_support_signed'],r['z'],r['y']
        max_axis_error=max(max_axis_error,max(abs(a-b) for a,b in zip(signed,z)))
        changed_pairs=[]
        for a,b in AXIS_PAIRS:
            if sign(gap(energy,a,b))!=sign(gap(signed,a,b)):
                changed_pairs.append([a,b])
        before,after=top(energy),top(signed)
        tally['rows_with_any_changed_pair']+=int(bool(changed_pairs))
        tally['six_pair_orders_changed']+=len(changed_pairs)
        tally['rows_with_changed_top2']+=int(before!=after)
        if before!=after or changed_pairs:
            changed.append(dict(slot=r['slot'],rep=r['rep'],episode=r['episode'],
                                pair_axis_indices=changed_pairs,energy_top2=before,signed_top2=after,
                                physical_top2=top(z),fine_top2=top(y),eval_r2=r['eval_r2'],
                                energy_score=energy,signed_score=signed))
    return dict(max_exact_reconstruction_abs_error=max_axis_error,total_rows=len(rows),counts=dict(tally),rows=changed)


def validate(rows):
    if not rows:raise ValueError('No outcome rows')
    for r in rows:
        vectors=[r['y'],r['z'],r['train_r2'],r['eval_r2']]+[vector(r,m) for m in METHODS]
        if any(len(v)!=4 or not all(math.isfinite(x) for x in v) for v in vectors):
            raise ValueError(f'Invalid score or R2 vector at {(r["slot"],r["rep"],r["episode"])}')
        if len(set(r['target']))!=2 or not set(r['target']).issubset(AXES):
            raise ValueError('Invalid target')


def main():
    infile=HERE/'results_rows.json'
    raw=json.loads(infile.read_text())
    rows=raw if isinstance(raw,list) else raw['rows']
    validate(rows)
    by_slot=defaultdict(list)
    for r in rows:by_slot[r['slot']].append(r)
    slot_stats={s:{m:method_stats(vals,m) for m in METHODS} for s,vals in sorted(by_slot.items())}
    pairs=match_rows(rows)
    matched={}
    for tag,values in [('train',pairs),('train_and_eval',[p for p in pairs if p['strict_also_eval']])]:
        matched[tag]=dict(overall=comparison_stats(values),by_slot={
            slot:comparison_stats([p for p in values if p['slot']==slot]) for slot in sorted(by_slot)})
    totals={m:method_stats(rows,m) for m in METHODS}
    macros={m:macro(slot_stats,m) for m in METHODS}
    out=dict(input_sha256=hashlib.sha256(infile.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
        input_rows=len(rows),physical_episodes=len({(r['slot'],r['episode']) for r in rows}),slots=sorted(by_slot),
        all_same_episode_seed_pairs=sum(len(f)*(len(f)-1)//2 for f in (
            [r for r in rows if (r['slot'],r['episode'])==k] for k in sorted({(r['slot'],r['episode']) for r in rows}))),
        pair_gap_orientation='higher index score minus lower index score; six pairs have equal weight',
        prescore_label_methods=list(FORECASTS),same_support_oracle_methods=list(ORACLE),
        correctness_positive_class='Fine top2 equals current physical top2',
        matched_margin_tie_abs=MARGIN_TIE,top2_tie_rule='Exact equality at boundary; tied cases abstain',
        class_macro_policy='Average defined slot rates and disclose the number defined; undefined BA when either label is absent',
        micro=totals,macro=macros,per_slot=slot_stats,matched=matched,matched_pairs=pairs,
        target_alignment=dict(micro=row_states(rows),per_slot={s:row_states(vals) for s,vals in sorted(by_slot.items())}),
        same_support_signed_attribution=posthoc_changes(rows))
    dest=HERE/'SUMMARY.json'
    temp=dest.with_suffix('.tmp')
    temp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    temp.replace(dest)
    print(json.dumps({'rows':len(rows),'episodes':out['physical_episodes'],
                      'macro_mae':{m:macros[m]['mae_all_six_gaps'] for m in FORECASTS},
                      'strict_discordant':matched['train_and_eval']['overall']['discordant']}))


if __name__=='__main__':
    main()
