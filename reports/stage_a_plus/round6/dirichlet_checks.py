"""Fixed analytic identities and existing JSON-only budget diagnostics. No samples/models."""
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
PREV = OUT.parent / 'round5'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def interval(score, cap, eta):
    alpha = math.asin(math.sqrt(score / cap))
    theta = 2 * math.asin(eta / 2)
    return (cap * math.sin(max(0., alpha-theta))**2,
            cap * math.sin(min(math.pi/2, alpha+theta))**2)

def qeta(eta):
    return math.sqrt(eta*eta*(1-eta*eta/4))

def dot(x, y):
    return sum(a*b for a,b in zip(x,y))

fixed = []
for eigs in [(0.,4.),(.5,.524),(.1,1.2)]:
    cap = max(eigs)
    for angle in [0., .2, .7, 1.4]:
        for change in [-.3, 0., .03, .3]:
            y=(math.cos(angle), math.sin(angle))
            z=(math.cos(angle+change), math.sin(angle+change))
            d=tuple(b-a for a,b in zip(y,z))
            ly=tuple(a*b for a,b in zip(eigs,y))
            sy=dot(y,ly); sz=sum(a*b*b for a,b in zip(eigs,z))
            residual=tuple(a-sy*b for a,b in zip(ly,y))
            rho=math.sqrt(dot(residual,residual)); eta=math.sqrt(dot(d,d)); q=qeta(eta)
            identity=2*dot(residual,d)+sum((a-sy)*b*b for a,b in zip(eigs,d))
            lo=sy-2*rho*q-sy*q*q; hi=sy+2*rho*q+(cap-sy)*q*q
            flo,fhi=interval(sy,cap,eta)
            passed=(abs((sz-sy)-identity)<1e-12 and lo-1e-12<=sz<=hi+1e-12
                    and flo-1e-12<=sz<=fhi+1e-12)
            assert passed
            fixed.append(dict(eigenvalues=eigs,angle=angle,change=change,identity_error=abs((sz-sy)-identity),all_bounds_pass=passed))

horizons=(4,16,32); ar_rates=(.985,.984,.9,.8); angle=.03; cap=4.
physical=[2*(1-sum(r**h for h in horizons)/len(horizons)) for r in ar_rates]
c=math.cos(angle); s=math.sin(angle)
fine=[c*c*physical[0]+s*s*physical[1],s*s*physical[0]+c*c*physical[1],*physical[2:]]
eta=2*math.sin(angle/2); rho=(physical[1]-physical[0])*s*c; q=qeta(eta)
gap=fine[1]-fine[0]
old_beta=lambda value:2*math.sqrt(cap*value)*eta+cap*eta*eta
old_lower=gap-old_beta(fine[0])-old_beta(fine[1])
new_lower=gap-4*rho*q-(cap-fine[0]+fine[1])*q*q
angle_lower=interval(fine[1],cap,eta)[0]-interval(fine[0],cap,eta)[1]
assert new_lower<=physical[1]-physical[0]+1e-12
example=dict(scope='one analytically specified stationary Gaussian AR process; selection-interface only',
    rates=ar_rates,horizons=horizons,rotation_radians=angle,physical_scores=physical,fine_scores=fine,
    physical_pair_gap=physical[1]-physical[0],fine_pair_gap=gap,eta=eta,rho=rho,
    old_norm_pair_lower=old_lower,normalized_geometry_pair_lower=angle_lower,
    observable_dynamic_residual_pair_lower=new_lower,
    old_norm_pass=old_lower>0,normalized_geometry_pass=angle_lower>0,dynamic_residual_pass=new_lower>0,
    posterior_or_actual_Fine_certificate=False)

paths=[PREV/'G0_ERROR_BUDGET_AUDIT.json',PREV/'ORACLE_GATE_FLOOR_AUDIT.json']
data=json.loads(paths[0].read_text()); floor_data=json.loads(paths[1].read_text())
floors={(r['slot'],r['rep']):r for r in floor_data['rows']}
rows=[]
for parent in data['rows']:
    key=(parent['slot'],parent['rep']); floor=floors[key]
    speeds=parent['speed_FP64']; cap=parent['finite_Q0_Qh_coverage_K']
    per_axis=floor['axis_specific_floor_by_native_axis']; pairs=[]
    for pair in parent['all_four_pairs']:
        a=pair['selected_native_axis']; b=pair['unselected_native_axis']; gap=speeds[b]-speeds[a]
        aa=math.asin(math.sqrt(speeds[a]/cap)); ab=math.asin(math.sqrt(speeds[b]/cap))
        threshold=2*math.sin((ab-aa)/4)
        interval_gap=interval(speeds[b],cap,per_axis[b])[0]-interval(speeds[a],cap,per_axis[a])[1]
        qa=qeta(per_axis[a]); qb=qeta(per_axis[b])
        optimistic_r0_gap=gap-(cap-speeds[a])*qa*qa-speeds[b]*qb*qb
        rho_cap=optimistic_r0_gap/(2*(qa+qb)) if qa+qb>0 else None
        pairs.append(dict(a=a,b=b,fine_gap=gap,normalized_common_eta_threshold=threshold,
            normalized_geometry_lower_at_numeric_floors=interval_gap,
            normalized_geometry_gate_numeric_veto=interval_gap<=0,
            optimistic_dynamic_r0_lower_at_numeric_floors=optimistic_r0_gap,
            optimistic_strict_common_rho_ceiling_at_floors=rho_cap))
    common=min(p['normalized_common_eta_threshold'] for p in pairs)
    rows.append(dict(slot=key[0],rep=key[1],old_common_eta_threshold=parent['strict_common_eta_threshold'],
        normalized_common_eta_threshold=common,
        common_gate_numeric_veto=floor['common_axis_RMS_error_lower_bound']>=common,
        axis_specific_gate_numeric_veto=any(p['normalized_geometry_gate_numeric_veto'] for p in pairs),
        optimistic_dynamic_r0_all_pairs_positive=all(p['optimistic_dynamic_r0_lower_at_numeric_floors']>0 for p in pairs),
        optimistic_strict_common_rho_ceiling_at_floors=min(p['optimistic_strict_common_rho_ceiling_at_floors'] for p in pairs),
        pairs=pairs,actual_eta_upper_bound_available=False,actual_rho_upper_bound_available=False,
        actual_Fine_certificate='NOT_ESTABLISHED'))

summary_by_slot={}
for slot in sorted({r['slot'] for r in rows}):
    group=[r for r in rows if r['slot']==slot]
    summary_by_slot[slot]=dict(
        normalized_eta_threshold_range=[min(r['normalized_common_eta_threshold'] for r in group),max(r['normalized_common_eta_threshold'] for r in group)],
        common_gate_numeric_veto_count=sum(r['common_gate_numeric_veto'] for r in group),
        axis_specific_gate_numeric_veto_count=sum(r['axis_specific_gate_numeric_veto'] for r in group),
        optimistic_rho_ceiling_range=[min(r['optimistic_strict_common_rho_ceiling_at_floors'] for r in group),max(r['optimistic_strict_common_rho_ceiling_at_floors'] for r in group)])

result=dict(scope='fixed formulas and old summary-only thresholds; numeric floors are not certified upper bounds',
    scope_sha256=sha(OUT/'DIRICHLET_SCOPE.md'),script_sha256=sha(__file__),
    inputs=[dict(path=str(f),sha256=sha(f)) for f in paths],
    fixed_matrix_checks=fixed,analytic_AR_selection_example=example,
    existing_parents=rows,summary_by_slot=summary_by_slot,
    all_formula_checks_pass=all(r['all_bounds_pass'] for r in fixed),
    cost=dict(existing_JSON_inputs=2,scientific_array_reads=0,model_execution=0,new_fits=0,PRNG=0,new_physical_samples=0,server_connections=0,GPU_jobs=0))
(OUT/'DIRICHLET_CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(formula_cases=len(fixed),example=example,summary_by_slot=summary_by_slot),ensure_ascii=False,indent=2))
