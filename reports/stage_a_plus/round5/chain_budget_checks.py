"""Compose saved analytic bounds and fixed score moments; not model validation."""
from pathlib import Path
import json,math,hashlib

OUT=Path(__file__).resolve().parent
source=OUT/'TAIL_POSTERIOR_CHECKS.json'
tail=json.loads(source.read_text())
C=128+192/math.e**2
scores=[.3,.5,1.4,1.8]
rows=[]
for case in tail['analytic_Gaussian_absorption_cases']:
    delta=case['statistic_span_L2_bound']
    L=1-delta*delta-24*delta
    E=delta*delta+9*C*delta*delta/L if L>0 else None
    ok=E is not None and E<1 and 2*E<1
    eta=math.sqrt(2*E/(1+math.sqrt(1-E))) if ok else None
    # Same-Q0-as-anchor, covariance I and K=4 are explicitly hypothetical here.
    # The score moments are separate fixed fixtures, not the Gaussian case's
    # measured dynamics or any stored Fine's scores.
    betas=[4*math.sqrt(s)*eta+4*eta*eta for s in scores] if ok else None
    margin=min(scores[b]-scores[a]-betas[a]-betas[b]for a in (0,1)for b in (2,3))if ok else None
    rows.append({'cube_radius':case['cube_radius'],'known_generative_joint_KL_bound':case['known_generative_joint_KL_upper_for_posterior_KL'],
                 'span_delta_bound':delta,'closure_L':L,'closure_E':E,
                 'hypothetical_anchor_covariance_lower_bound':1.,'closure_gate_pass':ok,
                 'same_anchor_normalized_axis_error_bound':eta,
                 'separate_fixed_score_moments':scores,'stationary_endpoint_K_assumed':4,
                 'score_gap_after_error_budget':margin,'conditional_numeric_chain_nonempty':margin is not None and margin>0,
                 'is_current_Fine_or_complete_joint_process_validation':False})
assert [r['conditional_numeric_chain_nonempty']for r in rows]==[False,False,True]
result={'scope':'analytic tail examples followed by conditional Gaussian closure and independent fixed score-moment fixtures; no actual Fine claim',
        'source_path':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rows':rows,'all_checks_pass':True,
        'limitations':['A supplied by known analytic generative parameters, not an actual estimated posterior excess',
                       'Q0 equals anchor and K=4 only in this budget illustration',
                       'Score moments not asserted to arise from the likelihood example or any real data',
                       'Nonempty arithmetic does not establish network learning, current inputs or novelty'],
        'cost':{'scientific_array_reads':0,'model_execution':0,'new_physical_samples':0,'PRNG':0,'training_GPU_server':0}}
dest=OUT/'CHAIN_BUDGET_CHECKS.json'
if dest.exists():raise SystemExit('Preserve completed output.')
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(rows,ensure_ascii=False,indent=2))
