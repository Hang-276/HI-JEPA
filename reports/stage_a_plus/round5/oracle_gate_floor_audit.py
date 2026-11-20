"""Existing truth-assisted OLS summary used only to refute one sufficient gate.

No new labels, arrays, fitting or model execution. This is not an unlabeled
acceptance certificate and does not reject the actual representation.
"""
from pathlib import Path
import hashlib
import json
import math

OUT=Path(__file__).resolve().parent
GATE=OUT/'G0_ERROR_BUDGET_AUDIT.json'
OLD=OUT.parent/'round2/NONLINEAR_REMAINDER_AUDIT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
gate=json.loads(GATE.read_text());old=json.loads(OLD.read_text());rows=[]
for g in gate['rows']:
    r=next(x for x in old['rows']if x['slot']==g['slot']and x['rep']==g['rep'])
    assert r['TRAIN_shape']==g['TRAIN_shape']
    assert r['deployed_G0_selected_raw']==g['native_selected_axes']
    mse=r['OLS']['TRAIN_residual_MSE_by_axis']
    n=r['TRAIN_shape'][0]*r['TRAIN_shape'][1]
    gram_min=min(r['OLS']['design_singular_values'])**2/n
    normal_cross=r['OLS']['max_static_predictor_residual_cross']
    normal_equation_correction=5*normal_cross**2/gram_min
    # Additional numerical sensitivity allowance. This is not interval arithmetic.
    sensitivity_allowance=1e-10
    corrected=[max(0.,x-normal_equation_correction-sensitivity_allowance)for x in mse]
    floor=math.sqrt(max(corrected));threshold=g['strict_common_eta_threshold']
    native_floor=[None]*4
    for aligned,native in enumerate(r['CAL_axis_mapping']['matched_indices']):
        native_floor[native]=math.sqrt(corrected[aligned])
    pair_vetoes=[];K=g['finite_Q0_Qh_coverage_K'];speeds=g['speed_FP64']
    for pair in g['all_four_pairs']:
        a=pair['selected_native_axis'];b=pair['unselected_native_axis']
        budget=sum(2*math.sqrt(K*speeds[j])*native_floor[j]+K*native_floor[j]**2 for j in (a,b))
        pair_vetoes.append({'a':a,'b':b,'minimum_possible_axis_specific_bound_from_numeric_floor':budget,
                            'finite_gap':pair['FP64_finite_gap'],'numeric_veto':budget>=pair['FP64_finite_gap']})
    rows.append({'slot':g['slot'],'rep':g['rep'],
                 'existing_truth_assisted_OLS_MSE_by_aligned_axis':mse,
                 'normal_equation_MSE_correction':normal_equation_correction,
                 'additional_MSE_sensitivity_allowance':sensitivity_allowance,
                 'corrected_MSE_floor_by_aligned_axis':corrected,
                 'is_interval_arithmetic_certificate':False,
                 'common_axis_RMS_error_lower_bound':floor,
                 'strict_common_eta_sufficient_gate':threshold,
                 'floor_over_threshold':floor/threshold,
                 'this_uniform_error_sufficient_gate_has_numeric_oracle_veto':floor>=threshold,
                 'axis_specific_floor_by_native_axis':native_floor,
                 'axis_specific_pair_vetoes':pair_vetoes,
                 'axis_specific_same_bound_has_numeric_oracle_veto':any(p['numeric_veto']for p in pair_vetoes),
                 'actual_model_failure_inferred':False})
result={'scope':'oracle lower-bound veto of the common-axis-L2 and axis-specific versions of one norm sufficient bound on the same finite TRAIN support; NOT unlabeled evidence of correctness or model failure',
        'proof':'Each normalized signed-permuted physical coordinate is in the full physical affine span. Exact OLS minimum residual MSE lower-bounds its squared discrepancy. Numerical fitted MSE is corrected by g.T Gram^-1 g <= 5*g_inf^2/lambda_min(Gram), plus an explicit 1e-10 MSE sensitivity allowance; stored inputs are FP64 and not interval certified.',
        'scope_sha256':sha(OUT/'FEASIBILITY_SCOPE.md'),'script_sha256':sha(__file__),
        'inputs':[{'path':str(p),'sha256':sha(p)}for p in (GATE,OLD)],'rows':rows,
        'summary':{'parents':12,'common_eta_gate_numeric_oracle_veto_count':sum(r['this_uniform_error_sufficient_gate_has_numeric_oracle_veto']for r in rows),
                   'axis_specific_same_bound_numeric_oracle_veto_count':sum(r['axis_specific_same_bound_has_numeric_oracle_veto']for r in rows),
                   'not_ruled_out_is_not_pass':True,'does_not_rule_out_other_sharper_axis_specific_or_signed_pair_bounds':True},
        'cost':{'existing_summary_JSON_reads':2,'new_physical_labels_or_arrays':0,'new_fits':0,'model_forward':0,'training':0,'server_GPU':0}}
dest=OUT/'ORACLE_GATE_FLOOR_AUDIT.json'
# Reviewed correction of this round's provisional summary; no prior-round file
# or original scientific result is changed. See REVIEW_FIXES.md.
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'summary':result['summary'],'rows':[{k:r[k]for k in ['slot','rep','normal_equation_MSE_correction','common_axis_RMS_error_lower_bound','strict_common_eta_sufficient_gate','floor_over_threshold','axis_specific_same_bound_has_numeric_oracle_veto']}for r in rows]},ensure_ascii=False,indent=2))
