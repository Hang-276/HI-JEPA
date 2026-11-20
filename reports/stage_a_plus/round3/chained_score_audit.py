"""Join completed affine and score-remainder audits; no array reads or fitting."""
from pathlib import Path
import hashlib,json,time

OUT=Path(__file__).resolve().parent
GEO=OUT/'AFFINE_GEOMETRY_AUDIT.json'
REM=OUT.parent/'round2/NONLINEAR_REMAINDER_AUDIT.json'
SCOPE=OUT/'CHAINED_SCORE_SCOPE.md'
TOL=1e-12

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert SCOPE.exists();scope_sha=sha(SCOPE);began=time.perf_counter()
g=json.loads(GEO.read_text());r=json.loads(REM.read_text())
assert any(a['path']==str(REM.resolve())and a['sha256']==sha(REM)for a in g['accesses'])
assert g['summary']['parents_total']==12 and len(r['rows'])==12
scenarios=[];compat_fail=[]
for gp in g['rows']:
    slot=gp['slot'];rep=gp['rep']
    rp=next(p for p in r['rows']if p['slot']==slot and p['rep']==rep)
    target=next(s['target']for s in g['slots']if s['slot']==slot)
    assert target==rp['target']and gp['original_A']==rp['OLS']['A']
    assert gp['original_Q']==rp['polar']['Q']
    for split in ('train','cal'):
        ge=gp['evaluations'][split];re=rp['evaluation'][split]['transitions']['affine_to_actual']
        assert ge['row_scale_score_max_difference']<=TOL
        assert rp['evaluation'][split]['Q0']=='TRAIN fixed for every denominator'
        assert len(ge['pairs_all4'])==len(re['pairs'])==4
        pairs=[]
        for p in ge['pairs_all4']:
            a=p['target_a'];b=p['other_b']
            pr=next(pp for pp in re['pairs']if pp['a']==a and pp['b']==b)
            gapdiff=abs(p['affine_core_gap']-pr['core_gap'])
            if gapdiff>TOL or p['finite_affine_gap_lower_bound_if_admissible']is None:
                rr={'target_a':a,'other_b':b,'status':'N/A_incompatible_saved_core_or_no_affine_bound',
                    'saved_core_gap_difference':gapdiff}
                pairs.append(rr);compat_fail.append({'slot':slot,'rep':rep,'split':split,**rr});continue
            lower=p['finite_affine_gap_lower_bound_if_admissible']
            cs=pr['CS_bound_measured_Q0_correction'];corr=pr['truth_assisted_pair_correlation_triangle_bound']
            lcs=lower-cs;lcorr=lower-corr;actual=pr['output_gap']
            assert actual+TOL>=lcs and actual+TOL>=lcorr
            assert abs(actual-pr['core_gap']-pr['signed_exact_pair_discrepancy'])<TOL
            pairs.append({'target_a':a,'other_b':b,'status':'compatible',
                'physical_gap':p['physical_gap'],'affine_gap_both_sources':pr['core_gap'],
                'saved_core_gap_difference':gapdiff,'affine_lower_bound':lower,
                'A_to_actual_CS_measured_Q0':cs,'A_to_actual_truth_correlation_triangle':corr,
                'chained_actual_lower_CS':lcs,'chained_actual_lower_corr':lcorr,
                'exact_actual_gap_saved':actual,
                'positive_CS_lower':lcs>0,'positive_corr_lower':lcorr>0,
                'exact_actual_gap_positive':actual>0})
        compatible=all(p['status']=='compatible'for p in pairs)
        scenarios.append({'slot':slot,'rep':rep,'split':split,'target':target,'pairs_all4':pairs,
            'compatible':compatible,
            'core_radius_certification_kind':{
                'uniform_radius_r_ball':gp['domains']['uniform_radius_r_ball_condition'],
                'pointwise_min_kappa':gp['domains']['pointwise_min_kappa_condition']},
            'all4_positive_affine_lower':compatible and all(p['affine_lower_bound']>0 for p in pairs),
            'all4_positive_chained_CS':compatible and all(p['positive_CS_lower']for p in pairs),
            'all4_positive_chained_corr':compatible and all(p['positive_corr_lower']for p in pairs),
            'all4_positive_exact_actual_gap':compatible and all(p['exact_actual_gap_positive']for p in pairs),
            'minimum_affine_lower':min(p['affine_lower_bound']for p in pairs)if compatible else None,
            'minimum_chained_CS_lower':min(p['chained_actual_lower_CS']for p in pairs)if compatible else None,
            'minimum_chained_corr_lower':min(p['chained_actual_lower_corr']for p in pairs)if compatible else None,
            'minimum_exact_actual_gap':min(p['exact_actual_gap_saved']for p in pairs)if compatible else None,
            'CAL_is_fixed_TRAIN_scale_support_diagnostic_not_deployed_G0':split=='cal'})

summary={}
for split in ('train','cal'):
    rows=[s for s in scenarios if s['split']==split]
    summary[split]={k:sum(s[k]for s in rows)for k in ['compatible','all4_positive_affine_lower',
        'all4_positive_chained_CS','all4_positive_chained_corr','all4_positive_exact_actual_gap']}
    summary[split].update({'parent_split_denominator':12,'pair_denominator':48,
        'pair_positive_CS':sum(p.get('positive_CS_lower',False)for s in rows for p in s['pairs_all4']),
        'pair_positive_corr':sum(p.get('positive_corr_lower',False)for s in rows for p in s['pairs_all4'])})
report={'scope':'truth-assisted finite-array chained score bounds; not J_actual-to-Fine population or high-probability certification',
    'scope_sha256_before_join':scope_sha,'script_sha256':sha(__file__),
    'inputs':[{'path':str(p.resolve()),'sha256':sha(p)}for p in (GEO,REM)],
    'unique_Fine_parents':12,'physical_slots':4,'parent_split_scenarios':24,'pair_rows':96,
    'summary':summary,'scenarios':scenarios,'compatibility_failures':compat_fail,
    'cost':{'new_array_reads':0,'new_fits_NN_forward_PRNG_data_FINAL_GPU':0,'elapsed_seconds':time.perf_counter()-began}}
assert len(scenarios)==24 and sum(len(s['pairs_all4'])for s in scenarios)==96 and sha(SCOPE)==scope_sha
(OUT/'CHAINED_SCORE_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'summary':summary,'compatibility_failure_count':len(compat_fail),
    'rows':[{k:s[k]for k in ['slot','rep','split','minimum_affine_lower','minimum_chained_CS_lower',
        'minimum_chained_corr_lower','minimum_exact_actual_gap']}for s in scenarios]},ensure_ascii=False,indent=2))
