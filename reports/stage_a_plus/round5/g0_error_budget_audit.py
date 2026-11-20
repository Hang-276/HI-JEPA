"""Read stored G0 metadata only; invert a conditional finite-support bound."""
from pathlib import Path
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2/execution/jobs/representation/OursFine4'
SLOTS=['ar_rich_fresh1','ar_rich_fresh2','fhn_fresh_eps008','fhn_fresh_eps05']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[];inputs=[]
for slot in SLOTS:
    for rep in range(3):
        path=BASE/slot/f'seed{rep}'/'G0.json'
        data=json.loads(path.read_text());inputs.append({'path':str(path),'sha256':sha(path)})
        assert data['physical_labels_used'] is False
        speed=data['speed_FP64'];selected=data['selected_pair'];other=[j for j in range(4)if j not in selected]
        length=data['TRAIN_shape'][1];horizons=data['horizons']
        assert horizons==[4,16,32]
        K=sum(4*length/(length-h)for h in horizons)/len(horizons)
        pairs=[]
        for a in selected:
            for b in other:
                gap=speed[b]-speed[a];assert gap>0
                linear=2*math.sqrt(K)*(math.sqrt(speed[a])+math.sqrt(speed[b]))
                quadratic=2*K
                eta=2*gap/(linear+math.sqrt(linear*linear+4*quadratic*gap))
                assert abs(linear*eta+quadratic*eta*eta-gap)<1e-12
                assert linear*(.99*eta)+quadratic*(.99*eta)**2<gap
                pairs.append({'selected_native_axis':a,'unselected_native_axis':b,
                              'FP64_finite_gap':gap,'linear_eta_coefficient':linear,
                              'quadratic_eta_coefficient':quadratic,
                              'strict_eta_threshold':eta})
        rows.append({'slot':slot,'rep':rep,'TRAIN_shape':data['TRAIN_shape'],
                     'native_selected_axes':selected,'physical_axis_mapping_used':False,
                     'speed_FP64':speed,'finite_Q0_Qh_coverage_K':K,
                     'all_four_pairs':pairs,'strict_common_eta_threshold':min(p['strict_eta_threshold']for p in pairs),
                     'minimum_FP64_gap':min(p['FP64_finite_gap']for p in pairs),
                     'FP32_rounding_max':max(abs(x-y)for x,y in zip(speed,data['speed_FP32'])),
                     'actual_representation_error_bound_available':False,
                     'actual_certificate_status':'UNESTABLISHED; threshold inversion only'})
result={'scope':'known finite TRAIN scores; conditional same-support normalized-axis-error budget; no physical truth or actual error estimated',
        'scope_sha256':sha(OUT/'FEASIBILITY_SCOPE.md'),'script_sha256':sha(__file__),
        'rows':rows,'inputs':inputs,
        'summary_by_slot':{slot:{'eta_threshold_range':[min(r['strict_common_eta_threshold']for r in rows if r['slot']==slot),max(r['strict_common_eta_threshold']for r in rows if r['slot']==slot)],
                                  'gap_range':[min(r['minimum_FP64_gap']for r in rows if r['slot']==slot),max(r['minimum_FP64_gap']for r in rows if r['slot']==slot)]}for slot in SLOTS},
        'independent_episode_witness_design_only':{str(n):{'delta':.05,'M':1,'K':9,'simultaneous_absolute_mean_error':math.sqrt(2*math.log(2*9/.05)/n)}for n in (24,32)},
        'cost':{'metadata_JSON_read':12,'scientific_array_reads':0,'model_load_or_forward':0,'fitting':0,'PRNG':0,'training':0,'server_GPU':0}}
dest=OUT/'G0_ERROR_BUDGET_AUDIT.json'
if dest.exists():raise SystemExit('Preserve completed result; refusing overwrite.')
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result['summary_by_slot'],ensure_ascii=False,indent=2))
