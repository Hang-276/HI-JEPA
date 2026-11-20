"""Deterministic Stage A+2 audit of existing TRAIN scores.

Reads saved physical/Fine arrays and frozen CAL maps, without importing any
model, fitting any map, drawing randomness, or changing historical outputs.
Truth-assisted bounds here are finite-array diagnostics, not unlabeled or
population certificates. The formulas are fixed for every parent/pair/horizon.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2'
H = (4, 16, 32)
U32 = 2.**-24
ACCESSES = []


def tracked(path, member=None):
    path = Path(path).resolve()
    ACCESSES.append({'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'member':member, 'read_only':True})
    if member is None:
        return json.loads(path.read_text())
    with np.load(path, allow_pickle=False) as z:
        return np.array(z[member], dtype=np.float64)


def norm(x):
    std=x.std((0,1),ddof=0)
    assert np.min(std)>0
    return (x-x.mean((0,1)))/std


def scores(z):
    return np.asarray([((z[:,h:]-z[:,:-h])**2).mean((0,1)) for h in H])


def selected(score):
    return sorted(np.lexsort((np.arange(4),np.asarray(score,np.float32)))[:2].tolist())


def cancel_ratio(parts):
    parts=np.asarray(parts,np.float64)
    denom=np.abs(parts).sum()
    return 0. if denom==0 else float(1.-abs(parts.sum())/denom)


began=time.perf_counter()
aggregation=tracked(BASE/'final_evaluation_v2/AGGREGATION_COMPLETED.json')
rows=[];oracles=[];geometry=[]
for group in aggregation['representation_groups']:
    if group['method']!='OursFine4':
        continue
    slot=group['slot_id'];target=[0,1] if slot.startswith('ar_') else [1,3]
    other=[j for j in range(4) if j not in target]
    physical=tracked(BASE/'private/TRAIN_CAL'/f'{slot}.npz','train__physical')
    z=norm(physical);qh=scores(z);sp=qh.mean(0)
    gamma=float(min(sp[other])-max(sp[target]))
    geometry_pairs=[]
    for a in target:
        for b in other:
            rho=float((z[...,a]*z[...,b]).mean())
            mab_h=[float(((z[:,h:,a]-z[:,:-h,a])*(z[:,h:,b]-z[:,:-h,b])).mean()) for h in H]
            mab=float(np.mean(mab_h));gap=float(sp[b]-sp[a])
            chi=2*mab-rho*float(sp[a]+sp[b])
            angle=.5*float(np.arctan2(gap,abs(chi)))
            geometry_pairs.append({'target_axis_a':a,'other_axis_b':b,'rho_Q0':rho,
                'M_ab_per_h':mab_h,'M_ab':mab,'S_a':float(sp[a]),'S_b':float(sp[b]),
                'gamma_ab':gap,'chi_ab':chi,'abs_chi_over_gamma':abs(chi)/gap,
                'worst_sign_pair_order_critical_angle_radians':angle,
                'worst_sign_pair_order_critical_angle_degrees':float(np.degrees(angle)),
                'hypothetical_rotation_only_not_actual_Fine_fit':True})
    geometry.append({'slot':slot,'physical_only':True,'pairs':geometry_pairs})
    oracles.append({'slot':slot,'TRAIN_shape':list(physical.shape),'target':target,
                    'physical_scores':sp.tolist(),'empirical_target_margin':gamma,
                    'selected':selected(sp),'target_selected':selected(sp)==target})
    for parent in group['rows']:
        rep=parent['profile']['logical_rep']
        folder=BASE/'execution/jobs/representation/OursFine4'/slot/f'seed{rep}'
        y=tracked(folder/'FEATURES.npz','train')
        g0=tracked(folder/'G0.json')
        mapping=tracked(parent['CAL_maps']['path'])['maps']['axis_affine']
        assert mapping['assignment_unique'] and not mapping['dense_mixing_allowed']
        m=mapping['matched_indices']
        # Existing CAL mapping only. No recomputation/refit/axis search.
        w=norm(y[...,m]*np.asarray(mapping['slopes'])+np.asarray(mapping['intercept']))
        e=w-z;qw=scores(w);sw=qw.mean(0)
        c=[];r=[];boundary=[];lag=[];perh=[]
        k=2*z*e+e*e
        for hidx,h in enumerate(H):
            dz=z[:,h:]-z[:,:-h];de=e[:,h:]-e[:,:-h]
            cc=(dz*de).mean((0,1));rr=(de*de).mean((0,1))
            bb=(k[:,h:]+k[:,:-h]).mean((0,1))
            ll=(z[:,h:]*e[:,:-h]+z[:,:-h]*e[:,h:]+e[:,h:]*e[:,:-h]).mean((0,1))
            c.append(cc);r.append(rr);boundary.append(bb);lag.append(ll)
            rho=np.divide(cc,np.sqrt(qh[hidx]*rr),out=np.zeros(4),where=rr>0)
            perh.append({'h':h,'q':qh[hidx].tolist(),'c':cc.tolist(),'r':rr.tolist(),
                          'normalized_cross_moment_cosine':rho.tolist(),'boundary':bb.tolist(),'lag':ll.tolist(),
                          'signed_axis_score_error':(qw[hidx]-qh[hidx]).tolist()})
        c=np.asarray(c);r=np.asarray(r);boundary=np.asarray(boundary);lag=np.asarray(lag)
        oldaxis=(2*np.sqrt(qh*r)+r).mean(0)
        globalold=float(2*oldaxis.max())
        rawscore=scores(norm(y)).mean(0)
        assert np.max(abs(rawscore-np.asarray(g0['speed_FP64'])))==0
        assert selected(rawscore)==g0['selected_pair']
        fp32=np.asarray(g0['speed_FP32'])[m]
        num=U32*np.abs(sw)
        assert np.all(np.abs(fp32-sw)<=num+1e-12)
        pairs=[]
        for a in target:
            for b in other:
                ch=c[:,b]-c[:,a];rh=r[:,b]-r[:,a]
                bd=boundary[:,b]-boundary[:,a];ld=lag[:,b]-lag[:,a]
                C=float(ch.mean());R=float(rh.mean());B=float(bd.mean());L=float(ld.mean())
                gap=float(sp[b]-sp[a]);D=float(sw[b]-sw[a]-gap)
                corr=2*abs(C)+abs(R)
                corr_no_hcancel=float(np.mean(2*np.abs(ch)+np.abs(rh)))
                signaware=2*max(0.,-C)+max(0.,-R)
                lagbound=abs(B)+2*abs(L)
                oldpair=float(oldaxis[a]+oldaxis[b])
                factorized=[]
                for h in H:
                    dz=z[:,h:]-z[:,:-h];de=e[:,h:]-e[:,:-h]
                    xm=dz[...,b]-dz[...,a];xp=dz[...,b]+dz[...,a]
                    em=de[...,b]-de[...,a];ep=de[...,b]+de[...,a]
                    qm=float((xm*xm).mean());qp=float((xp*xp).mean())
                    rm=float((em*em).mean());rp=float((ep*ep).mean())
                    factorized.append(np.sqrt(qm*rp)+np.sqrt(qp*rm)+np.sqrt(rm*rp))
                pol=float(np.mean(factorized));numerical=float(num[a]+num[b])
                signed_round=float(fp32[b]-fp32[a]-(sw[b]-sw[a]))
                assert abs(D-(2*C+R))<1e-12
                assert abs(D-(B-2*L))<1e-12
                assert abs(D)<=corr+1e-12 and abs(D)<=lagbound+1e-12
                assert abs(D)<=oldpair+1e-12 and abs(D)<=pol+1e-12
                assert -D<=signaware+1e-12
                pairs.append({'target_axis_a':a,'other_axis_b':b,'oracle_gap':gap,
                    'Fine_FP64_gap':float(sw[b]-sw[a]),'Fine_stored_FP32_gap':float(fp32[b]-fp32[a]),
                    'signed_exact_pair_error':D,'c_difference_per_h':ch.tolist(),'r_difference_per_h':rh.tolist(),
                    'signed_pair_error_per_h':(2*ch+rh).tolist(),'C':C,'R':R,
                    'boundary_difference_per_h':bd.tolist(),'lag_difference_per_h':ld.tolist(),'B':B,'L':L,
                    'old_global_Linfty_bound':globalold,'old_pair_specific_Cauchy_bound':oldpair,
                    'polarization_Cauchy_bound':pol,'correlation_pair_bound_no_h_cancel':corr_no_hcancel,
                    'correlation_pair_bound':corr,'one_sided_pair_bound':signaware,'lag_pair_bound':lagbound,
                    'cross_h_C_cancellation_fraction':cancel_ratio(ch),'cross_h_R_cancellation_fraction':cancel_ratio(rh),
                    'cross_component_cancellation_fraction':cancel_ratio([2*C,R]),
                    'FP32_pair_rounding_bound':numerical,'signed_FP32_pair_rounding_error':signed_round,
                    'pass_old_global':gap>globalold+numerical,
                    'pass_old_pair_specific':gap>oldpair+numerical,
                    'pass_polarization_Cauchy':gap>pol+numerical,
                    'pass_correlation_no_h_cancel':gap>corr_no_hcancel+numerical,
                    'pass_correlation':gap>corr+numerical,
                    'pass_one_sided':gap>signaware+numerical,
                    'pass_lag':gap>lagbound+numerical,
                    'signed_exact_rounded_gap_positive':gap+D+signed_round>0})
        method_fields=['old_global','old_pair_specific','polarization_Cauchy','correlation_no_h_cancel',
                       'correlation','one_sided','lag']
        passes={name:all(p['pass_'+name] for p in pairs) for name in method_fields}
        actualpair=selected(sw)
        rows.append({'slot':slot,'rep':rep,'target':target,'matched_indices':m,'CAL_slopes':mapping['slopes'],
                    'physical_scores':sp.tolist(),'Fine_aligned_scores':sw.tolist(),'empirical_target_margin':gamma,
                    'all_four_pair_pass':passes,'selected_physical':actualpair,
                    'stored_pair_matches_target':sorted(m.index(j) for j in g0['selected_pair'])==target,
                    'normalization_identity_max_abs':float(abs(k.mean((0,1))).max()),
                    'old_axis_bound':oldaxis.tolist(),'per_horizon':perh,'pairs':pairs})

hist=tracked(ROOT/'outputs/HCF_STABLE_ID_RUN_20261028_V2/oracle_FHN_G_target_v1/RESULTS.json')
history=[]
for rr in hist['rows']:
    ss=np.asarray(rr['G0']['selector']['speed_FP64']);gaps=[float(ss[b]-ss[a]) for a in [1,3] for b in [0,2]]
    history.append({'unit':rr['unit_id'],'t_start':rr['support_original_t_start'],
        'scores':ss.tolist(),'target':[1,3],'pair_gaps_a1b0_a1b2_a3b0_a3b2':gaps,
        'target_margin':min(gaps),'selected':rr['G0']['selector']['selected_pair'],
        'target_selected':rr['G0']['selector']['selected_pair']==[1,3],
        'perfect_representation_error':0.,'positive_target_margin_required_even_with_zero_error':True})
integrity=tracked(ROOT/'outputs/HCF_STAGE_A_20261104/source_integrity_audit.json')
sourcepaths=['work/hcf_method_finalize_20261027/v1/explicit_selector.py',
             'work/hcf_stable_id_20261028/confirmation_v2/ours_bridge.py',
             'work/hcf_stable_id_20261028/confirmation_v2/data_owner.py']
current_sources={str(ROOT/p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sourcepaths}
ownsha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
summary={name:{'all12':sum(r['all_four_pair_pass'][name] for r in rows),
               'AR6':sum(r['all_four_pair_pass'][name] for r in rows if r['slot'].startswith('ar_')),
               'FHN6':sum(r['all_four_pair_pass'][name] for r in rows if r['slot'].startswith('fhn_')),
               'pair_cells48':sum(p['pass_'+name] for r in rows for p in r['pairs'])}
         for name in method_fields}
report={'scope':'truth-assisted finite saved TRAIN audit; no unlabeled or population guarantee',
        'horizons':list(H),'standardization':'own full TRAIN mean/std, FP64 ddof0, per-h valid endpoints',
        'alignment':'existing frozen CAL axis permutation/scalar-affine; no fit',
        'normalized_cross_moment_definition':'c/sqrt(q*r), without endpoint demeaning; not Pearson correlation',
        'code_sha256':ownsha,'oracles':oracles,'rows':rows,'coverage':summary,
        'historical_FHN_oracles':history,'source_integrity_reference':integrity,
        'current_source_sha256':current_sources,
        'accesses':ACCESSES,'cost':{'new_training':0,'new_model_forward':0,'new_physical_data':0,'PRNG_calls':0,
            'new_FINAL_array_reads':0,'existing_aggregation_metadata_read':True,
            'elapsed_seconds':time.perf_counter()-began}}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'PAIRED_SCORE_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(OUT/'ROTATION_GEOMETRY.json').write_text(json.dumps({'scope':'physical TRAIN second moments; hypothetical pair rotation only',
    'horizons':list(H),'rows':geometry,'formula':'chi=2*M_ab-rho*(S_a+S_b); theta_crit=0.5*atan2(gamma,abs(chi))',
    'not_actual_Fine_rotation_model':True,'not_full_four_axis_selection_certificate':True,
    'physical_input_sources':[a for a in ACCESSES if a['member']=='train__physical'],
    'code_sha256':ownsha,'new_randomness_or_training':0},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'coverage':summary,'rows':[{'slot':r['slot'],'rep':r['rep'],
    'old2max':max(p['old_global_Linfty_bound'] for p in r['pairs']),
    'min_pair_safe_slack':min(p['oracle_gap']-p['correlation_pair_bound']-p['FP32_pair_rounding_bound'] for p in r['pairs']),
    'max_pair_bound':max(p['correlation_pair_bound'] for p in r['pairs']),
    'max_crossh_cancellation_C':max(p['cross_h_C_cancellation_fraction'] for p in r['pairs'])} for r in rows],
    'source_count':len(ACCESSES),'elapsed_seconds':report['cost']['elapsed_seconds']},ensure_ascii=False,indent=2))
