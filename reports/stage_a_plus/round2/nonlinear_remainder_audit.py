"""Fixed-protocol truth-assisted affine/polar remainder audit.

Only reads existing TRAIN/CAL arrays and original CAL maps/G0. Does not import
models, use randomness, generate physical data, read FINAL arrays, or modify
historical files. The one OLS and one polar decomposition per parent are
scientific diagnostics, never deployed changes to Fine or G0.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2'
PROTOCOL=OUT/'NONLINEAR_REMAINDER_PROTOCOL.md'
SLOTS=('ar_rich_fresh1','ar_rich_fresh2','fhn_fresh_eps008','fhn_fresh_eps05')
SPLITS=('train','cal')
H=(4,16,32)
ACCESS=[]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path,member=None):
    path=Path(path).resolve()
    assert 'FINAL' not in str(path)
    if member is not None:
        assert member in ('train','cal','train__physical','cal__physical')
    ACCESS.append({'path':str(path),'member':member,'sha256':sha(path),'read_only':True})
    if member is None:
        return json.loads(path.read_text())
    with np.load(path,allow_pickle=False) as z:
        x=np.asarray(z[member],np.float64)
    assert x.ndim==3 and x.shape[-1]==4 and np.isfinite(x).all()
    return x


def fit_train_scale(train):
    m=train.mean((0,1));s=train.std((0,1),ddof=0)
    assert np.min(s)>0
    return m,s


def statevar(x):
    return x.var((0,1),ddof=0)


def meaninc(x):
    return np.asarray([((x[:,h:]-x[:,:-h])**2).mean((0,1)) for h in H])


def pair_selected(s):
    return sorted(np.lexsort((np.arange(4),s))[:2].tolist())


def score_record(z,train,target):
    var=statevar(train)
    assert np.min(var)>0
    per_h=meaninc(z)/var
    s=per_h.mean(0);other=[j for j in range(4) if j not in target]
    pairs=[{'a':a,'b':b,'gap':float(s[b]-s[a])} for a in target for b in other]
    return {'scores':s.tolist(),'scores_per_h':per_h.tolist(),'Q0_TRAIN_variance':var.tolist(),
            'target_margin':min(p['gap'] for p in pairs),'all_target_pairs_positive':all(p['gap']>0 for p in pairs),
            'score_order_top2':pair_selected(s),'pairs':pairs,
            'CAL_ranking_is_support_diagnostic_not_deployed_G0':True}


def decompose(core_train,out_train,core_eval,out_eval,target):
    resid_train=out_train-core_train;resid=out_eval-core_eval
    V=statevar(core_train);vr=statevar(resid_train);Vout=statevar(out_train)
    z0=core_train-core_train.mean((0,1));r0=resid_train-resid_train.mean((0,1))
    c0=(z0*r0).mean((0,1));dv=2*c0+vr
    assert np.min(V)>0 and np.min(Vout)>0
    assert np.max(np.abs(Vout-(V+dv)))<1e-12
    a=meaninc(core_eval);rinc=meaninc(resid)
    c=np.asarray([((core_eval[:,h:]-core_eval[:,:-h])*(resid[:,h:]-resid[:,:-h])).mean((0,1)) for h in H])
    sz=a/V;sy=meaninc(out_eval)/Vout;direct=sy-sz
    termcross=2*c/Vout
    termenergy=rinc/Vout
    termnorm=-sz*dv/Vout
    formula=termcross+termenergy+termnorm
    residual=float(np.max(abs(direct-formula)))
    assert residual<1e-12
    # Norm bounds, not exact signed re-evaluation. All use TRAIN denominators.
    cs=(2*np.sqrt(a*rinc)+rinc+np.abs(sz)*np.abs(dv))/Vout
    dvcs=2*np.sqrt(V*vr)+vr
    allcs=(2*np.sqrt(a*rinc)+rinc+np.abs(sz)*dvcs)/Vout
    corr=(2*np.abs(c.mean(0))+rinc.mean(0)+np.abs(sz.mean(0)*dv))/Vout
    delta=direct.mean(0);zscore=sz.mean(0);yscore=sy.mean(0)
    alpha=termcross.mean(0);beta=termenergy.mean(0);nu=termnorm.mean(0)
    assert np.all(np.abs(delta)<=cs.mean(0)+1e-12)
    assert np.all(np.abs(delta)<=allcs.mean(0)+1e-12)
    assert np.all(np.abs(delta)<=corr+1e-12)
    pairs=[]
    for aa in target:
        for bb in range(4):
            if bb in target:
                continue
            gap=float(zscore[bb]-zscore[aa]);ygap=float(yscore[bb]-yscore[aa])
            d=float(delta[bb]-delta[aa])
            terms=[float(t[bb]-t[aa])for t in (alpha,beta,nu)]
            bcs=float(cs.mean(0)[aa]+cs.mean(0)[bb])
            bfull=float(allcs.mean(0)[aa]+allcs.mean(0)[bb])
            bcorr=float(sum(abs(t)for t in terms))
            assert abs(d-sum(terms))<1e-12 and abs(ygap-gap-d)<1e-12
            assert abs(d)<=bcs+1e-12 and abs(d)<=bcorr+1e-12
            pairs.append({'a':aa,'b':bb,'core_gap':gap,'output_gap':ygap,'signed_exact_pair_discrepancy':d,
                'cross_increment_contribution':terms[0],'residual_increment_contribution':terms[1],
                'Q0_variance_correction_contribution':terms[2],
                'CS_bound_measured_Q0_correction':bcs,'CS_bound_all_norms':bfull,
                'truth_assisted_pair_correlation_triangle_bound':bcorr,
                'margin_safe_CS_measured_Q0':gap>bcs,'margin_safe_CS_all_norms':gap>bfull,
                'margin_safe_truth_corr':gap>bcorr})
    evmean=resid.mean((0,1));evvar=statevar(resid)
    return {'core_Q0_variance':V.tolist(),'residual_Q0_variance':vr.tolist(),'cross_Q0_covariance':c0.tolist(),
        'Q0_variance_difference':dv.tolist(),'output_Q0_variance':Vout.tolist(),
        'evaluation_residual_mean':evmean.tolist(),'evaluation_residual_variance':evvar.tolist(),
        'evaluation_residual_MSE':(evvar+evmean**2).tolist(),
        'core_scores':zscore.tolist(),'output_scores':yscore.tolist(),'signed_score_discrepancy':delta.tolist(),
        'cross_increment_contribution':alpha.tolist(),'residual_increment_contribution':beta.tolist(),
        'Q0_variance_correction_contribution':nu.tolist(),
        'per_h':{'horizons':list(H),'core_raw_increment_second_moment':a.tolist(),
                 'residual_increment_second_moment':rinc.tolist(),'cross_increment_moment':c.tolist(),
                 'core_scores':sz.tolist(),'output_scores':sy.tolist(),
                 'cross_contribution':termcross.tolist(),'residual_energy_contribution':termenergy.tolist(),
                 'Q0_variance_correction_contribution':termnorm.tolist(),'direct_discrepancy':direct.tolist()},
        'bound_by_axis_CS_measured_Q0':cs.mean(0).tolist(),'bound_by_axis_CS_all_norms':allcs.mean(0).tolist(),
        'bound_by_axis_truth_corr':corr.tolist(),'pairs':pairs,
        'all_four_pairs_margin_safe_CS_measured_Q0':all(p['margin_safe_CS_measured_Q0']for p in pairs),
        'all_four_pairs_margin_safe_CS_all_norms':all(p['margin_safe_CS_all_norms']for p in pairs),
        'all_four_pairs_margin_safe_truth_corr':all(p['margin_safe_truth_corr']for p in pairs),
        'core_target_margin':min(p['core_gap']for p in pairs),'output_target_margin':min(p['output_gap']for p in pairs),
        'max_exact_formula_residual':residual}


assert PROTOCOL.exists()
protocol_sha=sha(PROTOCOL)
began=time.perf_counter()
rows=[];oracles=[];ols_count=0;polar_count=0
for slot in SLOTS:
    target=[0,1] if slot.startswith('ar_') else [1,3]
    pp=BASE/'private/TRAIN_CAL'/f'{slot}.npz'
    physical={s:load(pp,s+'__physical')for s in SPLITS}
    mv,sv=fit_train_scale(physical['train'])
    v={s:(physical[s]-mv)/sv for s in SPLITS}
    oracles.append({'slot':slot,'target':target,'scores_by_split':{s:score_record(v[s],v['train'],target)for s in SPLITS},
                    'physical_TRAIN_mean':mv.tolist(),'physical_TRAIN_std':sv.tolist()})
    for rep in range(3):
        folder=BASE/'execution/jobs/representation/OursFine4'/slot/f'seed{rep}'
        raw={s:load(folder/'FEATURES.npz',s)for s in SPLITS}
        maps=load(BASE/'evaluation_v2/CAL_maps/representation/OursFine4'/slot/f'seed{rep}/MAPS.json')['maps']['axis_affine']
        assert maps['assignment_unique'] and not maps['rank_degenerate'] and not maps['dense_mixing_allowed']
        g0=load(folder/'G0.json');perm=maps['matched_indices'];slopes=np.asarray(maps['slopes']);intercept=np.asarray(maps['intercept'])
        aligned={s:raw[s][...,perm]*slopes+intercept for s in SPLITS}
        mw,sw=fit_train_scale(aligned['train']);w={s:(aligned[s]-mw)/sw for s in SPLITS}
        # Verify the old G0 before fitting anything.
        mr,sr=fit_train_scale(raw['train']);old=meaninc((raw['train']-mr)/sr).mean(0)
        assert np.max(abs(old-np.asarray(g0['speed_FP64'])))==0
        assert sorted(np.lexsort((np.arange(4),old.astype(np.float32)))[:2].tolist())==g0['selected_pair']
        X=v['train'].reshape(-1,4);Y=w['train'].reshape(-1,4)
        design=np.column_stack((X,np.ones(len(X))))
        coeff,_,rank,design_sing=np.linalg.lstsq(design,Y,rcond=None);ols_count+=1
        if rank!=5:
            rows.append({'slot':slot,'rep':rep,'status':'N/A_rank_deficient_no_repair','design_rank':int(rank)})
            continue
        A=coeff[:4].T;b=coeff[4]
        ua,sing,vat=np.linalg.svd(A,full_matrices=False);polar_count+=1;Q=ua@vat
        cores={s:{'physical':v[s],'actual':w[s],'affine':v[s]@A.T+b,'polar':v[s]@Q.T+b}for s in SPLITS}
        ntrain=cores['train']['affine']-cores['train']['polar']
        rtrain=w['train']-cores['train']['affine']
        staticcross=(ntrain*rtrain).mean((0,1))
        evals={}
        for split in SPLITS:
            cc=cores[split];ct=cores['train']
            evals[split]={'shape':list(w[split].shape),'Q0':'TRAIN fixed for every denominator',
                'Qh':split+' legal endpoints for h=4/16/32',
                'scores':{k:score_record(cc[k],ct[k],target)for k in ('physical','actual','affine','polar')},
                'transitions':{name:decompose(ct[z],ct[o],cc[z],cc[o],target)for name,z,o in (
                    ('affine_to_actual','affine','actual'),('polar_to_affine','polar','affine'),
                    ('polar_to_actual','polar','actual'))}}
        rows.append({'slot':slot,'rep':rep,'status':'complete','target':target,
            'TRAIN_shape':list(w['train'].shape),'CAL_shape':list(w['cal'].shape),
            'CAL_axis_mapping':{'matched_indices':perm,'slopes':maps['slopes'],'intercept':maps['intercept'],'refitted':False},
            'Fine_aligned_TRAIN_mean':mw.tolist(),'Fine_aligned_TRAIN_std':sw.tolist(),
            'OLS':{'column_convention':'w=A v+b; NumPy rows W=V@A.T+b',
                'A':A.tolist(),'b':b.tolist(),'design_rank':int(rank),'design_singular_values':design_sing.tolist(),
                'design_condition_number':float(design_sing[0]/design_sing[-1]),'rcond':'numpy default None',
                'fit_support':'all full TRAIN states only','regularization':None,
                'TRAIN_residual_MSE_by_axis':(rtrain*rtrain).mean((0,1)).tolist(),
                'TRAIN_full_affine_explained_variance_by_axis':(1-statevar(rtrain)/statevar(w['train'])).tolist(),
                'max_static_predictor_residual_cross':float(abs(design.T@rtrain.reshape(-1,4)/len(X)).max())},
            'polar':{'Q':Q.tolist(),'A_singular_values':sing.tolist(),'Q_orthogonality_error_fro':float(np.linalg.norm(Q@Q.T-np.eye(4))),
                'detQ':float(np.linalg.det(Q)),'Q_minus_identity_fro':float(np.linalg.norm(Q-np.eye(4))),
                'A_minus_Q_fro':float(np.linalg.norm(A-Q)),'A_minus_identity_fro':float(np.linalg.norm(A-np.eye(4))),
                'TRAIN_nonorthogonal_affine_MSE_by_axis':(ntrain*ntrain).mean((0,1)).tolist(),
                'TRAIN_nonorthogonal_vs_OLS_residual_cross_by_axis':staticcross.tolist(),
                'closest_Q_metric':'Frobenius nearest orthogonal to A, not data-weighted Procrustes'},
            'deployed_G0_selected_raw':g0['selected_pair'],
            'deployed_G0_selected_physical':sorted(perm.index(k) for k in g0['selected_pair']),
            'evaluation':evals})

names=('affine_to_actual','polar_to_affine','polar_to_actual')
coverage={}
for split in SPLITS:
    coverage[split]={}
    for name in names:
        dd=[r['evaluation'][split]['transitions'][name]for r in rows if r['status']=='complete']
        coverage[split][name]={
            'core_positive_target_margin_parents':sum(d['core_target_margin']>0 for d in dd),
            'output_positive_target_margin_parents':sum(d['output_target_margin']>0 for d in dd),
            'all4_pair_CS_measured_Q0_parents':sum(d['all_four_pairs_margin_safe_CS_measured_Q0']for d in dd),
            'all4_pair_CS_all_norms_parents':sum(d['all_four_pairs_margin_safe_CS_all_norms']for d in dd),
            'all4_pair_truth_corr_parents':sum(d['all_four_pairs_margin_safe_truth_corr']for d in dd),
            'pair_cells_CS_measured_Q0':sum(p['margin_safe_CS_measured_Q0']for d in dd for p in d['pairs']),
            'pair_cells_truth_corr':sum(p['margin_safe_truth_corr']for d in dd for p in d['pairs'])}
report={'scope':'truth-assisted TRAIN fitted affine/polar decomposition, fixed-TRAIN-scale CAL diagnostic; no independent confirmation',
    'protocol_path':str(PROTOCOL),'protocol_sha256_before_fit':protocol_sha,'script_sha256':sha(__file__),
    'horizons':list(H),'oracles':oracles,'rows':rows,'coverage':coverage,'accesses':ACCESS,
    'numerical_scope':'ordinary FP64 arithmetic, identity and bound checks atol1e-12; not interval-certified floating-point proof',
    'cost':{'OLS_fits':ols_count,'polar_SVD4':polar_count,'CAL_fits':0,'new_NN_training':0,
        'new_model_forward':0,'checkpoint_deserializations':0,'new_physical_data':0,'PRNG_calls':0,
        'FINAL_array_reads':0,'GPU_jobs':0,'elapsed_seconds':time.perf_counter()-began}}
assert sha(PROTOCOL)==protocol_sha and ols_count==12 and polar_count<=12
(OUT/'NONLINEAR_REMAINDER_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
compact=[]
for r in rows:
    if r['status']!='complete':
        compact.append(r);continue
    compact.append({'slot':r['slot'],'rep':r['rep'],'OLS_residual_var_max':max(r['OLS']['TRAIN_residual_MSE_by_axis']),
        'A_singular_values':r['polar']['A_singular_values'],'A_minus_Q_fro':r['polar']['A_minus_Q_fro'],
        'Q_minus_I_fro':r['polar']['Q_minus_identity_fro'],
        'margins':{s:{k:e['target_margin']for k,e in r['evaluation'][s]['scores'].items()}for s in SPLITS}})
print(json.dumps({'coverage':coverage,'rows':compact,'cost':report['cost']},ensure_ascii=False,indent=2))
