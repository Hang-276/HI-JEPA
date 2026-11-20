"""Fixed-gauge affine Hessian/target-gradient finite-array diagnostic.

No OLS, polar fit, NN, random data, or FINAL reads. Reuses stored matrices and
innovation laws. Neighborhood constants are checked at prescribed actual B;
no radius search or threshold tuning. All results are truth-assisted.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
R2=OUT.parent/'round2';BASE=ROOT/'outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2'
SCOPE=OUT/'AFFINE_GEOMETRY_SCOPE.md';THEORY=OUT/'AFFINE_LOCAL_THEORY.md'
PAIRS=[(i,j)for i in range(4)for j in range(i+1,4)]
COORDS=[p for i,j in PAIRS for p in ((i,j),(j,i))]
ACCESS=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scalar_json(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def read(p,key=None):
    p=Path(p).resolve();assert 'FINAL'not in str(p)
    ACCESS.append({'path':str(p),'member':key,'sha256':sha(p)})
    if key is None:return json.loads(p.read_text())
    assert key in ('train__physical','cal__physical')
    with np.load(p,allow_pickle=False)as z:return np.asarray(z[key],np.float64)
def variance(x):
    xx=x.reshape(-1,4);xx=xx-xx.mean(0);return xx.T@xx/len(xx)
def incmatrix(x,h):
    dd=(x[:,h:]-x[:,:-h]).reshape(-1,4);return dd.T@dd/len(dd)
def core_risk(A,D):
    sg,ld=np.linalg.slogdet(A);assert sg!=0
    out=[]
    for du in D:
        v=(A*A)@du;assert np.min(v)>0
        out.append(float(.5*(np.log(v).sum()-2*ld-np.log(du).sum())))
    return float(np.mean(out))
def row_score(A,M,Sigma):
    num=np.einsum('ij,jk,ik->i',A,M,A)
    den=np.einsum('ij,jk,ik->i',A,Sigma,A)
    assert np.min(den)>0
    return num/den

assert SCOPE.exists()and THEORY.exists()
scope_sha=sha(SCOPE);theory_sha=sha(THEORY);start=time.perf_counter()
old=read(R2/'NONLINEAR_REMAINDER_AUDIT.json');noise=read(R2/'INNOVATION_BUDGET_AUDIT.json')
slots=[];lookup={}
for oracle in old['oracles']:
    slot=oracle['slot'];target=oracle['target'];other=[j for j in range(4)if j not in target]
    law=next(x for x in noise['noise_laws_and_state_gauge']if x['slot']==slot)
    D=np.asarray(law['innovation_variance_by_regime_state_gauge']);assert D.shape==(9,4)and np.min(D)>0
    mv=np.asarray(oracle['physical_TRAIN_mean']);sv=np.asarray(oracle['physical_TRAIN_std'])
    p=BASE/'private/TRAIN_CAL'/f'{slot}.npz'
    v={sp:(read(p,sp+'__physical')-mv)/sv for sp in ('train','cal')}
    Sigma=variance(v['train']);assert np.max(abs(np.diag(Sigma)-1))<1e-12
    K=float(np.linalg.eigvalsh(Sigma)[-1]);blocks=[];Hs=[]
    for i,j in PAIRS:
        a=float(np.mean(D[:,j]/D[:,i]));c=float(np.mean(D[:,i]/D[:,j]));det=a*c-1
        lm=.5*(a+c+np.hypot(a-c,2.));small=det/lm
        assert det>0 and small>0
        H=np.array([[a,1.],[1.,c]]);Hs.append(H)
        blocks.append({'i':i,'j':j,'coordinate_order':[[i,j],[j,i]],'H':H.tolist(),
            'kappa_rotation_angle_parameter':a+c-2,'kappa_rotation_unit_Frobenius':.5*(a+c-2),
            'kappa_unit_shear_i_receives_j':a,'kappa_unit_shear_j_receives_i':c,
            'lambda_max':float(lm),'determinant':float(det),'lambda_min_stable_det_over_max':float(small)})
    kappa=min(b['lambda_min_stable_det_over_max']for b in blocks)
    R=float(np.max(D[:,:,None]/D[:,None,:]))
    evaluations={}
    for split in ('train','cal'):
        Mh=[incmatrix(v[split],h)for h in (4,16,32)];M=np.mean(Mh,axis=0);s=np.diag(M)/np.diag(Sigma)
        gradients=[]
        for a in target:
            for b in other:
                g=np.zeros((4,4))
                g[b,:]=2*(M[b,:]-s[b]*Sigma[b,:])/Sigma[b,b]
                g[a,:]=-2*(M[a,:]-s[a]*Sigma[a,:])/Sigma[a,a]
                np.fill_diagonal(g,0.)
                ell=np.asarray([g[i,j]for i,j in COORDS]);psi=0.
                contributions=[]
                for k,H in enumerate(Hs):
                    l=ell[2*k:2*k+2];value=float(l@np.linalg.solve(H,l));assert value>=-1e-14
                    psi+=value;contributions.append(value)
                norm2=float(ell@ell);coarse=norm2/kappa
                assert psi<=coarse+1e-12
                gradients.append({'target_a':a,'other_b':b,'physical_gap':float(s[b]-s[a]),
                    'ell_offdiag':ell.tolist(),'ell_norm_squared':norm2,
                    'Psi_dual_Hessian_squared':psi,'coarse_squared_using_global_kappa':coarse,
                    'coarse_over_Psi_factor':coarse/psi if psi>0 else None,
                    'block_contributions_in_pair_order':contributions,
                    'scope':'identity derivative of finite score quotient, Q0=TRAIN; not actual global Flow modulus'})
        evaluations[split]={'state_shape':list(v[split].shape),'Q0':'TRAIN covariance fixed',
            'Sigma_used_for_denominators':Sigma.tolist(),'M_mean':M.tolist(),'M_per_h':{str(h):m.tolist()for h,m in zip((4,16,32),Mh)},
            'physical_scores':s.tolist(),'physical_target_margin':min(p['physical_gap']for p in gradients),
            'target_pairs':gradients}
    entry={'slot':slot,'target':target,'state_gauge':'stored physical full-TRAIN mean/std, unchanged on CAL',
        'D_by_regime':D.tolist(),'true_regime_weights':[1/9]*9,'H_blocks_all6':blocks,
        'kappa_affine_global_min_H_eigenvalue':kappa,'max_innovation_variance_ratio_R':R,
        'Sigma_TRAIN':Sigma.tolist(),'Sigma_operator_norm':K,'evaluations':evaluations}
    slots.append(entry);lookup[slot]={'entry':entry,'D':D,'H':Hs,'Sigma':Sigma,'K':K,'kappa':kappa,'R':R}

rows=[]
for prior in old['rows']:
    slot=prior['slot'];rep=prior['rep'];x=lookup[slot];A=np.asarray(prior['OLS']['A']);Q=np.asarray(prior['polar']['Q']);diag=np.diag(A)
    row={'slot':slot,'rep':rep,'original_A':A.tolist(),'original_Q':Q.tolist(),'A_diagonal_before_scaling':diag.tolist()}
    if np.min(abs(diag))<=1e-8:
        rows.append({**row,'status':'N/A_diagonal_chart_guard_no_repair'});continue
    Ab=A/diag[:,None];B=Ab-np.eye(4);np.fill_diagonal(B,0.)
    bv=np.asarray([B[i,j]for i,j in COORDS]);norm=float(np.linalg.norm(B));norm2=float(bv@bv)
    op=float(np.sqrt(max(0.,np.linalg.eigvalsh(B.T@B)[-1])))
    qform=sum(float(bv[2*k:2*k+2]@H@bv[2*k:2*k+2])for k,H in enumerate(x['H']))
    quad=.5*qform;exact=core_risk(A,x['D']);rowexact=core_risk(Ab,x['D']);FQ=core_risk(Q,x['D'])
    assert abs(exact-rowexact)<1e-12
    ref=next(r for r in noise['rows']if r['slot']==slot and r['rep']==rep)
    assert abs(exact-ref['cores']['affine']['Gaussian_core_contrast_uniform9'])<1e-12
    assert abs(FQ-ref['cores']['polar']['Gaussian_core_contrast_uniform9'])<1e-12
    # No radius optimization. r is the actual Frobenius norm fixed in scope.
    cr=norm/(3*(1-norm))+x['R']**2*norm2/4 if norm<1 else None
    uniform_bound=cr*norm2 if cr is not None else None
    uniform=cr is not None and cr<=x['kappa']/4
    q=np.asarray([((B*B)@du)/du for du in x['D']])
    assert np.min(q)>=0
    log_budget=float(.5*np.mean(np.sum(q-np.log1p(q),axis=1)))
    det_budget=norm2*op/(3*(1-op))if op<1 else None
    point_budget=log_budget+det_budget if det_budget is not None else None
    point_kappa=point_budget is not None and point_budget<=x['kappa']*norm2/4
    point_direction=point_budget is not None and point_budget<=qform/4
    rem=exact-quad
    if uniform_bound is not None:assert abs(rem)<=uniform_bound+1e-12
    if point_budget is not None:assert abs(rem)<=point_budget+1e-12
    if uniform or point_kappa or point_direction:assert exact+1e-12>=qform/4
    if norm2==0:
        uniform=point_kappa=point_direction=True
    domains={'r_equals_actual_B_Fro':norm,'B_operator_norm':op,'R':x['R'],'kappa_aff':x['kappa'],
        'uniform_c_r':cr,'uniform_remainder_budget':uniform_bound,'uniform_radius_r_ball_condition':uniform,
        'pointwise_q_by_u':q.tolist(),'pointwise_log_budget':log_budget,'pointwise_logdet_series_budget':det_budget,
        'pointwise_total_remainder_budget':point_budget,
        'pointwise_budget_per_norm2':point_budget/norm2 if norm2>0 and point_budget is not None else None,
        'pointwise_min_kappa_condition':point_kappa,'pointwise_directional_quadratic_condition':point_direction,
        'pointwise_pass_is_not_uniform_ball_pass':True}
    evals={}
    for split in ('train','cal'):
        geo=x['entry']['evaluations'][split];M=np.asarray(geo['M_mean']);Sigma=x['Sigma'];s=np.asarray(geo['physical_scores'])
        sa=row_score(A,M,Sigma);sr=row_score(Ab,M,Sigma)
        assert np.max(abs(sa-sr))<1e-12
        Cj=[];score_remainders=[]
        row_cond=all(np.sqrt(Sigma[j,j])-np.sqrt(x['K'])*norm>0 for j in range(4))
        for j in range(4):
            T=M-s[j]*Sigma;bj=B[j,:]
            lj=float(2*T[j,:]@bj/Sigma[j,j]);u=float(Sigma[j,:]@bj);vv=float(bj@Sigma@bj)
            rowrem=(float(bj@T@bj)-lj*(2*u+vv))/(Sigma[j,j]+2*u+vv)
            assert abs((sa[j]-s[j])-lj-rowrem)<1e-12
            score_remainders.append(rowrem)
            if row_cond:
                aa=float(np.linalg.norm(T[j,:]));tt=float(max(abs(np.linalg.eigvalsh(T))))
                # General denominator form; Sigma_jj=1 up to FP64 normalization.
                cj=(tt+4*aa*np.sqrt(x['K']/Sigma[j,j])+2*aa*x['K']*norm/Sigma[j,j])/(np.sqrt(Sigma[j,j])-np.sqrt(x['K'])*norm)**2
                assert abs(rowrem)<=cj*float(bj@bj)+1e-12;Cj.append(cj)
        pairs=[]
        can_use=(uniform or point_kappa or point_direction)and row_cond
        for grad in geo['target_pairs']:
            a=grad['target_a'];b=grad['other_b'];ell=np.asarray(grad['ell_offdiag'])
            lin=float(ell@bv);gap=float(sa[b]-sa[a]);physical=float(grad['physical_gap'])
            psi=grad['Psi_dual_Hessian_squared'];lower=None
            if can_use:
                lower=physical-2*np.sqrt(max(0.,psi*exact))-4*max(Cj[a],Cj[b])*exact/x['kappa']
                assert gap+1e-12>=lower
            pairs.append({'target_a':a,'other_b':b,'physical_gap':physical,'affine_core_gap':gap,
                'linear_gap_change':lin,'exact_gap_change':gap-physical,'nonlinear_score_quotient_remainder':gap-physical-lin,
                'Psi_dual_Hessian_squared':psi,'coarse_squared_using_global_kappa':grad['coarse_squared_using_global_kappa'],
                'coarse_over_Psi_factor':grad['coarse_over_Psi_factor'],
                'finite_affine_gap_lower_bound_if_admissible':lower,
                'positive_lower_bound':lower is not None and lower>0})
        evals[split]={'affine_scores':sa.tolist(),'row_scale_score_max_difference':float(np.max(abs(sa-sr))),
            'physical_target_margin':float(geo['physical_target_margin']),
            'affine_target_margin':min(p['affine_core_gap']for p in pairs),
            'score_denominator_radius_condition':row_cond,'row_C_constants':Cj if row_cond else None,
            'exact_score_row_remainders':score_remainders,'pairs_all4':pairs,
            'all4_finite_affine_lower_bounds_positive':all(p['positive_lower_bound']for p in pairs)}
    rows.append({**row,'status':'complete','row_scaled_A':Ab.tolist(),'B_offdiag_matrix':B.tolist(),'B_offdiag_vector':bv.tolist(),
        'F_original_A':exact,'F_row_scaled_A':rowexact,'row_scale_F_difference':rowexact-exact,'F_stored_Q':FQ,
        'quadratic_half_bHb':quad,'F_over_quadratic':exact/quad if quad>0 else None,
        'signed_F_minus_quadratic':rem,'domains':domains,'evaluations':evals})

summary={'parents_total':len(rows),'parents_complete':sum(r['status']=='complete'for r in rows),
    'N_A_parents':sum(r['status']!='complete'for r in rows)}
complete=[r for r in rows if r['status']=='complete']
for flag in ('uniform_radius_r_ball_condition','pointwise_min_kappa_condition','pointwise_directional_quadratic_condition'):
    summary[flag]=sum(r['domains'][flag]for r in complete)
summary['F_over_quadratic_range']=[min(r['F_over_quadratic']for r in complete),max(r['F_over_quadratic']for r in complete)]
summary['B_Fro_range']=[min(r['domains']['r_equals_actual_B_Fro']for r in complete),max(r['domains']['r_equals_actual_B_Fro']for r in complete)]
summary['finite_affine_all4_positive']={sp:sum(r['evaluations'][sp]['all4_finite_affine_lower_bounds_positive']for r in complete)for sp in ('train','cal')}
report={'scope':'truth-assisted fixed-gauge local affine risk and finite TRAIN/CAL target-gradient geometry; no actual Flow global kappa or risk certificate',
    'scope_sha256_before_calculation':scope_sha,'theory_sha256_used':theory_sha,'script_sha256':sha(__file__),
    'offdiag_coordinate_order':[list(p)for p in COORDS],'slots':slots,'rows':rows,'summary':summary,'accesses':ACCESS,
    'cost':{'physical_array_members_read':8,'new_OLS':0,'new_polar':0,'new_NN_training_forward_PRNG_data_FINAL_GPU':0,
        'prior_serialization_failure_attempts_with_same_scientific_logic':1,
        'physical_members_read_including_preserved_first_failed_attempt':16,
        'elapsed_seconds':time.perf_counter()-start},'numerical_tolerance':'ordinary FP64 checks atol1e-12, not interval arithmetic'}
assert sha(SCOPE)==scope_sha and len(rows)==12 and len(slots)==4
(OUT/'AFFINE_GEOMETRY_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=scalar_json)+'\n')
print(json.dumps({'summary':summary,'slots':[{'slot':s['slot'],'kappa_aff':s['kappa_affine_global_min_H_eigenvalue'],
    'R':s['max_innovation_variance_ratio_R'],'H_pairs':s['H_blocks_all6'],
    'dual_improvement':[{'split':sp,'factors':[p['coarse_over_Psi_factor']for p in s['evaluations'][sp]['target_pairs']]}for sp in ('train','cal')]}for s in slots],
    'parents':[{'slot':r['slot'],'rep':r['rep'],'F_over_quad':r.get('F_over_quadratic'),
    'r':r.get('domains',{}).get('r_equals_actual_B_Fro'),'domains':{k:v for k,v in r.get('domains',{}).items()if k in (
    'uniform_radius_r_ball_condition','pointwise_min_kappa_condition','pointwise_directional_quadratic_condition','uniform_c_r','pointwise_budget_per_norm2')}}for r in rows]},ensure_ascii=False,indent=2,default=scalar_json))
