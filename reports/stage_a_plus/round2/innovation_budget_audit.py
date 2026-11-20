"""Fixed-scope N10 innovation-scale empirical plug-in diagnostic.

Uses stored A/Q/scales, no repeated OLS/polar, conditional-mean model, NN,
randomness, new data, or FINAL arrays. Per-regime centering is only a sample
moment calculation; none of the empirical quantities is a population bound.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2'
SCOPE=OUT/'INNOVATION_BUDGET_SCOPE.md'
OLD=OUT/'NONLINEAR_REMAINDER_AUDIT.json'
LEGACY=Path('/Users/yuzihang/Documents/Codex/2026-10-23/referenced-chatgpt-conversation-this-is-an-3')
SOURCES=[ROOT/'work/hcf_stable_id_20261028/confirmation_v2/data_owner.py',
         ROOT/'work/hcf_method_freeze_20261026/identifiability_core_v1/data_owner.py',
         LEGACY/'work/hcf_identified_g_20261024/fine_stage.py',
         LEGACY/'work/hcf_8h_extension_20261025/fhn/env_fhn.py']
ACCESS=[]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path,member=None):
    path=Path(path).resolve()
    assert 'FINAL' not in str(path)
    if member is not None:
        assert member in ('train','cal','train__physical','cal__physical','train__regimes','cal__regimes')
    ACCESS.append({'path':str(path),'member':member,'sha256':sha(path),'read_only':True})
    if member is None:
        return json.loads(path.read_text())
    with np.load(path,allow_pickle=False)as f:
        arr=f[member]
    return np.asarray(arr,np.int64 if member.endswith('regimes')else np.float64)


def budget(nu):
    return float(4*np.log((1+nu)/(1-nu))) if 0<=nu<1 else None


assert SCOPE.exists()
scope_sha=sha(SCOPE);began=time.perf_counter()
old=read(OLD);assert old['cost']['OLS_fits']==12 and old['cost']['polar_SVD4']==12
instances=read(BASE/'registration/INSTANCE_LOCK.json')['instances']
prepared=read(BASE/'data/DATA_PREPARED.json')
release=read(BASE/'ROOT_TRAINING_RELEASE.json')
source_hashes={str(p):sha(p)for p in SOURCES}
assert all(release['source_sha256'].get(p)==h for p,h in source_hashes.items())
assert sha(OUT/'nonlinear_remainder_audit.py')==old['script_sha256']
data={};laws=[]
for rec in prepared['records']:
    slot=rec['slot_id'];instance=instances[slot]
    manifest_path=instance['physics_manifest']['path']
    assert sha(manifest_path)==instance['physics_manifest']['sha256']
    manifest=read(manifest_path)
    if slot.startswith('ar_'):
        assert instance['physics']['mode']=='rich'
        assert manifest['noise_std']==.12 and manifest['variance_ratio']==9.
        base_std=.12;high_multiplier=3.
        config={'family':'AR rich','noise_std':base_std,'std_high_multiplier':high_multiplier,
                'regime_law':manifest['regime'],'no_noise_law_modification':True}
    else:
        cfg=instance['physics']
        assert all(cfg[k]==manifest['config'][k]for k in cfg)
        assert cfg['kick_std']==.025 and cfg['variance_multiplier']==3. and cfg['contrast_strength']==1.
        assert cfg['rotational'] is False and cfg['measurement_noise']==0.
        base_std=cfg['kick_std'];high_multiplier=1+(cfg['variance_multiplier']-1)*cfg['contrast_strength']
        config={'family':'FHN post-integration Gaussian kicks','fixed_config':cfg,
                'std_high_multiplier':high_multiplier,'no_noise_law_modification':True}
    std=np.full((9,4),base_std,np.float64)
    for j in range(4):std[j+5,j]*=high_multiplier
    assert sha(rec['observations']['path'])==rec['observations']['sha256']
    assert sha(rec['private_targets']['path'])==rec['private_targets']['sha256']
    oo=next(r for r in old['oracles']if r['slot']==slot)
    mv=np.asarray(oo['physical_TRAIN_mean']);sv=np.asarray(oo['physical_TRAIN_std'])
    assert np.min(sv)>0
    dvar=std**2/sv[None,:]**2
    physical={s:read(rec['private_targets']['path'],s+'__physical')for s in ('train','cal')}
    regimes={s:read(rec['observations']['path'],s+'__regimes')for s in ('train','cal')}
    v={s:(physical[s]-mv)/sv for s in ('train','cal')}
    for s in ('train','cal'):
        assert regimes[s].shape==physical[s].shape[:1]+(physical[s].shape[1]-1,)
        assert regimes[s].min()==0 and regimes[s].max()==8
    data[slot]={'v':v,'regimes':regimes,'D_variance':dvar}
    laws.append({'slot':slot,'config':config,'physical_TRAIN_std':sv.tolist(),
        'innovation_std_by_regime_physical':std.tolist(),'innovation_variance_by_regime_state_gauge':dvar.tolist(),
        'transition_indexing':'regimes[:,t] drives physical[:,t+1]; residual uses next state only',
        'manifest':instance['physics_manifest']})

rows=[];cholesky_count=0;solve_count=0
for prior in old['rows']:
    assert prior['status']=='complete'
    slot=prior['slot'];rep=prior['rep'];cached=data[slot]
    folder=BASE/'execution/jobs/representation/OursFine4'/slot/f'seed{rep}'
    mapping=prior['CAL_axis_mapping'];perm=mapping['matched_indices']
    slopes=np.asarray(mapping['slopes']);intercept=np.asarray(mapping['intercept'])
    mw=np.asarray(prior['Fine_aligned_TRAIN_mean']);sw=np.asarray(prior['Fine_aligned_TRAIN_std'])
    raw={s:read(folder/'FEATURES.npz',s)for s in ('train','cal')}
    w={s:(raw[s][...,perm]*slopes+intercept-mw)/sw for s in ('train','cal')}
    A=np.asarray(prior['OLS']['A']);Q=np.asarray(prior['polar']['Q']);b=np.asarray(prior['OLS']['b'])
    entry={'slot':slot,'rep':rep,'cores':{}}
    for core_name,B in (('affine',A),('polar',Q)):
        C=[(B*du[None,:])@B.T for du in cached['D_variance']]
        L=[];g=[]
        for cov in C:
            ll=np.linalg.cholesky(cov);cholesky_count+=1;L.append(ll)
            g.append(float(.5*(np.log(np.diag(cov)).sum()-2*np.log(np.diag(ll)).sum())))
        residue={s:w[s]-(cached['v'][s]@B.T+b)for s in ('train','cal')}
        if core_name=='affine':
            check=(residue['train']**2).mean((0,1))
            assert np.max(abs(check-np.asarray(prior['OLS']['TRAIN_residual_MSE_by_axis'])))<1e-12
        core={'matrix':B.tolist(),'intercept':b.tolist(),
            'C_by_regime_state_gauge':[c.tolist()for c in C],
            'Gaussian_core_contrast_by_regime':g,'Gaussian_core_contrast_uniform9':float(np.mean(g)),
            'contrast_scope':'fixed stored core, analytic conditional Gaussian-head profile contrast; not measured actual Fine NLL',
            'splits':{}}
        for split in ('train','cal'):
            rr=residue[split][:,1:,:].reshape(-1,4)
            u=cached['regimes'][split].reshape(-1)
            assert len(rr)==len(u)
            groups=[]
            for regime in range(9):
                block=rr[u==regime];assert len(block)>0
                whitened=np.linalg.solve(L[regime],block.T).T;solve_count+=1
                sample_mean=block.mean(0);white_mean=whitened.mean(0)
                sq=float((whitened*whitened).sum(1).mean())
                centered=whitened-white_mean
                sq_centered=float((centered*centered).sum(1).mean())
                meanenergy=float(white_mean@white_mean)
                assert abs(sq-(sq_centered+meanenergy))<1e-11
                n=float(np.sqrt(sq));nc=float(np.sqrt(sq_centered))
                groups.append({'regime':regime,'next_states':len(block),'empirical_frequency':len(block)/len(rr),
                    'raw_nu2_plugin':sq,'raw_nu_plugin':n,'raw_N8_delta_plugin_if_nu_lt1':budget(n),
                    'centered_by_regime_nu2_plugin':sq_centered,'centered_by_regime_nu_plugin':nc,
                    'centered_N8_delta_plugin_if_nu_lt1':budget(nc),'sample_residual_mean':sample_mean.tolist(),
                    'whitened_sample_mean_energy':meanenergy,'Gaussian_core_contrast':g[regime],
                    'conditional_innovation_centering_estimated':False,
                    'raw_status':'empirical_plugin_nu_lt1'if n<1 else 'N/A_empirical_nu_ge1',
                    'centered_status':'empirical_plugin_nu_lt1'if nc<1 else 'N/A_empirical_nu_ge1'})
            freq=np.asarray([x['empirical_frequency']for x in groups])
            aggregate={}
            for label,nkey,dkey in [('raw','raw_nu_plugin','raw_N8_delta_plugin_if_nu_lt1'),
                 ('regime_centered','centered_by_regime_nu_plugin','centered_N8_delta_plugin_if_nu_lt1')]:
                ns=np.asarray([x[nkey]for x in groups]);ds=[x[dkey]for x in groups];valid=all(x is not None for x in ds)
                aggregate[label]={'regimes_nu_lt1':int(sum(ns<1)),'min_nu_plugin':float(ns.min()),
                    'max_nu_plugin':float(ns.max()),'nu_RMS_empirical_frequency':float(np.sqrt(freq@(ns*ns))),
                    'N8_delta_plugin_all9_empirical_frequency':float(freq@np.asarray(ds))if valid else None,
                    'N8_delta_plugin_all9_uniform9':float(np.mean(ds))if valid else None,
                    'full9_status':'empirical_plugin_only_not_population_bound'if valid else 'N/A_at_least_one_regime_nu_ge1'}
            core['splits'][split]={'shape_next':list(residue[split][:,1:,:].shape),'all_next_states':len(rr),
                'groups_all9':groups,'aggregate':aggregate,
                'Gaussian_core_contrast_empirical_regime_frequency':float(freq@np.asarray(g)),
                'empirical_N10_expressions_not_conditional_innovation_residual':True,
                'CAL_fixed_TRAIN_scales_and_previously_used_for_model_selection':split=='cal'}
        entry['cores'][core_name]=core
    rows.append(entry)

assert len(rows)==12 and cholesky_count==216 and solve_count==432 and sha(SCOPE)==scope_sha
summary={}
for core_name in ('affine','polar'):
    summary[core_name]={}
    for split in ('train','cal'):
        dd=[r['cores'][core_name]['splits'][split]for r in rows]
        summary[core_name][split]={label:{
            'parents_all9_nu_lt1':sum(d['aggregate'][label]['regimes_nu_lt1']==9 for d in dd),
            'groups_nu_lt1_of108':sum(d['aggregate'][label]['regimes_nu_lt1']for d in dd),
            'minimum_group_nu':min(d['aggregate'][label]['min_nu_plugin']for d in dd),
            'maximum_group_nu':max(d['aggregate'][label]['max_nu_plugin']for d in dd)}for label in ('raw','regime_centered')}
report={'scope':'empirical N10 upper-expression plug-ins on cached residuals, not conditional innovation residuals or population risk bounds',
    'scope_sha256_before_run':scope_sha,'script_sha256':sha(__file__),'stored_core_audit':str(OLD),
    'stored_core_audit_sha256':sha(OLD),'source_hashes_matching_training_release':source_hashes,
    'noise_laws_and_state_gauge':laws,'rows':rows,'summary':summary,'accesses':ACCESS,
    'cost':{'new_OLS':0,'new_polar':0,'conditional_mean_models_fitted':0,'Cholesky4':cholesky_count,
        'group_whitening_solve4':solve_count,'new_NN_training':0,'new_model_forward':0,'PRNG_calls':0,
        'new_physical_data':0,'checkpoint_deserializations':0,'FINAL_array_reads':0,'GPU_jobs':0,
        'elapsed_seconds':time.perf_counter()-began}}
(OUT/'INNOVATION_BUDGET_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
compact=[]
for r in rows:
    x={'slot':r['slot'],'rep':r['rep']}
    for core_name in ('affine','polar'):
        core=r['cores'][core_name];x[core_name]={'Fcore_uniform9':core['Gaussian_core_contrast_uniform9']}
        for split in ('train','cal'):
            a=core['splits'][split]['aggregate'];x[core_name][split]={k:{kk:v for kk,v in vv.items()if kk in (
                'regimes_nu_lt1','max_nu_plugin','N8_delta_plugin_all9_empirical_frequency')}for k,vv in a.items()}
    compact.append(x)
print(json.dumps({'summary':summary,'rows':compact,'cost':report['cost']},ensure_ascii=False,indent=2))
