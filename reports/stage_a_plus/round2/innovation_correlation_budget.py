"""Per-axis empirical N10 moment expressions for a correlation risk bridge.

No fitting, random draws, solves, neural forward, or new data. All covariance
cores and TRAIN scales come from completed audits. This is not certification.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2'
SCOPE=OUT/'INNOVATION_CORRELATION_SCOPE.md';ACCESS=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,key=None):
    p=Path(p).resolve();assert 'FINAL'not in str(p)
    ACCESS.append({'path':str(p),'member':key,'sha256':sha(p)})
    if key is None:return json.loads(p.read_text())
    assert key in ('train','cal','train__physical','cal__physical','train__regimes','cal__regimes')
    with np.load(p,allow_pickle=False)as f:return np.asarray(f[key],np.int64 if key.endswith('regimes')else np.float64)

assert SCOPE.exists();scope_sha=sha(SCOPE);began=time.perf_counter()
old=read(OUT/'NONLINEAR_REMAINDER_AUDIT.json');n8=read(OUT/'INNOVATION_BUDGET_AUDIT.json')
data={};rows=[]
for oracle in old['oracles']:
    slot=oracle['slot'];m=np.asarray(oracle['physical_TRAIN_mean']);s=np.asarray(oracle['physical_TRAIN_std'])
    data[slot]={'v':{sp:(read(BASE/'private/TRAIN_CAL'/f'{slot}.npz',sp+'__physical')-m)/s for sp in ('train','cal')},
        'u':{sp:read(BASE/'data/TRAIN_CAL'/f'{slot}.npz',sp+'__regimes')for sp in ('train','cal')}}
for parent in old['rows']:
    slot=parent['slot'];rep=parent['rep'];target=parent['target'];other=[j for j in range(4)if j not in target]
    source=next(r for r in n8['rows']if r['slot']==slot and r['rep']==rep)
    maps=parent['CAL_axis_mapping'];perm=maps['matched_indices'];slopes=np.asarray(maps['slopes']);inter=np.asarray(maps['intercept'])
    mw=np.asarray(parent['Fine_aligned_TRAIN_mean']);sw=np.asarray(parent['Fine_aligned_TRAIN_std'])
    folder=BASE/'execution/jobs/representation/OursFine4'/slot/f'seed{rep}'
    w={sp:(read(folder/'FEATURES.npz',sp)[...,perm]*slopes+inter-mw)/sw for sp in ('train','cal')}
    entry={'slot':slot,'rep':rep,'target':target,'cores':{}}
    for core_name in ('affine','polar'):
        core=source['cores'][core_name];B=np.asarray(core['matrix']);b=np.asarray(core['intercept']);C=np.asarray(core['C_by_regime_state_gauge'])
        diag=np.diagonal(C,axis1=1,axis2=2);assert np.min(diag)>0
        output={}
        for split in ('train','cal'):
            rr=(w[split]-(data[slot]['v'][split]@B.T+b))[:,1:,:].reshape(-1,4);u=data[slot]['u'][split].reshape(-1)
            tau=[];tauc=[];groups=[]
            for regime in range(9):
                r=rr[u==regime];assert len(r)>0
                raw2=(r*r).mean(0);var=r.var(0,ddof=0)
                tt=np.sqrt(raw2/diag[regime]);tc=np.sqrt(var/diag[regime]);tau.append(tt);tauc.append(tc)
                groups.append({'regime':regime,'next_states':len(r),'empirical_frequency':len(r)/len(rr),
                    'residual_second_moment_by_axis':raw2.tolist(),'residual_u_centered_variance_by_axis':var.tolist(),
                    'core_diagonal_variance':diag[regime].tolist(),'tau_raw_plugin':tt.tolist(),'tau_u_centered_plugin':tc.tolist()})
            tau=np.asarray(tau);tauc=np.asarray(tauc);freq=np.asarray([g['empirical_frequency']for g in groups]);pairs=[]
            for a in target:
                for baxis in other:
                    rho=C[:,a,baxis]/np.sqrt(diag[:,a]*diag[:,baxis]);pair={'a':a,'b':baxis,'core_rho_by_regime':rho.tolist()}
                    for lab,tauu in [('raw',tau),('u_centered',tauc)]:
                        ee=2*(tauu[:,a]+tauu[:,baxis]);cap=np.minimum(ee,2.)
                        pair[lab]={'e_by_regime':ee.tolist(),'e_capped2_by_regime':cap.tolist()}
                        for wname,weights in [('uniform9',np.full(9,1/9)),('empirical',freq)]:
                            rho_rms=float(np.sqrt(weights@(rho*rho)));er=float(np.sqrt(weights@(ee*ee)));ec=float(np.sqrt(weights@(cap*cap)))
                            signal=max(0.,rho_rms-ec)
                            pair[lab][wname]={'core_rho_rms':rho_rms,'e_rms_plugin':er,'e_capped2_rms_plugin':ec,
                                'bound_saturated_no_positive_risk_budget':ec>=rho_rms,
                                'formal_J_lower_evaluated_at_plugins_not_certificate':float(-.5*np.log1p(-signal*signal))}
                    pairs.append(pair)
            output[split]={'groups_all9':groups,'all_four_pairs':pairs,'actual_conditional_residual_or_actual_Fine_risk_measured':False}
        entry['cores'][core_name]=output
    rows.append(entry)
summary={}
for name in ('affine','polar'):
    summary[name]={}
    for split in ('train','cal'):
        pp=[p for r in rows for p in r['cores'][name][split]['all_four_pairs']]
        summary[name][split]={lab:{ww:{'pair_count':len(pp),
            'saturated_zero_positive_risk_budget':sum(p[lab][ww]['bound_saturated_no_positive_risk_budget']for p in pp),
            'minimum_e_rms_plugin':min(p[lab][ww]['e_rms_plugin']for p in pp),
            'maximum_e_rms_plugin':max(p[lab][ww]['e_rms_plugin']for p in pp),
            'minimum_core_rho_rms':min(p[lab][ww]['core_rho_rms']for p in pp),
            'maximum_core_rho_rms':max(p[lab][ww]['core_rho_rms']for p in pp)}for ww in ('uniform9','empirical')}for lab in ('raw','u_centered')}
report={'scope':'empirical per-axis N10-expression plug-ins, not true conditional residual bounds or actual 2D-Q certification',
    'scope_sha256':scope_sha,'script_sha256':sha(__file__),'source_audit_sha256':sha(OUT/'INNOVATION_BUDGET_AUDIT.json'),
    'rows':rows,'summary':summary,'accesses':ACCESS,'cost':{'new_OLS_polar_Cholesky_solve_NN_PRNG_data_FINAL_GPU':0,
        'elapsed_seconds':time.perf_counter()-began}}
assert len(rows)==12 and sha(SCOPE)==scope_sha
(OUT/'INNOVATION_CORRELATION_BUDGET.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
