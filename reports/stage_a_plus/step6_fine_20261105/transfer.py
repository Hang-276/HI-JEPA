"""Freeze physical-score forecasts from TRAIN and cached CAL Fine outputs, then score CAL truth."""
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
from itertools import permutations
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
BASE=ROOT/'outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2'
SLOTS=('ar_rich_fresh1','ar_rich_fresh2','fhn_fresh_eps008','fhn_fresh_eps05')
LAGS=(4,16,32)
ACCESS=[]
PHASE=None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_json(obj):
    if isinstance(obj,np.ndarray):
        return obj.tolist()
    if isinstance(obj,np.generic):
        return obj.item()
    raise TypeError(type(obj))


def write_new(name,obj):
    file=OUT/name
    if file.exists():
        raise FileExistsError(file)
    tmp=file.with_suffix('.part')
    tmp.write_text(json.dumps(obj,default=as_json,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    tmp.replace(file)


def read_json(path):
    return json.loads(path.read_text())


def array(path, key):
    allowed={'train','cal','train__physical'} if PHASE=='freeze' else {'cal__physical','cal'}
    assert key in allowed and BASE in path.parents and 'FINAL' not in str(path)
    with np.load(path,allow_pickle=False) as archive:
        x=archive[key]
    assert x.ndim==3 and x.shape[-1]==4 and np.isfinite(x).all()
    ACCESS.append(dict(path=str(path.relative_to(ROOT)),member=key,shape=x.shape,dtype=str(x.dtype),
                       member_bytes_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    return x.astype(np.float64)


def model_dir(slot,rep):
    return BASE/'execution/jobs/representation/OursFine4'/slot/f'seed{rep}'


def phys_file(slot):
    return BASE/'private/TRAIN_CAL'/f'{slot}.npz'


def target(slot):
    return [0,1] if slot.startswith('ar_') else [1,3]


def scale(x):
    mu=x.mean((0,1))
    sigma=x.std((0,1),ddof=0)
    assert min(sigma)>0
    return mu,sigma


def dynamic(x,y=None):
    if y is None:
        y=x
    return np.stack([((x[:,h:]-x[:,:-h])*(y[:,h:]-y[:,:-h])).mean(axis=1) for h in LAGS],axis=1)


def match(z,raw):
    C=z.reshape(-1,4).T @ raw.reshape(-1,4) / (z.shape[0]*z.shape[1])
    choices=sorted(((sum(C[j,k]**2 for j,k in enumerate(p)),p) for p in permutations(range(4))),
                   key=lambda item:(-item[0],item[1]))
    gap=choices[0][0]-choices[1][0]
    perm=np.array(choices[0][1]); sg=np.sign(C[np.arange(4),perm])
    assert np.all(sg!=0)
    return perm,sg,gap


def order(s):
    s=np.asarray(s)
    ids=np.argsort(s,kind='stable')
    return sorted(ids[:2].tolist()) if s[ids[1]] < s[ids[2]] else None


def r2_on_axis(z,y):
    zz=z-z.mean(1,keepdims=True)
    yy=y-y.mean(1,keepdims=True)
    denom=(zz*zz).mean(1)*(yy*yy).mean(1)
    assert np.min(denom)>0
    return (zz*yy).mean(1)**2/denom


def check_parent(slot,rep):
    folder=model_dir(slot,rep)
    comp=read_json(folder/'COMPLETED.json')
    info=read_json(folder/'model/Fine.json')
    rec=dict()
    for name,key in [('FEATURES.npz','features'),('G0.json','G0'),('model/Fine_latest.pt','selected_artifact')]:
        path=folder/name
        actual=sha(path)
        assert path.resolve()==Path(comp[key]['path']).resolve() and actual==comp[key]['sha256']
        rec[name]=actual
    assert rec['model/Fine_latest.pt']==info['checkpoint_sha256']
    assert info['selected_model_hash']==comp['selected_tensor_hashes']
    assert comp['valid_t_start']==0 and int(comp['logical_rep'])==rep and comp['slot_id']==slot
    rec['selected_step']=comp['selected_step']
    rec['model_metadata']=sha(folder/'model/Fine.json')
    return rec


def freeze():
    parents=[]
    for slot in SLOTS:
        z0=array(phys_file(slot),'train__physical')
        mz,sz=scale(z0)
        z=(z0-mz)/sz
        physical_train=dynamic(z).mean((0,1))
        for rep in range(3):
            folder=model_dir(slot,rep)
            provenance=check_parent(slot,rep)
            raw=array(folder/'FEATURES.npz','train'); raw_cal=array(folder/'FEATURES.npz','cal')
            assert raw.shape==z.shape
            my,sy=scale(raw)
            native=(raw-my)/sy
            nh=dynamic(native)
            g0=read_json(folder/'G0.json')
            train_native=nh.mean(0).mean(0)
            assert np.max(abs(train_native-np.array(g0['speed_FP64']))) < 1e-11
            assert order(train_native.astype('f4'))==sorted(g0['selected_pair'])
            perm,sign,assignment_gap=match(z,native)
            y=native[...,perm]*sign
            cal=((raw_cal-my)/sy)[...,perm]*sign
            r=(z*y).mean((0,1)); assert min(r)>0
            resid=z-r*y
            qtr=dynamic(y).mean(0)
            T=dynamic(resid).mean(0)
            cross=dynamic(y,resid).mean(0)
            den=np.sqrt(T*qtr)
            cosine=np.divide(cross,den,out=np.zeros_like(cross),where=den>0)
            assert np.max(abs(cosine))<1+1e-11
            tr_reconstruction=(r*r*qtr+T+2*r*cross).mean(0)
            assert np.max(abs(tr_reconstruction-physical_train))<1e-11
            ycal_h=dynamic(cal)
            ys=ycal_h.mean(1)
            size=(r*r*ycal_h+T).mean(1)
            direction=size+(2*r*cosine*np.sqrt(ycal_h*T)).mean(1)
            predictions={'size':size,'direction':direction,'identity':ys,
                         'persistence':np.broadcast_to(physical_train,ys.shape)}
            assert np.isfinite(direction).all()
            parents.append(dict(slot=slot,rep=rep,target=target(slot),provenance=provenance,
                train_mu_z=mz,train_sd_z=sz,train_mu_y=my,train_sd_y=sy,perm=perm,sign=sign,
                assignment_objective_lead=assignment_gap,train_r2=r*r,r=r,T=T,c=cosine,
                train_z=physical_train,train_y=qtr.mean(0),native_g0=g0['selected_pair'],
                train_frozen_target_prediction=order(qtr.mean(0)),cal_fine_scores=ys,predictions=predictions))
    write_new('forecast_lock.json',dict(utc=stamp(),protocol_sha=sha(OUT/'PROTOCOL.md'),
        script_sha=sha(Path(__file__)),stage='before step-6 CAL physical member reads',
        phase_access=ACCESS,parents=parents))
    print('Frozen forecasts',len(parents),'parents',sum(len(p['cal_fine_scores']) for p in parents),'parent-episodes')


def reveal():
    lock=read_json(OUT/'forecast_lock.json')
    assert lock['protocol_sha']==sha(OUT/'PROTOCOL.md') and lock['script_sha']==sha(Path(__file__))
    allrows=[]; parent_rows=[]
    for slot in SLOTS:
        phys=array(phys_file(slot),'cal__physical')
        for p in [row for row in lock['parents'] if row['slot']==slot]:
            rep=p['rep']; folder=model_dir(slot,rep)
            assert sha(folder/'FEATURES.npz')==p['provenance']['FEATURES.npz']
            raw=array(folder/'FEATURES.npz','cal')
            assert raw.shape==phys.shape
            z=(phys-np.array(p['train_mu_z']))/p['train_sd_z']
            y=((raw-p['train_mu_y'])/p['train_sd_y'])[...,p['perm']]*p['sign']
            r=np.array(p['r']);resid=z-y*r
            y_h=dynamic(y); z_h=dynamic(z)
            T=dynamic(resid); c=dynamic(y,resid)
            energy=(r*r*y_h+T).mean(1)
            signed=energy+2*r*c.mean(1)
            zz=z_h.mean(1); yy=y_h.mean(1)
            assert np.max(abs(signed-zz))<1e-11
            assert np.max(abs(yy-p['cal_fine_scores']))<1e-12
            rr=r2_on_axis(z,y)
            matches=read_json(BASE/'evaluation_v2/CAL_maps/representation/OursFine4'/slot/f'seed{rep}/MAPS.json')['maps']['axis_affine']
            pc=p['predictions']
            for i in range(len(y)):
                allrows.append(dict(slot=slot,rep=rep,episode=i,target=p['target'],z=zz[i],y=yy[i],
                    predictions={k:v[i] for k,v in pc.items()},train_r2=p['train_r2'],eval_r2=rr[i],
                    same_support_energy=energy[i],same_support_signed=signed[i],
                    actual_residual_increment_energy=T[i].mean(0),
                    actual_signed_correction=(2*r*c[i].mean(0))))
            parent_rows.append(dict(slot=slot,rep=rep,target=p['target'],
                train_r2=p['train_r2'],eval_r2=r2_on_axis(z.reshape(1,-1,4),y.reshape(1,-1,4))[0],
                train_z=p['train_z'],train_y=p['train_y'],z=zz.mean(0),y=yy.mean(0),
                named_train_y=order(p['train_y'])==p['target'],named_cal_y=order(yy.mean(0))==p['target'],
                named_cal_z=order(zz.mean(0))==p['target'],fine_cal_tracks_physical=order(zz.mean(0))==order(yy.mean(0)),
                train_forecast_fine_set=order(p['train_y']),cal_fine_set=order(yy.mean(0)),cal_physical_set=order(zz.mean(0)),
                previous_cal_assignment_agrees=(p['perm']==matches['matched_indices']),
                previous_cal_sign_agrees=(p['sign']==np.sign(matches['slopes']).tolist()),
                mapping_unique_train=p['assignment_objective_lead']>1e-12,
                predictions={k:np.mean(v,axis=0) for k,v in pc.items()},
                max_decomposition_error=np.max(abs(signed-zz))))
    write_new('results_rows.json',allrows)
    write_new('parent_rows.json',parent_rows)
    write_new('reveal_receipt.json',dict(utc=stamp(),lock_sha=sha(OUT/'forecast_lock.json'),
        truth_reads=ACCESS,rows=len(allrows),parent_rows=len(parent_rows),rows_sha=sha(OUT/'results_rows.json')))
    print('Scored',len(allrows),'existing parent-episodes')

if __name__=='__main__':
    PHASE=sys.argv[1]
    if PHASE=='freeze':
        freeze()
    elif PHASE=='reveal':
        reveal()
    else:
        raise ValueError('choose freeze or reveal')
