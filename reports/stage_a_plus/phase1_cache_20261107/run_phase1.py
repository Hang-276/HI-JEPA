"""Protocol-locked single-threaded analysis of the existing Fine caches."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time
import zipfile

for key in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
            'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'step6_fine_20261105'
PROTOCOL = HERE.parent / 'PHASE1_CACHE_REVIEW_PROTOCOL.md'
sys.path.insert(0, str(OLD))
from transfer import (BASE, ROOT, LAGS, SLOTS, as_json, dynamic, match, model_dir,
                      order, phys_file, r2_on_axis, scale, sha)

PILOT_SLOT = SLOTS[0]
B_METHODS = ('b_diag', 'b_fixed', 'b_current')
ATOL = 1e-12


def read(path):
    return json.loads(path.read_text())


def store(filename, obj):
    out = HERE / filename
    if out.exists():
        raise FileExistsError(out)
    part = out.with_suffix('.part')
    part.write_text(json.dumps(obj, default=as_json, ensure_ascii=False,
                               allow_nan=False, separators=(',', ':')) + '\n')
    part.replace(out)


def now():
    return datetime.now(timezone.utc).isoformat()


def max_rss():
    units = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return units if sys.platform == 'darwin' else 1024 * units


def nearly(actual, expected, tol=ATOL, label=''):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tol, err_msg=label)


def member_head(file):
    assert file.read(6) == b'\x93NUMPY'
    version = file.read(2)[0]
    assert version in (1, 2, 3)
    count = struct.unpack('<H' if version == 1 else '<I',
                          file.read(2 if version == 1 else 4))[0]
    assert count <= 8192
    return ast.literal_eval(file.read(count).decode('utf-8' if version == 3 else 'latin1'))


class ArchivedInputs:
    def __init__(self, phase):
        for line in (OLD / 'SHA256SUMS').read_text().splitlines():
            h, name = line.split(maxsplit=1)
            assert sha(OLD / name) == h, name
        self.old_lock = read(OLD / 'forecast_lock.json')
        reveal = read(OLD / 'reveal_receipt.json')
        assert reveal['lock_sha'] == sha(OLD / 'forecast_lock.json')
        self.allowed = ({'train__physical', 'train', 'cal'} if phase == 'freeze'
                        else {'cal', 'cal__physical'})
        self.members = {(r['path'], r['member']): r for r in
                        self.old_lock['phase_access'] + reveal['truth_reads']}
        self.access = []

    def array(self, path, key, first=False):
        rel = str(path.relative_to(ROOT))
        assert BASE in path.parents and key in self.allowed and 'FINAL' not in rel
        saved = self.members[(rel, key)]
        if first:
            with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as member:
                head = member_head(member)
                shape = tuple(head['shape'])
                dtype = np.dtype(head['descr'])
                assert list(shape) == saved['shape'] and not head['fortran_order']
                nbytes = shape[1] * shape[2] * dtype.itemsize
                episode = member.read(nbytes)
                assert len(episode) == nbytes
                digest = hashlib.sha256(episode)
                total = len(episode)
                while True:
                    chunk = member.read(64 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    total += len(chunk)
                assert total == shape[0] * nbytes
                value = np.frombuffer(episode, dtype=dtype).reshape(1, *shape[1:])
                checksum = digest.hexdigest()
        else:
            with np.load(path, allow_pickle=False) as archive:
                value = archive[key]
            assert list(value.shape) == saved['shape']
            checksum = hashlib.sha256(value.tobytes()).hexdigest()
        assert str(value.dtype) == saved['dtype'] and checksum == saved['member_bytes_sha256'], (rel, key)
        assert value.ndim == 3 and value.shape[-1] == 4 and np.isfinite(value).all()
        self.access.append(dict(path=rel, member=key, materialized_episodes=len(value),
                                full_member_sha256=checksum))
        return value.astype(np.float64)


def blocks(x):
    middle = x.shape[1] // 2
    assert min(middle, x.shape[1] - middle) > max(LAGS)
    return x[:, :middle], x[:, middle:]


def moments(y):
    out = []
    for h in LAGS:
        delta = y[:, h:] - y[:, :-h]
        out.append((delta.swapaxes(1, 2) @ delta) / delta.shape[1])
    return np.stack(out, axis=1)


def train_calibrator(y, z):
    yy = y.reshape(-1, 4)
    zz = z.reshape(-1, 4)
    coef, _, rank, singular = np.linalg.lstsq(yy-yy.mean(0), zz-zz.mean(0), rcond=1e-12)
    B = coef.T
    intercept = zz.mean(0)-B@yy.mean(0)
    MT = moments(y).mean(0)
    q = np.diagonal(MT, axis1=-2, axis2=-1)
    r = dict(rank=int(rank), singular_values=singular)
    if rank < 4 or np.any(q <= 0):
        r.update(available=False, reason='rank deficient or nonpositive TRAIN increment energy')
        return r
    root = np.sqrt(q)
    R = MT/(root[..., :, None]*root[..., None, :])
    nearly(np.diagonal(R, axis1=-2, axis2=-1), 1)
    assert np.linalg.eigvalsh(R).min() >= -ATOL
    E = z-(y@B.T+intercept)
    T = dynamic(E).mean(0)
    u = 2*dynamic(y@B.T, E).mean(0)
    nearly(dynamic(y@B.T).mean(0)+T+u, dynamic(z).mean(0))
    r.update(available=True, B=B, intercept=intercept, T=T, R=R,
             train_residual_cross=u)
    return r


def prediction(y, old, cal):
    y_h = dynamic(y)
    r, T, c = (np.array(old[k]) for k in ('r', 'T', 'c'))
    size = (r*r*y_h+T).mean(1)
    pred = dict(identity=y_h.mean(1), size=size,
                direction=size+(2*r*c*np.sqrt(y_h*T)).mean(1),
                persistence=np.broadcast_to(old['train_z'], size.shape))
    if not cal['available']:
        pred.update({k:[None]*len(y) for k in B_METHODS})
        return pred
    B, Tb, R = (np.array(cal[k]) for k in ('B', 'T', 'R'))
    M = moments(y)
    q = np.diagonal(M, axis1=-2, axis2=-1)
    nearly(q, y_h, label='un-centered second moment diagonal')
    root = np.sqrt(q)
    fixed = R[None, ...]*root[..., :, None]*root[..., None, :]
    diag = M*np.eye(4)
    for m in (M, fixed, diag):
        nearly(np.diagonal(m, axis1=-2, axis2=-1), q)
        assert np.linalg.eigvalsh(m).min() >= -ATOL

    def score(matrix):
        return (np.einsum('ji,nhik,jk->nhj', B, matrix, B)+Tb).mean(1)

    pred.update(b_diag=score(diag), b_fixed=score(fixed), b_current=score(M))
    nearly(pred['b_current'], (dynamic(y@B.T)+Tb).mean(1), label='B matrix prediction vs increments')
    expected = np.einsum('ji,nhik,jk->nhj', B, M-fixed, B).mean(1)
    nearly(pred['b_current']-pred['b_fixed'], expected, label='current-minus-fixed isolation')
    return pred


def select_row(pred, episode):
    return {method:values[episode] for method,values in pred.items()}


def freeze(inp, slot, pilot):
    z0 = inp.array(phys_file(slot), 'train__physical')
    mz, sz = scale(z0)
    z = (z0-mz)/sz
    truth_train = dynamic(z).mean(1)
    gamma = np.diff(np.sort(truth_train,axis=-1)[:, 1:3],axis=1).ravel()
    q25 = float(np.quantile(gamma, .25, method='linear'))
    parents = []
    for old in inp.old_lock['parents']:
        if old['slot'] != slot or (pilot and old['rep'] != 0):
            continue
        folder = model_dir(slot,old['rep'])
        raw = inp.array(folder/'FEATURES.npz','train')
        my,sy = scale(raw)
        for k,value in (('train_mu_z',mz),('train_sd_z',sz),('train_mu_y',my),('train_sd_y',sy)):
            nearly(value,old[k],label=k)
        native=(raw-my)/sy
        perm,sg,_=match(z,native)
        assert perm.tolist()==old['perm'] and sg.tolist()==old['sign']
        y=native[...,perm]*sg
        assert y.shape==z.shape
        g0=read(folder/'G0.json')
        g0_speed=dynamic(native).mean(0).mean(0)
        nearly(g0_speed,g0['speed_FP64'])
        assert order(g0_speed.astype('f4'))==old['native_g0']==sorted(g0['selected_pair'])
        nearly((z*y).mean((0,1))**2,old['train_r2'])
        nearly(truth_train.mean(0),old['train_z'])
        cal=train_calibrator(y,z)
        raw_eval=inp.array(folder/'FEATURES.npz','cal',first=pilot)
        ev=((raw_eval-my)/sy)[...,perm]*sg
        pred=prediction(ev,old,cal)
        for method in ('identity','size','direction','persistence'):
            nearly(pred[method],old['predictions'][method][:len(ev)],label=method)
        half_preds=[prediction(part,old,cal) for part in blocks(ev)]
        episodes=[dict(episode=i,fine_top2=order(pred['identity'][i]),
                       predictions=select_row(pred,i),
                       block_predictions=[select_row(x,i) for x in half_preds])
                  for i in range(len(ev))]
        parents.append(dict(rep=old['rep'], target=old['target'], train_r2=old['train_r2'],
                            cal=cal, episodes=episodes))
    assert len(parents)==(1 if pilot else 3)
    return dict(slot=slot,pilot=pilot,train_small_gap_q25=q25,parents=parents)


def margin(s,I):
    if s is None or I is None:
        return None
    return min(s[j] for j in range(4) if j not in I)-max(s[j] for j in I)


def evaluation_window(y,z,cal,predictions,sets):
    physical=dynamic(z).mean(1)
    fine=dynamic(y).mean(1)
    diagnostics=[]
    if cal['available']:
        B,ic,T=(np.array(cal[k]) for k in ('B','intercept','T'))
        core=y@B.T
        E=z-(core+ic)
        cross=2*dynamic(core,E).mean(1)
        d_energy=(dynamic(E)-T).mean(1)
        for n,pred in enumerate(predictions):
            nearly(physical[n]-pred['b_current'],cross[n]+d_energy[n],label='signed residual reconstruction')
    else:
        cross=d_energy=[None]*len(y)
    for n,I in enumerate(sets):
        comparisons=[dict(selected=a, unselected=b, gap=physical[n,b]-physical[n,a])
                     for a in I for b in range(4) if b not in I] if I is not None else []
        row=predictions[n]
        diagnostics.append(dict(physical_scores=physical[n], fine_scores=fine[n],
            physical_top2=order(physical[n]), fine_top2=order(fine[n]),
            physical_margin_original_fine_set=margin(physical[n],I),
            physical_cross_boundary_gaps=comparisons,
            predicted_margins={method:margin(s,I) for method,s in row.items()},
            current_minus_fixed=(np.array(row['b_current'])-row['b_fixed']
                                 if cal['available'] else None),
            true_remaining_correction=(physical[n]-row['b_current'] if cal['available'] else None),
            residual_cross=cross[n], residual_energy_shift=d_energy[n]))
    return diagnostics


def reveal(inp,lock):
    slot,pilot=lock['slot'],lock['pilot']
    z0=inp.array(phys_file(slot),'cal__physical',first=pilot)
    old_rows={(r['slot'],r['rep'],r['episode']):r for r in read(OLD/'results_rows.json')}
    rows=[]
    for parent in lock['parents']:
        rep=parent['rep']
        old=next(x for x in inp.old_lock['parents'] if (x['slot'],x['rep'])==(slot,rep))
        raw=inp.array(model_dir(slot,rep)/'FEATURES.npz','cal',first=pilot)
        assert raw.shape==z0.shape
        y=((raw-old['train_mu_y'])/old['train_sd_y'])[...,old['perm']]*old['sign']
        z=(z0-old['train_mu_z'])/old['train_sd_z']
        e=parent['episodes']
        sets=[x['fine_top2'] for x in e]
        whole=evaluation_window(y,z,parent['cal'],[x['predictions'] for x in e],sets)
        halves=[evaluation_window(yy,zz,parent['cal'],[x['block_predictions'][k] for x in e],sets)
                for k,(yy,zz) in enumerate(zip(blocks(y),blocks(z)))]
        r2=r2_on_axis(z,y)
        for n,x in enumerate(e):
            assert n==x['episode']
            reference=old_rows[(slot,rep,n)]
            ps,ys=whole[n]['physical_scores'],whole[n]['fine_scores']
            for v,k in ((ps,'z'),(ys,'y'),(r2[n],'eval_r2'),(parent['train_r2'],'train_r2')):
                nearly(v,reference[k],label='Step 6 '+k)
            assert order(ys)==sets[n] and order(ps)==order(reference['z'])
            gamma=float(np.sort(ps)[2]-np.sort(ps)[1])
            rows.append(dict(slot=slot,rep=rep,episode=n,target=parent['target'],
                y=ys,z=ps,train_r2=parent['train_r2'],eval_r2=r2[n],predictions=x['predictions'],
                physical_boundary_gap=gamma,small_gap=gamma<=lock['train_small_gap_q25'],
                full_diagnostic=whole[n],block_diagnostics=[h[n] for h in halves]))
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=('freeze','reveal'))
    ap.add_argument('slot',choices=SLOTS)
    ap.add_argument('--pilot',action='store_true')
    args=ap.parse_args()
    phase,slot,pilot=args.phase,args.slot,args.pilot
    if pilot and slot!=PILOT_SLOT:
        raise ValueError('pilot is frozen to the first slot, seed and episode')
    script_sha=sha(Path(__file__))
    protocol_sha=sha(PROTOCOL)
    if not pilot:
        gate=read(HERE/'pilot_reveal.json')
        assert gate['checks_passed'] and gate['script_sha']==script_sha and gate['protocol_sha']==protocol_sha
    sealname='pilot_freeze.json' if pilot else f'freeze_{slot}.json'
    outname=sealname if phase=='freeze' else ('pilot_reveal.json' if pilot else f'reveal_{slot}.json')
    if (HERE/outname).exists():
        raise FileExistsError(outname)
    start=time.perf_counter()
    inp=ArchivedInputs(phase)
    if phase=='freeze':
        output=freeze(inp,slot,pilot)
    else:
        if not pilot:
            for name in SLOTS:
                full=read(HERE/f'freeze_{name}.json')
                assert full['slot']==name and full['script_sha']==script_sha and full['protocol_sha']==protocol_sha
        seal=HERE/sealname
        lock=read(seal)
        assert lock['slot']==slot and lock['pilot']==pilot and lock['script_sha']==script_sha and lock['protocol_sha']==protocol_sha
        assert all(row['member']!='cal__physical' for row in lock['access'])
        output=dict(slot=slot,pilot=pilot,forecast_sha=sha(seal),rows=reveal(inp,lock))
    output.update(protocol_sha=protocol_sha,script_sha=script_sha,utc=now(),access=inp.access,
                  elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=max_rss(),checks_passed=True)
    store(outname,output)
    print(json.dumps(dict(result=outname,elapsed_seconds=output['elapsed_seconds'],
                          peak_rss_bytes=output['peak_rss_bytes'],passed=True)))


if __name__=='__main__':
    main()
