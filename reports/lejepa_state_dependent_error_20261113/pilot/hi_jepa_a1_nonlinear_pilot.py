import importlib.util
from pathlib import Path
import itertools
import json
import numpy as np
import torch

P=Path.cwd()/'reports/lejepa_natural_direction_repair_20261113/loaders.py'
spec=importlib.util.spec_from_file_location("a1_small_pilot_loaders",P)
L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
torch.set_num_threads(1);torch.set_num_interop_threads(1)
master=2026111396
role_ids={'cal':31,'validation':32,'windows':33}
rng=lambda role: np.random.default_rng(np.random.SeedSequence([master,role_ids[role]]))
zcal=rng('cal').standard_normal((16384,4))
zval=rng('validation').standard_normal((16384,4))
innov=rng('windows').standard_normal((64,193,4))
rates=np.array([.93,.906,.904,.88]);zw=innov.copy()
for t in range(1,193): zw[:,t]=zw[:,t-1]*rates+innov[:,t]*np.sqrt(1-rates*rates)
alpha=np.array([a for a in itertools.product(range(4),repeat=4) if sum(a)<=3])
degrees=alpha.sum(1)
def basis(z):
 f=np.stack([np.ones_like(z),z,(z*z-1)/np.sqrt(2),(z*z*z-3*z)/np.sqrt(6)],axis=-1)
 return np.prod(f[...,np.arange(4)[None,:],alpha],axis=-1)
def mse(x):return np.mean(x*x,axis=0)
def moments(x):return np.mean([np.mean((x[:,l:]-x[:,:-l])**2,axis=(0,1)) for l in (4,16,32)],axis=0)
def incs(x):return np.mean([np.mean((x[:,l:]-x[:,:-l])**2,axis=1) for l in (4,16,32)],axis=0)
def pack(x):return np.asarray(x).tolist()
bcal,bval=basis(zcal),basis(zval)
_,ids,cells=L.load_panel()
rows=[]
for cell in cells:
 if cell['layer']!='mlp' or cell['process']!='separated_narrow':continue
 f,_=L.load_readout(cell,ids)
 ycal,yval,yw=map(f,(zcal,zval,zw))
 lcoef=np.linalg.lstsq(np.column_stack([np.ones(len(zcal)),zcal]),ycal,rcond=None)[0]
 bias,scale=lcoef[0],lcoef[1:].diagonal()
 vcal,vval,vw=[(y-bias)/scale for y in (ycal,yval,yw)]
 ecal,eval_,ew=vcal-zcal,vval-zval,vw-zw
 coef=np.linalg.lstsq(bcal,ecal,rcond=None)[0]
 halves=[np.linalg.lstsq(bcal[i:i+8192],ecal[i:i+8192],rcond=None)[0] for i in (0,8192)]
 orth1,orth2=halves
 known_proxy_val=basis(vval)
 known_proxy_w=basis(vw)
 z_w_basis=basis(zw)
 cv_proxy=basis(vcal)
 for j in [1,2]:
  groups={
    'self': (degrees>=2)&(alpha[:,j]==degrees),
    'other_univariate': (degrees>=2)&((alpha>0).sum(1)==1)&(alpha[:,j]==0),
    'mixed': (degrees>=2)&((alpha>0).sum(1)>1),
    'linear_cross': (degrees==1)&(alpha[:,j]==0),
  }
  nonlinear=degrees>=2
  selected_nl=np.flatnonzero(nonlinear)[np.argmax(np.abs(coef[nonlinear,j]))]
  selected_l=np.flatnonzero(groups['linear_cross'])[np.argmax(np.abs(coef[groups['linear_cross'],j]))]
  sel={
    'linear_one': np.arange(len(alpha))==selected_l,
    'nonlinear_one': np.arange(len(alpha))==selected_nl,
    'nonlinear_self': groups['self'],
    'nonlinear_mixed': groups['mixed'],
    'nonlinear_all': nonlinear,
  }
  results={}
  for name, mask in sel.items():
   a=coef[mask,j]
   oracle=bval[:,mask]@a; proxy=known_proxy_val[:,mask]@a
   direct_coef=np.linalg.lstsq(cv_proxy[:,mask],ecal[:,j],rcond=None)[0]
   direct=known_proxy_val[:,mask]@direct_coef
   oracle_w=z_w_basis[...,mask]@a
   proxy_w=known_proxy_w[...,mask]@a
   err=eval_[:,j]; truth_w=zw[...,j]
   old_num=incs(vw[...,j]); phys=incs(truth_w)
   old_div=np.var(vval[:,j]); new_div=np.var(vval[:,j]-proxy)
   results[name]={
     'oracle_delta_mse':float(mse(err)-mse(err-oracle)),
     'proxy_delta_mse':float(mse(err)-mse(err-proxy)),
     'direct_proxy_delta_mse':float(mse(err)-mse(err-direct)),
     'proxy_relative_l2':float(np.sqrt(mse(proxy-oracle)/mse(oracle))) if mse(oracle)>0 else None,
     'oracle_increment_mse_drop':float(moments(ew[...,j])-moments(ew[...,j]-oracle_w)),
     'proxy_increment_mse_drop':float(moments(ew[...,j])-moments(ew[...,j]-proxy_w)),
     'proxy_slow_mae_drop':float(np.abs(old_num/old_div-phys).mean()-np.abs(incs(vw[...,j]-proxy_w)/new_div-phys).mean()),
   }
  evar=mse(eval_[:,j])
  rows.append({'name':cell['name'],'axis':j+1,'baseline_mse':float(evar),
               'basis_energy':{group:float(np.sum(coef[mask,j]**2)) for group,mask in groups.items()},
               'half_sign_stability': bool(np.sign(orth1[selected_nl,j])==np.sign(orth2[selected_nl,j])),
               'half_top_term_stability': bool(np.argmax(abs(orth1[nonlinear,j]))==np.argmax(abs(orth2[nonlinear,j]))),
               'mixed_debiased_energy':float(np.dot(orth1[groups['mixed'],j],orth2[groups['mixed'],j])),
               'self_debiased_energy':float(np.dot(orth1[groups['self'],j],orth2[groups['self'],j])),
               'largest_nl_term':pack(alpha[selected_nl]),'largest_nl_coef':float(coef[selected_nl,j]),
               'largest_linear_term':pack(alpha[selected_l]),'largest_linear_coef':float(coef[selected_l,j]),
               'results':results})
print('n rows',len(rows))
for j in (2,3):
 use=[x for x in rows if x['axis']==j]
 print('axis',j,'base mse median',np.median([q['baseline_mse'] for q in use]))
 for group in ('self','other_univariate','mixed','linear_cross'):
  print('energy',group,np.median([q['basis_energy'][group] for q in use]),'frac',np.median([q['basis_energy'][group]/q['baseline_mse'] for q in use]))
 for method in sel:
  a=[q['results'][method] for q in use]
  print('method',method,{k:float(np.median([x[k] for x in a])) for k in a[0]})
 from collections import Counter
 print('top nonlinear',Counter(tuple(q['largest_nl_term']) for q in use))
 print('top nlinear classification',Counter('self' if q['largest_nl_term'][j-1]==sum(q['largest_nl_term']) else 'mixed' if np.count_nonzero(q['largest_nl_term'])>1 else 'other' for q in use))
Path('/tmp/hi_jepa_a1_nonlinear_pilot.json').write_text(json.dumps(rows,indent=2))
