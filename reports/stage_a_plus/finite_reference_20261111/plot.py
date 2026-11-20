"""Render the sealed finite-reference results."""
from pathlib import Path
import json
import os
os.environ['MPLCONFIGDIR']='/private/tmp/hcf-mpl-20261111'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
summary=json.loads((root/'SUMMARY.json').read_text())
rows=[r for r in summary['results'] if r['length']==16385]
fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
colors=['#28618A','#C66D3D','#49836E']
for ax in axes:
    ax.set_xscale('log',base=2)
    ax.set_xticks([64,1024,16384,65536],['64','1,024','16,384','65,536'])
    ax.tick_params(axis='x',rotation=25)
    ax.set_xlabel('Independent reference points')
    ax.set_ylim(0,105)
    ax.set_ylabel('Windows (%)')
    ax.grid(axis='y',alpha=.18)
    ax.spines[['top','right']].set_visible(False)

def series(ax,rho,key,label,color):
    data=[r for r in rows if r['rho']==rho]
    n=np.array([r['n'] for r in data])
    y=np.array([r['metrics'][key]['estimate'] for r in data])*100
    ci=np.array([r['metrics'][key]['ci95'] for r in data])*100
    ax.plot(n,y,'o-',label=label,color=color,linewidth=1.8,markersize=4)
    ax.fill_between(n,ci[:,0],ci[:,1],alpha=.12,color=color)

series(axes[0],.995,'identity_error','Physical selection error',colors[0])
series(axes[0],.995,'hidden_disagreement','Hidden physical disagreement',colors[1])
axes[0].set_title('Finite reference errors at a tie')
series(axes[1],.995,'empirical_disagreement','Empirical disagreement',colors[0])
series(axes[1],.995,'physical_disagreement','Physical disagreement',colors[1])
axes[1].set_title('Agreement depends on normalization')
for rho,label,color in zip((.995,.99,.97),('Tied factors','Moderate gap','Large gap'),colors):
    series(axes[2],rho,'safe',label,color)
axes[2].set_title('Coverage with a reference confidence band')
for ax in axes:
    ax.legend(frameon=False,fontsize=8,loc='upper left')
fig.savefig(root/'RESULTS.png',dpi=180)
