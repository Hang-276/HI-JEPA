"""Deterministic matrix identities only; no random data or neural computation.

Verify an exact two-axis risk-to-ordering bridge and construct conditional
Gaussian VARs with the same innovation contrast and score margin but different
coupling geometry. These are analytic counterexamples, not HCF experiment runs.
"""
from pathlib import Path
import json
import math
import numpy as np

OUT = Path(__file__).resolve().parent
H = (4, 16, 32)

def risk(z, tau=.2):
    return .5 * np.log1p(tau*tau/(1-tau*tau)*np.asarray(z))

def c_eta(eta, r=.6):
    return np.mean([(r+eta)**h-(r-eta)**h for h in H])

rows=[]
max_identity_error=0.
for phi_deg in (0.,45.,75.):
    phi=math.radians(phi_deg)
    gamma_target=.01
    wanted=gamma_target/(2*math.cos(phi))
    low,high=0.,.3
    assert c_eta(high)>wanted
    for _ in range(90):
        mid=(low+high)/2
        if c_eta(mid)<wanted: low=mid
        else: high=mid
    eta=(low+high)/2
    N=np.array([[math.cos(phi),math.sin(phi)],
                [math.sin(phi),-math.cos(phi)]])
    A=.6*np.eye(2)+eta*N
    B_cov=np.eye(2)-A@A.T-.1*np.eye(2)
    assert np.linalg.eigvalsh(B_cov).min()>0
    # D(+/-) have mean .1 I; B_cov + mean D makes Sigma=I stationary.
    assert np.max(np.abs(A@A.T+B_cov+.1*np.eye(2)-np.eye(2)))<1e-14
    M=np.mean([2*np.eye(2)-np.linalg.matrix_power(A,h)
               -np.linalg.matrix_power(A.T,h) for h in H],axis=0)
    gamma=float(M[1,1]-M[0,0]); chi=float(2*M[0,1])
    assert abs(gamma-gamma_target)<1e-13
    zcrit=gamma**2/(gamma**2+chi**2)
    theta_crit=.5*math.atan2(gamma,abs(chi))
    for theta in np.linspace(-math.pi/4,math.pi/4,201):
        Q=np.array([[math.cos(theta),math.sin(theta)],
                    [-math.sin(theta),math.cos(theta)]])
        observed=Q@M@Q.T
        measured=observed[1,1]-observed[0,0]
        formula=gamma*math.cos(2*theta)-chi*math.sin(2*theta)
        max_identity_error=max(max_identity_error,abs(measured-formula))
    rows.append({'phi_deg':phi_deg,'eta':eta,'A':A.tolist(),
                 'observed_action_loading_covariance_BBT':B_cov.tolist(),
                 'stationary_covariance':np.eye(2).tolist(),'M':M.tolist(),
                 'gamma':gamma,'chi':chi,'abs_chi_over_gamma':abs(chi)/gamma,
                 'kappa_curv_theta':4*.2**2/(1-.2**2),
                 'kappa_barrier':float(risk(1)),
                 'zcrit':zcrit,'critical_angle_deg':math.degrees(theta_crit),
                 'sharp_population_risk_threshold':float(risk(zcrit))})

# The formula also holds with correlated, unequally normalized outputs.
for rho in (-.6,0.,.55):
    Sigma=np.array([[1.,rho],[rho,1.]])
    M=np.array([[.8,.27],[.27,1.1]])
    gamma=M[1,1]-M[0,0];chi=2*M[0,1]-rho*np.trace(M)
    for theta in np.linspace(-math.pi/4,math.pi/4,201):
        Q=np.array([[math.cos(theta),math.sin(theta)],
                    [-math.sin(theta),math.cos(theta)]])
        score=np.diag(Q@M@Q.T)/np.diag(Q@Sigma@Q.T)
        formula=(gamma*math.cos(2*theta)-chi*math.sin(2*theta))/(1-rho*rho*math.sin(2*theta)**2)
        max_identity_error=max(max_identity_error,abs(score[1]-score[0]-formula))

report={'scope':'analytic matrix checks, no sampled trajectories or model training',
        'H':list(H),'regime_law':'iid +/- with equal weights',
        'innovation_covariances':[[[.12,0],[0,.08]],[[.08,0],[0,.12]]],
        'known_exogenous_actions':'iid standard Gaussian; B B^T given per row',
        'counterexample_rows':rows,'max_matrix_formula_error':max_identity_error,
        'random_draws':0,'model_forward':0,'training_updates':0}
(OUT/'ANALYTIC_BRIDGE_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    theta=np.linspace(0,np.pi/4,400)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    for row in rows:
        phi=row['phi_deg'];g=row['gamma'];x=abs(row['chi'])
        axes[0].plot(np.degrees(theta),(g*np.cos(2*theta)-x*np.sin(2*theta))/g,label=f'phi={phi:g} deg')
        axes[1].scatter(row['critical_angle_deg'],row['sharp_population_risk_threshold'],s=45)
    axes[0].axhline(0,color='black',linewidth=.7)
    axes[0].set(xlabel='Worst-sign rotation angle (degrees)',ylabel='Pair score gap / original gap',title='Same kappa and gamma; different ordering boundary')
    axes[0].legend(frameon=False)
    axes[1].plot(np.degrees(theta),risk(np.sin(2*theta)**2),color='black')
    axes[1].set(xlabel='Rotation angle (degrees)',ylabel='Profiled excess NLL (nats/transition)',title='Identical identification-risk curve')
    fig.savefig(OUT/'analytic_bridge.png',dpi=180)
    plt.close(fig)
except ImportError:
    pass

assert max_identity_error<1e-12
print(json.dumps({'max_identity_error':max_identity_error,
                  'rows':[{k:r[k] for k in ('phi_deg','gamma','chi','critical_angle_deg','sharp_population_risk_threshold')} for r in rows]},indent=2))
