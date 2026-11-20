"""Decision rules with no simulator inputs except the shared known noise scale."""

import math
import numpy as np


def chi8_radius(alpha):
    """Invert the exact even-degree chi-square survival function."""
    lo, hi = 0., 100.
    for _ in range(60):
        x = (lo + hi)/2
        survival = math.exp(-x)*(1+x+x*x/2+x*x*x/6)
        if survival > alpha:
            lo = x
        else:
            hi = x
    return math.sqrt(2*hi)


def calibrate(initial, observed_23, sigma, alpha_beta):
    transpose = initial.swapaxes(1, 2)
    gram = transpose @ initial
    ginv = np.linalg.inv(gram)
    estimated_drift = np.linalg.solve(gram, transpose @ observed_23).swapaxes(1, 2)
    direction = estimated_drift[:,:,0]*np.array([-1.,1.])
    lengths = np.linalg.norm(direction,axis=1)
    radius = sigma*np.sqrt(2*math.log(1/alpha_beta)*ginv[:,0,0])
    bounded = lengths > radius
    beta = math.pi/4-np.arctan2(direction[:,1],direction[:,0])
    width = np.full_like(beta,np.nan)
    width[bounded] = np.arcsin(radius[bounded]/lengths[bounded])
    return estimated_drift,ginv,beta,width


def context_sums(states, contexts):
    w2 = np.where(contexts==3,1,np.where(contexts==4,-1,0))
    w3 = np.where(contexts==5,1,np.where(contexts==6,-1,0))
    active = (np.abs(w2)+np.abs(w3)).sum(0)
    stats = [(np.einsum('tb,tbi->bi',w,states[:-1]),
              np.einsum('tb,tbi->bi',w,states[1:,:,1:3])) for w in (w2,w3)]
    return active,stats


def scores(states, contexts, drift, ginv, sigma, alpha_drift, alpha_score):
    active, stats = context_sums(states, contexts)
    residual, leverage = [],[]
    for start,end in stats:
        residual.append(end-np.einsum('bji,bi->bj',drift,start))
        leverage.append(np.sqrt(np.maximum(np.einsum('bi,bij,bj->b',start,ginv,start),0)))
    m = np.maximum(active,1)
    direction = (residual[0]+residual[1][:,::-1]*np.array([1.,-1.]))/(sigma*m[:,None])
    drift_penalty = chi8_radius(alpha_drift)*(leverage[0]+leverage[1])/m
    sample_penalty = np.sqrt(2*math.log(1/alpha_score)/m)
    return direction,drift_penalty,sample_penalty,active


def decide(direction, beta, width, common_radius, active):
    projection = -np.sin(beta)*direction[:,0]+np.cos(beta)*direction[:,1]
    radius = common_radius+2*np.sin(width/2)*np.linalg.norm(direction,axis=1)
    eligible = (active>0)&np.isfinite(radius)
    sign = (eligible&(projection>radius)).astype(int)-(eligible&(projection< -radius)).astype(int)
    return sign, projection, radius
