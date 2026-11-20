"""Deterministic matrix check of the analytic increment-loss counterexample."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rates = np.array([.93, .9051, .9049, .88])
lags = np.array([4, 16, 32])
a = .1
r = 1 / np.sqrt(1 + a*a)
D = (2 - 2*rates[None]**lags[:, None]).mean(axis=0)
N = np.diag(D)
rows = []
for donor in (3, 0):
    transform = np.eye(4)
    transform[2] = 0
    transform[2, 2] = r
    transform[2, donor] = r*a
    error = transform - np.eye(4)
    covariance = transform @ transform.T
    state_mse = np.diag(error @ error.T)
    increment_mse = np.diag(error @ N @ error.T)
    predicted_scores = np.diag(transform @ N @ transform.T)/np.diag(covariance)
    assert np.allclose(np.diag(covariance), 1, atol=1e-14, rtol=0)
    assert abs(np.linalg.det(transform)-r)<1e-14
    assert abs(state_mse[2]-(2-2*r))<1e-14
    assert abs(increment_mse[2]-((r-1)**2*D[2]+r*r*a*a*D[donor]))<1e-14
    assert abs(predicted_scores[2]-(D[2]+a*a*D[donor])/(1+a*a))<1e-14
    rows.append({"donor_axis": donor+1, "state_r2": (1-state_mse).tolist(),
        "state_mse": state_mse.tolist(), "increment_mse": increment_mse.tolist(),
        "predicted_scores": predicted_scores.tolist(), "predicted_top2": (np.argsort(predicted_scores)[:2]+1).tolist(),
        "determinant": float(np.linalg.det(transform)),
        "state_loss": float(state_mse.mean()),
        "increment_half_loss": float(increment_mse.mean()/2),
        "lambda1_objective": float((state_mse.mean()+increment_mse.mean()/2)/2)})
assert np.allclose(rows[0]["state_mse"],rows[1]["state_mse"],atol=1e-14,rtol=0)
assert rows[0]["predicted_top2"]==[1,2] and rows[1]["predicted_top2"]==[1,3]
assert rows[1]["increment_half_loss"]<rows[0]["increment_half_loss"]
out={"status":"passed", "kind":"deterministic analytic verification, not a trained-model experiment",
    "rates":rates.tolist(),"lags":lags.tolist(),"a":a,"physical_scores":D.tolist(),
    "physical_top2":(np.argsort(D)[:2]+1).tolist(),"candidates":rows,
    "critical_a":float(np.sqrt((D[2]-D[1])/(D[1]-D[0])))}
(HERE/'THEORY_CHECK.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({"status":out['status'],"critical_a":out['critical_a']}))
