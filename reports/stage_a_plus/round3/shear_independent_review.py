"""Read-only independent review of saved shear moments and source root branches."""
from pathlib import Path
import importlib.util
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("source_shear_audit",ROOT/"shear_barrier_audit.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # main() is not run; original audit is not rewritten.
data=json.loads((ROOT/"SHEAR_BARRIER_AUDIT.json").read_text())
base={(x["slot"],x["split"]):x for x in data["base_moments"]}
max_score_error=0.0
max_coefficient_error=0.0
max_risk_error=0.0
max_saved_root_gap=0.0
rows_valid=0
rows_invalid=0
for row in data["rows"]:
    b=base[row["slot"],row["split"]]
    sigma=np.asarray(b["Sigma_Q0_TRAIN"],float)
    moment=np.asarray(b["M_Qh"],float)
    original=np.diag(moment)/np.diag(sigma)
    i,j=row["affected_axis"],row["added_axis"]
    target=set(row["target"])
    outsiders=set(range(4))-target
    margin=min(original[k]-original[a] for a in target for k in outsiders)
    if row["status"]=="BASE_TARGET_NOT_STRICT":
        assert margin<=0 and row["strict_risk_barrier"]==0
        assert row["constraints"]==[]
        rows_invalid+=1
        continue
    assert margin>0
    rows_valid+=1
    sign=-1 if i in target else 1
    for constraint in row["constraints"]:
        k=constraint["competing_axis"]
        coeff=np.array([sign*(moment[i,i]-original[k]*sigma[i,i]),
                        sign*(moment[i,j]-original[k]*sigma[i,j]),
                        sign*(moment[j,j]-original[k]*sigma[j,j])])
        saved=np.array([constraint["g"],constraint["beta"],constraint["alpha"]])
        max_coefficient_error=max(max_coefficient_error,float(np.max(abs(coeff-saved))))
        for t in (-1.7,-0.1,0.0,0.45,2.3):
            a=np.eye(4);a[i,j]=t
            transformed=np.diag(a@moment@a.T)/np.diag(a@sigma@a.T)
            poly=coeff[0]+2*coeff[1]*t+coeff[2]*t*t
            denominator=sigma[i,i]+2*t*sigma[i,j]+t*t*sigma[j,j]
            assert denominator>0
            max_score_error=max(max_score_error,
                                abs(sign*(transformed[i]-transformed[k])-poly/denominator))
        if constraint["root"] is not None:
            root=constraint["root"]
            polynomial=coeff[0]+2*coeff[1]*root+coeff[2]*root*root
            den=sigma[i,i]+2*root*sigma[i,j]+root*root*sigma[j,j]
            max_saved_root_gap=max(max_saved_root_gap,abs(polynomial/den))
    if row["status"]=="FINITE_STRICT_TIE":
        t=row["strict_parameter_radius"]
        independent_risk=float(np.log1p(t*t*np.array(row["variance_ratios"])).mean()/2)
        max_risk_error=max(max_risk_error,abs(independent_risk-row["strict_risk_barrier"]))

corner_cases=[
    (1,.4,-1,"crossing_tie"),
    (1,0,-1,"crossing_tie"),
    (1,.5,0,"crossing_tie"),
    (1,0,0,"no_finite_root_limiting_tie"),
    (1,0,1,"no_finite_root_positive_limit"),
    (1,1,1,"tangent_tie"),
    (1,2,1,"crossing_tie"),
    (1,-2,1,"crossing_tie"),
]
corner_results=[]
for g,beta,alpha,expected in corner_cases:
    result=module.nearest_root(g,beta,alpha)
    assert result["kind"]==expected
    if alpha!=0:
        roots=np.roots([alpha,2*beta,g])
    elif beta!=0:
        roots=np.array([-g/(2*beta)])
    else:
        roots=np.array([],complex)
    finite_real=[float(x.real) for x in roots if abs(x.imag)<1e-12]
    if expected=="tangent_tie":
        # Repeated roots are ill-conditioned for a generic companion solver;
        # check the exact rational tangent root rather than a split float pair.
        finite_real=[-beta/alpha]
    if finite_real:
        independent_radius=min(abs(x) for x in finite_real)
        assert abs(result["radius"]-independent_radius)<1e-12
    else:
        assert result["radius"] is None
    corner_results.append({"g_beta_alpha":[g,beta,alpha],"classification":result["kind"],
                           "nearest_radius":result["radius"]})

# A deliberately non-unit diagonal Sigma confirms the source's generalized formula.
sigma=np.array([[2,.3,.1,0],[.3,1.5,.05,.1],[.1,.05,1.2,.02],[0,.1,.02,.8]])
moment=np.array([[1.7,.12,.08,0],[.12,.6,.04,.02],[.08,.04,1.5,.06],[0,.02,.06,1.4]])
assert np.linalg.eigvalsh(sigma).min()>0
s=np.diag(moment)/np.diag(sigma)
general_error=0.0
for i,j,k in ((0,1,2),(1,3,0),(3,2,1)):
    for t in (-1.0,.2,1.7):
        transformed=module.score_at(sigma,moment,i,j,t)
        q=(moment[i,i]-s[k]*sigma[i,i]
           +2*t*(moment[i,j]-s[k]*sigma[i,j])
           +t*t*(moment[j,j]-s[k]*sigma[j,j]))
        denominator=sigma[i,i]+2*t*sigma[i,j]+t*t*sigma[j,j]
        general_error=max(general_error,abs(transformed[i]-transformed[k]-q/denominator))

checks={
    "all_96_rows_reviewed":rows_valid+rows_invalid==96,
    "negative_baseline_rows_are_zero":rows_invalid==12,
    "all_saved_general_coefficients":max_coefficient_error<1e-12,
    "finite_array_general_score_identity":max_score_error<1e-12,
    "saved_roots_are_gap_ties":max_saved_root_gap<1e-10,
    "profile_risk_identity":max_risk_error<1e-12,
    "all_eight_root_branch_fixtures":len(corner_results)==8,
    "nonunit_sigma_score_fixture":general_error<1e-12,
}
checks={key:bool(value) for key,value in checks.items()}
report={"kind":"independent_saved_moment_and_analytic_root_review",
        "original_audit_script_main_executed":False,
        "new_physical_array_reads":0,"new_model_forwards":0,"new_training":0,
        "valid_baseline_rows":rows_valid,"invalid_baseline_rows":rows_invalid,
        "max_general_score_error":max_score_error,
        "max_general_coefficient_error":max_coefficient_error,
        "max_saved_root_gap":max_saved_root_gap,"max_risk_error":max_risk_error,
        "nonunit_sigma_error":general_error,"corner_cases":corner_results,
        "checks":checks,"passed":all(checks.values())}
out=ROOT/"SHEAR_INDEPENDENT_CHECKS.json"
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"passed":report["passed"],"checks":checks,"path":str(out)},
                 ensure_ascii=False,indent=2))
assert report["passed"]
