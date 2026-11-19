#!/usr/bin/env python3
"""Outer-fold deletion sensitivity for matched OOF model differences.

Descriptive five-fold influence check, not a spatially valid confidence interval.
"""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss,brier_score_loss
BASE=Path("results/v623_matched_three_model")
def read(p):
    with p.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    assert json.loads((BASE/"manifest.json").read_text())["status"]=="SAME_OUTER_FOLDS_AND_BUFFER_VERIFIED"
    x=read(Path("results/v623_fixed_xgb_outer_oof/outer_oof_predictions.csv"))
    g=read(Path("results/v623_matched_glm_10km/glm_oof_predictions.csv"))
    ref={r["id"]:r for r in x}
    models={"xgboost":ref}
    for name in ("linear","quadratic"):
        rr=[r for r in g if r["variant"]==name]
        models["glm_"+name]={r["id"]:r for r in rr}
    assert len(ref)==9336 and all(set(d)==set(ref) and len(d)==len(ref) for d in models.values())
    taxa=sorted({r["taxon"] for r in x})
    folds=sorted({r["outer_fold"] for r in x})
    assert len(taxa)==8 and len(folds)==5
    for model,d in models.items():
        for id,r in d.items():
            if any(str(r[k])!=str(ref[id][k]) for k in ("y","taxon","outer_fold","plot")):
                raise ValueError("Misaligned model rows")
    rows=[]
    for excluded in ["NONE"]+folds:
        for taxon in taxa:
            ids=[r["id"] for r in x if r["taxon"]==taxon and (excluded=="NONE" or r["outer_fold"]!=excluded)]
            y=np.array([int(ref[id]["y"]) for id in ids])
            assert len(ids)>0
            for model,d in models.items():
                p=np.array([float(d[id]["p"]) for id in ids])
                rows.append(dict(excluded_fold=excluded,taxon=taxon,model=model,n=len(ids),
                    positives=int(y.sum()),logloss=float(log_loss(y,p,labels=[0,1])),
                    brier=float(brier_score_loss(y,p))))
    write(BASE/"leave_one_outer_fold_out_taxon_scores.csv",rows)
    idx={(r["excluded_fold"],r["taxon"],r["model"]):r for r in rows}
    summary=[]
    for comparator in ("glm_linear","glm_quadratic"):
        for metric in ("logloss","brier"):
            full=np.mean([idx["NONE",t,"xgboost"][metric]-idx["NONE",t,comparator][metric] for t in taxa])
            vals=[]
            for excluded in folds:
                delta=float(np.mean([idx[excluded,t,"xgboost"][metric]-idx[excluded,t,comparator][metric] for t in taxa]))
                vals.append(delta)
                summary.append(dict(comparison="xgboost_minus_"+comparator,metric=metric,
                    excluded_fold=excluded,equal_taxon_delta=delta,full_delta=float(full),
                    sign_reversal=bool(np.sign(delta)!=np.sign(full))))
            print(json.dumps(dict(comparison=comparator,metric=metric,full_delta=float(full),
                minimum=float(min(vals)),maximum=float(max(vals)),
                sign_reversals=int(sum(np.sign(v)!=np.sign(full) for v in vals)))))
    write(BASE/"leave_one_outer_fold_out_contrasts.csv",summary)
    (BASE/"fold_influence_notes.json").write_text(json.dumps(dict(
        status="DESCRIPTIVE_SPATIAL_FOLD_INFLUENCE",n_folds=len(folds),n_taxa=len(taxa),
        n_fold_taxon_model_rows=len(rows),n_contrasts=len(summary),
        caveats=["Removing an outer fold changes which test rows contribute; models are NOT retrained.",
                 "Five spatial folds are not independent replicates; extrema are NOT confidence intervals.",
                 "AUC and AP are omitted to avoid undefined metrics for remaining rare-taxon subsets.",
                 "Use only as influence sensitivity, not a spatial block bootstrap."]),indent=2)+"\n")
if __name__=="__main__":main()
