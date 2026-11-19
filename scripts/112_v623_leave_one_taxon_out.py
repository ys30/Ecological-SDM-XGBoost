#!/usr/bin/env python3
"""Leave-one-taxon-out robustness for equal-taxon matched OOF contrasts.

Descriptive influence diagnostics, NOT independent inferential confidence intervals.
"""
import csv,json
from pathlib import Path
import numpy as np
BASE=Path("results/v623_matched_three_model")
METRICS=("logloss","brier","auc","ap")
MODELS=("glm_linear","glm_quadratic")
def read(path):
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    manifest=json.loads((BASE/"manifest.json").read_text())
    assert manifest["status"]=="SAME_OUTER_FOLDS_AND_BUFFER_VERIFIED"
    data=read(BASE/"taxon_pooled_oof_scores.csv")
    taxa=sorted({r["taxon"] for r in data})
    assert len(taxa)==8 and len(data)==24
    idx={(r["taxon"],r["model"]):r for r in data}
    assert len(idx)==len(data)
    rows=[]
    for comparator in MODELS:
        for metric in METRICS:
            differences={t:float(idx[t,"xgboost"][metric])-float(idx[t,comparator][metric]) for t in taxa}
            full=float(np.mean(list(differences.values())))
            for excluded in taxa:
                remaining=[v for t,v in differences.items() if t!=excluded]
                reduced=float(np.mean(remaining))
                rows.append(dict(comparison="xgboost_minus_"+comparator,metric=metric,
                    excluded_taxon=excluded,full_eight_taxon_delta=full,
                    seven_taxon_delta=reduced,shift_from_full=reduced-full,
                    sign_reversed=(np.sign(reduced)!=np.sign(full)) if full!=0 else ""))
    write(BASE/"leave_one_taxon_out_influence.csv",rows)
    summary=[]
    for comparator in MODELS:
        for metric in METRICS:
            rr=[r for r in rows if r["comparison"]=="xgboost_minus_"+comparator and r["metric"]==metric]
            full=rr[0]["full_eight_taxon_delta"]
            summary.append(dict(comparison="xgboost_minus_"+comparator,metric=metric,
                full_delta=full,min_seven_taxon_delta=min(r["seven_taxon_delta"] for r in rr),
                max_seven_taxon_delta=max(r["seven_taxon_delta"] for r in rr),
                n_sign_reversals=sum(bool(r["sign_reversed"]) for r in rr),
                max_absolute_shift=max(abs(r["shift_from_full"]) for r in rr)))
    write(BASE/"leave_one_taxon_out_summary.csv",summary)
    print(json.dumps({"status":"DESCRIPTIVE_INFLUENCE_ONLY","taxa":len(taxa),
        "contrasts":len(summary),"leave_one_out_rows":len(rows),"summary":summary,
        "caveat":"Seven-taxon ranges are not confidence intervals; no independent spatial sampling is assumed."},indent=2))
if __name__=="__main__":main()
