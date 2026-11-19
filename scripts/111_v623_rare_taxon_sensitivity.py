#!/usr/bin/env python3
"""Rare-taxon descriptive sensitivity, using prespecified prevalence ordering only."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path("results/v623_matched_three_model")
def read(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))
def write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    meta=json.loads((ROOT/"manifest.json").read_text())
    assert meta["status"]=="SAME_OUTER_FOLDS_AND_BUFFER_VERIFIED"
    scores=read(ROOT/"taxon_pooled_oof_scores.csv")
    taxa=sorted({r["taxon"] for r in scores})
    assert len(taxa)==8 and len(scores)==24
    by={(r["taxon"],r["model"]):r for r in scores}
    for t in taxa:
        n={int(by[t,m]["n"]) for m in ("xgboost","glm_linear","glm_quadratic")}
        positives={int(by[t,m]["positives"]) for m in ("xgboost","glm_linear","glm_quadratic")}
        if n!={1167} or len(positives)!=1:raise ValueError("Unequal test labels")
    ordered=sorted(taxa,key=lambda t:(int(by[t,"xgboost"]["positives"]),t))
    rare=set(ordered[:4])
    rows=[]
    for t in ordered:
        ref=by[t,"xgboost"]
        for other in ("glm_linear","glm_quadratic"):
            a=by[t,other]
            row=dict(taxon=t,n=int(ref["n"]),positives=int(ref["positives"]),
                     prevalence=int(ref["positives"])/int(ref["n"]),
                     rarity_group="lower_four_prevalence" if t in rare else "upper_four_prevalence",
                     comparison="xgboost_minus_"+other)
            for metric in ("logloss","brier","auc","ap"):
                row["delta_"+metric]=float(ref[metric])-float(a[metric])
            rows.append(row)
    write(ROOT/"rare_taxon_paired_differences.csv",rows)
    summary=[]
    for other in ("glm_linear","glm_quadratic"):
        for group in ("lower_four_prevalence","upper_four_prevalence"):
            sub=[r for r in rows if r["comparison"]=="xgboost_minus_"+other and r["rarity_group"]==group]
            out=dict(comparison="xgboost_minus_"+other,group=group,n_taxa=len(sub),
                     positive_range=f'{min(r["positives"] for r in sub)}-{max(r["positives"] for r in sub)}')
            for metric in ("logloss","brier","auc","ap"):
                values=np.array([r["delta_"+metric] for r in sub])
                out["mean_delta_"+metric]=float(values.mean())
                out["taxa_favor_xgboost_"+metric]=int(np.sum(values<0 if metric in ("logloss","brier") else values>0))
            summary.append(out)
    write(ROOT/"rare_taxon_group_summary.csv",summary)
    (ROOT/"rare_taxon_analysis_notes.json").write_text(json.dumps(dict(
        status="DESCRIPTIVE_ONLY",definition="Four lowest observed prevalence taxa vs remaining four; selection uses outcomes, not model scores.",
        ordered_taxa=ordered,low_prevalence_taxa=sorted(rare),n_taxa_per_group=4,
        caveats=["Small strata: do not interpret group differences as significance.",
                 "AUC/AP pooled across folds within taxon; not eligible for some fold-specific cells.",
                 "Prevalence grouping descriptive, not an independently prespecified biological rarity threshold.",
                 "No external validation or causal algorithm superiority claim."]),indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
