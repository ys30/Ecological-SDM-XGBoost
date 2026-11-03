#!/usr/bin/env python3
"""Audit tied-score AP on the 23 archived matched taxon–fold groups.

Reproduces legacy stable-order AP, and computes score-threshold/grouped AP
equivalent to sklearn.average_precision_score for binary labels.
Does not overwrite archived results.
"""
import csv, math
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    with (ROOT/"results"/name).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def key(r):return tuple(r[k] for k in ("id","plot","fold","observed"))
def index(rows):
    d={key(r):r for r in rows}
    assert len(d)==len(rows),"duplicate prediction key"
    return d
bay=index(load("bayes_spatial_cv_predictions.csv"))
xgb=index(load("spatial_2d_block_cv_predictions.csv"))
maxent=index(load("maxent_spatial_2d_predictions.csv"))
matched=load("matched_bayesian_maxent_xgboost_folds.csv")
assert len(matched)==23 and set(maxent).issubset(bay) and set(maxent).issubset(xgb)
groups=defaultdict(list)
for k,r in maxent.items():
    groups[(r["species"],int(r["fold"]))].append((int(r["observed"]),{
      "bayesian":float(bay[k]["mean_probability"]),
      "maxent":float(r["relative_suitability"]),
      "xgboost":float(xgb[k]["predicted_probability"])}))
def legacy_ap(y,p):
    order=sorted(range(len(y)),key=lambda i:(-p[i],i))
    hits=0;total=0.
    for rank,i in enumerate(order,1):
        if y[i]:hits+=1;total+=hits/rank
    return total/sum(y)
def grouped_ap(y,p):
    pairs=sorted(zip(p,y),reverse=True)
    total=sum(y);tp=0;seen=0;acc=0.;ties=0;mixed=0
    i=0
    while i<len(pairs):
        j=i+1
        while j<len(pairs) and pairs[j][0]==pairs[i][0]:j+=1
        n=j-i;positive=sum(v for _,v in pairs[i:j])
        if n>1:ties+=n-1
        if n>1 and 0<positive<n:mixed+=1
        seen+=n;tp+=positive
        acc+=positive*tp/seen
        i=j
    return acc/total,ties,mixed
rows=[]
for rec in matched:
    cell=(rec["species"],int(rec["fold"]))
    obs=groups[cell];y=[v[0] for v in obs]
    assert len(obs)==int(rec["n"]) and sum(y)==int(rec["positives"])
    for model in ("bayesian","maxent","xgboost"):
        p=[v[1][model] for v in obs]
        assert all(math.isfinite(v) for v in p)
        legacy=legacy_ap(y,p)
        archived=float(rec[model+"_ap"])
        assert math.isclose(legacy,archived,abs_tol=1e-12,rel_tol=0),(cell,model,legacy,archived)
        grouped,ties,mixed=grouped_ap(y,p)
        rows.append(dict(species=cell[0],fold=cell[1],model=model,n=len(y),
          positives=sum(y),tied_extra_records=ties,mixed_label_tie_groups=mixed,
          archived_ap=archived,grouped_ap=grouped,
          grouped_minus_archived_ap=grouped-archived))
assert len(rows)==69
path=ROOT/"results"/"matched_ap_tie_sensitivity.csv"
with path.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for model in ("bayesian","maxent","xgboost"):
    a=[r for r in rows if r["model"]==model]
    print(model,"tied cells",sum(r["tied_extra_records"]>0 for r in a),
          "mixed-label tie cells",sum(r["mixed_label_tie_groups"]>0 for r in a),
          "max absolute AP shift",max(abs(r["grouped_minus_archived_ap"]) for r in a))
print("PASS: 69 legacy AP values reproduced; grouped-threshold AP sensitivity exported")
