#!/usr/bin/env python3
"""Verify original Bayesian OOF predictions reproduce published matched fold metrics."""
import csv
import math
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results"
with (root / "bayes_spatial_cv_predictions.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
with (root / "matched_bayesian_maxent_xgboost_folds.csv").open(newline="") as f:
    matched = list(csv.DictReader(f))
assert len(rows) == 9336, len(rows)
keys = [(r["id"],r["plot"],r["species"],r["fold"]) for r in rows]
assert len(keys) == len(set(keys)), "Duplicate OOF keys"
groups = defaultdict(list)
for r in rows:
    y = int(r["observed"])
    p = float(r["mean_probability"])
    lo, hi = float(r["lower_95"]),float(r["upper_95"])
    assert y in (0,1) and all(map(math.isfinite,(p,lo,hi)))
    assert 0 <= lo <= p <= hi <= 1, r
    groups[(r["species"],int(r["fold"]))].append((y,p))

def auc(y,p):
    npos=sum(y); nneg=len(y)-npos
    assert npos and nneg
    ordered=sorted(zip(p,y),key=lambda v:v[0])
    rank_sum=0.; start=0
    while start<len(ordered):
        end=start+1
        while end<len(ordered) and ordered[end][0]==ordered[start][0]: end+=1
        rank=(start+1+end)/2
        rank_sum+=rank*sum(v[1] for v in ordered[start:end])
        start=end
    return (rank_sum-npos*(npos+1)/2)/(npos*nneg)

def ap(y,p):
    order=sorted(range(len(y)),key=lambda i:(-p[i],i))
    npos=sum(y)
    return sum(sum(y[order[j]] for j in range(i+1))/(i+1)
               for i in range(len(y)) if y[order[i]])/npos

for r in matched:
    key=(r["species"],int(r["fold"]))
    vals=groups[key]
    y,p=zip(*vals)
    assert len(vals)==int(r["n"]) and sum(y)==int(r["positives"]),key
    a,b=auc(y,p),ap(y,p)
    assert math.isclose(a,float(r["bayesian_auc"]),abs_tol=1e-12,rel_tol=0),(key,"AUC",a,r["bayesian_auc"])
    assert math.isclose(b,float(r["bayesian_ap"]),abs_tol=1e-12,rel_tol=0),(key,"AP",b,r["bayesian_ap"])
    print(f"PASS {key[0]} fold {key[1]} n={len(vals)} AUC={a:.6f} AP={b:.6f}")
assert len(matched)==23
print("PASS: original Bayesian OOF scores reproduce all 23 matched AUC/AP entries")
