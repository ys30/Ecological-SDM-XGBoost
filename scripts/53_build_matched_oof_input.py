#!/usr/bin/env python3
"""Build generic evaluator input from archived NSW three-model OOF outputs.

Preserves anonymous taxon IDs; uses the MaxEnt-eligible observation intersection.
Joins by observation ID and validates plot, fold, and outcome across pipelines.
"""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    with (ROOT/"results"/name).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))
def idx(rows):
    out={}
    for r in rows:
        k=r["id"]
        if k in out:raise ValueError("Duplicate observation ID: "+k)
        out[k]=r
    return out
bay=idx(load("bayes_spatial_cv_predictions.csv"))
xgb=idx(load("spatial_2d_block_cv_predictions.csv"))
maxent=idx(load("maxent_spatial_2d_predictions.csv"))
assert len(bay)==len(xgb)==9336,(len(bay),len(xgb))
assert len(maxent)>0 and set(maxent).issubset(bay) and set(maxent).issubset(xgb)
rows=[]
for key,m in maxent.items():
    b=bay[key];x=xgb[key]
    for field in ("plot","fold","observed"):
        assert m[field]==b[field]==x[field],(key,field)
    assert m["species"]==b["species"],key
    rows.append(dict(observation_id=key,taxon=m["species"],fold=m["fold"],
                     observed=m["observed"],bayesian=b["mean_probability"],
                     maxent=m["relative_suitability"],xgboost=x["predicted_probability"]))
rows.sort(key=lambda r:(r["taxon"],int(r["fold"]),int(r["observation_id"])))
path=ROOT/"results"/"matched_oof_generic_input.csv"
with path.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print("PASS: verified",len(rows),"matched OOF rows across three pipelines")
