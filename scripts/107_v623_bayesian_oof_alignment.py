#!/usr/bin/env python3
"""Audit Bayesian historical held-out identity against v6.23 XGBoost; no superiority claim."""
import csv,json,math
from pathlib import Path
BASE=Path("results/v623_fixed_xgb_outer_oof")
def read(path):
    with open(path,newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def main():
    gate=json.loads((BASE/"integrity_gate.json").read_text())
    if gate.get("status")!="PASS":raise ValueError("OOF integrity gate missing")
    new=read(BASE/"outer_oof_predictions.csv")
    bayes=read("results/bayes_spatial_cv_predictions.csv")
    def indexed(rows):
        d={}
        for r in rows:
            if r["id"] in d:raise ValueError("Duplicate ID")
            d[r["id"]]=r
        return d
    a=indexed(new);b=indexed(bayes)
    common=set(a)&set(b);mismatch=[]
    for id in common:
        n,o=a[id],b[id]
        if not (n["plot"]==o["plot"] and n["taxon"]==o["species"] and
                int(n["outer_fold"])==int(o["fold"]) and int(n["y"])==int(o["observed"])):
            mismatch.append(id)
        p=float(o["mean_probability"])
        if not math.isfinite(p) or p<0 or p>1:raise ValueError("Invalid Bayesian probability")
    result=dict(status="ALIGNED_METADATA_ONLY" if len(common)==len(a)==len(b) and not mismatch else "NOT_FULLY_ALIGNED",
        new_rows=len(a),bayesian_rows=len(b),shared_ids=len(common),new_only=len(set(a)-set(b)),
        bayesian_only=len(set(b)-set(a)),metadata_mismatch_count=len(mismatch),
        first_mismatch_ids=sorted(mismatch)[:10],
        protocol_equivalence="NOT_ESTABLISHED",
        caveat="Bayesian historical training buffer, feature design and fitting protocol have not been shown identical to v6.23 XGBoost. No causal or superiority comparison.")
    (BASE/"bayesian_historical_alignment.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if result["status"]!="ALIGNED_METADATA_ONLY":raise ValueError("Bayesian metadata alignment failed")
if __name__=="__main__":main()
