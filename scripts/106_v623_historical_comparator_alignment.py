#!/usr/bin/env python3
"""Strict row-level alignment audit: archived v6.14 XGB versus v6.23 XGB.

Historical comparator is NOT assumed to share the 10-km training exclusion.
No performance difference is interpreted as a controlled model comparison.
"""
import csv,json,math
from pathlib import Path
BASE=Path("results/v623_fixed_xgb_outer_oof")
HIST=Path("results/v614_common_input/taxon_specific_xgboost_oof.csv")
def read(p):
    with open(p,newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def main():
    gate=json.loads((BASE/"integrity_gate.json").read_text())
    if gate.get("status")!="PASS":raise ValueError("v6.23 OOF integrity not verified")
    new=read(BASE/"outer_oof_predictions.csv")
    old=read(HIST)
    def index(rows):
        ids=[r["id"] for r in rows]
        if len(ids)!=len(set(ids)):raise ValueError("Duplicate comparator IDs")
        return {r["id"]:r for r in rows}
    a=index(new);b=index(old)
    shared=sorted(set(a)&set(b),key=int)
    differences=[]
    for id in shared:
        n,o=a[id],b[id]
        checks=(n["plot"]==o["plot"],n["taxon"]==o["Species"],
                int(n["outer_fold"])==int(o["fold"]),int(n["y"])==int(o["y"]))
        if not all(checks):differences.append(dict(id=id,plot=checks[0],taxon=checks[1],fold=checks[2],label=checks[3]))
        for r in (n,o):
            p=float(r["p"])
            if not math.isfinite(p) or not 0<=p<=1:raise ValueError("Invalid probability")
    result=dict(status="ALIGNED_METADATA_ONLY" if not differences and len(shared)==len(a)==len(b) else "NOT_FULLY_ALIGNED",
        historical_rows=len(old),new_rows=len(new),shared_ids=len(shared),
        new_only=len(set(a)-set(b)),historical_only=len(set(b)-set(a)),
        metadata_mismatch_count=len(differences),first_mismatches=differences[:10],
        comparison_scope="Historical taxon-specific XGB versus new fixed XGB; NOT Bayesian/GLM",
        protocol_equivalence="NOT_ESTABLISHED",
        restrictions=["Matching test rows and labels does not prove matching training data or 10-km buffers.",
            "Do not report model superiority or paired uncertainty as a controlled protocol comparison.",
            "Verify Bayesian/GLM predictions and training exclusion separately."])
    (BASE/"historical_comparator_alignment.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if result["status"]!="ALIGNED_METADATA_ONLY":raise ValueError("Historical comparator metadata mismatch")
if __name__=="__main__":main()
