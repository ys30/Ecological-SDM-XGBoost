#!/usr/bin/env python3
"""Validate v6.23 OOF outputs against canonical inputs; no model fitting."""
import csv,json,math
from collections import Counter
from pathlib import Path
BASE=Path("results/v623_fixed_xgb_outer_oof")
def read(path):
    with open(path,newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def main():
    source=read("data/train.csv")
    folds=read("results/spatial_2d_block_assignments.csv")
    pred=read(BASE/"outer_oof_predictions.csv")
    metrics=read(BASE/"outer_fold_metrics.csv")
    manifest=json.loads((BASE/"manifest.json").read_text())
    fmap={r["plot"]:int(r["fold"]) for r in folds}
    expected={r["id"]:(r["plot"],r["Species"],int(r["pres.abs"])) for r in source}
    assert len(expected)==len(source)==9336,"Canonical row IDs invalid"
    assert len(pred)==len(source) and len({r["id"] for r in pred})==len(pred),"OOF IDs missing/duplicated"
    assert {r["id"] for r in pred}==set(expected),"OOF coverage mismatch"
    counts=Counter()
    for r in pred:
        plot,taxon,y=expected[r["id"]]
        assert (r["plot"],r["taxon"],int(r["y"]))==(plot,taxon,y),"Label/plot/taxon mismatch"
        assert int(r["outer_fold"])==fmap[plot],"Outer-fold leakage or mismatch"
        p=float(r["p"])
        assert math.isfinite(p) and 0<=p<=1,"Invalid probability"
        counts[(int(r["outer_fold"]),r["taxon"])]+=1
    assert len(counts)==40,"Missing outer-fold/taxon cells"
    assert len(metrics)==40 and len({(int(r["outer_fold"]),r["taxon"]) for r in metrics})==40
    ineligible=0
    for r in metrics:
        key=(int(r["outer_fold"]),r["taxon"])
        assert int(r["n_test"])==counts[key],"Metric sample count mismatch"
        pos=sum(int(p["y"]) for p in pred if (int(p["outer_fold"]),p["taxon"])==key)
        assert int(r["positives_test"])==pos,"Metric positive count mismatch"
        eligible=0<pos<counts[key]
        assert r["auc_ap_eligible"].lower()==str(eligible).lower(),"Eligibility flag mismatch"
        if not eligible:
            ineligible+=1
            assert r["auc"]==r["ap"]=="","Undefined AUC/AP must be blank"
        for col in ("logloss","baseline_logloss","brier"):
            assert math.isfinite(float(r[col])),f"Nonfinite {col}"
    assert manifest["fit_count"]==40 and manifest["oof_rows"]==len(pred)
    assert ineligible==10,"Unexpected single-class count"
    result={"status":"PASS","oof_rows":len(pred),"outer_taxon_cells":len(counts),
            "auc_ap_ineligible":ineligible,
            "checks":["exact IDs","fold correspondence","class/plot correspondence",
                      "probability bounds","metric eligibility","manifest coverage"]}
    (BASE/"integrity_gate.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
