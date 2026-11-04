#!/usr/bin/env python3
"""Non-destructive audit of v6.14 single-class fold status and metric eligibility."""
import csv
from pathlib import Path
R=Path(__file__).resolve().parents[1]
src=R/"results/v614_common_input/pooled_vs_specific_four_input.csv"
with src.open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
assert len(rows)==40 and len({(r["species"],r["fold"]) for r in rows})==40
assert len({r["species"] for r in rows})==8
out=[]
for r in rows:
    n=int(r["n"]);p=int(r["positives"])
    assert 0<=p<=n
    eligible=0<p<n
    for model in ("specific","pooled","blind"):
        for metric in ("auc","ap"):
            val=r[model+"_"+metric]
            if eligible:assert val not in ("","NA","NaN"),(r["species"],r["fold"],model,metric)
            else:assert val in ("","NA","NaN"),(r["species"],r["fold"],model,metric,val)
        assert r[model+"_brier"] not in ("","NA","NaN")
    out.append(dict(species=r["species"],fold=r["fold"],n=n,positives=p,
        original_status=r["status"],
        corrected_metric_status="eligible_auc_ap_brier" if eligible else "undefined_single_class_auc_ap__brier_valid",
        auc_ap_eligible=int(eligible),brier_eligible=1))
assert sum(int(r["auc_ap_eligible"]) for r in out)==30
assert sum(int(r["brier_eligible"]) for r in out)==40
dst=R/"results/v614_common_input/metric_eligibility_audit.csv"
with dst.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print("PASS: 30 AUC/AP eligible; 10 single-class; 40 Brier eligible")
print("Wrote",dst)
