#!/usr/bin/env python3
"""Paper-ready descriptive summaries from validated v6.23 outer OOF predictions.

No model fitting. Taxa are equally weighted in macro summaries; metrics are
computed from pooled outer predictions per taxon, not averaged fold AUC.
"""
import csv,json,math
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score,average_precision_score
BASE=Path("results/v623_fixed_xgb_outer_oof")
def read(p):
    with open(p,newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    if not (BASE/"integrity_gate.json").exists():
        raise ValueError("Integrity gate must run before descriptive reporting")
    gate=json.loads((BASE/"integrity_gate.json").read_text())
    if gate.get("status")!="PASS":raise ValueError("OOF integrity gate failed")
    pred=read(BASE/"outer_oof_predictions.csv")
    taxa=sorted({r["taxon"] for r in pred})
    summaries=[]
    for taxon in taxa:
        group=[r for r in pred if r["taxon"]==taxon]
        y=np.asarray([int(r["y"]) for r in group])
        p=np.asarray([float(r["p"]) for r in group])
        eligible=len(np.unique(y))==2
        summaries.append(dict(taxon=taxon,n=len(y),positives=int(y.sum()),
            prevalence=float(y.mean()),pooled_logloss=float(log_loss(y,p,labels=[0,1])),
            pooled_brier=float(brier_score_loss(y,p)),
            pooled_auc=float(roc_auc_score(y,p)) if eligible else "",
            pooled_ap=float(average_precision_score(y,p)) if eligible else "",
            pooled_auc_ap_eligible=eligible))
    write(BASE/"taxon_pooled_oof_summary.csv",summaries)
    macro=dict(n_taxa=len(summaries),n_oof_rows=len(pred),
        mean_taxon_logloss=float(np.mean([r["pooled_logloss"] for r in summaries])),
        mean_taxon_brier=float(np.mean([r["pooled_brier"] for r in summaries])),
        mean_taxon_auc=float(np.mean([r["pooled_auc"] for r in summaries if r["pooled_auc"]!=""])),
        mean_taxon_ap=float(np.mean([r["pooled_ap"] for r in summaries if r["pooled_ap"]!=""])),
        auc_ap_taxa_eligible=sum(r["pooled_auc_ap_eligible"] for r in summaries),
        important_caveats=["Pooled per-taxon AUC uses OOF predictions from different fitted models and should not be interpreted as fold-level AUC.",
                           "Mean pooled AUC/AP across taxa is descriptive only; prevalence and spatial sampling differ.",
                           "Do not compare against historical Bayesian/GLM until identical rows, folds, and buffer rules are verified.",
                           "No confidence intervals or paired superiority claims have been computed."])
    (BASE/"macro_descriptive_summary.json").write_text(json.dumps(macro,indent=2)+"\n")
    print(json.dumps(macro,indent=2))
if __name__=="__main__":main()
