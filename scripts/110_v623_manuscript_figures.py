#!/usr/bin/env python3
"""Publication-oriented matched-model figures from verified OOF comparison CSVs."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
BASE=Path("results/v623_matched_three_model")
def read(p):
    with p.open(newline="") as f:return list(csv.DictReader(f))
def main():
    meta=json.loads((BASE/"manifest.json").read_text())
    if meta["status"]!="SAME_OUTER_FOLDS_AND_BUFFER_VERIFIED":raise ValueError("Unverified comparison")
    rows=read(BASE/"taxon_pooled_oof_scores.csv")
    contrasts=read(BASE/"paired_taxon_contrasts.csv")
    taxa=sorted({r["taxon"] for r in rows})
    models=["xgboost","glm_linear","glm_quadratic"]
    labels=["XGBoost","Linear GLM","Quadratic GLM"]
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    for ax,metric in zip(axes.flat,["logloss","brier","auc","ap"]):
        for model,label in zip(models,labels):
            vals=np.asarray([float(next(r[metric] for r in rows if r["taxon"]==t and r["model"]==model)) for t in taxa])
            ax.plot(range(len(taxa)),vals,marker="o",linewidth=1.4,label=label)
        ax.set_xticks(range(len(taxa)),taxa,rotation=35)
        ax.set_title(metric.upper() if metric in ("auc","ap") else metric.title())
        ax.set_xlabel("Anonymous taxon")
        ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Matched primary-fold 10-km buffered OOF performance",fontsize=12)
    fig.savefig(BASE/"figure_matched_taxon_scores.svg",bbox_inches="tight")
    fig.savefig(BASE/"figure_matched_taxon_scores.png",dpi=250,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,metric in zip(axes,["logloss","brier"]):
        for comp,marker in [("xgboost_minus_glm_linear","o"),("xgboost_minus_glm_quadratic","s")]:
            glm=comp.removeprefix("xgboost_minus_")
            diff=[float(next(r[metric] for r in rows if r["taxon"]==t and r["model"]=="xgboost"))-
                  float(next(r[metric] for r in rows if r["taxon"]==t and r["model"]==glm)) for t in taxa]
            ax.plot(range(len(taxa)),diff,marker=marker,label=comp.replace("xgboost_minus_","vs "))
        ax.axhline(0,color="black",linewidth=.8)
        ax.set_xticks(range(len(taxa)),taxa,rotation=35)
        ax.set_title("XGBoost minus GLM: "+metric)
        ax.set_ylabel("Negative favors XGBoost")
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    fig.savefig(BASE/"figure_paired_probability_scores.svg",bbox_inches="tight")
    fig.savefig(BASE/"figure_paired_probability_scores.png",dpi=250,bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"taxa":len(taxa),"models":len(models),"contrasts":len(contrasts),"figures":4,"status":"FIGURES_WRITTEN"},indent=2))
if __name__=="__main__":main()
