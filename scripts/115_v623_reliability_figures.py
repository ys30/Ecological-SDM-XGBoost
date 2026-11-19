#!/usr/bin/env python3
"""Publication-oriented descriptive OOF reliability panels; no posthoc calibration."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE=Path("results/v623_matched_three_model")
MODELS=[("xgboost","XGBoost"),("glm_linear","Linear GLM"),("glm_quadratic","Quadratic GLM")]
def read(path):
    with path.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def main():
    rows=read(BASE/"calibration_equal_count_deciles.csv")
    summary=read(BASE/"calibration_taxon_summary.csv")
    taxa=sorted({r["taxon"] for r in rows})
    assert len(taxa)==8 and len(rows)==240 and len(summary)==24
    lookup={(r["taxon"],r["model"]):r for r in summary}
    fig,axes=plt.subplots(2,4,figsize=(14,7),sharex=True,sharey=True)
    for ax,taxon in zip(axes.flat,taxa):
        ax.plot([0,1],[0,1],linestyle="--",linewidth=1,color="0.5",label="Ideal")
        for model,label in MODELS:
            rr=sorted((r for r in rows if r["taxon"]==taxon and r["model"]==model),key=lambda r:int(r["bin"]))
            assert len(rr)==10
            x=np.array([float(r["mean_prediction"]) for r in rr])
            y=np.array([float(r["observed_fraction"]) for r in rr])
            n=np.array([int(r["n"]) for r in rr])
            assert n.sum()==int(lookup[taxon,model]["n"])
            ax.plot(x,y,marker="o",markersize=3,linewidth=1.2,label=label)
        npos=int(round(float(lookup[taxon,"xgboost"]["prevalence"])*int(lookup[taxon,"xgboost"]["n"])))
        ax.set_title(f"{taxon} (positives={npos})",fontsize=10)
        ax.grid(alpha=.15)
        ax.set_xlim(0,1);ax.set_ylim(0,1)
    fig.supxlabel("Mean predicted probability within taxon-specific decile")
    fig.supylabel("Observed detection frequency")
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=4,bbox_to_anchor=(.5,1.01),frameon=False)
    fig.suptitle("Spatially held-out reliability by taxon (descriptive)",y=1.06)
    fig.tight_layout()
    fig.savefig(BASE/"figure_reliability_by_taxon.png",dpi=220,bbox_inches="tight")
    fig.savefig(BASE/"figure_reliability_by_taxon.svg",bbox_inches="tight")
    plt.close(fig)
    # Zoomed low-probability view: descriptive; retains full-range panel above.
    fig,axes=plt.subplots(2,4,figsize=(14,7),sharex=True,sharey=True)
    for ax,taxon in zip(axes.flat,taxa):
        ax.plot([0,.3],[0,.3],linestyle="--",linewidth=1,color="0.5")
        for model,label in MODELS:
            rr=sorted((r for r in rows if r["taxon"]==taxon and r["model"]==model),key=lambda r:int(r["bin"]))
            x=np.array([float(r["mean_prediction"]) for r in rr])
            y=np.array([float(r["observed_fraction"]) for r in rr])
            ax.plot(x,y,marker="o",markersize=3,linewidth=1.2,label=label)
        ax.set_title(taxon,fontsize=10);ax.grid(alpha=.15);ax.set_xlim(0,.3);ax.set_ylim(0,.3)
    fig.supxlabel("Mean predicted probability (0–0.3 view)")
    fig.supylabel("Observed detection frequency (0–0.3 view)")
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=3,bbox_to_anchor=(.5,1.01),frameon=False)
    fig.suptitle("Low-probability reliability detail (clipped display)",y=1.06)
    fig.tight_layout()
    fig.savefig(BASE/"figure_reliability_low_probability.png",dpi=220,bbox_inches="tight")
    fig.savefig(BASE/"figure_reliability_low_probability.svg",bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"status":"FIGURES_WRITTEN","taxa":len(taxa),"models":len(MODELS),"figures":4,
        "warning":"Descriptive reliability only; decile points for rare taxa can have zero positives."}))
if __name__=="__main__":main()
