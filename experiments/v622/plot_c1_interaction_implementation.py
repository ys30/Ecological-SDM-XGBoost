"""Generate publication-ready C1 interaction forest plots from immutable CI JSON artifacts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path("results/v622")
def main():
    original=json.loads((ROOT/"c1_tuning_sharing_interaction.json").read_text())
    nonfallback=json.loads((ROOT/"c1_interaction_nonfallback_sensitivity.json").read_text())
    r=json.loads((ROOT/"c1_r_baseline_interaction_sensitivity.json").read_text())
    print("nonfallback schema",list(nonfallback.keys()))
    # Two-design primary exploratory interaction; R implementation check is grid4 only.
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout="constrained")
    metrics=("auc","logloss","brier")
    for ax,metric in zip(axes,metrics):
        points=[]
        for design,label in (("grid4_s2026_b0","grid4 Python"),("km15_b0","km15 Python")):
            item=original["designs"][design]["metrics"][metric]
            points.append((label,item["mean"],item["ci95"]))
        item=r["results"][metric]["r_hist"]
        points.append(("grid4 R fixed",item["mean"],item["ci95"]))
        for i,(label,mean,ci) in enumerate(points):
            ax.errorbar(mean,2-i,xerr=[[mean-ci[0]],[ci[1]-mean]],fmt="o",capsize=3,color="black")
        ax.axvline(0,color="gray",linestyle="--",linewidth=1)
        ax.set_yticks([2,1,0],labels=[x[0] for x in points])
        ax.set_title({"auc":"AUC (primary)","logloss":"Log loss (secondary)","brier":"Brier (secondary)"}[metric])
        ax.set_xlabel("Tuning × sharing interaction")
        ax.grid(axis="x",alpha=.2)
    fig.suptitle("C1 exploratory interaction: taxon-cluster bootstrap 95% intervals",fontsize=12)
    out=ROOT/"c1_interaction_implementation_forest.png"
    fig.savefig(out,dpi=220,bbox_inches="tight")
    fig.savefig(ROOT/"c1_interaction_implementation_forest.pdf",bbox_inches="tight")
    plt.close(fig)
    print("FIGURE",out)
if __name__=="__main__":main()
