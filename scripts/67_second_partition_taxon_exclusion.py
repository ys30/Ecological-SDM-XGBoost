"""Second-partition taxon influence: recompute matched OOF metrics without refitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
paths=sorted(Path("downloaded").rglob("*_oof.csv"))
assert len(paths)==10,len(paths)
d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
assert set(d.variant)=={"linear","quadratic"}
assert len(d)==18672 and d.groupby("variant").size().eq(9336).all()
def score(g):
    y=g.observed.to_numpy();p=g.probability.to_numpy()
    return dict(n=len(g),positives=int(y.sum()),
        auc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
        ap=average_precision_score(y,p) if y.sum()>0 else np.nan,
        brier=np.mean((y-p)**2),logloss=log_loss(y,p,labels=[0,1]))
scenarios={"all_taxa":[],"exclude_nsw18":["nsw18"],"exclude_nsw24":["nsw24"],
           "exclude_nsw18_nsw24":["nsw18","nsw24"]}
rows=[];taxon_rows=[]
for scenario,removed in scenarios.items():
    q=d.loc[~d.species.isin(removed)].copy()
    assert q.groupby("variant").size().nunique()==1
    scores={v:score(g) for v,g in q.groupby("variant")}
    a,b=scores["linear"],scores["quadratic"]
    row=dict(scenario=scenario,remaining_taxa=q.species.nunique(),n_per_model=a["n"],positives=a["positives"])
    for k in ["auc","ap","brier","logloss"]:
        row["linear_"+k]=a[k];row["quadratic_"+k]=b[k];row["pooled_delta_"+k]=b[k]-a[k]
    by=q.groupby(["variant","species","fold"]).apply(lambda g:pd.Series(score(g)),include_groups=False).reset_index()
    left=by[by.variant=="linear"].drop(columns="variant")
    right=by[by.variant=="quadratic"].merge(left,on=["species","fold"],suffixes=("","_linear"),validate="one_to_one")
    for k in ["auc","ap","brier","logloss"]:
        right["delta_"+k]=right[k]-right[k+"_linear"]
        row["mean_cell_delta_"+k]=right["delta_"+k].mean()
        row["n_eligible_"+k]=int(right["delta_"+k].notna().sum())
    rows.append(row)
    taxon_rows.append(right.assign(scenario=scenario))
out=Path("results/alternative_spatial_validation");out.mkdir(parents=True,exist_ok=True)
summary=pd.DataFrame(rows);summary.to_csv(out/"taxon_exclusion_summary.csv",index=False)
pd.concat(taxon_rows).to_csv(out/"taxon_exclusion_paired_cells.csv",index=False)
print(summary[["scenario","remaining_taxa","n_per_model","pooled_delta_auc","pooled_delta_ap","pooled_delta_brier","mean_cell_delta_auc","mean_cell_delta_ap","mean_cell_delta_brier","mean_cell_delta_logloss"]].to_string(index=False))
with open(out/"TAXON_EXCLUSION_VERIFIED.md","w") as f:
    f.write("# Second-partition taxon-exclusion sensitivity\n\n")
    f.write("This is a **post-hoc evaluation subset analysis**, not retraining after removing species. Both Bayesian hierarchical models were trained using all eight taxa; excluding taxa here removes only their held-out predictions when recomputing metrics. Thus results cannot establish how a model trained without those taxa would perform.\n\n")
    f.write("## Scenario-level results\n\n"+summary.to_markdown(index=False,floatfmt=".6f")+"\n\n")
    f.write("Positive pooled ΔAUC/ΔAP and negative pooled ΔBrier/ΔLogloss favor quadratic; taxon-fold mean differences use eligible cells and are a different estimand. AUC/AP are undefined for one-class/no-positive cells. Within-dataset exploratory analysis, not external validation.\n")
