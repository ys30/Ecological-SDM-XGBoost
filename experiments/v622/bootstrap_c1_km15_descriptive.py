"""Taxon-cluster bootstrap uncertainty for descriptive C3d km15 paired AUC.

Resample taxa (not individual records) with replacement, preserving all
evaluable outer folds for each sampled taxon. Descriptive only: one design,
not a confirmatory two-design C3d result.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    root=Path("results/v622")
    p=root/"c1_km15_tuned_pooled_vs_species_descriptive.csv"
    df=pd.read_csv(p)
    col="matched_auc_delta_pooled_minus_species"
    if len(df)!=40 or df["Species"].nunique()!=8 or df[["Species","outer_fold"]].duplicated().any():
        raise ValueError("expected complete 8-taxon x 5-fold km15 descriptive table")
    observed=df[col].dropna()
    if len(observed)==0:
        raise ValueError("no evaluable matched AUC cells")
    taxa=np.array(sorted(df["Species"].unique()))
    groups={taxon:df.loc[df["Species"]==taxon,col].dropna().to_numpy(float) for taxon in taxa}
    rng=np.random.default_rng(622)
    draws=[]
    for _ in range(2000):
        sample=rng.choice(taxa,size=len(taxa),replace=True)
        vals=np.concatenate([groups[t] for t in sample if len(groups[t])])
        draws.append(float(vals.mean()) if len(vals) else np.nan)
    draws=np.asarray(draws)
    finite=draws[np.isfinite(draws)]
    if len(finite)!=2000:
        raise ValueError("bootstrap draws without evaluable taxa; require explicit coverage policy")
    lo,hi=np.quantile(finite,[0.025,0.975])
    summary={
        "status":"DESCRIPTIVE_SINGLE_DESIGN_C3D_NOT_CONFIRMATORY",
        "design":"km15_b0",
        "estimand":"unweighted mean of evaluable taxon-fold paired AUC differences",
        "direction":"pooled tuned minus species tuned",
        "mean":float(observed.mean()),"ci95_taxon_cluster_percentile":[float(lo),float(hi)],
        "bootstrap_draws":2000,"seed":622,
        "evaluable_cells":int(len(observed)),"total_cells":int(len(df)),
        "evaluable_taxa":int(sum(len(v)>0 for v in groups.values())),
        "missing_auc_cells":int(df[col].isna().sum()),
        "interpretation":"Descriptive uncertainty only; C1 fixed baselines absent; taxon count is only eight."
    }
    (root/"c1_km15_descriptive_taxon_bootstrap.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":
    main()
