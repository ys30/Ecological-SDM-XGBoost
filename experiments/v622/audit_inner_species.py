"""Non-training audit of species positive/negative counts in spatial nested folds."""
import json
from pathlib import Path
import pandas as pd
from experiments.v622.spatial_inner_folds import spatial_inner_folds

d=pd.read_csv("data/train.csv")
a=pd.read_csv("results/spatial_2d_block_assignments.csv")
assert set(a["plot"])==set(d["plot"])
assert d[["plot","Species"]].duplicated().sum()==0
assert set(d["pres.abs"].dropna().unique()) <= {0,1}
records=[]
for outer_fold in sorted(a["fold"].unique()):
    outer=set(a.loc[a["fold"]==outer_fold,"plot"])
    inner=spatial_inner_folds(d,outer,n_clusters=12,n_folds=4,seed=2026)
    mapping=dict(zip(inner["plot"],inner["inner_fold"]))
    train=d.loc[~d["plot"].isin(outer)].copy()
    train["inner_fold"]=train["plot"].map(mapping)
    assert train["inner_fold"].notna().all()
    for f in sorted(inner["inner_fold"].unique()):
        for partition,rows in [("train",train.loc[train["inner_fold"]!=f]),("validation",train.loc[train["inner_fold"]==f])]:
            for species,g in rows.groupby("Species"):
                pos=int(g["pres.abs"].sum())
                neg=int(len(g)-pos)
                records.append({"outer_fold":int(outer_fold),"inner_fold":int(f),
                                "partition":partition,"Species":species,
                                "n":int(len(g)),"positive":pos,"negative":neg,
                                "auc_defined":bool(pos>0 and neg>0)})
out=pd.DataFrame(records)
Path("results/v622").mkdir(parents=True,exist_ok=True)
out.to_csv("results/v622/inner_species_class_counts.csv",index=False)
summary={"outer_folds":int(a["fold"].nunique()),"inner_folds_per_outer":4,
 "species":int(d["Species"].nunique()),"records":int(len(out)),
 "undefined_auc_rows":int((~out["auc_defined"]).sum()),
 "undefined_validation_auc_rows":int(((out["partition"]=="validation")&(~out["auc_defined"])).sum()),
 "undefined_training_auc_rows":int(((out["partition"]=="train")&(~out["auc_defined"])).sum()),
 "smallest_validation_positives":int(out.loc[out["partition"]=="validation","positive"].min()),
 "scope":"Diagnostic only. Do not silently drop invalid taxon-fold AUC values."}
Path("results/v622/inner_species_class_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
