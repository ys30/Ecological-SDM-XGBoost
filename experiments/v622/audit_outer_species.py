"""Audit outer-test class support for the historical five-fold grid design.

Diagnostic only: do not compute, substitute, or pool undefined AUC values.
"""
import json
from pathlib import Path
import pandas as pd

d=pd.read_csv("data/train.csv")
a=pd.read_csv("results/spatial_2d_block_assignments.csv")
assert a["plot"].is_unique
assert set(a["plot"])==set(d["plot"])
assert d[["plot","Species"]].duplicated().sum()==0
assert d["pres.abs"].notna().all()
assert set(d["pres.abs"].unique())=={0,1}
records=[]
for fold in sorted(a["fold"].unique()):
    test_plots=set(a.loc[a["fold"]==fold,"plot"])
    for partition,rows in (("outer_train",d.loc[~d["plot"].isin(test_plots)]),
                           ("outer_test",d.loc[d["plot"].isin(test_plots)])):
        for species,g in rows.groupby("Species"):
            pos=int(g["pres.abs"].sum())
            neg=int(len(g)-pos)
            records.append({"outer_fold":int(fold),"partition":partition,"Species":species,
                            "n":int(len(g)),"positive":pos,"negative":neg,
                            "auc_defined":bool(pos>0 and neg>0)})
out=pd.DataFrame(records)
assert len(out)==5*2*8
test=out.loc[out["partition"]=="outer_test"]
train=out.loc[out["partition"]=="outer_train"]
summary={"outer_folds":5,"species":8,"outer_test_species_folds":int(len(test)),
 "outer_test_auc_undefined":int((~test["auc_defined"]).sum()),
 "outer_train_auc_undefined":int((~train["auc_defined"]).sum()),
 "outer_test_zero_positive":int((test["positive"]==0).sum()),
 "outer_test_zero_negative":int((test["negative"]==0).sum()),
 "per_species":test.groupby("Species").agg(
   outer_test_folds=("auc_defined","size"),
   valid_outer_auc=("auc_defined","sum"),
   total_test_positives=("positive","sum")).reset_index().to_dict(orient="records"),
 "interpretation":"Outer AUC support only. No predictions or model performance calculated."}
p=Path("results/v622");p.mkdir(parents=True,exist_ok=True)
out.to_csv(p/"outer_species_class_counts.csv",index=False)
(p/"outer_species_class_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
