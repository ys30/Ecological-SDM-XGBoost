"""Prepare one leakage-safe C2a outer OOF smoke partition, without fitting."""
from pathlib import Path
import pandas as pd
from experiments.v622.audit_c2a_all_partitions import HASHES
from experiments.v622.design_matrix import FoldDesign
from experiments.v622.spatial_inner_folds import spatial_inner_folds
import hashlib

def main():
    for p,h in HASHES.items():
        if Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:
            raise RuntimeError("source hash mismatch "+p)
    d=pd.read_csv("data/train.csv")
    a=pd.read_csv("results/spatial_2d_block_assignments.csv")
    outer=set(a.loc[a.fold==1,"plot"])
    inner=spatial_inner_folds(d,outer,outer_fold=1)
    species="nsw18"
    train=d.loc[(d.Species==species)&~d["plot"].isin(outer)].merge(inner[["plot","inner_fold"]],on="plot",validate="one_to_one")
    test=d.loc[(d.Species==species)&d["plot"].isin(outer)].copy()
    out=Path("results/v622/c2a_outer_smoke");out.mkdir(parents=True,exist_ok=True)
    for model in ("glmnet_lin","glmnet_quad_rt"):
        for f in range(4):
            tr=train.loc[train.inner_fold!=f];va=train.loc[train.inner_fold==f]
            design=FoldDesign(model,standardize=True).fit(tr)
            design.transform(tr).assign(y=tr["pres.abs"].to_numpy(int)).to_csv(out/f"{model}_inner{f}_train.csv",index=False)
            design.transform(va).assign(y=va["pres.abs"].to_numpy(int)).to_csv(out/f"{model}_inner{f}_valid.csv",index=False)
        design=FoldDesign(model,standardize=True).fit(train)
        design.transform(train).assign(y=train["pres.abs"].to_numpy(int)).to_csv(out/f"{model}_outer_train.csv",index=False)
        design.transform(test).assign(y=test["pres.abs"].to_numpy(int),plot=test["plot"].to_numpy()).to_csv(out/f"{model}_outer_test.csv",index=False)
    print("PASS C2a outer smoke prepared",len(train),"train",len(test),"test; no model fitted",flush=True)
if __name__=="__main__":main()
