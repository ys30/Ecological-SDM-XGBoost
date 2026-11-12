"""Prepare training-only real C2a fold data for native R glmnet path audit."""
import json,hashlib
from pathlib import Path
import pandas as pd
from experiments.v622.spatial_inner_folds import spatial_inner_folds
from experiments.v622.design_matrix import FoldDesign
def main():
    data=pd.read_csv("data/train.csv")
    assignments=pd.read_csv("results/spatial_2d_block_assignments.csv")
    outer=set(assignments.loc[assignments.fold==1,"plot"])
    if not outer:raise RuntimeError("missing fold 1")
    inner=spatial_inner_folds(data,outer,outer_fold=1)
    frame=data.loc[(data.Species=="nsw18")&~data["plot"].isin(outer)].merge(inner[["plot","inner_fold"]],on="plot",validate="one_to_one")
    out=Path("results/v622/c2a_r_native");out.mkdir(parents=True,exist_ok=True)
    for model in ("glmnet_lin","glmnet_quad_rt"):
        for fold in range(4):
            train=frame.loc[frame.inner_fold!=fold]
            valid=frame.loc[frame.inner_fold==fold]
            design=FoldDesign(model,standardize=True).fit(train)
            a=design.transform(train);b=design.transform(valid)
            a.assign(y=train["pres.abs"].to_numpy(int)).to_csv(out/f"{model}_fold{fold}_train.csv",index=False)
            b.assign(y=valid["pres.abs"].to_numpy(int)).to_csv(out/f"{model}_fold{fold}_valid.csv",index=False)
    print("PASS training-only native R glmnet input partitions",flush=True)
if __name__=="__main__":main()
