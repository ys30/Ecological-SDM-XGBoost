"""Prepare all eight taxa for one audited C2a outer fold; diagnostic batch only."""
import argparse,hashlib
from pathlib import Path
import pandas as pd
from experiments.v622.audit_c2a_all_partitions import HASHES,DESIGNS
from experiments.v622.design_matrix import FoldDesign
from experiments.v622.spatial_inner_folds import spatial_inner_folds

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--design",choices=list(DESIGNS),default="grid4_s2026_b0")
    ap.add_argument("--fold",type=int,choices=range(1,6),default=1)
    args=ap.parse_args()
    for p,h in HASHES.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError("frozen source hash mismatch "+p)
    data=pd.read_csv("data/train.csv")
    assignment=pd.read_csv(DESIGNS[args.design])
    outer=set(assignment.loc[assignment.fold==args.fold,"plot"])
    inner=spatial_inner_folds(data,outer,outer_fold=args.fold)
    root=Path("results/v622/c2a_batch_DIAGNOSTIC_ONLY")/args.design/f"fold{args.fold}"
    for taxon in sorted(data.Species.unique()):
        sub=data.loc[data.Species==taxon]
        train=sub.loc[~sub["plot"].isin(outer)].merge(inner[["plot","inner_fold"]],on="plot",validate="one_to_one")
        test=sub.loc[sub["plot"].isin(outer)]
        target=root/taxon
        target.mkdir(parents=True,exist_ok=True)
        for model in ("glmnet_lin","glmnet_quad_rt"):
            for f in range(4):
                tr=train.loc[train.inner_fold!=f];va=train.loc[train.inner_fold==f]
                design=FoldDesign(model,standardize=True).fit(tr)
                design.transform(tr).assign(y=tr["pres.abs"].to_numpy(int)).to_csv(target/f"{model}_inner{f}_train.csv",index=False)
                design.transform(va).assign(y=va["pres.abs"].to_numpy(int)).to_csv(target/f"{model}_inner{f}_valid.csv",index=False)
            design=FoldDesign(model,standardize=True).fit(train)
            design.transform(train).assign(y=train["pres.abs"].to_numpy(int)).to_csv(target/f"{model}_outer_train.csv",index=False)
            design.transform(test).assign(y=test["pres.abs"].to_numpy(int),plot=test["plot"].to_numpy()).to_csv(target/f"{model}_outer_test.csv",index=False)
        print("PREPARED",args.design,args.fold,taxon,"train",len(train),"test",len(test),flush=True)
    print("PASS eight-taxon diagnostic batch preparation; no fits",flush=True)
if __name__=="__main__":main()
