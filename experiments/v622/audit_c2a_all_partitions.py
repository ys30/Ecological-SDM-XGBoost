"""Preflight C2a nested spatial partitions for both frozen outer designs.
No model fitting, no outer-held-out predictions, no modification of frozen inputs.
"""
import hashlib,json
from pathlib import Path
import pandas as pd
from experiments.v622.spatial_inner_folds import spatial_inner_folds

HASHES={
 "data/train.csv":"f7eb81447e0e6deab1a10579ec1a6694587c6c61aa56947fe6a0d6012c3b1763",
 "results/spatial_2d_block_assignments.csv":"4e9c2d698ab44cf11282d12429d7a8e016f95e4e50baa1cf2653f558a7af2a81",
 "results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv":"04686634473d51b5c63dcbd4474659fe864a114ae0885c5f55736522c798ce",
}
DESIGNS={"grid4_s2026_b0":"results/spatial_2d_block_assignments.csv",
         "km15_b0":"results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv"}
def main():
    frozen=json.loads(Path("experiments/v622/docs/FROZEN_C1_FIXED.json").read_text())
    manifest=Path("experiments/v622/docs/EMS_v6.22_manifest_sha256.txt")
    if frozen["status"]!="FROZEN" or hashlib.sha256(manifest.read_bytes()).hexdigest()!=frozen["manifest_sha256"]:
        raise RuntimeError("frozen manifest mismatch")
    for line in manifest.read_text().splitlines():
        if line.strip():
            sha,path=line.split("  ",1)
            if hashlib.sha256((Path("experiments/v622")/path).read_bytes()).hexdigest()!=sha:
                raise RuntimeError("frozen file mismatch: "+path)
    for path,sha in HASHES.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
            raise RuntimeError("source hash mismatch: "+path)
    data=pd.read_csv("data/train.csv")
    if len(data)!=9336 or data.groupby(["plot","Species"]).size().ne(1).any():
        raise RuntimeError("invalid plot-taxon keys")
    taxa=sorted(data.Species.unique())
    if len(taxa)!=8 or data["plot"].nunique()!=1167:raise RuntimeError("unexpected frozen data dimensions")
    output=[]
    for design,path in DESIGNS.items():
        assignments=pd.read_csv(path)
        if len(assignments)!=1167 or not assignments["plot"].is_unique or set(assignments["plot"])!=set(data["plot"]):
            raise RuntimeError("outer assignment mismatch")
        folds=sorted(assignments.fold.unique())
        if len(folds)!=5:raise RuntimeError("not five outer folds")
        for fold in folds:
            outer=set(assignments.loc[assignments.fold==fold,"plot"])
            inner=spatial_inner_folds(data,outer,outer_fold=int(fold))
            inner_again=spatial_inner_folds(data,outer,outer_fold=int(fold))
            if not inner.equals(inner_again):raise RuntimeError("non-deterministic inner fold")
            if set(inner["plot"])!=set(data["plot"])-outer or set(inner.inner_fold)!=set(range(4)):
                raise RuntimeError("inner plot coverage or leakage")
            for taxon in taxa:
                sub=data.loc[(data.Species==taxon)&~data["plot"].isin(outer)].merge(
                    inner[["plot","inner_fold"]],on="plot",validate="one_to_one")
                if len(sub)!=len(inner):raise RuntimeError("taxon inner join mismatch")
                for inner_fold in range(4):
                    tr=sub.loc[sub.inner_fold!=inner_fold]
                    va=sub.loc[sub.inner_fold==inner_fold]
                    if set(tr["plot"])&set(va["plot"]):raise RuntimeError("inner leakage")
                    output.append(dict(design=design,outer_fold=int(fold),taxon=taxon,
                        inner_fold=inner_fold,n_train=len(tr),n_valid=len(va),
                        n_train_positive=int(tr["pres.abs"].sum()),
                        n_valid_positive=int(va["pres.abs"].sum()),
                        requires_class_support_fallback=bool(tr["pres.abs"].sum()<5 or tr["pres.abs"].nunique()<2)))
    out=pd.DataFrame(output)
    if len(out)!=320:raise RuntimeError("expected 2 designs x 5 outer x 8 taxa x 4 inner")
    target=Path("results/v622/c2a_partition_preflight.csv")
    target.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(target,index=False)
    print("PASS C2a all-design nested partition audit",len(out),"taxon-inner partitions",flush=True)
    print("class-support fallback partitions:",int(out.requires_class_support_fallback.sum()),flush=True)
    print(out.groupby("design").agg(partitions=("taxon","size"),fallbacks=("requires_class_support_fallback","sum")).to_string(),flush=True)
if __name__=="__main__":main()
