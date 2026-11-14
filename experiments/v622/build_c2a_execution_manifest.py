"""C2a full-scale execution manifest from audited spatial folds; no model fitting."""
from pathlib import Path
import pandas as pd

def main():
    p=Path("results/v622/c2a_partition_preflight.csv")
    d=pd.read_csv(p)
    keys=["design","outer_fold","taxon"]
    if len(d)!=320 or d.duplicated(keys+["inner_fold"]).any():
        raise RuntimeError("invalid 320-row nested partition audit")
    if d.groupby(keys).size().ne(4).any():raise RuntimeError("missing inner folds")
    group=d.groupby(keys,sort=True).agg(
        n_inner_fallbacks=("requires_class_support_fallback","sum"),
        n_inner_train_min=("n_train","min"),
        n_inner_valid_min=("n_valid","min"),
        n_inner_positive_min=("n_train_positive","min"),
    ).reset_index()
    if len(group)!=80:raise RuntimeError("expected 80 taxon-design-outer cells")
    models=pd.DataFrame({"model_id":["glmnet_lin","glmnet_quad_rt"]})
    manifest=group.merge(models,how="cross")
    manifest["status"]="NOT_RUN"
    manifest["diagnostic_only"]=True
    manifest["run_order"]=range(1,len(manifest)+1)
    if len(manifest)!=160:raise RuntimeError("expected 160 model fits")
    out=Path("results/v622/c2a_full_execution_manifest_DIAGNOSTIC_ONLY.csv")
    out.parent.mkdir(parents=True,exist_ok=True)
    manifest.to_csv(out,index=False)
    print("PASS C2a execution manifest: 160 planned model fits, 80 paired cells; NO new fits",flush=True)
    print("Inner support fallbacks across 320 partitions:",int(d.requires_class_support_fallback.sum()),flush=True)
    print("Model-arm cells with >=1 inner fallback:",int((manifest.n_inner_fallbacks>0).sum()),flush=True)
    print(group.groupby("design").agg(cells=("taxon","size"),cells_with_fallback=("n_inner_fallbacks",lambda x:int((x>0).sum()))).to_string(),flush=True)
if __name__=="__main__":main()
