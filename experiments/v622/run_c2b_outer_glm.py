"""C2b: frozen-input gated, paired spatial outer-fold GLM predictions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.v622.design_matrix import FoldDesign
from experiments.v622.glm_adapter import run_glm_inner

HASHES={"data/train.csv":"f7eb81447e0e6deab1a10579ec1a6694587c6c61aa56947fe6a0d6012c3b1763","results/spatial_2d_block_assignments.csv":"4e9c2d698ab44cf11282d12429d7a8e016f95e4e50baa1cf2653f558a7af2a81","results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv":"04686634473d51b5c63dcbd4474659fe864a114ae0885c5c5f55736522c798ce"}
def verify(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=HASHES[path]:
        raise RuntimeError("SOURCE HASH MISMATCH: "+path)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--design",required=True,choices=["grid4_s2026_b0","km15_b0"])
    parser.add_argument("--output-dir",default="results/v622")
    args=parser.parse_args()
    frozen=json.loads(Path("experiments/v622/docs/FROZEN_C1_FIXED.json").read_text())
    manifest=Path("experiments/v622/docs/EMS_v6.22_manifest_sha256.txt")
    if frozen["status"]!="FROZEN" or hashlib.sha256(manifest.read_bytes()).hexdigest()!=frozen["manifest_sha256"]:
        raise RuntimeError("FROZEN MANIFEST MISMATCH")
    for line in manifest.read_text().splitlines():
        if line.strip():
            sha,path=line.split("  ",1)
            if hashlib.sha256((Path("experiments/v622")/path).read_bytes()).hexdigest()!=sha:
                raise RuntimeError("MANIFEST INPUT MISMATCH: "+path)
    assign_path="results/spatial_2d_block_assignments.csv" if args.design=="grid4_s2026_b0" else "results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv"
    verify("data/train.csv");verify(assign_path)
    frame=pd.read_csv("data/train.csv")
    assignments=pd.read_csv(assign_path)
    if len(assignments)!=1167 or not assignments["plot"].is_unique or set(assignments["plot"])!=set(frame["plot"]):
        raise RuntimeError("INVALID FOLD ASSIGNMENTS")
    if frame.groupby(["plot","Species"]).size().ne(1).any() or len(frame)!=9336:
        raise RuntimeError("INVALID BASE DATA KEYS")
    fold_map=assignments.set_index("plot")["fold"]
    if fold_map.isna().any() or fold_map.nunique()!=5:raise RuntimeError("INVALID OUTER FOLDS")
    frame["outer_fold"]=frame["plot"].map(fold_map)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    for model in ("glm_lin","glm_quad_rt"):
        rows=[]
        for fold in sorted(frame.outer_fold.unique()):
            train=frame.loc[frame.outer_fold!=fold]
            test=frame.loc[frame.outer_fold==fold]
            if set(train["plot"])&set(test["plot"]):raise RuntimeError("OUTER LEAKAGE")
            for taxon in sorted(frame.Species.unique()):
                tr=train.loc[train.Species==taxon]
                te=test.loc[test.Species==taxon]
                design=FoldDesign(model,standardize=True,missing_policy="error").fit(tr)
                a=design.transform(tr).to_numpy()
                b=design.transform(te).to_numpy()
                outcome=run_glm_inner(tr["pres.abs"].to_numpy(),te["pres.abs"].to_numpy(),a,b)
                for (_,rec),prob in zip(te.iterrows(),outcome.predictions):
                    rows.append({"plot":rec["plot"],"Species":taxon,"outer_fold":int(fold),"pres.abs":int(rec["pres.abs"]),"pred":float(prob),"fallback":int(outcome.fallback),"fit_status":outcome.status,"attempt_count":outcome.attempt_count,"first_error":outcome.first_error,"retry_error":outcome.retry_error,"n_training_positive":outcome.n_training_positive,"model_id":model})
        result=pd.DataFrame(rows)
        if len(result)!=9336 or result.duplicated(["plot","Species","outer_fold"]).any() or not np.isfinite(result.pred).all() or not result.pred.between(0,1).all():
            raise RuntimeError("INVALID OOF: "+model)
        expected=frame[["plot","Species","outer_fold","pres.abs"]]
        if not result[["plot","Species","outer_fold","pres.abs"]].sort_values(["plot","Species"]).reset_index(drop=True).equals(expected.sort_values(["plot","Species"]).reset_index(drop=True)):
            raise RuntimeError("OOF LABEL OR FOLD MISMATCH")
        target=output/f"oof_{model}_{args.design}.csv.gz"
        if target.exists():raise FileExistsError(target)
        result.to_csv(target,index=False,compression="gzip")
        print("PASS C2B",model,args.design,len(result),"fallback",int(result.fallback.sum()),"retry",int((result.attempt_count==2).sum()),flush=True)
if __name__=="__main__":main()
