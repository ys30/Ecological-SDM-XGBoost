"""C2a real-data, training-only spatial nested tuning smoke; NO outer predictions."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.v622.design_matrix import FoldDesign
from experiments.v622.spatial_inner_folds import spatial_inner_folds
from experiments.v622.elasticnet_prototype import run_elasticnet_inner
from experiments.v622.class_support_policy import relative_log_loss_skill
HASHES={"data/train.csv":"f7eb81447e0e6deab1a10579ec1a6694587c6c61aa56947fe6a0d6012c3b1763","results/spatial_2d_block_assignments.csv":"4e9c2d698ab44cf11282d12429d7a8e016f95e4e50baa1cf2653f558a7af2a81"}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outer-fold",type=int,default=1)
    ap.add_argument("--taxon",default="nsw18")
    ap.add_argument("--output-dir",default="results/v622")
    args=ap.parse_args()
    marker=json.loads(Path("experiments/v622/docs/FROZEN_C1_FIXED.json").read_text())
    manifest=Path("experiments/v622/docs/EMS_v6.22_manifest_sha256.txt")
    if marker["status"]!="FROZEN" or hashlib.sha256(manifest.read_bytes()).hexdigest()!=marker["manifest_sha256"]:raise RuntimeError("freeze mismatch")
    for line in manifest.read_text().splitlines():
        if line.strip():
            sha,path=line.split("  ",1)
            if hashlib.sha256((Path("experiments/v622")/path).read_bytes()).hexdigest()!=sha:raise RuntimeError("manifest mismatch "+path)
    for path,sha in HASHES.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise RuntimeError("input mismatch "+path)
    data=pd.read_csv("data/train.csv")
    assign=pd.read_csv("results/spatial_2d_block_assignments.csv")
    if args.taxon not in set(data.Species):raise ValueError("unknown taxon")
    outer=set(assign.loc[assign.fold==args.outer_fold,"plot"])
    if not outer:raise ValueError("outer fold absent")
    inner=spatial_inner_folds(data,outer,outer_fold=args.outer_fold)
    frame=data.loc[(data.Species==args.taxon)&~data["plot"].isin(outer)].merge(inner[["plot","inner_fold"]],on="plot",validate="one_to_one")
    if len(frame)==0 or frame.inner_fold.nunique()!=4:raise ValueError("invalid inner fold coverage")
    records=[]
    for model in ("glmnet_lin","glmnet_quad_rt"):
        for alpha in (0.0,0.5,1.0):
            for lam in (0.001,0.01,0.1,1.0):
                scores=[];failures=0;fallbacks=0
                start=time.monotonic()
                for fold in range(4):
                    tr=frame.loc[frame.inner_fold!=fold];val=frame.loc[frame.inner_fold==fold]
                    design=FoldDesign(model,standardize=True).fit(tr)
                    a=design.transform(tr).to_numpy();b=design.transform(val).to_numpy()
                    ytr=tr["pres.abs"].to_numpy(int);yval=val["pres.abs"].to_numpy(int)
                    outcome=run_elasticnet_inner(ytr,yval,a,b,alpha=alpha,lam=lam)
                    if outcome.fallback:fallbacks+=1
                    if outcome.status=="fit_failure_fallback":failures+=1
                    scores.append(float(relative_log_loss_skill(ytr,yval,outcome.predictions)))
                records.append({"model":model,"taxon":args.taxon,"outer_fold":args.outer_fold,"alpha":alpha,"lambda":lam,"mean_inner_relative_logloss_skill":float(np.mean(scores)),"fallback_folds":fallbacks,"fit_failure_folds":failures,"elapsed_seconds":round(time.monotonic()-start,4)})
    result=pd.DataFrame(records)
    if len(result)!=24 or result.isna().any().any():raise RuntimeError("invalid tuning trace")
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/"c2a_real_inner_pilot.csv",index=False)
    winners=result.sort_values(["model","mean_inner_relative_logloss_skill"],ascending=[True,False]).groupby("model").head(1)
    print("C2A TRAINING-ONLY INNER PILOT; NOT OUTER TEST RESULTS")
    print(winners.to_json(orient="records"),flush=True)
    print("PASS candidates",len(result),"inner fits",len(result)*4,flush=True)
if __name__=="__main__":main()
