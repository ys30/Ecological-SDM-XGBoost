"""Fixed historical v6.14 XGBoost controls, grid4 and km15; no tuning."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.v622.xgb_backend import XGBoostC1Backend
from experiments.v622.xgb_inner_runner import _features
from experiments.v622.validate_c1_oof import validate_oof

HASHES={"data/train.csv":"f7eb81447e0e6deab1a10579ec1a6694587c6c61aa56947fe6a0d6012c3b1763",
"results/spatial_2d_block_assignments.csv":"4e9c2d698ab44cf11282d12429d7a8e016f95e4e50baa1cf2653f558a7af2a81",
"results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv":"04686634473d51b5c63dcbd4474659fe864a114ae0885c5c5f55736522c798ce"}
FIXED={"candidate_id":"v614_fixed","n_estimators":300,"max_depth":4,"learning_rate":.03,
"min_child_weight":1.0,"subsample":.8,"colsample_bytree":.8}

def main():
    p=argparse.ArgumentParser();p.add_argument("--design",required=True,choices=["grid4_s2026_b0","km15_b0"]);p.add_argument("--output-dir",default="results/v622");args=p.parse_args()
    marker=json.loads(Path("experiments/v622/docs/FROZEN_C1_FIXED.json").read_text())
    if marker.get("status")!="FROZEN" or args.design not in marker.get("designs",[]) or marker.get("fixed_config")!=FIXED:
        raise SystemExit("REFUSED: fixed-control freeze missing or inconsistent")
    manifest=Path("experiments/v622/docs/EMS_v6.22_manifest_sha256.txt")
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=marker["manifest_sha256"]:
        raise SystemExit("REFUSED: preregistration manifest changed")
    for line in manifest.read_text().splitlines():
        if line.strip():
            sha,rel=line.split("  ",1)
            if hashlib.sha256((Path("experiments/v622")/rel).read_bytes()).hexdigest()!=sha:raise SystemExit("REFUSED: manifest mismatch "+rel)
    path="results/spatial_2d_block_assignments.csv" if args.design=="grid4_s2026_b0" else "results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv"
    for x in ("data/train.csv",path):
        if hashlib.sha256(Path(x).read_bytes()).hexdigest()!=HASHES[x] or marker["inputs"][x]!=HASHES[x]:
            raise SystemExit("REFUSED: source SHA mismatch "+x)
    data=pd.read_csv("data/train.csv");assign=pd.read_csv(path)
    if len(assign)!=1167 or not assign["plot"].is_unique or set(assign["plot"])!=set(data["plot"]):raise ValueError("bad assignments")
    taxa=np.array(sorted(data.Species.unique()),dtype=object)
    outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=True)
    for variant in ("species","pooled"):
        rows=[]
        for fold in sorted(assign.fold.unique()):
            test_plots=set(assign.loc[assign.fold==fold,"plot"])
            tr=data.loc[~data["plot"].isin(test_plots)]
            te=data.loc[data["plot"].isin(test_plots)]
            backend=XGBoostC1Backend(variant,seed=1000+int(fold))
            for taxon in taxa:
                trt=tr.loc[tr.Species==taxon];tet=te.loc[te.Species==taxon]
                fit=trt if variant=="species" else tr
                a,b=_features(fit,tet,variant,taxa)
                y=fit["pres.abs"].to_numpy();fallback=int(y.sum()<5 or np.unique(y).size<2);error=""
                if not fallback:
                    try:pred=backend.primary(FIXED,a,y,b,int(fold),str(taxon))
                    except Exception as e:
                        error=type(e).__name__
                        try:pred=backend.retry(FIXED,a,y,b,int(fold),str(taxon))
                        except Exception as e2:fallback=1;error+=";"+type(e2).__name__
                if fallback:
                    yt=trt["pres.abs"].to_numpy()
                    pred=np.full(len(tet),(yt.sum()+.5)/(len(yt)+1))
                for (_,record),prob in zip(tet.iterrows(),pred):
                    rows.append({"plot":record["plot"],"Species":str(taxon),"outer_fold":int(fold),
                                 "pres.abs":int(record["pres.abs"]),"pred":float(prob),
                                 "fallback":fallback,"fit_error":error,"candidate_id":FIXED["candidate_id"]})
        df=pd.DataFrame(rows)
        print("FIXED AUDIT",args.design,variant,validate_oof(df,data,assign),flush=True)
        target=outdir/f"oof_xgb_{variant}_fixed_{args.design}.csv.gz"
        if target.exists():raise FileExistsError(str(target))
        df.to_csv(target,index=False,compression="gzip")
        print("PASS FIXED",args.design,variant,len(df),int(df.fallback.sum()),flush=True)
if __name__=="__main__":main()
