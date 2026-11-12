"""C2a shared-grid diagnostic: boundary winners and candidate-path audit.
Does not refit or evaluate outer-held-out data.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace",default="results/v622/c2a_r_native/native_glmnet_shared_grid_trace.csv")
    ap.add_argument("--output",default="results/v622/c2a_r_native/shared_grid_boundary_audit.json")
    a=ap.parse_args()
    d=pd.read_csv(a.trace)
    required={"model","alpha","lambda","candidate_index","inner_fold","skill"}
    if not required.issubset(d):raise ValueError("missing trace columns")
    if len(d)!=720:raise ValueError(f"expected 720 candidate-fold records, got {len(d)}")
    result={}
    for model,g in d.groupby("model"):
        if len(g)!=360:raise ValueError("expected 360 records/model")
        for alpha,part in g.groupby("alpha"):
            if len(part)!=120 or set(part.inner_fold)!=set(range(4)):raise ValueError("incomplete alpha/fold grid")
            for fold,f in part.groupby("inner_fold"):
                if len(f)!=30 or set(f.candidate_index)!=set(range(1,31)):raise ValueError("incomplete lambda path")
            if part.groupby("candidate_index").lambda.nunique().max()!=1:raise ValueError("lambda grid differs across folds")
        agg=g.groupby(["alpha","candidate_index","lambda"],as_index=False).skill.mean()
        agg=agg.sort_values(["skill","alpha","candidate_index"],ascending=[False,True,True])
        best=agg.iloc[0]
        chosen=agg[(agg.alpha==best.alpha)].sort_values("candidate_index")
        result[model]={
            "best_alpha":float(best.alpha),"best_lambda":float(best["lambda"]),
            "best_index":int(best.candidate_index),"best_skill":float(best.skill),
            "on_min_lambda_boundary":bool(best.candidate_index==30),
            "on_max_lambda_boundary":bool(best.candidate_index==1),
            "alpha_path_skill_at_min_lambda":float(chosen.iloc[-1].skill),
            "alpha_path_skill_at_penultimate_lambda":float(chosen.iloc[-2].skill),
            "min_lambda_skill_minus_penultimate":float(chosen.iloc[-1].skill-chosen.iloc[-2].skill),
            "full_30_grid_verified":True
        }
    Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    if any(x["on_min_lambda_boundary"] for x in result.values()):
        print("BOUNDARY WARNING: selected minimum lambda; do not release formal OOF until sensitivity grid and path convention are audited.",flush=True)
    print("PASS trace completeness and fold-shared numeric lambda grid",flush=True)
if __name__=="__main__":main()
