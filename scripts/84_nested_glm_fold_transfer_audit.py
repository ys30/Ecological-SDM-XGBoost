"""Audit per-outer-fold threshold transfer from archived nested GLM artifact."""
from pathlib import Path
import pandas as pd
p=Path("results/alternative_spatial_validation/nested_glm_threshold/outer_fold_threshold_counts.csv")
d=pd.read_csv(p)
assert len(d)==3*5*3*3
assert (d.tp+d.fn==d.positives).all()
assert (d.tp+d.fp+d.fn+d.tn==d.n).all()
d["outer_fpr"]=d.fp/(d.fp+d.tn)
d["recall"]=d.tp/(d.tp+d.fn).replace(0,float("nan"))
d["fpr_ratio"]=d.outer_fpr/d.inner_fp_rate_target
d["fpr_excess"]=d.outer_fpr-d.inner_fp_rate_target
assert d.groupby(["buffer_km","species","inner_fp_rate_target"]).outer_fold.nunique().eq(5).all()
out=Path("results/alternative_spatial_validation/nested_glm_fold_audit")
out.mkdir(parents=True,exist_ok=True)
d.to_csv(out/"outer_fold_transfer_audit.csv",index=False)
summary=d.groupby(["buffer_km","species","inner_fp_rate_target"],as_index=False).agg(
 folds_exceeding_target=("fpr_excess",lambda x:int((x>0).sum())),
 worst_fold_fpr=("outer_fpr","max"),
 median_fold_fpr=("outer_fpr","median"),
 max_fpr_ratio=("fpr_ratio","max"),
 min_train_threshold=("threshold","min"),
 max_train_threshold=("threshold","max"),
 min_inner_negative_count=("inner_n_neg","min"),
 max_inner_negative_count=("inner_n_neg","max"))
summary.to_csv(out/"fold_transfer_summary.csv",index=False)
print("10KM FOLD TRANSFER SUMMARY")
print(summary[summary.buffer_km==10].to_string(index=False))
print("10KM FOLD DETAILS")
print(d[d.buffer_km==10][["outer_fold","species","inner_fp_rate_target","positives","tp","fp","tn","outer_fpr","threshold","inner_n_neg"]].to_string(index=False))
