"""Sensitivity summary: complete AUC coverage versus sparse-fold exclusions."""
from pathlib import Path
import pandas as pd
root=Path("results/v622/c2a_combined_DIAGNOSTIC_ONLY")
d=pd.read_csv(root/"taxon_design_robustness_DIAGNOSTIC_ONLY.csv")
assert len(d)==16 and d.groupby("taxon").design.nunique().eq(2).all()
for minfold in [1,3,5]:
    eligible=d[d.valid_auc_folds>=minfold]
    summary=eligible.groupby("design").agg(taxa=("taxon","nunique"),
        mean_taxon_delta_auc=("mean_delta_auc","mean"),
        median_taxon_delta_auc=("mean_delta_auc","median"),
        positive_taxa=("mean_delta_auc",lambda x:int((x>0).sum()))).reset_index()
    summary.insert(0,"minimum_valid_folds",minfold)
    if minfold==1:all_rows=summary
    else:all_rows=pd.concat([all_rows,summary],ignore_index=True)
all_rows.to_csv(root/"auc_coverage_sensitivity_DIAGNOSTIC_ONLY.csv",index=False)
print("PASS AUC fold-coverage sensitivity",flush=True)
print(all_rows.to_string(index=False),flush=True)
print("DIAGNOSTIC ONLY; fold coverage changes estimand, not a significance test",flush=True)
