"""Audit low-positive taxon-fold metric instability in buffered model comparison."""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
base=Path("results/alternative_spatial_validation/buffered_four_model")
d=pd.read_csv(base/"paired_taxon_fold_differences.csv")
train=pd.read_csv("data/train.csv")
fold=pd.read_csv("results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv")
truth=train.merge(fold[["plot","fold"]],on="plot",validate="many_to_one")
assert len(truth)==9336
counts=truth.groupby(["Species","fold"]).agg(n_test=("pres.abs","size"),positives_test=("pres.abs","sum")).reset_index().rename(columns={"Species":"species"})
assert counts.groupby("species").positives_test.sum().sum()==747
targets=["nsw19","nsw20","nsw22"]
out=d[d.species.isin(targets)].merge(counts,on=["species","fold"],validate="many_to_one")
assert set(out.species)==set(targets)
out.to_csv(base/"rare_taxon_fold_differences_with_counts.csv",index=False)
summary=(out.groupby(["buffer_km","variant","species","metric"],as_index=False)
  .agg(eligible_folds=("fold","nunique"),mean_difference=("glm_minus_bayesian","mean"),
       min_difference=("glm_minus_bayesian","min"),max_difference=("glm_minus_bayesian","max")))
summary.to_csv(base/"rare_taxon_fold_instability.csv",index=False)
print("RARE TAXON FOLD POSITIVE COUNTS")
print(counts[counts.species.isin(targets)].to_string(index=False))
print("QUADRATIC AP/AUC FOLD DIFFERENCES, 10KM")
print(out[(out.buffer_km==10)&(out.variant=="quadratic")&out.metric.isin(["auc","ap"])].sort_values(["species","fold","metric"]).to_string(index=False,float_format=lambda x:f"{x:.5f}"))
print("RANGE OF QUADRATIC AP DIFFERENCES BY BUFFER AND TAXON")
print(summary[(summary.variant=="quadratic")&(summary.metric=="ap")].to_string(index=False,float_format=lambda x:f"{x:.5f}"))
