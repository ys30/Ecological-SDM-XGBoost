"""Taxon-level heterogeneity in buffered GLM vs hierarchical Bayesian performance.

Descriptive associations only: eight taxa, repeated folds and three buffers.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
base=Path("results/alternative_spatial_validation/buffered_four_model")
d=pd.read_csv(base/"paired_taxon_fold_differences.csv")
train=pd.read_csv("data/train.csv")
counts=train.groupby("Species").agg(n_observations=("pres.abs","size"),n_detections=("pres.abs","sum"))
counts.index.name="species"
counts=counts.reset_index()
assert len(counts)==8 and counts.n_observations.eq(1167).all()
# Average across eligible folds for each taxon, model specification, radius and metric.
summary=(d.groupby(["buffer_km","variant","species","metric"],as_index=False)
         .agg(mean_glm_minus_bayesian=("glm_minus_bayesian","mean"),
              eligible_folds=("glm_minus_bayesian","size"))
         .merge(counts,on="species",validate="many_to_one"))
assert summary.species.nunique()==8
summary.to_csv(base/"taxon_heterogeneity.csv",index=False)
assoc=[]
for (km,v,metric),g in summary.groupby(["buffer_km","variant","metric"]):
    assert len(g)==8
    rho,p=spearmanr(g.n_detections,g.mean_glm_minus_bayesian)
    assoc.append(dict(buffer_km=km,variant=v,metric=metric,n_taxa=8,
        spearman_rho=float(rho),nominal_p=float(p),
        positive_taxa=int((g.mean_glm_minus_bayesian>0).sum()),
        negative_taxa=int((g.mean_glm_minus_bayesian<0).sum())))
a=pd.DataFrame(assoc)
a.to_csv(base/"detection_heterogeneity_associations.csv",index=False)
print("Taxon detection counts:")
print(counts.to_string(index=False))
print("Quadratic taxon-level AUC/AP at 5,10,20 km:")
print(summary[(summary.variant=="quadratic")&summary.metric.isin(["auc","ap"])].to_string(index=False,float_format=lambda x:f"{x:.5f}"))
print("Exploratory detection-count associations (NOT confirmatory):")
print(a.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
