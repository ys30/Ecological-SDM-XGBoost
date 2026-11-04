"""Taxon-cluster uncertainty for buffered paired GLM–Bayesian differences.
Bootstrap eight anonymized taxa, retaining their five fold cells together.
Descriptive percentile intervals, not inferential significance tests.
"""
from pathlib import Path
import numpy as np
import pandas as pd
p=Path("results/alternative_spatial_validation/buffered_four_model")
d=pd.read_csv(p/"paired_taxon_fold_differences.csv")
assert set(d.buffer_km)=={5,10,20}
assert set(d.variant)=={"linear","quadratic"}
assert set(d.metric)=={"auc","ap","brier","logloss"}
assert d.species.nunique()==8
rng=np.random.default_rng(20261104)
rows=[]
for (km,variant,metric),g in d.groupby(["buffer_km","variant","metric"]):
    taxa=g.groupby("species").glm_minus_bayesian.mean()
    vals=taxa.to_numpy()
    assert len(vals)==8
    draws=rng.choice(vals,size=(20000,8),replace=True).mean(axis=1)
    rows.append(dict(buffer_km=km,variant=variant,metric=metric,
                     n_taxa=len(vals),n_cells=len(g),
                     mean_taxon_difference=vals.mean(),
                     ci_025=np.quantile(draws,.025),ci_975=np.quantile(draws,.975)))
out=pd.DataFrame(rows).sort_values(["buffer_km","variant","metric"])
out.to_csv(p/"taxon_cluster_bootstrap_20000.csv",index=False)
print(out.to_string(index=False,float_format=lambda v:f"{v:.6f}"))
