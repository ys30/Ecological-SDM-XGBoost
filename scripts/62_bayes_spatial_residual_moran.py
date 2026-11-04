"""Descriptive 5-nearest-neighbor Moran's I of held-out Bernoulli residuals.
Within each taxon and fold only; coordinates from public training table.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
files=sorted(Path("downloaded").rglob("*_oof.csv"))
assert len(files)==10
ext=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
base=pd.read_csv("results/bayes_spatial_cv_predictions.csv").rename(columns={"mean_probability":"probability"})
base["variant"]="baseline"
cols=["id","plot","species","fold","observed","probability","variant"]
df=pd.concat([base[cols],ext[cols]],ignore_index=True)
xy=pd.read_csv("data/train.csv")[["id","easting","northing"]]
assert xy.id.is_unique
df=df.merge(xy,on="id",validate="many_to_one")
assert df.groupby("variant").size().eq(9336).all()
rows=[]
for (variant,species,fold),g in df.groupby(["variant","species","fold"]):
    coords=g[["easting","northing"]].to_numpy(dtype=float)
    resid=(g.observed-g.probability).to_numpy(dtype=float)
    n=len(g);k=min(5,n-1)
    assert k>=1
    tree=cKDTree(coords)
    _,indices=tree.query(coords,k=k+1)
    nbr=indices[:,1:]
    centered=resid-resid.mean()
    denom=np.dot(centered,centered)
    moran=float(np.sum(centered[:,None]*centered[nbr])/(k*denom)) if denom>1e-15 else np.nan
    rows.append(dict(variant=variant,species=species,fold=int(fold),n=n,moran_i=moran,
                     mean_residual=float(resid.mean()),residual_variance=float(np.var(resid))))
r=pd.DataFrame(rows)
assert len(r)==120
out=Path("results/bayes_extensions");out.mkdir(parents=True,exist_ok=True)
r.to_csv(out/"spatial_residual_moran5_by_taxon_fold.csv",index=False)
wide=r.pivot(index=["species","fold"],columns="variant",values="moran_i")
paired=[]
for variant in ["quadratic","no_pool"]:
    d=(wide[variant]-wide["baseline"]).dropna()
    paired.append(dict(variant=variant,eligible_cells=len(d),mean_moran_i=float(wide[variant].mean()),
                       baseline_mean_moran_i=float(wide["baseline"].mean()),
                       mean_paired_delta_moran_i=float(d.mean()),
                       n_cells_reduced=int((d<0).sum())))
s=pd.DataFrame(paired);s.to_csv(out/"spatial_residual_moran5_summary.csv",index=False)
with open(out/"SPATIAL_RESIDUAL_REPORT.md","w") as f:
    f.write("# Held-out residual spatial autocorrelation\n\n")
    f.write("Moran's I calculated on Bernoulli residual y-p with row-standardized directed 5-nearest-neighbor weights, within each taxon and held-out fold. This is a descriptive diagnostic, not a permutation-based significance test. Coordinates are projected easting/northing from the public training data. Spatial block holdout is unbuffered.\n\n")
    f.write(s.to_markdown(index=False,floatfmt=".6f")+"\n")
print(s.to_string(index=False))
