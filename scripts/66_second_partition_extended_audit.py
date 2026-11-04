"""Verify diagnostics, calibration, taxon heterogeneity and residual spatial dependence.
Second spatial partition; paired 10 refit artifacts; descriptive internal validation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

root=Path("downloaded")
def read_parts(suffix):
    paths=sorted(root.rglob("*_"+suffix+".csv"))
    assert len(paths)==10,(suffix,len(paths),[str(x) for x in paths])
    return pd.concat([pd.read_csv(p).assign(source=p.name) for p in paths],ignore_index=True)
diag=read_parts("diagnostics");sam=read_parts("sampler");pred=read_parts("oof")
assert diag.source.nunique()==10 and sam.source.nunique()==10
assert np.isfinite(diag[["rhat","ess_bulk","ess_tail"]]).all().all()
assert diag.rhat.max()<=1.01 and diag.ess_bulk.min()>=400 and diag.ess_tail.min()>=400
assert sam.divergences.sum()==0 and sam.treedepth_limit_hits.sum()==0 and sam.bfmi.min()>=.3
assert len(pred)==18672
plots=pd.read_csv("data/train.csv")[["plot","easting","northing"]].drop_duplicates("plot")
assert plots.plot.is_unique
pred=pred.merge(plots,on="plot",validate="many_to_one")
def ece(x):
    bins=np.minimum((x.probability.to_numpy()*10).astype(int),9)
    y=x.observed.to_numpy();p=x.probability.to_numpy()
    return sum(np.sum(bins==j)/len(x)*abs(y[bins==j].mean()-p[bins==j].mean()) for j in range(10) if np.any(bins==j))
cal=[]
for (variant,species),g in pred.groupby(["variant","species"]):
    cal.append(dict(variant=variant,species=species,n=len(g),positives=int(g.observed.sum()),brier=np.mean((g.observed-g.probability)**2),ece10=ece(g),bias=(g.probability-g.observed).mean()))
for variant,g in pred.groupby("variant"):
    cal.append(dict(variant=variant,species="ALL_POOLED",n=len(g),positives=int(g.observed.sum()),brier=np.mean((g.observed-g.probability)**2),ece10=ece(g),bias=(g.probability-g.observed).mean()))
cal=pd.DataFrame(cal)
a=cal[cal.variant=="linear"].drop(columns="variant")
b=cal[cal.variant=="quadratic"].merge(a,on="species",suffixes=("","_linear"),validate="one_to_one")
for k in ["brier","ece10","bias"]:b["delta_"+k]=b[k]-b[k+"_linear"]
spatial=[]
for (variant,species,fold),g in pred.groupby(["variant","species","fold"]):
    assert len(g)>=7
    coords=g[["easting","northing"]].to_numpy(dtype=float)
    neighbors=NearestNeighbors(n_neighbors=6).fit(coords).kneighbors(coords,return_distance=False)[:,1:]
    residual=g.observed.to_numpy()-g.probability.to_numpy()
    z=residual-residual.mean()
    I=float(np.mean(z[:,None]*z[neighbors])/np.mean(z*z)) if np.mean(z*z)>0 else np.nan
    spatial.append(dict(variant=variant,species=species,fold=fold,n=len(g),moran_i_5nn=I))
spatial=pd.DataFrame(spatial)
sp=spatial[spatial.variant=="quadratic"].merge(spatial[spatial.variant=="linear"].drop(columns="variant"),on=["species","fold"],suffixes=("","_linear"),validate="one_to_one")
sp["delta_moran_i"]=sp.moran_i_5nn-sp.moran_i_5nn_linear
out=Path("results/alternative_spatial_validation");out.mkdir(parents=True,exist_ok=True)
cal.to_csv(out/"calibration_by_taxon.csv",index=False)
b.to_csv(out/"calibration_paired.csv",index=False)
sp.to_csv(out/"spatial_residual_paired.csv",index=False)
pd.DataFrame([dict(max_rhat=diag.rhat.max(),min_ess_bulk=diag.ess_bulk.min(),min_ess_tail=diag.ess_tail.min(),min_bfmi=sam.bfmi.min(),divergences=int(sam.divergences.sum()),treedepth_hits=int(sam.treedepth_limit_hits.sum()))]).to_csv(out/"mcmc_diagnostics_summary.csv",index=False)
pooled=b[b.species=="ALL_POOLED"].iloc[0]
print("DIAGNOSTICS",diag.rhat.max(),diag.ess_bulk.min(),diag.ess_tail.min(),sam.bfmi.min())
print("CALIBRATION",b[["species","delta_brier","delta_ece10"]].to_string(index=False))
print("SPATIAL",sp[["moran_i_5nn","moran_i_5nn_linear","delta_moran_i"]].mean().to_dict(),"reduced",int((sp.delta_moran_i<0).sum()),"of",len(sp))
with open(out/"SECOND_PARTITION_EXTENDED_AUDIT.md","w") as f:
    f.write("# Second-partition extended audit: diagnostics, calibration, spatial residuals\n\n")
    f.write("Source: ten successful linear/quadratic four-chain refits, alternative 15-cluster spatial partition. All results are **within-dataset**, descriptive and exploratory.\n\n")
    f.write(f"## MCMC diagnostics\n\nMax Rhat {diag.rhat.max():.5f}; min bulk ESS {diag.ess_bulk.min():.1f}; min tail ESS {diag.ess_tail.min():.1f}; min BFMI {sam.bfmi.min():.4f}; divergences {int(sam.divergences.sum())}; treedepth hits {int(sam.treedepth_limit_hits.sum())}.\n\n")
    f.write("## Calibration (quadratic vs linear)\n\n"+b[["species","n","delta_brier","delta_ece10","delta_bias"]].to_markdown(index=False,floatfmt=".6f")+"\n\n")
    f.write(f"## Five-nearest-neighbor held-out residual Moran's I\n\nLinear mean {sp.moran_i_5nn_linear.mean():.6f}; quadratic mean {sp.moran_i_5nn.mean():.6f}; paired mean difference {sp.delta_moran_i.mean():+.6f}; reduced in {int((sp.delta_moran_i<0).sum())}/40 taxon-fold cells.\n\n")
    f.write("Moran's I is computed on plot coordinates within each held-out taxon-fold, using row-standardized 5-NN weights and residual y-p. No spatial permutation test or external geographic validation.\n")
