"""Nested buffered spatial GLM: inner-OOF thresholds, untouched outer-fold evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
d=pd.read_csv("data/train.csv")
a=pd.read_csv("results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv")
d=d.merge(a[["plot","fold"]],on="plot",validate="many_to_one")
feat=["rainann","tempann","soildepth","topo"]
plots=d[["plot","easting","northing","fold"]].drop_duplicates("plot")
rows=[]
score_rows=[]
def fit_predict(train,test,taxon):
 x=train[train.Species==taxon];v=test[test.Species==taxon]
 assert len(x)>0 and len(v)>0
 scaler=StandardScaler().fit(x[feat])
 z=scaler.transform(x[feat]);q=scaler.transform(v[feat])
 z=np.column_stack((z,z[:,0]**2,z[:,1]**2))
 q=np.column_stack((q,q[:,0]**2,q[:,1]**2))
 y=x["pres.abs"].to_numpy(dtype=int)
 if np.unique(y).size<2:
  return np.full(len(v),(y.sum()+1)/(len(y)+2))
 model=LogisticRegression(C=100,solver="lbfgs",max_iter=3000,tol=1e-8,random_state=20261104)
 model.fit(z,y)
 return model.predict_proba(q)[:,1]
for km in (5,10,20):
 for fold in range(1,6):
  outertest=plots[plots.fold==fold]
  cand=plots[plots.fold!=fold].copy()
  dist=cKDTree(outertest[["easting","northing"]]).query(cand[["easting","northing"]],k=1)[0]
  cand=cand.loc[dist>=km*1000].copy()
  assert len(cand)>100
  xy=cand[["easting","northing"]].to_numpy(float)
  xy=(xy-xy.mean(axis=0))/xy.std(axis=0)
  cand["inner_fold"]=KMeans(n_clusters=4,n_init=10,random_state=20261104+fold).fit_predict(xy)
  inner=[]
  for inner_fold in range(4):
   valplots=cand[cand.inner_fold==inner_fold]
   trainplots=cand[cand.inner_fold!=inner_fold]
   dd=cKDTree(valplots[["easting","northing"]]).query(trainplots[["easting","northing"]],k=1)[0]
   trainplots=trainplots.loc[dd>=km*1000]
   assert len(trainplots)>0
   assert set(trainplots["plot"]).isdisjoint(set(valplots["plot"]))
   tr=d[d["plot"].isin(trainplots["plot"])]
   va=d[d["plot"].isin(valplots["plot"])]
   for taxon in ("nsw19","nsw20","nsw22"):
    prob=fit_predict(tr,va,taxon)
    vv=va[va.Species==taxon]
    inner.append(pd.DataFrame(dict(species=taxon,prob=prob,observed=vv["pres.abs"].to_numpy(dtype=int))))
  inner=pd.concat(inner,ignore_index=True)
  outertrain=d[d["plot"].isin(cand["plot"])]
  outer=d[d.fold==fold]
  for taxon in ("nsw19","nsw20","nsw22"):
   ii=inner[inner.species==taxon]
   negative=ii.loc[ii.observed==0,"prob"].to_numpy()
   assert len(negative)>0
   oo=outer[outer.Species==taxon]
   pp=fit_predict(outertrain,oo,taxon)
   yy=oo["pres.abs"].to_numpy(dtype=int)
   negouter=pp[yy==0]
   for rate in (.01,.02,.05):
    # Strictly greater than max allowed negative order statistic, using nextafter.
    allowance=int(np.floor(rate*len(negative)))
    sortedneg=np.sort(negative)[::-1]
    threshold=np.nextafter(sortedneg[allowance],np.inf) if allowance<len(sortedneg) else 0.
    pred=pp>=threshold
    score_rows.append(dict(buffer_km=km,outer_fold=fold,species=taxon,inner_fp_rate_target=rate,
      threshold=float(threshold),inner_negative_q50=float(np.quantile(negative,.5)),
      inner_negative_q90=float(np.quantile(negative,.9)),inner_negative_q98=float(np.quantile(negative,.98)),
      outer_negative_q50=float(np.quantile(negouter,.5)),outer_negative_q90=float(np.quantile(negouter,.9)),
      outer_negative_q98=float(np.quantile(negouter,.98)),
      inner_negative_above_threshold=int(np.sum(negative>=threshold)),
      outer_negative_above_threshold=int(np.sum(negouter>=threshold)),
      n_inner_negative=len(negative),n_outer_negative=len(negouter)))
    rows.append(dict(buffer_km=km,outer_fold=fold,species=taxon,inner_fp_rate_target=rate,
      threshold=float(threshold),inner_n_neg=len(negative),inner_allowed_fp=allowance,
      n=len(yy),positives=int(yy.sum()),tp=int((pred&(yy==1)).sum()),
      fp=int((pred&(yy==0)).sum()),fn=int((~pred&(yy==1)).sum()),tn=int((~pred&(yy==0)).sum())))
out=Path("results/alternative_spatial_validation/nested_glm_threshold")
out.mkdir(parents=True,exist_ok=True)
z=pd.DataFrame(rows)
z.to_csv(out/"outer_fold_threshold_counts.csv",index=False)
agg=z.groupby(["buffer_km","species","inner_fp_rate_target"],as_index=False)[["tp","fp","fn","tn"]].sum()
agg["recall"]=agg.tp/(agg.tp+agg.fn)
agg["empirical_fpr"]=agg.fp/(agg.fp+agg.tn)
agg.to_csv(out/"nested_glm_pooled_metrics.csv",index=False)
score=pd.DataFrame(score_rows)
score.to_csv(out/"outer_fold_score_shift.csv",index=False)
print("10 KM SCORE SHIFT, 2% TARGET")
print(score[(score.buffer_km==10)&(score.inner_fp_rate_target==.02)].to_string(index=False))
print("NESTED GLM 10 KM")
print(agg[agg.buffer_km==10].to_string(index=False))
print("No Bayesian nested comparator has been fitted. Inner rate targets do not guarantee outer rates.")
