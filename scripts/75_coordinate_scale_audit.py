"""Audit whether projected XY distances agree with WGS84 geodesic distances.
This diagnoses scale consistency; it cannot establish a formal CRS or datum.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
d=pd.read_csv("data/train.csv")
p=d[["plot","Longitude","Latitude","easting","northing"]].drop_duplicates("plot")
assert len(p)==1167
assert p.notna().all().all()
xy=p[["easting","northing"]].to_numpy(float)
lon=np.deg2rad(p.Longitude.to_numpy(float));lat=np.deg2rad(p.Latitude.to_numpy(float))
# Sample nearest neighbors in XY, compare great-circle distances in metres.
dist,idx=cKDTree(xy).query(xy,k=2)
j=idx[:,1]
dlon=lon[j]-lon;dlat=lat[j]-lat
h=np.sin(dlat/2)**2+np.cos(lat)*np.cos(lat[j])*np.sin(dlon/2)**2
geo=6371008.8*2*np.arcsin(np.sqrt(np.clip(h,0,1)))
projected=dist[:,1]
valid=(geo>1)&(projected>1)
ratio=projected[valid]/geo[valid]
summary=pd.DataFrame([dict(n_plots=len(p),n_pairs=int(valid.sum()),
  xy_to_geodesic_median=float(np.median(ratio)),
  xy_to_geodesic_p05=float(np.quantile(ratio,.05)),
  xy_to_geodesic_p95=float(np.quantile(ratio,.95)),
  median_nearest_xy=float(np.median(projected[valid])),
  median_nearest_geodesic_m=float(np.median(geo[valid])),
  plausible_metre_scale=bool(.9<np.median(ratio)<1.1))])
out=Path("results/alternative_spatial_validation/coordinate_scale_audit")
out.mkdir(parents=True,exist_ok=True)
summary.to_csv(out/"coordinate_scale_summary.csv",index=False)
print(summary.to_string(index=False))
print("WARNING: scale agreement is NOT proof of EPSG/CRS; inspect authoritative metadata.")
