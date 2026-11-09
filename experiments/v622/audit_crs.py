#!/usr/bin/env python3
"""v6.22 coordinate audit: diagnostic only, never invent an EPSG code.

Run from repository root:
  python experiments/v622/audit_crs.py
Outputs a JSON report; exit 2 means distance-scale inconsistency.
No model training or outer metrics are performed.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

d = pd.read_csv("data/train.csv", usecols=["plot","long","lat","easting","northing"])
p = d.drop_duplicates("plot").sort_values("plot")
assert len(p) == 1167, "unexpected number of plots"
assert p.notna().all().all(), "missing coordinates"
assert p[["plot"]].duplicated().sum() == 0
xy = p[["easting","northing"]].to_numpy(float)
lon = np.deg2rad(p["long"].to_numpy(float))
lat = np.deg2rad(p["lat"].to_numpy(float))
dist, idx = cKDTree(xy).query(xy,k=2)
j = idx[:,1]
dlon = lon[j]-lon
dlat = lat[j]-lat
h = np.sin(dlat/2)**2 + np.cos(lat)*np.cos(lat[j])*np.sin(dlon/2)**2
geo = 6371008.8*2*np.arcsin(np.sqrt(np.clip(h,0,1)))
valid = (geo>1)&(dist[:,1]>1)
ratio = dist[valid,1]/geo[valid]
if not valid.any():
    raise SystemExit("No valid nearest-neighbor coordinate pairs")
report = {
  "n_plots":len(p),"n_pairs":int(valid.sum()),
  "median_xy_over_geodesic":float(np.median(ratio)),
  "p05":float(np.quantile(ratio,.05)),
  "p95":float(np.quantile(ratio,.95)),
  "metre_scale_plausible":bool(0.9<float(np.median(ratio))<1.1),
  "formal_crs_verified":False,
  "interpretation":"XY/geodesic ratio is diagnostic, not proof of EPSG or datum; verify authoritative metadata before nominal 10-km buffer."
}
out=Path("results/v622/crs_diagnostic.json")
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
if not report["metre_scale_plausible"]:
    raise SystemExit(2)
