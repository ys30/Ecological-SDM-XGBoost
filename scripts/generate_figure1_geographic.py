#!/usr/bin/env python3
"""Figure 1: geographically grounded NSW sample and spatial holdout map."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "spatial_2d_block_assignments.csv"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
with CSV.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 1167, f"Expected 1167 samples; got {len(rows)}"
lon = np.array([float(r["long"]) for r in rows])
lat = np.array([float(r["lat"]) for r in rows])
fold = np.array([int(r["fold"]) for r in rows])
assert set(fold) == {1,2,3,4,5}
assert all(np.isfinite(lon)) and all(np.isfinite(lat))
print("Fold counts:", {i:int((fold==i).sum()) for i in range(1,6)})
pc = ccrs.PlateCarree()
fig = plt.figure(figsize=(13.5, 6.5), constrained_layout=True)
ax1 = fig.add_subplot(1, 2, 1, projection=pc)
ax2 = fig.add_subplot(1, 2, 2, projection=pc)
# Natural Earth is downloaded by Cartopy when missing; failure is fatal, not replaced by a sketch.
land = cfeature.NaturalEarthFeature("physical", "land", "10m", facecolor="#f2f4f2", edgecolor="none")
coast = cfeature.NaturalEarthFeature("physical", "coastline", "10m", facecolor="none", edgecolor="#56616a")
states = shapereader.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces")
records = list(shapereader.Reader(states).records())
nsw = [r for r in records if r.attributes.get("name") == "New South Wales" and
       (r.attributes.get("admin") == "Australia" or r.attributes.get("name_en") == "New South Wales")]
if not nsw:
    raise RuntimeError("New South Wales state polygon not found in Natural Earth admin-1 data")
for ax in (ax1, ax2):
    ax.add_feature(land, zorder=0)
    ax.add_feature(coast, linewidth=.6, zorder=1)
    ax.gridlines(draw_labels={"bottom":"x","left":"y"}, linewidth=.35, color="#a6afb4", alpha=.55)
ax1.set_extent([112,155,-44,-10],crs=pc)
for r in nsw:
    ax1.add_geometries([r.geometry], pc, facecolor="#ffcd8a", edgecolor="#c65d28", linewidth=1.2, zorder=3)
ax1.scatter(lon,lat,s=1.4,c="#182c4c",transform=pc,zorder=4)
ax1.set_title("A  Australia and New South Wales",loc="left",weight="bold")
# True sample extent, not a fabricated survey-area polygon.
padx,pady=.35,.35
ax2.set_extent([lon.min()-padx,lon.max()+padx,lat.min()-pady,lat.max()+pady],crs=pc)
for r in nsw:
    ax2.add_geometries([r.geometry],pc,facecolor="none",edgecolor="#b26a41",linewidth=.8,zorder=2)
palette=["#2d6ca2","#ed7d31","#4d9b74","#9467bd","#d4525c"]
for i,color in enumerate(palette,1):
    ax2.scatter(lon[fold==i],lat[fold==i],s=7,alpha=.8,color=color,label=f"Fold {i} (n={(fold==i).sum()})",rasterized=True,transform=pc,zorder=5,linewidths=0)
ax2.legend(loc="upper right",fontsize=8,framealpha=.93,markerscale=1.7)
ax2.set_title("B  NSW survey plots and spatial holdout folds",loc="left",weight="bold")
fig.suptitle("Geographic context and within-region spatial holdout assignment",fontsize=13,weight="bold")
fig.text(.5,.01,"Coordinates: WGS84 (longitude/latitude). Geographic boundaries: Natural Earth 10m.\n"
         "Points are survey plots; fold colors indicate validation assignment, not contiguous fold polygons.",
         ha="center",va="bottom",fontsize=8)
for ext in ("png","pdf"):
    dest=OUT / f"figure1_geographic_nsw_holdouts.{ext}"
    fig.savefig(dest,dpi=350,bbox_inches="tight",pad_inches=.16)
    print("Saved",dest)
plt.close(fig)
