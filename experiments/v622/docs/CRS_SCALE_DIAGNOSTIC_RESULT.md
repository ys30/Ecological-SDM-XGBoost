# EMS v6.22 coordinate distance-scale diagnostic — verified run

- GitHub Actions run: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42715512368
- Source commit: 14f5992b0d6de65fbfb88c24efe17c3d437eeb32
- Outcome: SUCCESS; pinned data/train.csv SHA-256 matched.
- Plot count: 1,167; valid nearest-neighbor pairs: 1,167.
- Median projected-XY / WGS84 spherical nearest-neighbor distance: **1.0041124146694431**.
- 5th percentile: **1.0011640298615623**.
- 95th percentile: **1.006895762778911**.
- metre_scale_plausible: true.
- formal_crs_verified: false.

Interpretation: coordinate scale is consistent with metres at nearest-neighbor distances (median deviation about +0.411%). This does not identify a projection, EPSG code or geodetic datum, and is not a direct test of error over 10-km distances. Obtain authoritative coordinate metadata before claiming a verified 10-km geographic buffer. The original run's JSON is archived as GitHub Actions artifact ems-v622-coordinate-scale-diagnostic (artifact ID 13716954078).

No models were trained and no outer-fold predictions were generated.
