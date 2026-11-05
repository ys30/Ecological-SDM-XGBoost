# Coordinate-distance scale audit — verified

Successful [GitHub Actions run #41960211768](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41960211768), artifact `coordinate-distance-scale-audit`, script [75_coordinate_scale_audit.py](../scripts/75_coordinate_scale_audit.py).

| Statistic | Observed |
|---|---:|
| Distinct plots | 1,167 |
| Valid nearest-neighbor pairs | 1,167 |
| Projected XY / WGS84 spherical distance, median | 1.004112 |
| Ratio p05 | 1.001164 |
| Ratio p95 | 1.006896 |
| Median projected XY nearest-neighbor distance | 650.004614 |
| Median spherical nearest-neighbor distance (m) | 648.139421 |
| Meter-scale plausibility flag | True |

Each plot is paired with its nearest neighbor under the provided projected XY coordinates; distances are compared with great-circle distances computed from the supplied `long`/`lat` coordinates. The ratios support **approximately metre-scale projected units** for local distances. They do **not** identify an EPSG code, projection, geodetic datum, or formal accuracy of 5/10/20 km exclusion buffers. The spherical metric and projected metric have different geometry, so a ratio near 1 is not a proof of exact distance equivalence.

**Recommended manuscript wording:** “The supplied planar coordinates were used for Euclidean training-set exclusion at nominal 5, 10 and 20 km radii. A separate audit of 1,167 nearest-neighbor plot pairs showed a median planar-to-spherical distance ratio of 1.0041 (5th–95th percentile 1.0012–1.0069), supporting an approximately metre-scale interpretation; the original projection metadata remain unverified.”

Source-data site identity is independently documented in [NSW disdat provenance audit](NSW_DISDAT_EXECUTED_PROVENANCE_AUDIT.md). The buffer sensitivity results are [here](BUFFERED_FOUR_MODEL_MATCHED_VERIFIED.md), with [taxon bootstrap uncertainty](BUFFERED_TAXON_BOOTSTRAP_VERIFIED.md).
