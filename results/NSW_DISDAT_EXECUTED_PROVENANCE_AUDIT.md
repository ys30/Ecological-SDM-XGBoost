# Empirical NSW disdat provenance verification — GitHub Actions evidence

## Verified execution

- [Run 41286199826](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41286199826): successful compare job and uploaded `nsw-disdat-provenance-audit` artifact, source commit `6219ba1`.
- [Run 41286053395](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41286053395): successful compare job and uploaded artifact, source commit `b581993`.

Both artifact reports state:

| Diagnostic | Result |
|---|---:|
| Local species–plot records | 9,336 |
| Local distinct plots | 1,167 |
| Reference NSW open-forest PA rows | 2,075 |
| Reference NSW open-forest ENV rows | 2,075 |
| Shared environmental covariates | disturb, rainann, soildepth, soilfert, tempann, topo |
| Local plots matching a reference six-covariate signature | **1,167 / 1,167** |
| Distinct local six-covariate signatures | 1,165 |
| Distinct reference six-covariate signatures | 2,070 |
| Local positive species–plot records | 747 |
| Reference geographic CRS | `+proj=longlat +datum=WGS84` |

**Interpretation:** All local plot environmental signatures occur in the reference NSW dataset. This strongly supports the NSW `disdat` source hypothesis. It is **not yet** a coordinate-and-species-label identity proof: reference species are coded `nsw18`–`nsw25`, and the earlier reports did not compare geographic coordinates because their `x/y` column names were not recognized. The revised audit now recognizes geographic `x/y` when `disCRS("NSW")` reports `longlat`, and attempts joint coordinate-plus-environment matching.

## Remaining before source identity is declared

1. Inspect the new audit run for geographic and joint coordinate/environment match counts.
2. Resolve `nsw18`–`nsw25` to the eight taxonomic names using the authoritative `disdat` metadata.
3. Join reference PA labels by verified geographic site identity and compare all 9,336 binary observations.
4. Verify original source publication, license and predictor units.

Until step 3, manuscript language should say **environmental signatures matched**, not **all species observations were independently verified**. If full identity is established, describe this as a spatial-CV reanalysis of the reference presence–absence survey, not an external test on an untouched dataset.
