# v6.23 data provenance and leakage register

**Status:** Code-trace review, not a verified external-source reconstruction. Read alongside the CI-verified feature audit (run #44525359745).

## Confirmed from checked repository code

| Item | Evidence in repository | Assessment |
|---|---|---|
| Canonical data | `data/train.csv` and `scripts/95_v622_c1_input_preflight.py` | Eight taxa, 1,167 plots and 9,336 taxon–plot rows; checks labels, nonfinite predictors and fold coverage |
| Model inputs | `scripts/102_v623_fixed_xgb_outer_oof.py` | Four numeric variables: `rainann`, `soildepth`, `tempann`, `topo`; XGBoost receives unscaled training values |
| GLM transformation | `scripts/108_v623_matched_buffer_glm.py` | `StandardScaler().fit(xtr)` within each taxon and outer training split; quadratic terms are squared standardized rainfall and temperature |
| Spatial fold construction | `scripts/11_spatial_2d_block_cv.R` | Plot-coordinate median, longitude/latitude 4×4 quantile grid, deterministic shuffled greedy assignment to five folds; labels not used by the displayed assignment algorithm |
| Outer buffer | `scripts/102_v623_fixed_xgb_outer_oof.py`, `scripts/108_v623_matched_buffer_glm.py` | Excludes any candidate training plot within 10 km Haversine distance of an outer test plot |
| Source candidate | `scripts/33_audit_disdat_nsw_provenance.R` | Tests a possible match with `disdat::disPa('NSW','ot')` and `disdat::disEnv('NSW','ot')`; **the script is an investigation, not proof of identity** |
| Fit-time IDs | `scripts/117_v623_exact_training_id_audit.py` | Both models' recorded train/test ID digests match independently reconstructed ID sets in successful full CI #44463218295 |
| Feature integrity | `scripts/118_v623_feature_provenance_audit.py` | Passed in CI #44525359745; verifies numeric integrity and fold-local scaler reconstruction, **not upstream provenance** |

## Outstanding questions and publication gates

1. **Original data identity:** locate the saved output of `scripts/33_audit_disdat_nsw_provenance.R`; verify plot coordinate and environment signatures and species labels against the named source. Do not call the dataset `disdat`-derived solely because the script targets that package.
2. **Environmental feature origins:** document the units, temporal coverage, spatial resolution, missing-value handling, and original data provider for `rainann`, `soildepth`, `tempann`, and `topo`. A variable name does not establish these.
3. **Upstream target leakage:** check whether any predictor, site selection, imputation, aggregation or filtering used the presence/absence labels or the eventual held-out plots. Current fold-local GLM scaling cannot resolve this.
4. **Spatial fold artifact reproducibility:** rerun the geographic fold assignment script with a pinned R version, compare the generated plot-to-fold mapping to `results/spatial_2d_block_assignments.csv`, and report whether the assignment is reproduced byte-for-byte or as an identical mapping. The assignment script also trains a historical pooled model, which is **not** the v6.23 matched comparison.
5. **CRS:** the v6.23 10-km exclusion uses geographic longitude/latitude in radians with Haversine distance. Explicitly verify source CRS rather than assuming it from numeric coordinate ranges.
6. **Hyperparameters:** v6.23 uses a fixed XGBoost configuration and fixed GLM regularization; no nested tuning. State this prominently in the manuscript.
7. **Independent validation:** no external region or year has been tested under the v6.23 matched protocol.

## Manuscript-ready wording (conditional on source verification)

> We used four numeric environmental predictors (annual rainfall, soil depth, annual temperature, and topography) and an outcome-blind five-fold geographic block assignment derived from plot coordinates. For each outer test fold, candidate training plots within 10 km of any test plot were excluded. XGBoost used the four numeric predictors directly, whereas GLMs used a StandardScaler fitted exclusively to each outer training set. Fit-time train/test observation IDs were hashed independently for each model and checked against reconstructed memberships. Original environmental-data provenance and potential upstream leakage remain under audit.

**Important:** The descriptive English labels for the four predictors are provisional interpretations of variable names until the upstream data dictionary confirms definitions and units.

## Supporting runs

- [Fit-time membership and full three-model workflow #44463218295](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44463218295)
- [Feature audit #44525359745](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44525359745)
