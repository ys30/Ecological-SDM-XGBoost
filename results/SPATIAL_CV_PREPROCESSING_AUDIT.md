# Spatial CV preprocessing and comparison audit (2026-11-01)

## Code inspected
- `scripts/11_spatial_2d_block_cv.R` (XGBoost)
- `scripts/13_maxent_spatial_2d_cv.R` (MaxEnt)
- `scripts/15_bayes_hierarchical_spatial.R` (Bayesian)

## Verified implementation details
1. All three methods reuse the five spatial folds from `results/spatial_2d_block_assignments.csv`; the XGBoost source generates those folds from plot coordinates without using response labels.
2. XGBoost constructs its `model.matrix` **once on the full dataset before splitting** into training and test folds. This uses full-sample factor levels for `Species`, `disturb`, and `soilfert`. No response labels are used in the matrix construction, so this is **not evidence of target leakage**; nevertheless, it is transductive feature-schema construction, and fold-local preprocessing is preferable for a strict inductive evaluation.
3. XGBoost fits a pooled classifier with `Species` as an input feature; MaxEnt fits separate species-specific models; Bayesian fits a hierarchical model with species-specific random effects. Differences in architecture and input features mean the matched comparisons evaluate *pipelines*, not algorithms under identical feature sets.
4. MaxEnt uses `rainann`, `soildepth`, `tempann`, `topo` and survey non-detections as a background approximation; its cloglog output should not be described as a calibrated occurrence probability.
5. Bayesian standardizes four predictors separately within each training fold and applies the training means and standard deviations to held-out data. This avoids test-set information in the scaling parameters.
6. Folds are geographic 4x4 quantile-grid blocks grouped into five folds, **not distance-buffered spatial CV**. A held-out block can border a training block, and no independent external test set is demonstrated.

## Actionable sensitivity analyses (not yet completed)
- Refit XGBoost using training-only feature encoding and an explicit unseen-category policy; compare held-out AUC/AP to the archived baseline before claiming equivalence.
- Consider a common-four-predictor XGBoost sensitivity model to isolate the effect of the additional predictors and species/categorical encoding.
- Quantify train/test nearest-neighbor geographic distances by fold, ideally in a suitable projected CRS, before describing geographic separation.
- Do not claim data leakage was found or ruled out exhaustively: this is a targeted source-code audit, not an end-to-end provenance audit.

## Manuscript language
Use *five-fold geographic block cross-validation* and *comparison of modeling pipelines*. Note that the MaxEnt output is relative suitability, not calibrated occurrence probability. Keep manuscripts private; this audit contains no manuscript text or confidential species identities.
