# v6.23 comparator protocol audit — 2026-11-18

## Confirmed identity audit
- Historical Bayesian OOF `results/bayes_spatial_cv_predictions.csv` versus v6.23 fixed XGBoost: **9,336/9,336** matching IDs; 0 metadata mismatches (taxon, plot, outer fold, observed label). CI: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44426156093.
- Historical taxon-specific XGBoost OOF `results/v614_common_input/taxon_specific_xgboost_oof.csv` versus v6.23: **9,336/9,336** matching IDs; 0 metadata mismatches. CI: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44417670616.
- These checks establish **held-out identity only**, not equal training exclusion or model inputs.

## Critical comparator limitation: existing 10-km GLM uses a DIFFERENT fold design
- v6.23 XGBoost `scripts/102_v623_fixed_xgb_outer_oof.py`: **primary** `results/spatial_2d_block_assignments.csv`, geodesic haversine 10-km exclusion using `long,lat`, four covariates, taxon-specific XGBoost.
- Existing 10-km GLM `scripts/71_buffered_alternative_spatial_glm.py`: **alternative** `results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv`, projected easting/northing Euclidean 10-km exclusion (CRS meter assumption), fold-local standardization, taxon-specific regularized logistic regression.
- **Do not directly compare** existing 10-km GLM summary (`results/BUFFERED_GLM_ALTERNATIVE_SPATIAL_VERIFIED.md`) to v6.23 XGBoost as a controlled same-fold experiment. Even if held-out row coverage is the same, held-out membership and buffer geometry are not established as identical.
- The historical Bayesian OOF file contains matching held-out rows but the 10-km training exclusion has not been demonstrated. It is **not** a verified buffered comparator.

## Required controlled next experiment
1. Refit taxon-specific linear and quadratic GLMs on the **primary** frozen five outer folds, using **the same haversine >10 km retained training plots** as v6.23 XGBoost. Freeze features, scaling, regularization, and single-class fallback before running.
2. Save 9,336 row-level OOF predictions per GLM variant, fold-level class counts, buffer exclusions, manifest with hashes, and a strict row-level alignment gate.
3. Compare proper scores using per-taxon OOF log loss and Brier. Report AUC/AP only for eligible outer test cells (30/40) and distinguish pooled per-taxon descriptive ranking.
4. Use paired plot-level/taxon-level uncertainty, recognizing spatial dependence and only eight taxa. No confirmatory superiority claims from naive independent-row bootstrap.
5. A Bayesian comparison needs a separately verified identical-buffer refit or must be clearly marked as historical non-equivalent.

## Historical result protection
Keep v6.14 frozen and all v6.23 artifacts in new paths. Do not relabel the alternative-fold GLM experiment as primary-fold.
