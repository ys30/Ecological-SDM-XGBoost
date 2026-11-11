# C1 historical R versus Python fixed-control equivalence audit

Status: **NOT EQUIVALENTLY VERIFIED** (2026-11-11). This is a source-code audit, not a numerical R/Python comparison.

## Inspected sources
- `scripts/42_common_input_xgboost.R`
- `scripts/45_v614_pooled_four_species_xgboost.R`
- `experiments/v622/xgb_backend.py`
- `experiments/v622/run_c1_fixed_controls.py`

## Exact matches by inspection
- Binary logistic objective, logloss evaluation metric, 300 boosting rounds, learning rate 0.03, max depth 4, subsample 0.8, colsample_bytree 0.8, two CPU threads.
- Four environmental predictors: rainann, soildepth, tempann, topo.
- Pooled model adds a one-hot indicator for each taxon.
- Fixed Python uses seed `1000 + outer_fold`, matching the seed formula in the historical R scripts.
- Fixed Python retains the same grid4 outer split; km15 is a prospective additional spatial design, not part of the historical v6.14 R scripts.

## Important differences / unverified equivalence
1. Historical R uses `xgb.train` and `xgb.DMatrix`; Python uses `XGBClassifier`. API defaults and library versions have not been numerically reconciled.
2. Python explicitly specifies `tree_method="hist"`; the historical R scripts do not specify a tree method. Their effective tree method is version-dependent.
3. Historical per-taxon R script emits `p=NA` for single-class training, whereas v6.22 Python uses training-only smoothed taxon-prevalence fallback when there are fewer than five positives or insufficient class support. This is an intentional v6.22 protocol difference, not numerical equivalence.
4. Historical R prediction clipping is [1e-15,1-1e-15]; v6.22 analysis clips at [1e-6,1-1e-6] per the prospective plan.
5. R version, XGBoost R version, Python XGBoost version, default values for unlisted booster parameters, RNG details, and row/column ordering have not been established as identical.
6. Historical R scripts operate only on grid4. The km15 fixed arm has no corresponding historical R artifact.

## Appropriate claims
- **Supported:** “We evaluated fixed XGBoost controls in Python using the archived v6.14 hyperparameter values under the v6.22 spatial-validation and fallback protocol.”
- **Not supported:** “We reproduced the original v6.14 R predictions exactly” or “the Python and R implementations are numerically identical.”

## Recommended numerical equivalence gate
Run original R and Python on identical grid4 nonfallback folds, pin package versions, align tree_method, all booster defaults, seed, row/column ordering and training matrix construction. Export row-level predictions and compare max absolute probability difference, mean absolute difference, rank concordance, and model metrics; report differences instead of assuming equivalence. Keep fallback rows separate because their protocols differ. Do not replace existing frozen OOF results silently.

## Current scientific interpretation
The C1 interaction remains an **exploratory within-Python comparison**, and log loss/Brier are secondary endpoints. The preregistered AUC support gate is not met. The nonfallback sensitivity (Actions 43050953967) and LOTO analysis (Actions 42967072873) improve internal robustness but do not establish R/Python equivalence.

## Executed numerical parity experiment — 2026-11-11

The first run (43057838997) failed during R installation due to Ubuntu apt DNS errors, **before modeling**. A pinned Ubuntu 24.04 / R 4.4.3 rerun **43079394306 succeeded**, artifact **13884406679**. R and Python used the same fixed hyperparameters and explicitly specified histogram tree construction; the comparison is a **hist-aligned diagnostic**, not a literal rerun of the historical unspecified-tree-method R environment.

| Per-record grid4 comparison | Per-species nonfallback | Pooled |
|---|---:|---:|
| Matched records | 8,866 | 9,336 |
| Mean absolute probability difference | 0.00758112 | 0.00630022 |
| Maximum absolute probability difference | 0.42882314 | 0.10920951 |
| Fraction abs diff < 1e-3 | 0.588653 | 0.328085 |
| Pearson correlation | 0.994077 | 0.996139 |

**Decision: numeric equivalence rejected at tight tolerances.** High correlation is not numerical equivalence. These diagnostics alone do not identify whether default parameters, software versions, or RNG differences dominate. Do not infer which is responsible without targeted ablation.

The R-side 18,202 rows are 8,866 nonfallback per-species and 9,336 pooled, with R OOF and diagnostic JSON uploaded as the artifact. The Python fixed control should continue to be described as a reimplementation sensitivity, not an exact historical replication. The interaction's proper-score findings remain within-Python and exploratory; the prespecified AUC gate is not met.
