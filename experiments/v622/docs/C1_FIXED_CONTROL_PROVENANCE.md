# C1a/C1b fixed-control provenance (v6.22)

The original v6.14 **fixed** common-four-predictor XGBoost arms are
`scripts/42_common_input_xgboost.R` (per-taxon) and
`scripts/45_v614_pooled_four_species_xgboost.R` (pooled with one-hot taxon).
Both explicitly use:

- `objective=binary:logistic`, `eval_metric=logloss`
- `eta=0.03`, `max_depth=4`, `subsample=0.8`,
  `colsample_bytree=0.8`, `nrounds=300`, `nthread=2`
- `min_child_weight` is not specified; XGBoost default is 1.
- Their original R fitting seed is `1000 + outer_fold`; v6.22
  tuned Python runner uses `100 + outer_fold`. These are **not**
  numerically identical implementations and seed differences must be
  disclosed if comparing Python reimplementations.

The fixed model IDs in the locked model matrix are
`xgb_species_fixed` and `xgb_pooled_fixed`. Their v6.21 settings
are documented as inherited from these archived implementations.

## Formal comparison design

Run each fixed arm on both existing outer spatial designs with identical
outer train/test splits, four predictors, one-hot species features for
pooled only, and explicit fallback/coverage logging. **Do not tune**
fixed arms on outer test data. Compute paired differences on matching
nonfallback taxon-fold cells and report all-record proper scores separately.

The tuned arms are already archived at:
grid4 Actions `42955194925`, km15 Actions `42960556320`.

Important: comparing a Python reimplementation of the R fixed arms
to tuned Python arms is a *reimplementation sensitivity* unless
equivalence (library versions, seeds, defaults) is established. Prefer
R-original fixed settings with a documented version or show both.

C1a = species tuned minus species fixed; C1b = pooled tuned minus
pooled fixed. The tuned-pooled minus tuned-species result is a distinct
sharing contrast and cannot substitute for C1a/C1b.
