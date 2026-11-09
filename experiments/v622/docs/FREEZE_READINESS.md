# EMS v6.22 freeze candidate checkpoint

Status: PRE-FREEZE pending final methodological audit.

Verified full GitHub Actions preflight:
- Run: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42704284461
- Tested commit: 0b18435f359bfb41dc942ffadde7449bb0fd2d2a
- Conclusion: SUCCESS
- Independent matrix gate previously passed (19 models / 20 contrasts).

The full preflight confirms file presence, manifest checksums, pinned base and matrix referential integrity. It does **not** confirm that the planned statistical comparisons are scientifically valid, that the software implementations reproduce the plan, or that a runtime pilot has completed.

## Open scientific audit item before freeze
Contrast C2a in config/v622_contrasts.csv compares glmnet_quad_rt against glmnet_lin. This changes both quadratic response specification and regularization/tuning assumptions? Both are glmnet, so the main difference is quadratic terms; however the linear and quadratic models have separately tuned alpha/lambda and should be interpreted as the effect of the *modeling pipeline* with that feature specification, not a pure fixed-hyperparameter contrast. Likewise C4 comparisons hold raw predictor identity but not equivalent objective functions or output probability semantics.

## Required freeze decision
Review all 20 contrast descriptions, model matrix definitions, projected CRS and whether the six fixed inputs are available to the eventual execution environment. Confirm the current candidate unchanged or amend and rerun CI before creating a separate freeze commit. No pilot or formal fitting until freeze.
