# EMS v6.22 scientific audit — execution readiness
Status: PRE-FREEZE; full preflight is green but scientific readiness is not yet established.

## Evidence checked
- Full preflight run 42704984933: SUCCESS for commit c6e5cb456e7d512dce9e0519ae9633ce5aa775cd.
- results/SPATIAL_CV_PREPROCESSING_AUDIT.md: archived XGBoost constructs model.matrix before splitting; full-sample factor-level discovery is transductive (not proven target leakage). The v6.22 implementation must instead fit feature encodings on each training fold.
- results/NESTED_BAYESIAN_PILOT_PROTOCOL.md: nominal XY-to-km scale is not formally CRS-verified; no independent geographic external test.

## Specific implementation gates
1. **CRS:** inspect the source coordinate columns and authoritative spatial metadata; prove projected CRS and linear units before interpreting any 10-km buffer. A successful CI does not establish this.
2. **Train-only preprocessing:** encode taxon/categorical features using only inner training data, with an explicit unseen-level policy. Refit scaling and imputation per inner training split.
3. **MaxEnt comparability:** archived MaxEnt uses survey non-detections as background approximation. Keep the presence-background caveat and avoid treating cloglog as calibrated occurrence probability.
4. **Contrast C2a:** glmnet_quad_rt vs glmnet_lin isolates feature-specification *pipelines*, not fixed-hyperparameter quadratic effects, since alpha/lambda are retuned separately.
5. **Sparse taxon folds:** log all fallback events; do not interpret undefined AUC cells as zero differences. Report matched coverage.
6. **Bootstrap:** only eight taxa are available; the 2,000 taxon-cluster bootstrap intervals must be accompanied by explicit small-cluster limitations and leave-one-taxon-out sensitivity where feasible.
7. **Execution implementation:** no v6.22 model runner or runtime-only pilot workflow has been verified in the branch. Passing config preflight is not evidence of an executable experiment.

## Release gate
Do not freeze and do not start formal fitting until CRS verification, model runner tests, and a documented implementation review are complete. Amend plan/manifest and rerun preflight if any substantive specification changes.
