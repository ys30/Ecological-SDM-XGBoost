# EMS reference material: selective integration decision

The separately supplied `EMS_revision_results.md` and `ems-revision.bundle` are **reference material**, not verified current-project outputs. No reference branch is merged and no reference metrics are imported into current manuscript tables.

## Useful ideas to retain

1. **Information-set harmonization:** compare per-taxon versus pooled XGBoost, MaxEnt, per-taxon quadratic GLM, and hierarchical GLM/GLMM using identical held-out records and explicitly listed predictors. The distinction between *algorithm*, *response specification*, and *taxon pooling* is essential. Proposed quadratic GLM is an **additional baseline to evaluate**, not a verified winner.
2. **Spatial design sensitivity:** report multiple fold geometries and train–test buffers separately. The current project has already independently verified a second five-fold partition for Bayesian models. Do not transfer the reference's 5/10/20-km or 15-grid-allocation performance claims without running and verifying those experiments in the current codebase.
3. **Sparse-taxon utility:** supplement AUC/AP with fixed-budget top-k recall and Brier/log loss against a training-fold prevalence baseline. Avoid presenting a favorable overall metric as universal across taxa.
4. **Prediction provenance:** record source commit, data fingerprint, fold assignment, feature set, model parameters, seed and output checksum for every regenerated OOF prediction. A reference report alleges a version mismatch in historical XGBoost predictions; this is a **provenance question to audit**, not an established correction to current verified tables.

## Existing verified current-project findings that remain authoritative

- Second-partition linear vs quadratic Bayesian: 10 successful matched refits, 9,336 OOF predictions per variant, 34 evaluable AUC/AP taxon-fold cells; see [verified results](ALTERNATIVE_SPATIAL_BAYESIAN_VERIFIED.md).
- Quadratic gains are heterogeneous. Excluding nsw18 and nsw24 from evaluation reverses pooled AUC/AP/Brier differences; see [taxon-exclusion audit](SECOND_PARTITION_TAXON_EXCLUSION_VERIFIED.md).
- No external-validation or universal-model-superiority claim.

## Not adopted

- Numerical rankings, mixed-model confidence intervals, and historical XGBoost corrections from the reference report without reproducible re-execution.
- The external branch's files, scripts, and manuscript edits as-is.
- Any non-anonymized species labels or private manuscript files.

## Suggested next verified experiment

Run a **per-taxon quadratic GLM** and **per-taxon linear GLM** on the same alternative five-fold assignments and the same four environmental predictors used in the Bayesian refits, with fold-local preprocessing. Archive all 9,336 OOF predictions per variant and compare against the existing Bayesian linear/quadratic OOF records. This tests whether hierarchical pooling improves over a transparent per-taxon nonlinear baseline; it does not establish external generalization.

Keep the full manuscript private. This note is a research planning and provenance record, not a new numerical result.
