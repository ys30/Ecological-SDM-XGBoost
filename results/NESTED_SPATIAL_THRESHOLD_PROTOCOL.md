# Nested spatial threshold selection protocol (pre-implementation specification)

## Why the existing OOF predictions are insufficient

The existing buffered GLM and Bayesian OOF predictions are valid for comparing held-out predictions, but **cannot be recycled as inner-validation predictions** for a leakage-free decision threshold: predictions for other outer folds were produced by models whose training sets may contain the current outer test fold. Similarly, choosing the cutoff using the current outer-fold test labels is the oracle diagnostic documented in [RARE_TAXON_FP_BUDGET_VERIFIED.md](RARE_TAXON_FP_BUDGET_VERIFIED.md), not a deployment protocol.

## Required nested design

For each outer spatial fold (five), buffer distance (5/10/20 km), species, and model:

1. Fix the outer test plots and exclusion buffer. Build the outer training set by removing any training plot within the specified buffer of an outer test plot, exactly as in the existing buffered pipelines.
2. Create **inner spatial folds only among outer training plots**. Freeze inner fold assignments before using the species outcome. An inner validation plot must not be used for training, preprocessing, feature selection, calibration, or threshold tuning for that inner fold.
3. For each inner fold, additionally remove training plots within the chosen spatial buffer of the inner validation plots. Fit the GLM or Bayesian model using **only inner-training data**, including fold-local feature standardization and any quadratic terms.
4. Concatenate genuine inner out-of-fold probabilities and outcomes. Predeclare the decision rule (e.g. select a probability threshold to keep inner false positives below a specified *rate*, with a deterministic tie policy). A fixed absolute FP count cannot be transported between different inner and outer sample sizes without a prespecified scaling rule.
5. Refit the model once using the complete buffered outer-training set. Apply the selected inner threshold **unchanged** to the untouched outer test fold. Save only aggregate TP, FP, FN, TN and the selected threshold for public reports.
6. Aggregate outer-fold confusion counts for each species, buffer, model and prespecified operating budget. Report empirical outer false positives as an **outcome**, not a guaranteed cap. Do not tune the rule after seeing outer results.

## Required safeguards and acceptance tests

- Assert outer test IDs disjoint from every inner fit and inner threshold-selection record.
- Assert inner validation IDs disjoint from inner fitting IDs; enforce spatial buffers with the same distance convention as the published model pipelines.
- Refit standardization separately within every inner fit; never use full-data scaling.
- Verify inner spatial folds are nonempty, and record when rare species have no positive inner-validation observations; avoid silently selecting arbitrary cutoffs.
- For a fixed FP-rate budget, specify denominator, quantile/tie handling, threshold candidates, and fallback when validation predictions are degenerate.
- Preserve identical outer folds, training exclusions, predictor sets and outcome labels for GLM and Bayesian; compare their actual **outer** confusion counts.
- Log all seeds, split hashes, model convergence checks and the selected threshold provenance.
- Keep source species identities, plot-level predictions and restricted input data out of the public artifacts.

## Status

**Protocol only. Not executed.** The prior oracle results are exploratory ranking diagnostics, not nested threshold estimates. Implementing this properly requires new inner-fold fits, including computationally expensive Bayesian fits, and verification of inner spatial partitioning and leakage guards. Do not report a prospective decision advantage before those runs succeed.
