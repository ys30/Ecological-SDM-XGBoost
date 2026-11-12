# C2a formal execution gate: lambda path and solver parity

**Status: BLOCKED FOR FORMAL OUTER OOF.** This is a methodological safeguard, not a failed CI run.

Frozen [v6.22 config](../config/v622_experiment.yaml) specifies alpha = [0, 0.5, 1] and a **30-value lambda path per alpha using glmnet's default ratio**, chosen jointly using training-only spatial inner folds. The successful [engineering smoke run 43286664343](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/43286664343) evaluated four hand-picked lambda values per alpha (0.001, 0.01, 0.1, 1.0). Its 96 fits and selected parameters are **not** a substitute for the 30-value paths. Do not use smoke winners for formal outer predictions.

## Required implementation decisions

1. **Path generation:** For each relevant training partition, generate the glmnet-style lambda maximum from the training response/design and alpha, with the appropriate glmnet default minimum ratio and path length 30. For alpha=0 (ridge), glmnet uses a near-zero alpha convention to construct its automatic path. Document the convention and software version. The path must not use held-out observations.
2. **Comparable candidates across inner folds:** Specify a leakage-safe common candidate grid for each outer training partition, or generate fold-specific paths and compare a predeclared common lambda fraction parameterization. Do not select candidates using different numeric lambda scales without documenting the mapping.
3. **Solver:** Prefer actual R `glmnet` for the frozen analysis, with `standardize=FALSE` after training-fold scaling, or demonstrate the Python adapter's numerical agreement at matching design/alpha/lambda settings. The current Python SAGA prototype is explicitly not certified R glmnet-equivalent.
4. **Selection metric:** Use the frozen relative log-loss skill, computed with null prevalence from each inner-training partition; four spatial inner folds, 12 train-only kmeans clusters. Select jointly over alpha and lambda with deterministic tie-breaking.
5. **Support and trace:** Enforce fewer-than-five-positive fallback separately in each inner-training partition; preserve convergence, retry, fallback, chosen candidate and per-fold scores.
6. **Formal scope:** 8 taxa × 5 outer folds × 2 co-primary spatial designs × 2 models, with fully independent nested tuning and OOF predictions. Do not run formal inference until the above gate passes.

## Current evidence

- C2b unpenalized GLM OOF and paired inference are completed, with primary AUC conclusion **INCONCLUSIVE**.
- C2b AP heterogeneity and nonfallback sensitivities are completed.
- C2a real-data inner-only smoke completed successfully, but is not a formal C2a analysis.

**No alteration of frozen C1/C2b data or results.** Any intentional deviation from the original glmnet path or solver must be explicitly recorded and labeled before analysis.
