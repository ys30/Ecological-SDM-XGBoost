# SDM manuscript: reproducibility and submission gates

Updated 2026-11-01. This document distinguishes completed source verification from model and publication readiness.

## Completed: exact data identity

[Run 41287426532](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41287426532) verified 1,167/1,167 site coordinates in reference PA/ENV, all six environmental fields, a unique eight-species mapping, and zero disagreements among 9,336 binary labels. This establishes the identity of the working dataset as the NSW open-forest survey records in `disdat`, attributed to Valavi et al. (2023).

## Still open: computational reproducibility

The repository's `scripts/32_capture_sdm_environment.R` and `capture-sdm-environment.yml` produce a **current CI environment inventory** for R, xgboost, maxnet, pROC and ggplot2. The workflow does **not** install brms, rstan or posterior; missing versions are expected and must not be described as a complete Bayesian runtime record. Even when all packages are present, a newly captured version is **not evidence of the version used for the historical model results**.

Minimum evidence required for a reproducible final benchmark:

1. Persist per-fold XGBoost, MaxEnt and Bayesian predictions or regenerate them from a pinned environment.
2. Preserve train/test plot membership and assert no plot crosses train/test within a fold.
3. Verify per-record keys (`id,plot,fold,species,observed`) and compute matched metrics from held-out predictions.
4. Document fold eligibility and the reason each rare species or fold is excluded from ROC AUC.
5. Capture the actual Bayesian fit configuration, package/Stan versions, RNG seeds and diagnostics for the models whose per-record predictions appear in the matched comparison.
6. Validate numerical tables, supplementary figures and manuscript against regenerated outputs, rather than against copied text.

## Still open: original-source metadata

- Survey collection year(s) and geographic coverage: confirm from the original source, not inferred from GitHub file timestamps.
- Dataset license and any redistribution conditions for `data/train.csv`: check the `disdat` package metadata and original provider.
- Environmental raster/source datasets and measurement units: document per variable from original metadata.
- Data Availability: state precisely that the present work **reuses the published survey evaluation presence–absence records for internal spatial CV**; it does not perform a new external validation using the original presence-only/background training split.

## Manuscript interpretation

The observed model rankings and rare-species retrieval counts are descriptive; five spatial folds are not five independent datasets, and no statistical significance is implied. Top-25 within each fold screens 125 plots overall, not 25. The three rare species are Bayesian-only retrieval analyses, not part of the five-species matched benchmark.

**Decision:** The source-identity gate is closed. The computational reproducibility and original metadata gates remain open. Do not label the manuscript submission-ready until those checks and authorship/declarations are completed.
