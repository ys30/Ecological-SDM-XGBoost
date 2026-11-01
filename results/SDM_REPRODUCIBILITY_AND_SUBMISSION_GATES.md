# SDM manuscript: reproducibility and submission gates

Updated 2026-11-01. This document distinguishes completed source verification from model and publication readiness.

## Completed: exact data identity

[Run 41287426532](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41287426532) verified 1,167/1,167 site coordinates in reference PA/ENV, all six environmental fields, a unique eight-species mapping, and zero disagreements among 9,336 binary labels. This establishes the identity of the working dataset as the NSW open-forest survey records in `disdat`, attributed to Valavi et al. (2023).

## Completed: independently recomputed three-model held-out metrics

[GitHub Actions run 41288740752](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41288740752) completed successfully, including the step **Recalculate 138 matched ROC AUC and AP values**. The committed [independent Python verifier](../scripts/37_validate_three_model_matched_metrics.py) joins the saved Bayesian, MaxEnt and XGBoost out-of-fold predictions by identical held-out observation keys and recomputes ROC AUC and average precision for 23 eligible species–fold cells (23 × 3 models × 2 metrics = 138 comparisons). The CI result establishes agreement between **saved predictions and published matched-fold summary values**; it does not establish that all three model-training procedures have been rerun.

## Bayesian computation: implementation verified, historical runtime pending

The committed [Bayesian fitting script](../scripts/15_bayes_hierarchical_spatial.R) explicitly loads `brms`, calls `brms::brm` through `brm(...)` with Bernoulli-logit likelihood and partial-pooling species effects, and calls `posterior::summarise_draws` and `rstan::get_sampler_params` for posterior and Stan sampler diagnostics. The original mode specifies two chains, 1,200 iterations per chain and 600 warmup; the `--confirm` mode specifies four chains, 2,000 iterations and 1,000 warmup. Five spatial folds are fitted separately with seeds `2026 + fold`.

The [saved convergence audit](BAYESIAN_FIVE_FOLD_CONVERGENCE_FINAL.md) documents 175/175 parameter-fold checks meeting its R-hat and ESS thresholds, with zero divergent transitions and zero treedepth-limit hits across 20 confirmation chains, and minimum BFMI 0.645. These are **repository-reported diagnostic results**, not a new independent rerun performed during this review.

**Important distinction:** `capture-sdm-environment.yml` explicitly installs only `xgboost`, `maxnet`, `pROC` and `ggplot2`; `scripts/32_capture_sdm_environment.R` checks availability of `brms`, `rstan` and `posterior` but does not install them. This separate inventory workflow is **not the Bayesian training environment**. Missing explicit installation in the inventory workflow is **not evidence that the historical Bayesian fits lacked these packages or were fabricated**. The actual historical Bayesian training runtime, package versions and end-to-end reproducibility are still unverified.

## Still open: computational reproducibility

The repository's `scripts/32_capture_sdm_environment.R` and `capture-sdm-environment.yml` produce a **current CI environment inventory** for R, xgboost, maxnet, pROC and ggplot2. The workflow does **not** install brms, rstan or posterior; missing versions are expected and must not be described as a complete Bayesian runtime record. Even when all packages are present, a newly captured version is **not evidence of the version used for the historical model results**.

Minimum evidence required for a reproducible final benchmark:

1. Persist per-fold XGBoost, MaxEnt and Bayesian predictions or regenerate them from a pinned environment.
2. Preserve train/test plot membership and assert no plot crosses train/test within a fold.
3. **Completed for the saved predictions:** verify per-record keys (`id,plot,fold,species,observed`) and recompute all 138 matched AUC/AP values; [CI run 41288740752](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41288740752) succeeded. Regeneration from fitted models remains separate.
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
