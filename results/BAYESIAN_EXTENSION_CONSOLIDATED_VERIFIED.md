# Bayesian extension — consolidated verified findings (2026-11-04)

The manuscript and supplementary material remain private. This file is a public results summary.

## Design and comparability

- Five plot-blocked, unbuffered spatial folds; 1,167 plots, eight anonymous taxa, 9,336 held-out plot–taxon predictions per Bayesian variant.
- Main three-pipeline XGBoost/MaxEnt/Bayesian comparison uses **23 eligible taxon-fold cells**.
- Bayesian specification extension uses **30 eligible AUC/AP taxon-fold cells**; Brier and log loss use all **40** cells. These are **not** an independent second validation dataset.
- A separate matched sensitivity calculation on the **original 23 cells** has now passed [CI run 41772919959](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41772919959).

## Matched 23-cell extension relative to archived Bayesian baseline

| Variant | ΔAUC | ΔAP | ΔBrier |
|---|---:|---:|---:|
| Quadratic hierarchical Bayesian | +0.045253 | +0.083092 | -0.011821 |
| Species-specific nonpooled Bayesian | +0.005476 | +0.001278 | +0.001176 |

Positive ΔAUC/ΔAP and negative ΔBrier are improvements. These differences do not imply an independent replicate of the main comparison.

## Full extension (30 eligible AUC/AP cells; 40 Brier/log-loss cells)

| Variant | ΔAUC | ΔAP | ΔBrier | ΔLog loss |
|---|---:|---:|---:|---:|
| Quadratic | +0.037831 | +0.081273 | -0.006775 | -0.020342 |
| Nonpooled | +0.010708 | +0.013707 | +0.000389 | -0.000658 |

Taxon-cluster exploratory sign-flip p for quadratic: AUC 0.007812, AP 0.015625, Brier 0.062500, log loss 0.007812; eight taxa only, no multiplicity correction.

## Probability calibration and spatial diagnostics

| Model | Pooled ECE (10 bins) | Pooled Brier | Mean 5-NN residual Moran's I |
|---|---:|---:|---:|
| Archived Bayesian | 0.019853 | 0.060842 | 0.341020 |
| Quadratic | 0.012553 | 0.054037 | 0.291470 |
| Nonpooled | 0.020273 | 0.061226 | 0.282463 |

Quadratic reduced residual Moran's I in 29/40 taxon-fold cells; nonpooled in 24/40. Moran's I results are descriptive, not spatial permutation significance tests. Positive residual spatial autocorrelation remains.

## Taxon influence and rare-taxon retrieval

Excluding nsw18, quadratic still improves mean paired AUC by +0.028635, AP by +0.066314, Brier by -0.001343 and log loss by -0.006358. Quadratic gains retain favorable signs when excluding any single taxon. However, calibration gains are heterogeneous and nsw18 is influential for Brier.

Top-25 rare-taxon recorded detections across five folds (baseline / quadratic / nonpooled): nsw19 **3 / 4 / 8**, nsw20 **17 / 17 / 17**, nsw22 **4 / 5 / 4**. These are retrospective retrieval results, not prospective survey yields.

## Appropriate manuscript claim

Quadratic environmental response terms improve the archived Bayesian pipeline's matched within-region holdout performance, calibration and mean residual spatial autocorrelation. This is a **model-specification sensitivity analysis**, not a second independent cohort or external spatial validation. The archived baseline used shorter-chain OOF predictions; the extension has separate four-chain diagnostics. The three main SDM pipelines use different input representations and should not be described as controlled algorithm-only comparisons.

**Remaining work:** alternative spatial block geometry and/or genuinely independent geographic transfer; possible spatial permutation tests; update private manuscript and citations after access to its editable source.

## Evidence

- [Five-fold paired metrics](BAYESIAN_EXTENSION_FIVE_FOLD_VERIFIED.md)
- [Taxon-level robustness](BAYESIAN_EXTENSION_ROBUSTNESS_VERIFIED.md)
- [Probability calibration](BAYESIAN_EXTENSION_CALIBRATION_VERIFIED.md)
- [Spatial residual diagnostics](BAYESIAN_EXTENSION_SPATIAL_RESIDUAL_VERIFIED.md)
- [Original 23-cell verification](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41772919959)
- [Taxon influence verification](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41772517013)
