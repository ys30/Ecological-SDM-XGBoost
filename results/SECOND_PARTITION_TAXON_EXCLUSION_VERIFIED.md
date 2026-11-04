# Alternative spatial partition: taxon-exclusion sensitivity (verified)

Source: [successful GitHub Actions run 41863297113](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41863297113). Models are **not refitted** after excluding taxa: this analysis filters held-out predictions from two Bayesian models originally trained on all eight anonymous taxa. The result tests influence on the evaluation summary, **not** training-set leave-taxon-out transfer.

## Pooled out-of-fold differences (quadratic minus linear)

| Evaluation subset | Taxa | N/model | ΔROC AUC | ΔAverage precision | ΔBrier |
|---|---:|---:|---:|---:|---:|
| All taxa | 8 | 9336 | +0.014325 | +0.072722 | -0.006473 |
| Exclude nsw18 | 7 | 8169 | +0.003139 | -0.003838 | -0.001023 |
| Exclude nsw24 | 7 | 8169 | +0.008831 | +0.065439 | -0.006135 |
| Exclude both | 6 | 7002 | -0.006879 | -0.031416 | +0.000279 |

## Mean taxon-fold paired differences

| Evaluation subset | ΔROC AUC | ΔAP | ΔBrier | ΔLog loss |
|---|---:|---:|---:|---:|
| All taxa | +0.016753 | +0.023018 | -0.006479 | -0.017405 |
| Exclude nsw18 | +0.009075 | +0.014658 | -0.000959 | -0.001762 |
| Exclude nsw24 | +0.018534 | +0.014995 | -0.006166 | -0.015242 |
| Exclude both | +0.009629 | +0.003221 | +0.000325 | +0.003369 |

## Interpretation and limitation

The pooled quadratic advantage is substantially concentrated in nsw18 and nsw24. When both taxa are excluded from evaluation, pooled discrimination and probability metrics favor the linear model, although equal-cell mean AUC/AP differences remain slightly positive. This is evidence of **heterogeneous, taxon-dependent nonlinear response benefits**, not universal superiority. Pooled and equal-cell estimands must be distinguished. All results remain within one dataset and two spatial fold geometries. No inferential significance claim is made. Public reporting uses anonymous taxon IDs only.
