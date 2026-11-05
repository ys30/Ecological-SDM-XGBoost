# Verified rare-taxon threshold sweep

**Successful CI:** [#42103238234](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42103238234). Script: [81_rare_taxon_threshold_sweep.py](../scripts/81_rare_taxon_threshold_sweep.py). Artifact `rare-taxon-threshold-sweep` ID 13450871218 contains fold-level confusion counts and pooled threshold metrics. Predictions are matched by OOF record ID, with observed outcomes verified equal. All published outputs are aggregates.

Eight exploratory probability thresholds: 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50. Three taxa nsw19/nsw20/nsw22; 5, 10, 20 km buffers. No thresholds were selected using held-out outcomes for a deployment claim.

## 10 km buffered quadratic OOF model threshold comparisons

| Species | Threshold | GLM TP | GLM FP | Bayesian TP | Bayesian FP |
|---|---:|---:|---:|---:|---:|
| nsw19 | .02 | 7 | 64 | 5 | 126 |
| nsw19 | .10 | 2 | 21 | 0 | 0 |
| nsw20 | .01 | 11 | 44 | 14 | 138 |
| nsw20 | .02 | 11 | 27 | 10 | 82 |
| nsw20 | .05 | 10 | 21 | 3 | 33 |
| nsw20 | .10 | 8 | 15 | 1 | 8 |
| nsw20 | .20 | 6 | 11 | 0 | 1 |
| nsw22 | .02 | 1 | 43 | 4 | 59 |
| nsw22 | .10 | 0 | 10 | 1 | 1 |

For nsw20, GLM has greater recall than Bayesian at each *sampled* cutoff from .02 through .50, while Bayesian has higher recall at .005 and .01. At .02 GLM achieves 11 TP/27 FP versus Bayesian 10 TP/82 FP. For nsw22, the contrast reverses at .10: Bayesian 1 TP/1 FP versus GLM 0 TP/10 FP.

**Caveats:** Shared probability cutoffs are not matched false-positive budgets; these eight sampled points do not prove whole-curve precision–recall dominance. With 6–18 positives, operating-point estimates are unstable. This is within-dataset buffered spatial OOF validation, not external validation. No claim of statistical significance or universal superiority. See [calibration and error audit](RARE_TAXON_CALIBRATION_ERROR_VERIFIED.md) and [taxon bootstrap](BUFFERED_TAXON_BOOTSTRAP_VERIFIED.md).
