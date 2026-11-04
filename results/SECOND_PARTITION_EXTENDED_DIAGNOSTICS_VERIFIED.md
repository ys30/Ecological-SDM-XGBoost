# Second spatial partition: extended Bayesian diagnostic audit (verified)

Evidence: [successful workflow 41858553200](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41858553200). Ten independently refitted Bayesian models (linear and quadratic × five alternative geographic folds), 9,336 paired held-out predictions per model. Anonymous taxa only. **Within-dataset sensitivity**, not external validation.

## MCMC diagnostics across all 10 fits

| Diagnostic | Observed |
|---|---:|
| Maximum Rhat | 1.008223 |
| Minimum bulk ESS | 813.955 |
| Minimum tail ESS | 1411.514 |
| Minimum BFMI | 0.705190 |
| Divergences | 0 |
| Maximum-treedepth hits | 0 |

All specified thresholds passed.

## Calibration by taxon: quadratic minus linear

Negative ΔBrier and ΔECE10 indicate improvement.

| Taxon | ΔBrier | ΔECE10 |
|---|---:|---:|
| nsw18 | -0.044618 | -0.121587 |
| nsw19 | -0.000071 | -0.000990 |
| nsw20 | +0.000964 | +0.003906 |
| nsw21 | -0.000399 | -0.002490 |
| nsw22 | -0.000113 | -0.000691 |
| nsw23 | +0.000508 | +0.006541 |
| nsw24 | -0.008837 | -0.034529 |
| nsw25 | +0.000785 | +0.004774 |
| All pooled | -0.006473 | -0.005811 |

Brier improves for five of eight taxa. Most pooled improvement is concentrated in nsw18 and nsw24; do not imply universal taxon benefit.

## Held-out spatial residual autocorrelation

5-nearest-neighbor residual Moran's I (40 taxon-fold cells):
- Linear mean: 0.281513
- Quadratic mean: 0.247125
- Paired mean difference: -0.034388
- Moran's I lower in 24/40 cells.

Positive residual spatial autocorrelation persists. No permutation test or spatial independence claim. These descriptive Moran's I values are not interchangeable with a permutation p-value.

## Research interpretation

The quadratic Bayesian specification has favorable pooled discrimination, probability calibration and mean residual spatial structure under the second spatial partition, with all MCMC diagnostics meeting prespecified thresholds. Taxon-level heterogeneity and spatial residual dependence remain important limitations. The second fold geometry is not an independent external dataset. Manuscript files are kept private.
