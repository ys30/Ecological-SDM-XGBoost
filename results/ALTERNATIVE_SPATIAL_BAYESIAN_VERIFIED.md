# Second spatial partition: independently refitted Bayesian comparison (verified)

**Evidence:** [Successful ten-model refit](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41773784743); [successful paired OOF audit](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41779523515).

The second partition was constructed without outcomes: 15 spatial k-means clusters grouped into five folds, with 227, 229, 228, 239 and 244 plots, respectively. Adjusted Rand index versus the original partition: 0.095486. Both linear and quadratic hierarchical Bayesian specifications were independently refitted on each new training fold using four chains, 2,000 iterations and 1,000 warmup iterations per chain. Each model produced 9,336 paired held-out plot–taxon predictions, with 747 observed positives.

## Pooled out-of-fold metrics

| Model | ROC AUC | Average precision | Brier | Log loss |
|---|---:|---:|---:|---:|
| Linear hierarchical Bayesian | 0.885563 | 0.335965 | 0.062382 | 0.204572 |
| Quadratic hierarchical Bayesian | **0.899888** | **0.408686** | **0.055909** | **0.187302** |

## Taxon–fold paired mean differences (quadratic minus linear)

| Metric | Mean difference | Eligible cells | Favorable cells |
|---|---:|---:|---:|
| ROC AUC | +0.016753 | 34 | 22 |
| Average precision | +0.022993 | 34 | 25 |
| Brier | −0.006479 | 40 | 28 |
| Log loss | −0.017405 | 40 | 29 |

AUC/AP are not defined for every taxon–fold due to limited positive cases. Pooled and taxon–fold averaged metrics are different estimands and must not be conflated.

## Interpretation

Quadratic environmental response terms retain favorable average discrimination and probability scores under a materially different five-fold spatial partition. This provides **alternative-partition within-dataset robustness**, not independent external validation. Fold-level observations and the two partition designs are correlated; no independent-replication p-value is claimed. The original spatial partition's quadratic extension used an archived shorter-chain baseline, whereas the second partition refitted both specifications with matching MCMC settings. Residual spatial dependence, calibration by taxon, spatial buffers, and external transfer still require examination.

Manuscript is private; only anonymous public results and code are stored here.
