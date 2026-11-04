# Alternative spatial GLM buffer sensitivity — verified

Evidence: [successful 0/5/10/20-km buffered refit run 41874313829](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41874313829).

For each geographic test fold, training plots closer than the specified radius to **any test plot** were excluded before refitting fold-local standardization and per-taxon L2-regularized logistic models. Evaluation stays fixed at 9,336 OOF records with 747 detections at each radius.

| Buffer | Linear AUC | Linear AP | Quadratic AUC | Quadratic AP | Quadratic Brier | Quadratic log loss |
|---|---:|---:|---:|---:|---:|---:|
| 0 km | 0.889224 | 0.341927 | 0.902159 | 0.428426 | 0.056302 | 0.196065 |
| 5 km | 0.889448 | 0.342662 | 0.902657 | 0.432188 | 0.056115 | 0.195095 |
| 10 km | 0.886364 | 0.338635 | 0.899413 | 0.423774 | 0.056561 | 0.197525 |
| 20 km | 0.877064 | 0.328384 | 0.891263 | 0.415261 | 0.056655 | 0.209761 |

Number of excluded training plots by fold at 5 km: 19, 12, 7, 2, 12; at 10 km: 39, 25, 24, 8, 22; at 20 km: 107, 50, 92, 108, 102. Training sample sizes shrink at larger radii; this is part of the intervention, so performance change cannot be attributed solely to spatial leakage removal.

The quadratic GLM has higher pooled AUC and AP than the linear GLM at all four radii. Its 20-km AUC is 0.891263, versus 0.902159 at 0 km. The apparent gain at 5 km is small and not evidence that buffering improves generalization.

**Limitations:** These are GLM-only refits. Bayesian, MaxEnt and XGBoost have not been rerun under the same buffers in this experiment. Projected easting/northing units are assumed to be meters; source CRS metadata still needs independent verification. Spatial holdouts are within the same dataset, not external validation. No inferential confidence interval is claimed.
