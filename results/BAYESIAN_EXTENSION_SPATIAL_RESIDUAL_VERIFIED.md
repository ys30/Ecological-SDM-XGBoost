# Verified Bayesian extension spatial residual diagnostics

Source: successful GitHub Actions run https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41772741169

| Model | Mean held-out residual Moran's I | Paired delta vs baseline | Taxon-fold cells with reduced Moran's I |
|---|---:|---:|---:|
| Archived linear hierarchical Bayesian | 0.341020 | — | — |
| Quadratic hierarchical Bayesian | 0.291470 | -0.049550 | 29/40 |
| Species-specific nonpooled Bayesian | 0.282463 | -0.058556 | 24/40 |

Method: 5-nearest-neighbor row-standardized directed weights, within each held-out spatial fold and taxon, applied to Bernoulli residuals (observed minus predicted probability) using projected easting/northing. Mean across 40 taxon-fold cells. This is a descriptive diagnostic, **not** a spatial permutation significance test. Positive spatial autocorrelation remains substantial in all models.

Interpretation: Quadratic environmental response terms reduce mean residual spatial autocorrelation while improving ranking and calibration metrics. The no-pooling variant reduces mean Moran's I further but does not consistently improve predictive scores. Do not equate reduced Moran's I with external spatial generalization or claim the residual spatial structure is resolved.

The private manuscript is not published in this repository.
