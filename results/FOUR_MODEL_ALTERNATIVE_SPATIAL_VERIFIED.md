# Four-model alternative spatial comparison — verified

Evidence: [successful matched OOF audit, run 41872107223](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41872107223). Four variants have exactly matching 9,336 held-out plot–taxon records each (747 positives). Alternative within-dataset geographic five-fold partition. AUC/AP taxon-fold metrics defined in 34/40 cells; Brier/logloss in all 40. No external validation.

## Pooled metrics

| Variant | AUC | AP | Brier | Logloss |
|---|---:|---:|---:|---:|
| Bayesian linear | 0.885563 | 0.335964 | 0.062382 | 0.204572 |
| Bayesian quadratic | 0.899888 | 0.408686 | **0.055909** | **0.187302** |
| GLM linear | 0.889224 | 0.341927 | 0.062431 | 0.204186 |
| GLM quadratic | **0.902159** | **0.428426** | 0.056302 | 0.196065 |

## Mean paired taxon-fold differences (first model minus second)

| Comparison | ΔAUC (34 cells) | ΔAP (34 cells) | ΔBrier (40 cells) | ΔLogloss (40 cells) |
|---|---:|---:|---:|---:|
| GLM quadratic − Bayesian quadratic | +0.006830 | +0.024598 | +0.000313 | +0.008431 |
| GLM linear − Bayesian linear | +0.016529 | +0.031371 | +0.000045 | −0.000435 |
| GLM quadratic − GLM linear | +0.007054 | +0.016246 | −0.006211 | −0.008539 |
| Bayesian quadratic − Bayesian linear | +0.016753 | +0.023018 | −0.006479 | −0.017405 |

GLM: per-taxon L2-regularized logistic regression with C=100; Bayesian: partially pooled hierarchical brms Bernoulli regression with regularizing priors. Both use the same four environmental variables; quadratic adds standardized rainfall and temperature squares. Comparisons do not isolate pooling from priors and estimation, and pooled metrics differ from equal-cell means. All findings are descriptive and exploratory. Taxon-exclusion sensitivity has shown heterogeneity and potential reversals. No significance or universal superiority claim.

This public report contains no private manuscript content.
