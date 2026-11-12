# v6.22 C2b next execution gate — quadratic versus linear GLM

**Status: engineering-ready components, formal outer OOF not yet run.** This is a scoped execution checklist, not a result.

## Frozen contrast
`C2b` is a preregistered **primary** response-shape contrast: `glm_quad_rt` versus `glm_lin`, evaluated separately on `grid4_s2026_b0` and `km15_b0`. Both are unpooled taxon-specific binomial logistic GLMs with the same four environmental inputs. Quadratic terms apply only to rainfall and temperature. Do not substitute `glm_quad_env` or `glmnet_quad_rt`.

## Verified code inventory
- `experiments/v622/glm_adapter.py`: statsmodels binomial IRLS fit, primary 100 iterations, retry 200, integrated fallback policy; currently an engineering adapter, not verified historical R equivalence.
- `experiments/v622/check_spatial_glm_wiring.py`: synthetic training-only inner-fold smoke; explicitly **does not** produce real outer OOF.
- `experiments/v622/design_matrix.py`: training-only predictor design; inspect scaling and quadratic construction before any formal outer fitting.
- `experiments/v622/config/v622_contrasts.csv`: authoritative C2b comparator identifiers.

## Requirements before formal execution
1. Validate source input hashes and both immutable outer fold assignments against the frozen manifest; abort on mismatch.
2. For each outer fold, fit each taxon-specific GLM on **outer training only**, with the exact frozen four predictors and response terms. No outer-fold response values enter feature construction or scaling. Neither model requires nested hyperparameter search.
3. Apply frozen less-than-five-positive/single-class fallback and one-retry policy. Preserve failure type, fallback flag, training positives, outer fold and taxon per OOF row.
4. Produce one row per plot × taxon for each model and spatial design: exactly 9,336 OOF rows per arm per design, unique `(plot, Species, outer_fold)` keys and identical labels.
5. Audit GLM rank deficiency, complete/quasi separation, extreme predicted probabilities, missingness, and the R-versus-Python GLM numerical comparison before treating it as a historical reproduction.
6. Compute 40 paired taxon-fold cells per design, with ROC AUC only for paired eligible cells, secondary AP/log loss/Brier, and 2,000 taxon-cluster bootstrap draws seed 622. Report coverage and all exclusions.
7. Apply the preregistered AUC support gate **without alteration**: same direction, absolute mean difference >= 0.01 in both designs, and at least one design interval excluding zero. Otherwise report inconclusive or negligible only if the specified interval rule is met.
8. Publish immutable CI artifact and exact run IDs before adding any numeric C2b claim to the manuscript.

## Editorial dependency
The existing C1 figure and supplementary insert are separate exploratory work. Do not relabel C1's secondary log-loss/Brier findings as a supported preregistered C2b result. Do not conflate R-hist XGBoost parity with R GLM parity.

**Next implementation:** a frozen-manifest-gated `run_c2b_outer_glm.py` and dedicated Actions workflow, followed by paired C2b inference. No C2b numerical results are asserted in this document.
