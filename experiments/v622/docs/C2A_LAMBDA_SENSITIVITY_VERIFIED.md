# C2a native glmnet lower-bound sensitivity — verified diagnostic

**Scope:** nsw18 only, grid4 outer fold 1, four inner spatial validation folds. **Not an outer OOF result.** The frozen C2a primary model comparison and parameter search remain unchanged.

## GitHub Actions evidence
- Native automatic path: run 43312177874 — success; 625 candidate-fold rows (automatic glmnet paths may truncate).
- Shared 30-candidate lambda grid: run 43314823983 — success; 720 candidate-fold scores.
- Shared-grid completeness and boundary check: run 43316767578 — success.
- Lower-bound extension: run 43323312648 — success; 24 sensitivity-only scores.

## Selected shared-grid candidates
| model | alpha | lambda | mean inner relative log-loss skill | grid index |
|---|---:|---:|---:|---:|
| glmnet_lin | 1.0 | 0.03500977 | 0.2514381 | 10 / 30 |
| glmnet_quad_rt | 0.5 | 0.00008598026783 | 0.4466996 | 30 / 30 |

The quadratic model selects the minimum lambda on the shared grid. Its skill advantage over the penultimate point is only 0.0001001012.

## Independent lower-bound sensitivity (quadratic only; alpha=0.5)
| lambda multiplier | mean inner skill |
|---:|---:|
| 1.00 | 0.4466977 |
| 0.50 | 0.4468379 |
| 0.25 | 0.4469025 |
| 0.10 | 0.4469396 |
| 0.05 | 0.4469520 |
| 0.01 | 0.4469606 |

The extension shows a small monotone improvement of ~0.000263 from the original lower bound to 1% of it. The 1x score differs by ~0.000002 from the original grid fit because the explicit lambda path differs; this is not an outer prediction and must not be presented as a substantive effect.

## Methodological gates before official C2a
1. The shared-grid diagnostic obtains endpoints from the full **outer-training** set, then evaluates identical numeric lambdas in all inner folds. That is a documented operational choice; compare against the frozen 30-value glmnet-default-ratio specification before calling it the preregistered implementation.
2. The Python design builder uses training-fold standardization before quadratic expansion; native R glmnet uses `standardize=FALSE` on the constructed matrix. Check equivalence to the intended specification and preserve train-only transformations.
3. Validate folds and label joins for all 8 taxa and both spatial designs, including class-support fallback and reproducible seeds.
4. Implement and audit complete outer-held-out OOF; only then compute paired AUC/CI under the frozen inferential gate.
5. The lambda extension is sensitivity-only. **Do not retrospectively expand the primary tuning grid** on the basis of these scores.

**Status:** Diagnostic gate passed; full formal C2a outer OOF not yet executed.
