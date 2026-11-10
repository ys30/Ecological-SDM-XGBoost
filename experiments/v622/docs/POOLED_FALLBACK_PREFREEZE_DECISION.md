# v6.22 C1/C3 pooled rare-taxon rule — pre-freeze decision proposal

**Status: PROPOSED, NOT ADOPTED.** This file does not change the active YAML or executable policy.

## Empirical diagnostic (outer grid fold 1, training data only)

Verified CI: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42923004038

Of 32 taxon × inner-fold cells, six have fewer than five taxon-level training positives, yet all six have adequate pooled fitting support:

| Taxon | Inner fold | Taxon positives | Pooled positives |
|---|---:|---:|---:|
| nsw19 | 0 | 4 | 431 |
| nsw19 | 1 | 3 | 462 |
| nsw19 | 2 | 2 | 477 |
| nsw20 | 2 | 0 | 477 |
| nsw19 | 3 | 3 | 472 |
| nsw22 | 3 | 1 | 472 |

The current implementation falls back to taxon training prevalence for these cells in **both** per-taxon and pooled XGBoost. The pooled model is actually fitted four times and its predictions are used for the other 26 cells.

## Proposed prospective rule

1. **Per-taxon fits:** require at least five training positives for the target taxon; otherwise use a constant predictor equal to that taxon's training prevalence and flag fallback.
2. **Pooled fits:** assess fitting support using the complete pooled inner-training partition. If the pooled partition has at least five positives and both classes, fit once per candidate and inner fold; predict for **every** validation taxon, including taxa with zero training positives.
3. **Pooled fit failure:** retry once using the specified XGBoost retry protocol; if unsuccessful, fall back separately for each validation taxon using its own inner-training prevalence and flag fallback.
4. **Scoring:** for both variants, calculate each taxon's log-loss skill against a null based only on that taxon's inner-training prevalence. Keep all eight taxa equally weighted. Record taxon-level training positives even if pooled predictions are not fallback.
5. **Evaluation:** retain the current paired AUC eligibility rule, and additionally stratify diagnostics by taxon training support (<5 versus >=5). A pooled prediction for a zero-positive taxon is not proof of reliable transfer.
6. **Scope:** this is a *prospective amendment* requiring a revised analysis plan, YAML, runner tests and freeze commit **before** formal nested tuning; it must not be backfilled as if preregistered originally.

## Why not switch automatically?

The current draft literally states that *taxon-level training groups with fewer than five positives* use a constant predictor. Interpreting pooled training as a distinct fitting group is scientifically reasonable but changes the estimand for the sharing comparison. The choice needs explicit sign-off, not an unnoticed implementation edit.

## Other required checks

- Test pooled-model failure behavior and cache reuse when rare taxa occur first in iteration order.
- Validate score-null separation: pooled fitting labels versus taxon-only null prevalence.
- Fix or document XGBoost-specific retry semantics (doubling rounds versus optimizer).
- Do not claim CRS/buffer verification from nearest-neighbor scale checks.
- Do not run full 30-candidate model selection or outer-fold evaluation until the plan is frozen.

**Decision required:** Adopt proposed pooled fitting-group rule, or retain the literal taxon-level fallback rule and acknowledge that rare taxa cannot measure sharing benefit under the latter.
