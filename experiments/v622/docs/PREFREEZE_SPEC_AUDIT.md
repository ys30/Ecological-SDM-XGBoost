# v6.22 pre-freeze specification audit (2026-11-09)

**Status: BLOCKED — not frozen; no new model fitting.**

## Verified GitHub checks
- Matrix gate run 42675818296: SUCCESS (19 models / 20 contrasts).
- Full preflight run 42675818325: FAILURE (the candidate YAML, plan, and matching manifest have not all been committed).

## Reconciliation required before freeze
1. Candidate plan section 7 says a negligible AUC effect is |mean| < 0.01 with both 95% intervals inside +/-0.02. Candidate YAML and final amendment instead require both intervals wholly inside +/-0.01. **Adopt the stricter +/-0.01 rule** in both files; reserve +/-0.02 for sensitivity.
2. Candidate plan section 6 describes mixed models as primary and bootstrap as fallback, while YAML and final amendment specify **taxon-cluster bootstrap as primary** and mixed models as auxiliary. Make the entire plan consistently reflect bootstrap primary.
3. Candidate plan section 3 says nine models are new; verify this against the actual 19-row matrix before making a numerical claim.
4. Candidate model matrix has an added glmnet_quad_rt and contrast C2f. Preserve these, but ensure all contrast definitions in the committed files match exactly.
5. Full preflight's SHA256 manifest must be generated from the exact committed YAML, matrix, contrasts and plan. Never reuse a manifest generated for different bytes.

## Freeze gate
Commit reconciled config/v622_experiment.yaml and docs/EMS_v6.22_analysis_plan.md, regenerate docs/EMS_v6.22_manifest_sha256.txt from exact bytes, pass full preflight, then freeze in a distinct SHA. Pilot is permitted only afterward.
