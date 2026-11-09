# v6.22 pre-freeze reconciliation — 2026-11-09

## Confirmed CI failure
GitHub Actions run 42660152847, job 129309235239 failed at `Validate plan, references and manifest`. The exact log states:
`NOT FROZEN: missing candidate files: config/v622_experiment.yaml, config/v622_model_matrix.csv, config/v622_contrasts.csv, docs/EMS_v6.22_analysis_plan.md, docs/EMS_v6.22_manifest_sha256.txt`.

Checkout, Python setup and PyYAML installation succeeded. This is **not** a model failure.

## Candidate version discrepancy
The locally accessible `v622_plan_draft.zip` contains **18 models and 19 contrasts**, and its YAML pins the old `0c159b9...` base. The later candidate previously reported in the working session contains **19 models and 20 contrasts** and is intended for base `4c8cb67c654c379f0c8768fcbbbd9d4e682a7a3d`. The two must not be silently interchanged.

## Gate decision
- Keep branch **PRE-FREEZE**.
- Do not weaken CI assertions to fit the older draft without an explicit documented design decision.
- Recover or reconstruct the exact 19-model / 20-contrast candidate, with its full analysis plan, YAML and manifest.
- Confirm changes relative to the 18/19 draft, then commit all files atomically.
- Re-run preflight and check actual success before freezing; pilot follows only after freeze.

No model training, nested tuning, outer predictions or manuscript conclusion updates are authorized by this note.
