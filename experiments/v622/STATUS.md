# EMS v6.22 — verified inputs, freeze pending

Branch: `ems-v622-experiments`
Pinned baseline: `4c8cb67c654c379f0c8768fcbbbd9d4e682a7a3d`

- Two primary baseline CSVs: SHA-256 **PASS**; see `INPUT_VERIFICATION.md`.
- Four archived supplementary inputs: prior preflight reported **PASS**.
- Complete candidate analysis plan, YAML, model matrix, contrasts: **NOT YET COMMITTED** to this branch.
- Pilot and new outer-fold predictions: **NOT RUN**.
- Preregistration status: **PRE-FREEZE**.

Next: commit the exact candidate files; inspect execution code for nested spatial CV, fallback handling and score definitions; freeze the commit before pilot.
