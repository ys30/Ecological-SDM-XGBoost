# EMS v6.22 experiment branch — preparation gate

Status: **PRE-FREEZE / NO NEW EXPERIMENTS EXECUTED**.

- Branch: `ems-v622-experiments`
- Pinned baseline commit: `4c8cb67c654c379f0c8768fcbbbd9d4e682a7a3d`
- Previous unresolvable baseline `0c159b9368ddd0f38db28d0b859c10258ab101a8` is **not required**.
- Candidate plan is maintained separately as `EMS_v622_NEW_BASE_CANDIDATE.zip`; it has **not yet been committed to this branch**.
- Candidate includes 19 model configurations and 20 planned contrasts.
- Before freezing: verify SHA-256 of all six inputs against the exact baseline snapshot; resolve any differences by documented amendments; commit the complete plan, YAML, model matrix, contrasts and changelog.
- Only after freezing: run the runtime-only pilot, then the preregistered analyses. No outer-test metrics may inform tuning or plan amendments.
- Existing 23 spatial schemes remain fixed-parameter sensitivity evidence; do not confuse them with new nested-tuning results.

This document records the actual branch state and must not be interpreted as a frozen preregistration.
