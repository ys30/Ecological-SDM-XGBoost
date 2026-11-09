# v6.22 Input verification — 2026-11-09

Baseline commit: `4c8cb67c654c379f0c8768fcbbbd9d4e682a7a3d`

| Path | Expected SHA-256 | Actual SHA-256 | Status |
|---|---|---|---|
| `data/train.csv` | `f7eb81447e0e6deab1a10579ec1a6694587c6c61aa56947fe6a0d6012c3b1763` | same | PASS |
| `results/spatial_2d_block_assignments.csv` | `4e9c2d698ab44cf11282d12429d7a8e016f95e4e50baa1cf2653f558a7af2a81` | same | PASS |

These two files were fetched from the pinned GitHub commit and their raw contents SHA-256 hashed. Hash algorithm self-test: SHA-256('abc') = `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`.

The other four archived inputs were reported as verified in the earlier preflight artifact; this report does not independently rehash those four.

**Gate:** Input version blocker resolved. **Not a freeze declaration.** Full candidate plan/config/matrix/contrasts must still be committed, validated against the baseline implementation, and frozen before runtime pilot or new analyses.
