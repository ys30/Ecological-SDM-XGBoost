#!/usr/bin/env python3
"""v6.22 input preflight: run from repository root before freezing."""
import hashlib
import json
from pathlib import Path

BASE = "4c8cb67c654c379f0c8768fcbbbd9d4e682a7a3d"
EXPECTED = {
    "data/train.csv": "f7eb81447e0e6deab1a10579ec1a6694587c6c61aa56947fe6a0d6012c3b1763",
    "results/spatial_2d_block_assignments.csv": "4e9c2d698ab44cf11282d12429d7a8e016f95e4e50baa1cf2653f558a7af2a81",
    "results/ems_revision/kmeans15_five_fold_assignments.csv": "04686634473d51b5c63dcbd4474659fe864a114ae0885c5c5f55736522c798ce",
    "results/ems_revision/fold_schemes.csv": "8c988b33fd66e487063913a4adea37aec0bd383e9ac8e5eebe59ecbbe409e73a",
    "results/ems_revision/train_exclusions.csv": "6aea19418a629926f11b1195f0cd6e721bd30bbfdcbc0be59371bb2e7d63ad5e",
    "results/ems_revision/oof_bayes_km15.csv.gz": "556a58e80350b7472d3aa064a610d21ff5e253c914fd54e5939e62287d1455db",
}
def main():
    report = {"base_commit": BASE, "files": {}, "pass": True}
    for name, expected in EXPECTED.items():
        path = Path(name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        ok = actual == expected
        report["files"][name] = {"expected_sha256": expected, "actual_sha256": actual, "pass": ok}
        report["pass"] &= ok
        print(("PASS" if ok else "FAIL"), name, actual or "MISSING")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("PRE-FREEZE BLOCKED: investigate mismatches; do not replace expected hashes silently")
if __name__ == "__main__":
    main()
