#!/usr/bin/env python3
"""Validate EMS v6.22 candidate without running model experiments."""
from pathlib import Path
import csv, hashlib, io, sys
import yaml

root = Path("experiments/v622")
required = [
    "config/v622_experiment.yaml",
    "config/v622_model_matrix.csv",
    "config/v622_contrasts.csv",
    "docs/EMS_v6.22_analysis_plan.md",
    "docs/EMS_v6.22_manifest_sha256.txt",
]
missing = [p for p in required if not (root / p).is_file()]
if missing:
    sys.exit("NOT FROZEN: missing candidate files: " + ", ".join(missing))
for line in (root / "docs/EMS_v6.22_manifest_sha256.txt").read_text().splitlines():
    if not line.strip():
        continue
    expected, relative = line.split("  ", 1)
    target = root / relative
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        sys.exit("MANIFEST MISMATCH: " + relative)
config = yaml.safe_load((root / "config/v622_experiment.yaml").read_text())
assert config["inputs"]["base_commit"] == "4c8cb67c654c379f0c8768fcbbbd9d4e682a7a3d"
with (root / "config/v622_model_matrix.csv").open(newline="") as f:
    models = list(csv.DictReader(f))
with (root / "config/v622_contrasts.csv").open(newline="") as f:
    contrasts = list(csv.DictReader(f))
ids = [m["model_id"] for m in models]
assert len(models) == 19 and len(set(ids)) == 19
assert len(contrasts) == 20 and len(set(c["contrast_id"] for c in contrasts)) == 20
assert all(c["model_a"] in ids and c["model_b"] in ids for c in contrasts)
print("PASS: manifest, YAML, 19 models, 20 contrasts, pinned base")
