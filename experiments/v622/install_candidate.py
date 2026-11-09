#!/usr/bin/env python3
"""Install a reconciled EMS v6.22 ZIP into a local repository checkout.

Usage:
  python experiments/v622/install_candidate.py /path/to/EMS_v622_RECONCILED_PREFREEZE.zip
  python experiments/v622/validate_candidate.py

Then review git diff, commit and push to ems-v622-experiments. Never mark
FROZEN before remote CI succeeds. This script does not push or run models.
"""
import hashlib
from pathlib import Path
import sys
from zipfile import ZipFile

if len(sys.argv) != 2:
    sys.exit("usage: install_candidate.py RECONCILED_PREFREEZE.zip")
repo = Path(__file__).resolve().parents[2]
prefix = "experiments/v622/"
needed = [
    "config/v622_experiment.yaml",
    "config/v622_model_matrix.csv",
    "config/v622_contrasts.csv",
    "docs/EMS_v6.22_analysis_plan.md",
]
archive = Path(sys.argv[1]).expanduser().resolve()
with ZipFile(archive) as z:
    missing = [name for name in needed if prefix + name not in z.namelist()]
    if missing:
        sys.exit("archive missing: " + ", ".join(missing))
    content = {name: z.read(prefix + name) for name in needed}
    # Do not trust the archive manifest: regenerate from installed bytes.
    for name, data in content.items():
        dest = repo / prefix / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print("STAGED", dest.relative_to(repo))
manifest = "".join(
    hashlib.sha256(content[name]).hexdigest() + "  " + name + "\n"
    for name in needed
)
dest = repo / prefix / "docs/EMS_v6.22_manifest_sha256.txt"
dest.write_text(manifest)
print("REGENERATED", dest.relative_to(repo))
print("NOT FROZEN. Review diff and run validate_candidate.py before commit.")
