#!/usr/bin/env python3
"""Guard public working-tree text against accidental taxonomic-name disclosure.

This checks only the current checkout, not Git history, images, or Actions artifacts.
No code-to-name lookup table is included.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SCAN_EXTENSIONS = {".md", ".csv", ".tsv", ".txt", ".r", ".py", ".yml", ".yaml", ".svg", ".json"}
EXCLUDED = {".git", ".venv", "venv", "__pycache__"}
# The publicly distributed NSW open-forest data uses anonymous nsw species IDs.
# Detect scientific binomials from the genera used in the nonpublic course labels,
# without reproducing any species epithet or mapping.
PATTERN = re.compile(r"\b(?:Angophora|Corymbia|Eucalyptus)[ ._]+[a-z][a-z-]+\b", re.I)
findings = []
checked = 0
for path in ROOT.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in SCAN_EXTENSIONS:
        continue
    if any(part in EXCLUDED for part in path.relative_to(ROOT).parts):
        continue
    checked += 1
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError:
        continue
    for number, line in enumerate(text.splitlines(), 1):
        if PATTERN.search(line):
            findings.append((path.relative_to(ROOT), number))
print(f"Checked {checked} text files; possible scientific-name disclosures: {len(findings)}")
for path, number in findings:
    print(f"  {path}:{number} (content suppressed)")
if findings:
    sys.exit(1)
print("PASS: no matching scientific binomials in scanned text files.")
