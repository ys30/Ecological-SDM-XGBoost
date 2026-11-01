#!/usr/bin/env python3
"""Check publication SVG bar heights against committed rare-species random-baseline CSV."""
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/bayesian_rare_species_random_baseline.csv"
FIGURE = ROOT / "results/figures/24_rare_species_retrieval_publication.svg"
COLORS = {"nsw19": "#286e99",
          "nsw20": "#a04e7b",
          "nsw22": "#32866c"}
PANEL_X = {"nsw19": 60,
           "nsw20": 415,
           "nsw22": 770}
NS = {"svg": "http://www.w3.org/2000/svg"}

def main():
    root = ET.parse(FIGURE).getroot()
    bars = []
    for node in root.findall(".//svg:rect", NS):
        if "x" in node.attrib and "height" in node.attrib:
            bars.append({k: node.attrib[k] for k in ("x", "height", "fill")})
    with DATA.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 9, f"Expected 9 data rows, got {len(rows)}"
    # Each data row has two bars; colored legend rectangles must not count.
    checked = set()
    for row in rows:
        species = row["species"]
        k = int(row["top_k_per_fold"])
        j = {10: 0, 25: 1, 50: 2}[k]
        center = PANEL_X[species] + 60 + 96 * j
        for x, color, value in (
            (center - 29.5, COLORS[species], float(row["recall"])),
            (center + 2.5, "#aebbc5",
             float(row["expected_random_captured"]) / int(row["total_positives"]))
        ):
            found = [b for b in bars if b["fill"] == color and
                     math.isclose(float(b["x"]), x, abs_tol=1e-6)]
            assert len(found) == 1, (species, k, color, x, found)
            assert math.isclose(float(found[0]["height"]), value * 500,
                                abs_tol=0.00002), (species, k, value, found[0])
            checked.add((species, k, color))
    assert len(checked) == 18
    print("PASS: 18 SVG data bars match 9 CSV rows; legends excluded")

if __name__ == "__main__":
    main()
