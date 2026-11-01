#!/usr/bin/env python3
"""Check that publication SVG labels and bar widths match the committed CSV."""
import csv
import pathlib
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/matched_bayesian_maxent_xgboost_species.csv"
MODELS = ("bayesian", "maxent", "xgboost")

with SOURCE.open(newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
assert len(rows) == 5
assert sum(int(r["folds"]) for r in rows) == 23
for metric in ("auc", "ap"):
    svg_path = ROOT / f"results/figures/23_three_model_{metric}_publication.svg"
    tree = ET.parse(svg_path)
    ns = {"s": "http://www.w3.org/2000/svg"}
    rects = [e for e in tree.findall(".//s:rect", ns)
             if e.attrib.get("fill", "").lower() in
             {"#4b78a8", "#e0a458", "#50a08b"}
             and e.attrib.get("x") == "330"
             and e.attrib.get("height") == "16"]
    assert len(rects) == 15, (metric, len(rects))
    texts = [e.text or "" for e in tree.findall(".//s:text", ns)]
    for i, row in enumerate(rows):
        species = row["species"].replace(".", " ")
        assert species in texts, (metric, species)
        assert f'{int(row["folds"])} folds · {int(row["positives"])} presences' in texts
        for j, model in enumerate(MODELS):
            value = float(row[f"{model}_{metric}"])
            assert 0 <= value <= 1
            rect = rects[3 * i + j]
            expected_width = 650 * value
            assert abs(float(rect.attrib["width"]) - expected_width) <= 0.011, (
                metric, species, model, rect.attrib["width"], expected_width)
            assert f"{value:.3f}" in texts, (metric, species, model)
    print(f"PASS {svg_path.relative_to(ROOT)}: 15 bars, 5 species, values and widths")
