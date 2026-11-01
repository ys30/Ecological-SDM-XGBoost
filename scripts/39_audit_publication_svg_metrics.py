#!/usr/bin/env python3
"""Check committed SVG chart labels and bar lengths against published species metrics."""
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "results"
with (base / "matched_bayesian_maxent_xgboost_species.csv").open(newline="", encoding="utf-8") as f:
    species = list(csv.DictReader(f))
models = ("bayesian", "maxent", "xgboost")
ns = "{http://www.w3.org/2000/svg}"
for metric, filename, title in (
    ("auc", "23_three_model_auc_publication.svg", "ROC AUC"),
    ("ap", "23_three_model_ap_publication.svg", "Average precision"),
):
    tree = ET.parse(base / "figures" / filename)
    root = tree.getroot()
    texts = [(node.text or "") for node in root.iter(ns + "text")]
    bars = [node for node in root.iter(ns + "rect") if node.get("x") == "330"]
    assert len(bars) == 15, (filename, len(bars))
    assert any(title in text for text in texts)
    assert all(any(m in text for text in texts) for m in ("Bayesian", "MaxEnt", "XGBoost"))
    assert any("23 matched species-fold" in text for text in texts)
    assert any("original-fit Bayesian" in text for text in texts)
    for i, row in enumerate(species):
        assert any(row["species"].replace(".", " ") == text for text in texts)
        assert any(f'{row["folds"]} folds' in text and f'{row["positives"]} presences' in text for text in texts)
        for j, model in enumerate(models):
            expected = float(row[f"{model}_{metric}"])
            bar = bars[3 * i + j]
            width = float(bar.attrib["width"])
            assert math.isclose(width, expected * 650, abs_tol=0.012, rel_tol=0), (filename, i, model, width, expected)
            assert f"{expected:.3f}" in texts, (filename, row["species"], model)
    print(f"PASS {filename}: 15 bar lengths, 15 values, species labels and captions")
print("PASS: both publication SVGs match 30 source metrics within graphical rounding")
