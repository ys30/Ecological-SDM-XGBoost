#!/usr/bin/env python3
"""Rebuild publication PNG previews from anonymous SVGs; no model fitting."""
from pathlib import Path
import cairosvg

base = Path("results/figures")
for stem in (
    "22_three_model_auc",
    "22_three_model_ap",
    "23_three_model_auc_publication",
    "23_three_model_ap_publication",
):
    source = base / (stem + ".svg")
    target = base / (stem + ".png")
    if not source.exists():
        raise FileNotFoundError(source)
    cairosvg.svg2png(url=str(source), write_to=str(target), output_width=1800)
    print(f"Regenerated {target} from anonymous SVG")
