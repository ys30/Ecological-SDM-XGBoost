#!/usr/bin/env python3
"""Technical figure QA for v6.23. Does not replace human visual inspection."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.image as mpimg
P=Path("results/v623_matched_three_model")
FIGS=("figure_matched_taxon_scores","figure_paired_probability_scores")
def main():
    result={"status":"PASS_FIGURE_TECHNICAL_QC","figures":{},"limitation":"Technical checks only; manually review visual layout, legend collisions, clipping and accessibility."}
    for stem in FIGS:
        svg=P/(stem+".svg");png=P/(stem+".png")
        root=ET.parse(svg).getroot()
        assert root.tag.endswith("svg")
        assert svg.stat().st_size>10000
        assert png.stat().st_size>10000
        a=mpimg.imread(png)
        h,w=a.shape[:2]
        assert w>=1500 and h>=700,(stem,w,h)
        assert a.ndim==3 and a.shape[-1] in (3,4)
        # Matplotlib SVG stores labels as text or glyph references; ensure accessible labels
        # are represented in the serialized vector output.
        source=svg.read_text(encoding="utf-8")
        assert "XGBoost" in source,stem
        assert "GLM" in source,stem
        assert "10-km" in source if stem=="figure_matched_taxon_scores" else "Negative favors XGBoost" in source
        result["figures"][stem]={"png_width_px":w,"png_height_px":h,"svg_bytes":svg.stat().st_size,"png_bytes":png.stat().st_size}
    (P/"figure_technical_qc.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
