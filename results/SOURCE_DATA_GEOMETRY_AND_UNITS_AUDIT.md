# Source-data audit

Direct inspection of `data/train.csv` gives 9,336 species–plot rows, 1,167 plots, eight species per plot, and 747 positive records.

| Field | Minimum | Maximum |
|---|---:|---:|
| Longitude | 151.2229 | 153.3644 |
| Latitude | -32.7068 | -28.3287 |
| `rainann` | 626 | 2632 |
| `tempann` | 90 | 194 |
| `soildepth` | 800 | 1350 |
| `topo` | -56 | 145 |
| `easting` | -240119.249 | -49543.158 |
| `northing` | 6359036.486 | 6849736.098 |

These are observed CSV values, not independently verified measurement units or coordinate reference system. The coordinates indicate an eastern Australian study region; the original survey source, years, field protocols, predictor units and CRS remain unverified. Do not interpret `tempann` directly as Celsius or the negative `easting` values as conventional UTM eastings without original documentation.

## New external provenance lead (not yet a confirmed exact dataset match)

Valavi et al. (2023), *Flexible species distribution modelling methods perform well on spatially separated testing data*, provide NSW environmental covariate definitions and direct readers to the `disdat` R package and the Elith et al. (2020) dataset. Their NSW variable names exactly overlap several fields in this repository: `disturb`, `rainann`, `soildepth`, `soilfert`, `tempann` and `topo`. See [Valavi et al. 2023 Appendix S1](https://rvalavi.github.io/SDM-Spatial-Validation/) and its [paper DOI](https://doi.org/10.1111/geb.13639).

Their Table 3 identifies `tempann` as annual mean temperature recorded in **degrees Celsius multiplied by 10**, `soildepth` as soil depth recorded as **metres multiplied by 1000**, and `topo` as topographic position in metres. Their `rainann` table prints 'Mm', which needs clarification before applying units. These definitions are **provisional for this repository** until original records are matched against `disdat` or its source archive. This finding upgrades the provenance investigation from unknown to a concrete candidate; it does not establish exact survey dates, license or row-level identity.
