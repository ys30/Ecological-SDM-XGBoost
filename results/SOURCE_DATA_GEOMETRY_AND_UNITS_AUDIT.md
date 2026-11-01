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
