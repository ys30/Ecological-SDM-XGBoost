# NSW source and metadata verification for v6.24

The working data match 1,167 NSW open-forest survey sites and all 9,336 binary observations by anonymous taxon and site in the verified disdat PA/ENV audit [41287426532](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41287426532). Cite the dataset publication by [Elith et al. (2020)](https://doi.org/10.17161/bi.v15i2.13384) and the [SDM_NCEAS archive](https://osf.io/kwc4v/). Simon Ferrier and colleagues compiled the NSW survey records. Cite [Valavi et al. (2023)](https://doi.org/10.1111/geb.13639) for the spatial validation benchmark.

The [NSW FGDC metadata record](https://api.osf.io/v2/files/5df1bcb5d039df0009606c4c/) gives WGS 84 (EPSG:4326) for the source geographic coordinates. Its methods describe collection before 2002. The record uses the keyword “1990s” and lists exact beginning and ending dates as unknown. Do not infer exact dates for the 1,167-plot subset. Use the matched longitude and latitude fields for the great-circle buffer.

| Field | Source definition | Source encoding |
| --- | --- | --- |
| rainann | Mean annual rainfall | mm |
| soildepth | Model-estimated mean soil depth | m × 1,000 |
| tempann | Annual mean temperature | °C × 10 |
| topo | Topographic position relative to surrounding cells within 1 km | m |

The [disdat NSW manual](https://cran.csiro.au/web/packages/disdat/refman/disdat.html#NSW) gives these units explicitly.

The [OSF project API](https://api.osf.io/v2/nodes/kwc4v/) resolves its project-level license to [CC BY 4.0](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/). The NSW FGDC metadata lists no access or use constraints for the archived data. Cite the archive and identify modifications if republishing an archive-derived subset. CRAN separately lists the disdat package as GPL ≥3 and the dataset article has its own terms. The project-level license here covers the public anonymized archive. This does not resolve any rights in course-specific fields, transformations of unknown origin, or original taxonomic names. Use public anonymous taxon codes and cite the deposited source.
