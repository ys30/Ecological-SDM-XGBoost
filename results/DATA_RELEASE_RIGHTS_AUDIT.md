# Public data release and licensing status

**Status: redistribution permission not yet verified.** The project's code license (MIT) does not itself grant rights to redistribute third-party survey data, course materials, or original taxonomic identities.

## What has been verified
- The executed NSW `disdat` identity audit matched 1,167 sites and 9,336 binary species-plot records to the reference, using anonymous labels. See [executed audit](NSW_DISDAT_EXECUTED_PROVENANCE_AUDIT.md).
- The public repository currently includes `data/train.csv` and `data/test.csv`; their presence in Git history is independent of any future removal from the main branch.
- The automated taxonomic-name guard checks current working-tree text only. It does not certify Git history, raster images, artifacts, or data rights.

## Outstanding decisions before claiming data may be redistributed
1. Obtain and archive the authoritative source dataset's **license/terms**, including whether republishing derived/subset CSVs and coordinates is allowed, and the required attribution.
2. Confirm that any course-provided enhancements or labels were not subject to additional restrictions; keep the identity mapping private.
3. Decide whether public `train.csv` and `test.csv` can remain. If permission cannot be documented, stop distributing them, provide reproducible download/processing instructions for authorized users, and assess Git-history remediation separately.
4. Audit image labels, generated artifacts and public history for residual identifiers before claiming comprehensive anonymization.
5. Make a separate decision on whether derived metrics and maps are permissible to publish under the applicable terms.

**No license determination is made here.** Do not assume that a publicly accessible R package dataset is automatically licensed for unrestricted redistribution. This note is a compliance checklist, not legal advice.

## External source check (2026-11-01)

- CRAN lists `disdat` version 1.1-0 (published 2026-04-28) under **GPL (>= 3)**: https://cran.r-project.org/web/packages/disdat/index.html . This is a package-level license declaration; do not silently treat it as a separate blanket permission for course-provided enhancements or non-package materials.
- The upstream `rspatial/disdat` repository recommends citing **Elith et al. (2020), _Presence-only and presence-absence data for comparing species distribution modeling methods_, Biodiversity Informatics 15:69–80**: https://github.com/rspatial/disdat .
- The package vignette says the data were released through both OSF and the R package. Confirm any OSF dataset-specific terms and attribution requirements before deciding how to license repository CSV subsets.
- **Practical status:** public reference dataset existence and package GPL declaration are verified; the precise rights covering this repository's derived CSVs and any course additions remain to be determined. No claim of a licensing violation or unrestricted redistribution is made.

## Confirmed historical identity exposure (2026-11-01)

The original `data/train.csv` and `data/test.csv` at commit `a320a8bd2733` were inspected directly through GitHub's commit-specific file API. Both contain an explicit, non-anonymous scientific-name value in the `Species` column. The exact names are intentionally **not repeated here**. Both CSVs were later anonymized on 2026-11-01, but their older versions remain in Git history.

**Risk classification: confirmed historical taxonomic-name disclosure**, not merely a hypothetical risk. Current-branch text-scanning CI cannot detect it. This check inspected historical CSV content only; it did not exhaustively audit every commit, PNG image, or workflow artifact.

**Remediation requires an explicit release decision:** (a) obtain permission to disclose the original labels and to redistribute the survey data, or (b) coordinate a Git history rewrite/removal of the historical files, force-push, collaborator re-clones, and review GitHub's retention/cached views and forks. A history rewrite can disrupt clones and links and cannot guarantee deletion of copies already downloaded. No destructive action has been taken.
