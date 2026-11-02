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
