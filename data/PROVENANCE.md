# Data provenance

The pipeline runs from two derived cohorts committed in this repository. Raw source data are **not** redistributed.

## Data availability

Only derived (curated) cohorts are included here; all raw source data remain available from their original public sources:

- Fibrolamellar-spectrum cohort:
  - Fusion-positive: Francisco et al., *JCI Insight* 2022 — raw RNA-seq from GEO **GSE181922** (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181922).
  - Fusion-negative: Hirsch et al., *J Hepatol* 2020 — BAP1-driven fibrolamellar-like HCC.
- Cholangiocarcinoma validation cohort: **TCGA-CHOL** via cBioPortal (https://www.cbioportal.org/study/summary?id=chol_tcga_pan_can_atlas_2018).

The assembled cohorts are the authors' curated artifacts; re-derivation from the original sources is documented in the construction scripts below.

## Fibrolamellar-spectrum cohort (main, n = 50)

File: `fibrolamellar/cohort.tsv`

| Subtype | n | Source |
|---|---|---|
| Fusion-positive (DNAJB1-PRKACA) | 32 | Francisco et al., *JCI Insight* 2022 (GSE181922, bulk RNA-seq + clinical features) |
| Fusion-negative (BAP1-mutant) | 18 | Hirsch et al., *J Hepatol* 2020 (Table 2, BAP1-driven fibrolamellar-like hepatocellular carcinoma) |

Columns: `patient_id`, `cohort` (Francisco2022 / Hirsch2020), `SEX`, `AGE`, `fusion_status` (positive / negative), `histology`, `sample_type`, `FFPM`.

## Cholangiocarcinoma cohort (validation, n = 36)

File: `cohort.tsv`

Source: TCGA-CHOL (`chol_tcga_pan_can_atlas_2018`) via cBioPortal. Clinical features and driver mutations (IDH1, IDH2, FGFR2, KRAS, BAP1, PBRM1, TP53, and others) were extracted; three rare molecular subtypes are IDH-mutant, FGFR2-mutant, and KRAS-mutant.

## Cohort construction scripts

- `code/build_flc_cohort.py` — builds the fusion-positive subset from Francisco 2022 Table S1.
- `code/prepare_cohort.py` — builds the CCA cohort from TCGA-CHOL clinical + mutation matrices.

The merged 50-tumor fibrolamellar-spectrum cohort is the curated artifact `fibrolamellar/cohort.tsv` (Francisco 32 + Hirsch 18); re-extraction from the original tables is documented here but not part of the reproducible pipeline.
