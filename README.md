# FairSparse

Subtype-stratified synthetic data generation that preserves rare molecular subtypes and their feature associations under differential privacy, for rare liver tumors.

## Run

```bash
bash run_all.sh
```

Requires Python 3.10+ with the packages in `requirements.txt` (numpy, pandas, scikit-learn, scipy, sdv, torch, opacus, lifelines, matplotlib, seaborn).

## Reproducibility

All random seeds are fixed. `run_all.sh` regenerates the figures in `figures_regen/` and the tables in `tables/` from the data in `data/`, matching the results reported in the paper:

- Fusion-negative subtype prevalence held at **36%** (vs 22.8–23% for TVAE)
- Age–fusion correlation recovered to **-0.809** (vs -0.060 for CTGAN)
- Downstream subtype-classification AUC **0.874** (vs 0.543 for CTGAN)

## Data

Assembled cohorts under `data/`:

- Fibrolamellar-spectrum cohort (GSE181922 + LICA-FR, n = 50)
- Cholangiocarcinoma validation cohort (TCGA-CHOL, n = 36)

Source provenance in `data/PROVENANCE.md`.

**FairSparse forms.** The main fibrolamellar-spectrum analysis uses the subtype-stratified form (a separate generator trained per subtype). The cholangiocarcinoma validation cohort uses the conditional-generation form (sampling conditioned on the subtype marginal, without separate generators), which restores the marginal but does not repair within-subtype feature distributions; this is consistent with the paper's framing that only the stratified form repairs the within-subtype distribution.

## License

MIT
