#!/usr/bin/env bash
# Reproducible end-to-end pipeline: cohorts -> synthesis -> evaluation -> figures -> tables.
# All random seeds are fixed; outputs are deterministic.
set -euo pipefail

cd "$(dirname "$0")"

PY=${PYTHON:-$(command -v python3 || command -v python)}

echo "== [1/4] Main cohort: synthesis + evaluation =="
"$PY" code/generate_results.py
"$PY" code/fibrolamellar_eval_v3.py
"$PY" code/dp_sgd_vae.py

echo "== [2/4] Validation cohort (CCA) =="
"$PY" code/experiments.py
"$PY" code/nstar_empirical.py

echo "== [3/4] Figures (Fig1-5 + Supplementary S1-S2) =="
"$PY" code/figures_v2.py
"$PY" code/figures_supp.py

echo "== [4/4] Tables =="
"$PY" code/tables_fibrolamellar.py

echo "DONE. See figures_regen/ and tables/."
