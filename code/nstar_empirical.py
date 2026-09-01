"""Empirical n*-eps: measure subtype relative-prevalence error under DP across
three cohorts (chol/meso/lihc), spanning subtype prevalence 1%-30%."""
import json
import os
import warnings
import numpy as np
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

warnings.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

CAT = ["SEX", "RACE", "ETHNICITY", "AJCC_STAGE", "PATH_T", "PATH_N", "PATH_M",
       "GRADE", "TUMOR_TYPE", "OS_STATUS", "DSS_STATUS"]
NUM = ["AGE", "OS_MONTHS", "DSS_MONTHS", "MUTATION_COUNT", "TMB_NONSYNONYMOUS",
       "FRACTION_GENOME_ALTERED", "ANEUPLOIDY_SCORE"]
EPS = [None, 10.0, 2.0, 0.5]


def load(name):
    df = pd.read_csv(os.path.join(DATA, name, "cohort.tsv"), sep="\t")
    genes = [c for c in df.columns if c.startswith("mut_")]
    df = df[CAT + NUM + genes].copy()
    for c in CAT:
        df[c] = df[c].astype(str).fillna("NA")
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
    for c in genes:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df, genes


def apply_dp(synth, real, eps, genes):
    out = synth.copy()
    if eps is None:
        return out
    for c in NUM:
        std = float(real[c].std())
        if std > 0 and not np.isnan(std):
            out[c] = out[c].astype(float) + np.random.laplace(0, std / eps, len(out))
    p = 1.0 / (1.0 + np.exp(eps))
    for c in genes:
        flip = np.random.rand(len(out)) < p
        out.loc[flip, c] = 1 - out.loc[flip, c]
    return out


def main():
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    results = []
    for name in ["chol", "meso", "lihc"]:
        real, genes = load(name)
        md = SingleTableMetadata()
        md.detect_from_dataframe(real)
        model = CTGANSynthesizer(metadata=md, epochs=150, verbose=False)
        model.fit(real)
        base = model.sample(500)
        for eps in EPS:
            syn = apply_dp(base, real, eps, genes)
            for g in genes:
                tp = float(real[g].mean())
                if tp < 0.01 or tp > 0.5:
                    continue
                sp = float(syn[g].mean())
                err = abs(sp - tp) / tp
                results.append({"cohort": name, "n": int(len(real)), "subtype": g[4:],
                                "prevalence": tp,
                                "eps": "inf" if eps is None else str(eps),
                                "rel_error": err})
    with open(os.path.join(DATA, "nstar_empirical.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {len(results)} points")
    # quick summary: rare subtypes (p<10%) at eps=2
    rare = [r for r in results if r["prevalence"] < 0.10 and r["eps"] in ("2.0", "inf")]
    for r in sorted(rare, key=lambda x: x["prevalence"])[:10]:
        print(f"  {r['cohort']:5s} {r['subtype']:8s} p={r['prevalence']:.2%} eps={r['eps']:>3s} rel_err={r['rel_error']:.2f}")


if __name__ == "__main__":
    main()
