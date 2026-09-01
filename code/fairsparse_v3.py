"""FairSparse v3 (separate generator per subtype) vs v2 (conditional generation): can both the marginal and the subtype-feature association be repaired?"""
import os, warnings, random
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "fibrolamellar")
SEEDS = [42, 123, 7, 2024, 99]

def load():
    df = pd.read_csv(os.path.join(DATA, "cohort.tsv"), sep="\t")
    df = df[["AGE","SEX","fusion_status"]].copy()
    df["AGE"] = pd.to_numeric(df["AGE"]); df["SEX"]=df["SEX"].astype(str)
    df["fusion_status"]=(df["fusion_status"]=="positive").astype(int)
    return df

def fit(real, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    md = SingleTableMetadata(); md.detect_from_dataframe(real)
    m = CTGANSynthesizer(metadata=md, epochs=200, verbose=False); m.fit(real)
    return m

def v2_conditional(real, n, seed):
    m = fit(real, seed)
    prev = float(real["fusion_status"].mean()); n_neg = max(1,int(round(n*(1-prev))))
    c0 = Condition(num_rows=n_neg, column_values={"fusion_status":0})
    c1 = Condition(num_rows=n-n_neg, column_values={"fusion_status":1})
    return m.sample_from_conditions(conditions=[c0,c1])

def v3_stratified(real, n, seed):
    neg = real[real.fusion_status==0]; pos = real[real.fusion_status==1]
    m_neg = fit(neg, seed); m_pos = fit(pos, seed + 1000)
    prev = float(real["fusion_status"].mean()); n_neg = int(round(n*(1-prev))); n_pos = n-n_neg
    return pd.concat([m_neg.sample(n_neg), m_pos.sample(n_pos)], ignore_index=True)

def main():
    real = load()
    print(f"real: neg_rate={1-real.fusion_status.mean():.1%}  neg_age_med={real[real.fusion_status==0].AGE.median():.1f}  pos_age_med={real[real.fusion_status==1].AGE.median():.1f}\n")
    print(f"{'method':18s} {'neg_rate':>8s} {'neg_age_med':>10s} {'pos_age_med':>10s}")
    for name, fn in [("v2 conditional", v2_conditional), ("v3 stratified", v3_stratified)]:
        nr=[]; na=[]; pa=[]
        for s in SEEDS:
            syn = fn(real, 500, s)
            nr.append(1-syn.fusion_status.mean())
            na.append(syn[syn.fusion_status==0].AGE.median())
            pa.append(syn[syn.fusion_status==1].AGE.median())
        print(f"{name:18s} {np.mean(nr):>7.1%}±{np.std(nr):.1%} {np.mean(na):>9.1f} {np.mean(pa):>9.1f}")

if __name__ == "__main__":
    main()
