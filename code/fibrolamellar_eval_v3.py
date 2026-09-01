"""Four-dimensional evaluation: features = AGE + SEX + fusion_status; histology used only for cohort description.
utility = AGE/SEX -> molecular subtype classification; privacy = membership inference on AGE/SEX;
fairness = per-subtype AGE fidelity and association preservation."""
import os, warnings, random
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "fibrolamellar")
SEEDS = [42, 123, 7, 2024, 99]
CAT = ["SEX"]; SUB = "fusion_status"

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

def add_dp(synth, real, eps, seed):
    rng = np.random.default_rng(seed); out = synth.copy()
    std = float(real["AGE"].std())
    if eps is not None:
        out["AGE"] = out["AGE"].astype(float) + rng.laplace(0, std/eps, len(out))
        p = 1.0/(1.0+np.exp(eps)); flip = rng.random(len(out)) < p
        out.loc[flip, "fusion_status"] = 1 - out.loc[flip, "fusion_status"]
    return out

def ctgan(real, n, eps, seed):
    return add_dp(fit(real, seed).sample(n), real, eps, seed)

def fairsparse_v3(real, n, eps, seed):
    neg = real[real.fusion_status==0]; pos = real[real.fusion_status==1]
    m_neg, m_pos = fit(neg, seed), fit(pos, seed + 1000)
    prev = float(real["fusion_status"].mean()); n_neg = int(round(n*(1-prev)))
    syn = pd.concat([m_neg.sample(n_neg), m_pos.sample(n-n_neg)], ignore_index=True)
    return add_dp(syn, real, eps, seed)

def featurize(df):
    X = pd.get_dummies(df[CAT].astype(str), columns=CAT); X["AGE"] = df["AGE"].astype(float).values
    return X

def fidelity(real, synth):
    return float(ks_2samp(real["AGE"], synth["AGE"]).statistic)

def utility(real, synth, seed):
    Xr, yr = featurize(real), real[SUB].values
    Xs, ys = featurize(synth), synth[SUB].values
    if len(set(ys)) < 2: return 0.5
    rf = RandomForestClassifier(n_estimators=100, random_state=seed); rf.fit(Xs, ys)
    return float(roc_auc_score(yr, rf.predict_proba(Xr)[:,1]))

def privacy(real, synth, seed):
    Xr, Xs = featurize(real), featurize(synth)
    X = pd.concat([Xr, Xs]).values; y = np.array([1]*len(Xr)+[0]*len(Xs))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed); aucs=[]
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=seed); rf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], rf.predict_proba(X[te])[:,1]))
    return float(np.mean(aucs))

def fairness_subgroup(real, synth):
    out = {}
    for s, lab in [(0,"rare_neg"),(1,"common_pos")]:
        rs = real[real[SUB]==s]; ss = synth[synth[SUB]==s]
        out[lab] = float(ks_2samp(rs["AGE"], ss["AGE"]).statistic) if len(ss)>=2 else None
    return out

def assoc_preserve(real, synth):
    return float(synth["fusion_status"].corr(synth["AGE"]))

def main():
    real = load()
    real_corr = float(real["fusion_status"].corr(real["AGE"]))
    # real-data upper bound (train and test on real data)
    from sklearn.model_selection import cross_val_score
    upper = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42),
                            featurize(real), real[SUB].values, cv=5, scoring="roc_auc").mean()
    print(f"real n={len(real)} fusion-neg={1-real[SUB].mean():.1%} AGE-fusion corr={real_corr:.3f}")
    print(f"utility real upper (AGE/SEX->subtype, 5-fold CV) = {upper:.3f}\n")
    print(f"{'method':14s}{'eps':>5s} {'fid_KS':>7s} {'util_AUC':>9s} {'priv_AUC':>9s} {'fair_rare':>8s} {'fair_common':>8s} {'AGE_corr':>8s}")
    print("-"*72)
    for eps in [None, 2.0]:
        e = "inf" if eps is None else str(eps)
        for name, fn in [("CTGAN", ctgan), ("FairSparse_v3", fairsparse_v3)]:
            fk=[]; ut=[]; pr=[]; fr=[]; fc=[]; ac=[]
            for seed in SEEDS:
                syn = fn(real, 500, eps, seed)
                fk.append(fidelity(real, syn)); ut.append(utility(real, syn, seed)); pr.append(privacy(real, syn, seed))
                fg = fairness_subgroup(real, syn); fr.append(fg["rare_neg"]); fc.append(fg["common_pos"])
                ac.append(assoc_preserve(real, syn))
            print(f"{name:14s}{e:>5s} {np.mean(fk):7.3f} {np.mean(ut):9.3f} {np.mean(pr):9.3f} "
                  f"{np.mean(fr):8.3f} {np.mean(fc):8.3f} {np.mean(ac):8.3f}")

if __name__ == "__main__":
    main()
