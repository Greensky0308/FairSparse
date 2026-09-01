"""Generate all fibrolamellar-spectrum main-cohort results (single source of truth).

numpy/torch/random seeds are fixed before each fit for reproducibility.
"""
import os, warnings, random, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from scipy.stats import ks_2samp, pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "fibrolamellar")
SEEDS = [42, 123, 7, 2024, 99]


def load():
    df = pd.read_csv(os.path.join(DATA, "cohort.tsv"), sep="\t")
    df = df[["AGE", "SEX", "fusion_status"]].copy()
    df["AGE"] = pd.to_numeric(df["AGE"])
    df["SEX"] = df["SEX"].astype(str)
    df["fusion_status"] = (df["fusion_status"] == "positive").astype(int)
    return df


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def fit(cls, df, s, **kw):
    seed_all(s)
    md = SingleTableMetadata(); md.detect_from_dataframe(df)
    m = cls(metadata=md, **kw); m.fit(df)
    return m


def fairsparse(real, n, s):
    neg = real[real.fusion_status == 0]; pos = real[real.fusion_status == 1]
    m_neg = fit(CTGANSynthesizer, neg, s, epochs=200, verbose=False)
    m_pos = fit(CTGANSynthesizer, pos, s + 1000, epochs=200, verbose=False)
    n_neg = int(round(n * (1 - real.fusion_status.mean())))
    return pd.concat([m_neg.sample(n_neg), m_pos.sample(n - n_neg)], ignore_index=True)


def add_noise(syn, real, eps, s):
    rng = np.random.default_rng(s); out = syn.copy()
    std = float(real["AGE"].std())
    if eps is not None:
        out["AGE"] = out["AGE"].astype(float) + rng.laplace(0, std / eps, len(out))
        p = 1.0 / (1.0 + np.exp(eps))
        flip = rng.random(len(out)) < p
        out.loc[flip, "fusion_status"] = 1 - out.loc[flip, "fusion_status"]
    return out


def featurize(df):
    X = pd.get_dummies(df[["SEX"]].astype(str), columns=["SEX"])
    X["AGE"] = df["AGE"].astype(float).values
    return X


def main():
    real = load()
    n = len(real)
    neg_rate = 1 - real.fusion_status.mean()
    real_corr = float(real.fusion_status.corr(real.AGE))
    real_neg_med = float(real[real.fusion_status == 0].AGE.median())
    real_pos_med = float(real[real.fusion_status == 1].AGE.median())

    # bootstrap CI (real data, 2000 resamples)
    age = real.AGE.to_numpy(float); fs = real.fusion_status.to_numpy()
    rng = np.random.default_rng(42)
    boots = [pearsonr(age[i := rng.integers(0, n, n)], fs[i])[0] for _ in range(2000)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    # utility upper bound on real data
    upper = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42),
                            featurize(real), real.fusion_status.values, cv=5,
                            scoring="roc_auc").mean()

    # fusion-negative prevalence per generator (mean over 5 seeds)
    prev = {"Real": neg_rate * 100}
    for name, cls, kw in [("TVAE", TVAESynthesizer, {"epochs": 200, "verbose": False}),
                          ("GaussCopula", GaussianCopulaSynthesizer, {}),
                          ("CTGAN", CTGANSynthesizer, {"epochs": 200, "verbose": False})]:
        rates = []
        for s in SEEDS:
            m = fit(cls, real, s, **kw)
            syn = m.sample(500)
            rates.append((1 - syn.fusion_status.mean()) * 100)
        prev[name] = float(np.mean(rates))
    fs_rates = []
    for s in SEEDS:
        syn = fairsparse(real, 500, s)
        fs_rates.append((1 - syn.fusion_status.mean()) * 100)
    prev["FairSparse"] = float(np.mean(fs_rates))

    # prevalence under output noise perturbation (CTGAN)
    prev_dp = {}
    for eps in [10.0, 2.0, 0.5]:
        rates = []
        for s in SEEDS:
            m = fit(CTGANSynthesizer, real, s, epochs=200, verbose=False)
            syn = add_noise(m.sample(500), real, eps, s)
            rates.append((1 - syn.fusion_status.mean()) * 100)
        prev_dp[f"eps{eps}"] = float(np.mean(rates))

    # fusion-negative age median + between-seed SD (Fig4b)
    neg_age = {"Real": [real_neg_med, float(real[real.fusion_status == 0].AGE.std())]}
    for name, gen in [("CTGAN", lambda s: fit(CTGANSynthesizer, real, s, epochs=200, verbose=False).sample(500)),
                      ("FairSparse", lambda s: fairsparse(real, 500, s))]:
        meds = []
        for s in SEEDS:
            syn = gen(s)
            meds.append(float(syn[syn.fusion_status == 0].AGE.median()))
        neg_age[name] = [float(np.mean(meds)), float(np.std(meds))]

    # four-dimension metrics (mean over 5 seeds)
    def metrics(gen):
        fk, ut, pr, fr, ac = [], [], [], [], []
        for s in SEEDS:
            syn = gen(s)
            fk.append(ks_2samp(real.AGE, syn.AGE).statistic)
            Xr, yr = featurize(real), real.fusion_status.values
            Xs, ys = featurize(syn), syn.fusion_status.values
            rf = RandomForestClassifier(n_estimators=100, random_state=s); rf.fit(Xs, ys)
            ut.append(roc_auc_score(yr, rf.predict_proba(Xr)[:, 1]))
            X = pd.concat([Xr, Xs]).values; y = np.array([1] * len(Xr) + [0] * len(Xs))
            skf = StratifiedKFold(3, shuffle=True, random_state=s); pa = []
            for tr, te in skf.split(X, y):
                r = RandomForestClassifier(n_estimators=100, random_state=s); r.fit(X[tr], y[tr])
                pa.append(roc_auc_score(y[te], r.predict_proba(X[te])[:, 1]))
            pr.append(float(np.mean(pa)))
            rs = real[real.fusion_status == 0]; ss = syn[syn.fusion_status == 0]
            fr.append(ks_2samp(rs.AGE, ss.AGE).statistic if len(ss) >= 2 else None)
            ac.append(syn.fusion_status.corr(syn.AGE))
        return dict(fid=float(np.mean(fk)), util=float(np.mean(ut)), priv=float(np.mean(pr)),
                    fair=float(np.mean(fr)), corr=float(np.mean(ac)))

    m_ctgan = metrics(lambda s: fit(CTGANSynthesizer, real, s, epochs=200, verbose=False).sample(500))
    m_fs = metrics(lambda s: fairsparse(real, 500, s))

    results = {
        "real": dict(n=n, neg_rate=neg_rate, neg_age_median=real_neg_med, pos_age_median=real_pos_med,
                     age_fusion_corr=real_corr, corr_ci=[ci_lo, ci_hi], utility_upper=upper),
        "prevalence": prev,
        "prevalence_dp": prev_dp,
        "neg_age": neg_age,
        "metrics": {"CTGAN": m_ctgan, "FairSparse": m_fs},
    }
    out = os.path.join(DATA, "fibrolamellar_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
