"""Complete experiment suite -> results.json.

Metrics: (1) naive collapse, (2) DP amplification, (3) FairSparse recovery,
(4) analytical n*-eps floor, (5) downstream Cox survival utility,
(6) membership-inference privacy, (7) fidelity (KS / correlation MAE).
"""
import json
import os
import warnings
import random
import numpy as np
import pandas as pd
import torch
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

CAT = ["SEX", "RACE", "ETHNICITY", "AJCC_STAGE", "PATH_T", "PATH_N", "PATH_M",
       "GRADE", "TUMOR_TYPE", "OS_STATUS", "DSS_STATUS"]
NUM = ["AGE", "OS_MONTHS", "DSS_MONTHS", "MUTATION_COUNT", "TMB_NONSYNONYMOUS",
       "FRACTION_GENOME_ALTERED", "ANEUPLOIDY_SCORE"]
BIN = ["IDH_mutant", "FGFR2_mutant", "KRAS_mutant", "BAP1_mutant",
       "PBRM1_mutant", "TP53_mutant"]
SUBTYPES = ["IDH_mutant", "FGFR2_mutant", "KRAS_mutant"]
EPS = [None, 10.0, 2.0, 0.5]
SEEDS = [42, 123]


def load():
    df = pd.read_csv(os.path.join(DATA, "cohort.tsv"), sep="\t")
    df = df[CAT + NUM + BIN].copy()
    for c in CAT:
        df[c] = df[c].astype(str).fillna("NA")
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
    for c in BIN:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def add_dp_noise(synth, train, eps, seed):
    rng = np.random.default_rng(seed)
    out = synth.copy()
    for c in NUM:
        std = float(train[c].std())
        if std > 0 and not np.isnan(std):
            out[c] = out[c].astype(float) + rng.laplace(0, std / eps, len(out))
    return out


def rr_binary(synth, eps, seed):
    rng = np.random.default_rng(seed)
    out = synth.copy()
    p = 1.0 / (1.0 + np.exp(eps))
    for c in BIN:
        flip = rng.random(len(out)) < p
        out.loc[flip, c] = 1 - out.loc[flip, c]
    return out


def apply_dp(synth, real, eps, seed):
    if eps is None:
        return synth
    synth = add_dp_noise(synth, real, eps, seed)
    synth = rr_binary(synth, eps, seed)
    return synth


def make_model(name, metadata, seed):
    if name == "TVAE":
        return TVAESynthesizer(metadata=metadata, epochs=150, verbose=False)
    if name == "GaussianCopula":
        return GaussianCopulaSynthesizer(metadata=metadata)
    return CTGANSynthesizer(metadata=metadata, epochs=150, verbose=False)


def fit_sample(name, real, n, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    md = SingleTableMetadata()
    md.detect_from_dataframe(real)
    model = make_model(name, md, seed)
    model.fit(real)
    return model.sample(n)


def fairsparse_sample(real, n, subtype, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    md = SingleTableMetadata()
    md.detect_from_dataframe(real)
    model = CTGANSynthesizer(metadata=md, epochs=150, verbose=False)
    model.fit(real)
    prev = float(real[subtype].mean())
    n_rare = max(1, int(round(n * prev)))
    c1 = Condition(num_rows=n_rare, column_values={subtype: 1})
    c0 = Condition(num_rows=n - n_rare, column_values={subtype: 0})
    return model.sample_from_conditions(conditions=[c1, c0])


def prev(df):
    return {s: float(df[s].mean()) for s in SUBTYPES}


def cox_features(df):
    cats = [c for c in CAT if c not in ("OS_STATUS", "DSS_STATUS")]
    X = pd.get_dummies(df[cats], columns=cats, drop_first=True)
    for c in ["AGE", "MUTATION_COUNT", "ANEUPLOIDY_SCORE"]:
        X[c] = df[c].astype(float)
    return X


def cox_cindex(real, synth, seed):
    event = (real["OS_STATUS"].astype(str).str.startswith("1")).astype(int).values
    duration = real["OS_MONTHS"].astype(float).values
    syn_event = (synth["OS_STATUS"].astype(str).str.startswith("1")).astype(int)
    syn_dur = synth["OS_MONTHS"].astype(float)
    Xr = cox_features(real)
    Xs = cox_features(synth).reindex(columns=Xr.columns, fill_value=0)
    s = pd.concat([Xs, pd.DataFrame({"t": syn_dur.values, "e": syn_event.values})], axis=1)
    s = s[s["t"] > 0]
    if s["e"].sum() < 2:
        return 0.5
    try:
        cph = CoxPHFitter(penalizer=0.5)
        cph.fit(s, duration_col="t", event_col="e")
        risk = cph.predict_partial_hazard(Xr)
        return float(concordance_index(duration, -risk.values, event))
    except Exception:
        return 0.5


def membership_auc(real, synth, seed):
    Xr, Xs = cox_features(real), cox_features(synth)
    Xs = Xs.reindex(columns=Xr.columns, fill_value=0)
    X = pd.concat([Xr, Xs], axis=0).values
    y = np.array([1] * len(Xr) + [0] * len(Xs))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def ks_fidelity(real, synth):
    vals = [ks_2samp(real[c], synth[c]).statistic for c in NUM
            if c in real and c in synth]
    return float(np.mean(vals))


def corr_mae(real, synth):
    rc = real[NUM].corr().fillna(0).values
    sc = synth[NUM].corr().fillna(0).values
    return float(np.mean(np.abs(rc - sc)))


def analytical_nstar(eps_grid, tau):
    r = 1.0 / (1.0 + np.exp(eps_grid))
    pmin = r / (tau + 2 * r)
    return pmin


def main():
    real = load()
    res = {"cohort": {"n": int(len(real)),
                      "subtypes": {s: float(real[s].mean()) for s in SUBTYPES},
                      "tumor_type": real["TUMOR_TYPE"].value_counts().to_dict(),
                      "os_status": real["OS_STATUS"].value_counts().to_dict()}}

    # (1) collapse (eps=inf)
    res["collapse"] = {}
    for name in ["TVAE", "GaussianCopula", "CTGAN"]:
        syn = fit_sample(name, real, 500, 42)
        res["collapse"][name] = prev(syn)

    # (2) DP amplification + (3) FairSparse recovery (CTGAN baseline + FairSparse)
    res["dp"] = {}
    res["fairsparse"] = {s: {} for s in SUBTYPES}
    for eps in EPS:
        ekey = "inf" if eps is None else str(eps)
        syn = apply_dp(fit_sample("CTGAN", real, 500, 42), real, eps, 42)
        res["dp"][ekey] = prev(syn)
        for st in SUBTYPES:
            syn_fs = apply_dp(fairsparse_sample(real, 500, st, 42), real, eps, 42)
            res["fairsparse"][st][ekey] = prev(syn_fs)

    # (4) analytical n*-eps floor
    eps_grid = np.linspace(0.1, 10, 200)
    res["nstar"] = {f"tau{t}": analytical_nstar(eps_grid, t).tolist()
                    for t in [0.25, 0.5]}
    res["nstar_eps"] = eps_grid.tolist()

    # (5)-(7) utility / privacy / fidelity for CTGAN vs FairSparse vs TVAE(inf)
    methods = {"CTGAN": lambda r, n, s: fit_sample("CTGAN", r, n, s),
               "FairSparse": lambda r, n, s: fairsparse_sample(r, n, "IDH_mutant", s),
               "TVAE": lambda r, n, s: fit_sample("TVAE", r, n, s)}
    eps_for_eval = [None, 10.0, 2.0, 0.5]
    res["utility"], res["privacy"], res["fidelity"], res["corr"] = {}, {}, {}, {}
    for mname, fn in methods.items():
        res["utility"][mname] = {}
        res["privacy"][mname] = {}
        res["fidelity"][mname] = {}
        res["corr"][mname] = {}
        for eps in eps_for_eval:
            if mname == "TVAE" and eps is not None:
                continue
            ekey = "inf" if eps is None else str(eps)
            ut, pr, ks_, cm = [], [], [], []
            for seed in SEEDS:
                syn = apply_dp(fn(real, 500, seed), real, eps, seed)
                ut.append(cox_cindex(real, syn, seed))
                pr.append(membership_auc(real, syn, seed))
                ks_.append(ks_fidelity(real, syn))
                cm.append(corr_mae(real, syn))
            res["utility"][mname][ekey] = ut
            res["privacy"][mname][ekey] = pr
            res["fidelity"][mname][ekey] = ks_
            res["corr"][mname][ekey] = cm

    out = os.path.join(DATA, "results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved results.json")
    print("cohort subtypes:", {k: f"{v:.1%}" for k, v in res['cohort']['subtypes'].items()})
    print("utility (C-index) CTGAN vs FairSparse @inf:",
          {m: f"{np.mean(v['inf']):.3f}" for m, v in res['utility'].items() if 'inf' in v})
    print("privacy (MIA AUC) CTGAN vs FairSparse @inf:",
          {m: f"{np.mean(v['inf']):.3f}" for m, v in res['privacy'].items() if 'inf' in v})


if __name__ == "__main__":
    main()
