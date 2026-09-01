"""Supplementary CCA figures (merged: FigS-A cohort+collapse+DP, FigS-B evaluation+n*)."""
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica"],
    "font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,
    "xtick.direction":"out","ytick.direction":"out","axes.edgecolor":"#303030",
    "text.color":"#303030","axes.labelcolor":"#303030","xtick.color":"#303030","ytick.color":"#303030"})
C_REAL="#333333"; C_TVAE="#C87A4F"; C_GC="#5B7FB4"; C_CTGAN="#DD8452"; C_FAIR="#4C72B0"
SUB_C = ["#4C72B0", "#DD8452", "#55A868"]  # IDH/FGFR2/KRAS blue/orange/green
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures_regen")

def save(fig, name):
    fig.savefig(os.path.join(OUT, name+".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name+".tif"), dpi=300, bbox_inches="tight", pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig); print("saved", name)

res = json.load(open(os.path.join(DATA, "results.json")))
subs = ["IDH_mutant","FGFR2_mutant","KRAS_mutant"]
real = [res["cohort"]["subtypes"][s]*100 for s in subs]
eps = ["inf","10.0","2.0","0.5"]

df = pd.read_csv(os.path.join(DATA, "cohort.tsv"), sep="\t")
genes = ["BAP1","PBRM1","IDH1","TP53","ARID1A","IDH2","FGFR2","KRAS","CDKN2A","PIK3CA","BRAF","NRAS","STK11","PTEN","SMAD4"]
freq = {g: int(df[f"mut_{g}"].sum()) for g in genes}
freq = dict(sorted(freq.items(), key=lambda x: -x[1]))

# ============ FigS-A: cohort + collapse + DP (3 panels, a-b-c) ============
fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.9), gridspec_kw={"width_ratios":[0.9, 1.15, 0.95]})
# a) cohort mutation-frequency scatter
for i, g in enumerate(freq):
    axs[0].scatter(freq[g], i, s=36, color="#333333", zorder=3)
axs[0].set_yticks(range(len(freq))); axs[0].set_yticklabels(list(freq.keys()), fontsize=6.5)
axs[0].invert_yaxis(); axs[0].set_xlabel("Mutated patients (n = 36)", fontsize=7)
axs[0].set_xlim(-0.6, max(freq.values())+0.6)
axs[0].set_xticks(range(0, max(freq.values())+1))
axs[0].grid(axis="x", ls=":", alpha=0.3)
# b) subtype collapse
methods = ["Real","TVAE","Gauss-\nCopula","CTGAN","Fair-\nSparse"]
x = np.arange(len(methods)); w = 0.22
data3 = {
    "IDH": [real[0], res["collapse"]["TVAE"]["IDH_mutant"]*100, res["collapse"]["GaussianCopula"]["IDH_mutant"]*100, res["collapse"]["CTGAN"]["IDH_mutant"]*100, res["fairsparse"]["IDH_mutant"]["inf"]["IDH_mutant"]*100],
    "FGFR2": [real[1], res["collapse"]["TVAE"]["FGFR2_mutant"]*100, res["collapse"]["GaussianCopula"]["FGFR2_mutant"]*100, res["collapse"]["CTGAN"]["FGFR2_mutant"]*100, res["fairsparse"]["FGFR2_mutant"]["inf"]["FGFR2_mutant"]*100],
    "KRAS": [real[2], res["collapse"]["TVAE"]["KRAS_mutant"]*100, res["collapse"]["GaussianCopula"]["KRAS_mutant"]*100, res["collapse"]["CTGAN"]["KRAS_mutant"]*100, res["fairsparse"]["KRAS_mutant"]["inf"]["KRAS_mutant"]*100],
}
for j,(k,v) in enumerate(data3.items()):
    for i,val in enumerate(v):
        axs[1].scatter(i + (j-1)*w, val, s=40, color=SUB_C[j], zorder=3, edgecolor="white", lw=0.6)
axs[1].set_xticks(x); axs[1].set_xticklabels(methods, fontsize=6.5)
axs[1].set_ylabel("Prevalence (%)", fontsize=7)
handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=5) for c in SUB_C]
axs[1].legend(handles, ["IDH","FGFR2","KRAS"], frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
# c) DP
for s,c,lab in zip(subs,[C_FAIR,C_GC,C_TVAE],["IDH","FGFR2","KRAS"]):
    y = [res["dp"][e][s]*100 for e in eps]
    axs[2].plot(range(4), y, "o-", color=c, lw=1.3, ms=3.5, label=lab)
    axs[2].axhline(res["cohort"]["subtypes"][s]*100, color=c, ls=":", lw=1)
axs[2].set_xticks(range(4)); axs[2].set_xticklabels(["∞","10","2","0.5"], fontsize=7)
axs[2].set_xlabel("Noise level ε", fontsize=7); axs[2].set_ylabel("Prevalence (%)", fontsize=7)
axs[2].legend(frameon=False, fontsize=6.5)
for i, lab in enumerate("abc"):
    axs[i].text(-0.3, 1.03, lab, transform=axs[i].transAxes, fontsize=11, va="top", ha="left")
fig.tight_layout(); save(fig, "Supplementary_Fig_S2")

# ============ FigS-B: utility + privacy + n* (3 panels, a-b-c) ============
fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.7))
# a) utility (blues: CTGAN light / FairSparse dark)
for m,c,lab in [("CTGAN","#9ECAE1","CTGAN"),("FairSparse","#2166AC","FairSparse")]:
    mu = [np.mean(res["utility"][m][e]) for e in eps]
    sd = [np.std(res["utility"][m][e]) for e in eps]
    axs[0].errorbar(range(4), mu, yerr=sd, color=c, marker="o", capsize=3, lw=1.3, ms=4, label=lab)
axs[0].axhline(0.5, color="#999", ls="--", lw=1)
axs[0].set_xticks(range(4)); axs[0].set_xticklabels(["∞","10","2","0.5"], fontsize=7)
axs[0].set_ylabel("Cox C-index", fontsize=7); axs[0].set_xlabel("Noise level ε", fontsize=7)
axs[0].set_ylim(0.2, 0.8); axs[0].legend(frameon=False, fontsize=6.5)
# b) privacy (orange/red: CTGAN light / FairSparse dark)
for m,c,lab in [("CTGAN","#FDAE6B","CTGAN"),("FairSparse","#D62728","FairSparse")]:
    mu = [np.mean(res["privacy"][m][e]) for e in eps]
    axs[1].plot(range(4), mu, "o-", color=c, lw=1.3, ms=4, label=lab)
axs[1].axhline(0.5, color="#999", ls="--", lw=1)
axs[1].set_xticks(range(4)); axs[1].set_xticklabels(["∞","10","2","0.5"], fontsize=7)
axs[1].set_ylabel("MIA AUC", fontsize=7); axs[1].set_xlabel("Noise level ε", fontsize=7)
axs[1].set_ylim(0.3, 1.0); axs[1].legend(frameon=False, fontsize=6.5)
# c) n* curve
pts = json.load(open(os.path.join(DATA, "nstar_empirical.json")))
chol = [p for p in pts if p["cohort"] == "chol"]
EPS_C = {"inf": "#333333", "10.0": "#4C72B0", "2.0": "#DD8452", "0.5": "#55A868"}
EPS_L = {"inf": "∞", "10.0": "10", "2.0": "2", "0.5": "0.5"}
p_grid = np.linspace(0.01, 0.5, 200)
for eps_, r in [(10.0, 1/(1+np.exp(10))), (2.0, 1/(1+np.exp(2))), (0.5, 1/(1+np.exp(0.5)))]:
    err = r * (1 - 2*p_grid) / p_grid
    axs[2].plot(p_grid*100, err, color=EPS_C[str(eps_)], lw=1.2, ls="--", alpha=0.6)
for eps_ in ["inf", "10.0", "2.0", "0.5"]:
    xs = [p["prevalence"]*100 for p in chol if p["eps"] == eps_]
    ys = [p["rel_error"] for p in chol if p["eps"] == eps_]
    axs[2].scatter(xs, ys, s=20, color=EPS_C[eps_], alpha=0.7, edgecolors="white", linewidths=0.3, zorder=3, label=f"ε={EPS_L[eps_]}")
axs[2].set_xscale("log"); axs[2].set_yscale("log")
axs[2].set_xlabel("Subtype prevalence (%)", fontsize=7)
axs[2].set_ylabel("Relative prevalence error", fontsize=7)
axs[2].set_xlim(1, 60); axs[2].set_ylim(0.005, 20)
axs[2].grid(True, which="both", ls=":", alpha=0.3)
axs[2].legend(frameon=False, fontsize=6, ncol=2, loc="upper right")
for i, lab in enumerate("abc"):
    axs[i].text(-0.3, 1.03, lab, transform=axs[i].transAxes, fontsize=11, va="top", ha="left")
fig.subplots_adjust(wspace=0.5)
fig.tight_layout(); save(fig, "Supplementary_Fig_S1")
print("DONE supp merged")
