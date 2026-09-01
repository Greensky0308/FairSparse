"""Publication-quality figures v2: violin/point/scatter/radar (Nature-style, Okabe-Ito colour-blind-safe)."""
import sys, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica"],
    "font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,
    "xtick.direction":"out","ytick.direction":"out","axes.edgecolor":"#303030",
    "text.color":"#303030","axes.labelcolor":"#303030","xtick.color":"#303030","ytick.color":"#303030"})
OKABE = ['#000000','#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7']
FIG = os.path.join(os.path.dirname(__file__), "..", "figures_regen")

def save(fig, name):
    fig.savefig(os.path.join(FIG, name+".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name+".tif"), dpi=300, bbox_inches="tight", pil_kwargs={"compression":"tiff_lzw"})
    fig.savefig(os.path.join(FIG, name+".png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved", name)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "fibrolamellar", "cohort.tsv"), sep="\t")
df["AGE"] = pd.to_numeric(df["AGE"]); df["fusion_status"] = (df["fusion_status"]=="positive")

import json as _json
R = _json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "fibrolamellar", "fibrolamellar_results.json")))

# ===== Fig1: framework diagram (unified flow) =====
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
def fig1():
    fig, ax = plt.subplots(figsize=(2.8, 3.3)); ax.axis("off")
    ax.set_xlim(0, 2.8); ax.set_ylim(0, 3.3)
    W, H, GAP = 2.2, 0.44, 0.16
    X = (2.8 - W) / 2
    labels = [
        ("Fibrolamellar-spectrum\ncohort (n = 50)", "#eef2f7"),
        ("Naive synthesis\nCTGAN / TVAE", "#fbe9e7"),
        ("Two threats:\nerasure + decoupling", "#fff3e0"),
        ("FairSparse\nsubtype-stratified", "#e8f5e9"),
        ("Privacy-preserving data\nassociation preserved", "#e3f2fd"),
    ]
    n = len(labels)
    for i, (txt, c) in enumerate(labels):
        y = 0.15 + (n-1-i)*(H+GAP)
        ax.add_patch(FancyBboxPatch((X, y), W, H, boxstyle="round,pad=0.015", fc=c, ec="#999999", lw=0.5))
        ax.text(X+W/2, y+H/2, txt, ha="center", va="center", fontsize=7.5)
    for i in range(n-1):
        y0 = 0.15 + (n-1-i)*(H+GAP)
        y1 = 0.15 + (n-2-i)*(H+GAP) + H
        ax.annotate("", xy=(X+W/2, y1), xytext=(X+W/2, y0),
                    arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))
    return fig
save(fig1(), "Fig1_framework")

# ===== Fig2: cohort age separation (a violin + b bootstrap correlation histogram) =====
from scipy.stats import pearsonr
C_FUS, C_BAP = "#5B7FB4", "#C87A4F"
fig = plt.figure(figsize=(7.2, 3.2))
ax = fig.add_subplot(1, 2, 1)
sns.violinplot(data=df, x="fusion_status", y="AGE", inner=None, cut=0,
               palette=[C_FUS, C_BAP], linewidth=0.5, saturation=0.85, bw=0.4, ax=ax)
sns.stripplot(data=df, x="fusion_status", y="AGE", color="#333333",
              size=2.0, alpha=0.6, edgecolor="none", jitter=0.12, ax=ax)
ax.set_xticks([0,1]); ax.set_xticklabels(["Fusion+\n(n=32)","BAP1 fusion−\n(n=18)"], fontsize=8)
ax.set_ylabel("Age (years)", fontsize=8); ax.set_xlabel("")
ax.set_ylim(0, 82)
ax.plot([0,0,1,1], [76,78,78,76], color="#333333", lw=0.8)
ax.text(0.5, 79, "p < 0.001", ha="center", va="bottom", fontsize=7)
ax.text(-0.10, 1.10, "a", transform=ax.transAxes, fontsize=11, va="top")

ax2 = fig.add_subplot(1, 2, 2)
age = df["AGE"].to_numpy(float)
fusion = df["fusion_status"].astype(int).to_numpy()
rng = np.random.default_rng(42)
n = len(age)
boots = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    boots.append(pearsonr(age[idx], fusion[idx])[0])
boots = np.array(boots)
observed = pearsonr(age, fusion)[0]
ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
ax2.hist(boots, bins=25, color="#9DB6D4", edgecolor="white", linewidth=0.4)
ax2.axvline(observed, color=OKABE[0], lw=1.2)
ax2.axvline(ci_lo, color=OKABE[0], ls="--", lw=0.8)
ax2.axvline(ci_hi, color=OKABE[0], ls="--", lw=0.8)
ax2.set_xlabel("Age–fusion correlation (bootstrap)")
ax2.set_ylabel("Frequency")
ax2.set_xlim(-0.95, -0.45)
ax2.text(0.98, 0.95, f"observed {observed:.3f}\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
         transform=ax2.transAxes, ha="right", va="top", fontsize=7)
ax2.text(-0.10, 1.10, "b", transform=ax2.transAxes, fontsize=11, va="top")
fig.tight_layout(); save(fig, "Fig2_cohort")

# ===== Fig3: rare-subtype double failure (point plot + reference lines) =====
# synthetic (eps=inf) + uniform DP + DP-SGD
cats1 = ["Real","TVAE","Gauss-\nCopula","CTGAN","Fair-\nSparse"]
vals1 = [R["prevalence"]["Real"], R["prevalence"]["TVAE"], R["prevalence"]["GaussCopula"], R["prevalence"]["CTGAN"], R["prevalence"]["FairSparse"]]
cols1 = [OKABE[0], OKABE[6], OKABE[5], OKABE[3], OKABE[2]]
fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.axhline(36.0, color=OKABE[0], ls="--", lw=1, alpha=0.7)
for i,(v,c) in enumerate(zip(vals1, cols1)):
    ax.scatter(i, v, s=95, color=c, zorder=3, edgecolor="white", linewidth=0.8)
    ax.text(i, v+2.5, f"{v:.1f}", ha="center", fontsize=7)
cats2 = ["ε=10","ε=2","ε=0.5","DP-SGD\nσ=3"]
vals2 = [R["prevalence_dp"]["eps10.0"], R["prevalence_dp"]["eps2.0"], R["prevalence_dp"]["eps0.5"], R["dp_sgd"]["sigma3_neg_rate"]]
for i,v in enumerate(vals2):
    ax.scatter(5+i, v, s=95, color=OKABE[7], zorder=3, edgecolor="white", linewidth=0.8)
    ax.text(5+i, v+2.5, f"{v:.1f}", ha="center", fontsize=7)
ax.axvline(4.5, color="0.7", ls=":", lw=0.8)
ax.set_xticks(list(range(5))+list(range(5,9)))
ax.set_xticklabels(cats1+cats2, fontsize=7)
ax.set_ylabel("Fusion-negative prevalence (%)"); ax.set_ylim(-3, 58); ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.text(0.25, 0.02, "naive synthesis (ε=∞)", transform=ax.transAxes, ha="center", fontsize=7, color=OKABE[6])
ax.text(0.78, 0.02, "noise perturbation / DP-SGD", transform=ax.transAxes, ha="center", fontsize=7, color=OKABE[7])
fig.tight_layout(); save(fig, "Fig3_collapse")

# ===== Fig4: association severed -> downstream recovery (a scatter + b fusion-negative age bars+scatter) =====
fig = plt.figure(figsize=(7.2, 3.2))
ax = fig.add_subplot(1, 2, 1)
_m_fs = R["metrics"]["FairSparse"]; _m_ct = R["metrics"]["CTGAN"]
pts = [("Real (upper bound)", R["real"]["age_fusion_corr"], R["real"]["utility_upper"], OKABE[0], (14, 14), "left"),
       ("FairSparse", _m_fs["corr"], _m_fs["util"], OKABE[2], (0, -20), "center"),
       ("CTGAN", _m_ct["corr"], _m_ct["util"], OKABE[6], (16, -12), "left")]
for lab, x, y, c, (dx, dy), ha in pts:
    ax.scatter(x, y, s=100, color=c, zorder=3, edgecolor="white", linewidth=1)
    ax.annotate(lab, (x,y), textcoords="offset points", xytext=(dx, dy), fontsize=8, ha=ha)
ax.axhline(0.5, color="0.6", ls=":", lw=1)
ax.set_xlabel(f"AGE–fusion correlation (real = {R['real']['age_fusion_corr']:.3f})")
ax.set_ylabel("Downstream subtype-classifier AUC")
ax.set_xlim(-0.95, 0.2); ax.set_ylim(0.30, 1.05)
ax.annotate("", xy=(-0.80, 0.87), xytext=(-0.10, 0.56),
            arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))
ax.text(-0.10, 1.10, "a", transform=ax.transAxes, fontsize=11, va="top")

ax2 = fig.add_subplot(1, 2, 2)
labels = ["CTGAN", "FairSparse"]
cols = [OKABE[6], OKABE[2]]
ks_vals = [R["metrics"]["CTGAN"]["fair"], R["metrics"]["FairSparse"]["fair"]]
bars = ax2.bar(labels, ks_vals, color=cols, width=0.45, edgecolor="#303030", linewidth=0.8, alpha=0.95, zorder=2)
for bar, v in zip(bars, ks_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax2.axhline(0, color=OKABE[0], ls="--", lw=0.8)
ax2.text(0.5, 0.03, "Real (KS = 0)", transform=ax2.transAxes, ha="center", fontsize=7)
ax2.set_ylabel("Fusion-negative age KS distance")
ax2.set_ylim(0, 0.6)
ax2.text(-0.10, 1.10, "b", transform=ax2.transAxes, fontsize=11, va="top")
fig.tight_layout(); save(fig, "Fig4_utility_fairness")

# ===== Fig5: four-dimension evaluation radar =====
dims = ["Fidelity\n(1−KS)","Utility\n(AUC)","Privacy\n(1−MIA)","Fairness\n(1−KS)"]
ctgan = [1-_m_ct["fid"], _m_ct["util"], 1-_m_ct["priv"], 1-_m_ct["fair"]]
v3    = [1-_m_fs["fid"], _m_fs["util"], 1-_m_fs["priv"], 1-_m_fs["fair"]]
ang = np.linspace(0, 2*np.pi, len(dims), endpoint=False).tolist(); ang += ang[:1]
ctgan += ctgan[:1]; v3 += v3[:1]
fig = plt.figure(figsize=(3.6, 3.2)); ax = fig.add_subplot(111, polar=True)
ax.plot(ang, v3, color=OKABE[2], lw=1.8, label="FairSparse"); ax.fill(ang, v3, color=OKABE[2], alpha=0.12)
ax.plot(ang, ctgan, color=OKABE[6], lw=1.8, label="CTGAN"); ax.fill(ang, ctgan, color=OKABE[6], alpha=0.06)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(dims, fontsize=7)
ax.set_ylim(0, 1); ax.set_yticks([0.5,1.0]); ax.set_yticklabels(["0.5","1.0"], fontsize=6)
ax.yaxis.set_zorder(10)
ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.15), frameon=False, fontsize=8)
fig.tight_layout(); save(fig, "Fig5_privacy")

# ===== Fig5 with numbers (non-overwriting) =====
dims2 = ["Fidelity\n(1−KS)","Utility\n(AUC)","Privacy\n(1−MIA)","Fairness\n(1−KS)"]
ctgan2 = [1-_m_ct["fid"], _m_ct["util"], 1-_m_ct["priv"], 1-_m_ct["fair"]]
v3_2   = [1-_m_fs["fid"], _m_fs["util"], 1-_m_fs["priv"], 1-_m_fs["fair"]]
ang2 = np.linspace(0, 2*np.pi, len(dims2), endpoint=False).tolist(); ang2 += ang2[:1]
ctgan2c = ctgan2 + ctgan2[:1]; v3_2c = v3_2 + v3_2[:1]
fig = plt.figure(figsize=(4.2, 3.6)); ax = fig.add_subplot(111, polar=True)
ax.plot(ang2, v3_2c, color=OKABE[2], lw=1.8, label="FairSparse"); ax.fill(ang2, v3_2c, color=OKABE[2], alpha=0.12)
ax.plot(ang2, ctgan2c, color=OKABE[6], lw=1.8, label="CTGAN"); ax.fill(ang2, ctgan2c, color=OKABE[6], alpha=0.06)
ax.set_xticks(ang2[:-1]); ax.set_xticklabels(dims2, fontsize=7)
ax.set_ylim(0, 1); ax.set_yticks([0.5,1.0]); ax.set_yticklabels(["0.5","1.0"], fontsize=6)
ax.yaxis.set_zorder(10)
for i in range(4):
    a = ang2[i]
    ax.text(a, v3_2[i]+0.07, f"{v3_2[i]:.2f}", ha="center", va="center", fontsize=6.5, color=OKABE[2])
    ax.text(a, ctgan2[i]-0.10, f"{ctgan2[i]:.2f}", ha="center", va="center", fontsize=6.5, color=OKABE[6])
ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.15), frameon=False, fontsize=8)
fig.tight_layout(); save(fig, "Fig5_privacy_numbers")

print("DONE")
