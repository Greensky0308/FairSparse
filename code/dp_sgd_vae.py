"""DP-SGD (opacus) baseline: DP-VAE generates synthetic data; evaluates whether the rare subtype (fusion-negative) is destroyed."""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "fibrolamellar")

def load_tensor():
    df = pd.read_csv(os.path.join(DATA, "cohort.tsv"), sep="\t")
    age = pd.to_numeric(df["AGE"]).values.astype(np.float32)
    age_mean, age_std = age.mean(), age.std()
    age = (age - age_mean) / age_std
    sex = pd.get_dummies(df["SEX"].astype(str)).values.astype(np.float32)      # F/M
    fs = ((df["fusion_status"] == "positive").astype(int).values).reshape(-1,1).astype(np.float32)
    X = np.concatenate([age.reshape(-1,1), sex, fs], axis=1)
    meta = {"age_mean": age_mean, "age_std": age_std, "n_cat": sex.shape[1]+1}
    return torch.tensor(X), meta

class VAE(nn.Module):
    def __init__(self, d_in, d_hid=32, d_z=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, d_hid), nn.ReLU())
        self.mu = nn.Linear(d_hid, d_z); self.logvar = nn.Linear(d_hid, d_z)
        self.dec = nn.Sequential(nn.Linear(d_z, d_hid), nn.ReLU(), nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Linear(d_hid, d_in))
    def encode(self, x):
        h = self.enc(x); return self.mu(h), self.logvar(h)
    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv); e = torch.randn_like(std); return mu + e*std
    def forward(self, x):
        mu, lv = self.encode(x); z = self.reparam(mu, lv); return self.dec(z), mu, lv

def loss_fn(x, xr, mu, lv, n_cat):
    recon_cont = nn.functional.mse_loss(xr[:, :1], x[:, :1])          # AGE
    recon_cat = nn.functional.binary_cross_entropy_with_logits(xr[:, 1:], x[:, 1:])  # one-hot + fusion
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
    return recon_cont + recon_cat + 0.1*kl

def train_dp(X, noise_multiplier, epochs=300, seed=42, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    d_in = X.shape[1]
    model = VAE(d_in)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X), batch_size=min(50, len(X)), shuffle=True)
    from opacus import PrivacyEngine
    pe = PrivacyEngine()
    model, opt, loader = pe.make_private(module=model, optimizer=opt, data_loader=loader,
                                         noise_multiplier=noise_multiplier, max_grad_norm=1.0)
    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            opt.zero_grad()
            xr, mu, lv = model(xb)
            loss = loss_fn(xb, xr, mu, lv, 1)
            loss.backward(); opt.step()
    eps = float('inf') if noise_multiplier == 0 else pe.get_epsilon(delta=1.0/len(X)**2)
    model.eval()
    return model, eps

def sample(model, meta, n, seed=42):
    torch.manual_seed(seed)
    d_z = 8
    with torch.no_grad():
        z = torch.randn(n, d_z)
        xr = model.dec(z)
    xr = xr.numpy()
    age = xr[:, 0]*meta["age_std"] + meta["age_mean"]
    fs_logit = xr[:, -1]
    fs = (1.0/(1.0+np.exp(-fs_logit)) > 0.5).astype(int)
    return age, fs

def main():
    X, meta = load_tensor()
    print(f"real n={len(X)} fusion-positive={X[:,-1].mean().item():.1%} fusion-negative={1-X[:,-1].mean().item():.1%}")
    print(f"real fusion-negative median AGE: {X[X[:,-1]==0][:,0].median().item()*meta['age_std']+meta['age_mean']:.1f}\n")
    print(f"{'noise_mult':>10s} {'eps':>8s} {'neg_rate':>8s} {'neg_age_med':>10s}")
    dp = {}
    for sigma in [0.0, 1.0, 3.0, 6.0]:
        model, eps = train_dp(X, sigma)
        age, fs = sample(model, meta, 500)
        neg_rate = 1 - fs.mean()
        neg_age = float(np.median(age[fs==0])) if (fs==0).sum()>0 else None
        print(f"{sigma:>10.1f} {eps:>8.2f} {neg_rate:>8.1%} {np.nan if neg_age is None else neg_age:>10.1f}")
        tag = f"sigma{sigma:.0f}"
        dp[f"{tag}_neg_rate"] = float(neg_rate)
        dp[f"{tag}_neg_age"] = neg_age
        dp[f"{tag}_eps"] = None if eps == float("inf") else float(eps)
    out = os.path.join(DATA, "fibrolamellar_results.json")
    with open(out) as f:
        results = json.load(f)
    results["dp_sgd"] = dp
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved dp_sgd -> {out}")

if __name__ == "__main__":
    main()
