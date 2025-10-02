# Prob-Mamba evaluation utilities

# Import libraries

import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pyarrow.feather as feather
from prob_mamba.utils import param_count


# Build seq2seq loaders for Probabilistic Mamba
def build_seq2seq_loaders(prefix, key_in_all, all_processed_data, batch_size=64):
    """
    prefix: "cleaned_NYSE" | "cleaned_IXIC" | "BTC_USDT-5m"
    key_in_all: "NYSE" | "IXIC" | "BTC"  (key in all_processed_data)
    """
    ddir = Path("/home/ubuntu/michael/MSc-Machine-Learning-Project/Datasets/Processed")
    train_df = feather.read_feather(ddir / f"{prefix}_train.feather")
    val_df   = feather.read_feather(ddir / f"{prefix}_val.feather")
    test_df  = feather.read_feather(ddir / f"{prefix}_test.feather")

    X_train = all_processed_data[key_in_all]['X_train']
    X_val   = all_processed_data[key_in_all]['X_val']
    X_test  = all_processed_data[key_in_all]['X_test']

    L = X_train.shape[1]
    input_dim = X_train.shape[2]

    def make_seq2seq_targets(df, L, target_col="y_next"):
        y = df[target_col].values
        Y = []
        for i in range(0, len(y) - L + 1):
            Y.append(y[i:i+L])
        return np.array(Y, dtype=np.float32)[..., None]  # (B, L, 1)

    Y_train = make_seq2seq_targets(train_df, L)
    Y_val   = make_seq2seq_targets(val_df,   L)
    Y_test  = make_seq2seq_targets(test_df,  L)

    trainL = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(Y_train, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    valL   = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                      torch.tensor(Y_val, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    testL  = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                      torch.tensor(Y_test, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    return trainL, valL, testL, input_dim, L

def prob_head_params(d_feat: int, n_state: int, d_y: int = 1) -> int:
    """
    Return the number of parameters in the probabilistic head of Prob-Mamba, i.e.
    the probabilistic component of the model
    """
    F, n, dy = d_feat, n_state, d_y
    return (
        n*F*F               # map_B weights
        + 3*n*F             # map_B bias + map_C weights + map_sig weights  (dy=1)
        + 2*F               # gate weights + map_R weights (dy=1)
        + (5*n + 3*dy + 1)  # gate b + map_C b + map_sig b + map_R b + a_raw + q_raw + r_raw + p0
    )

# Function to build prob-Mamba with a given parameter budget
def build_prob_mamba_budget_fixed_n(
    input_dim: int,
    target: int = 100_000,
    n_state: int = 16,
    n_layers: int = 1,
    d_state: int = 16, d_conv: int = 4, expand: int = 2,
    d_feat_min: int = 32, d_feat_max: int = 192, d_feat_step: int = 8,
    tol_frac: float = 0.03, verbose: bool = True,
):
    from prob_mamba.models import ProbabilisticMamba
    best = None
    mcfg = {"d_state": d_state, "d_conv": d_conv, "expand": expand}

    for F in range(d_feat_min, d_feat_max + 1, d_feat_step):
        m = ProbabilisticMamba(
            d_in=input_dim, d_feat=F, d_y=1, n_state=n_state,
            n_mamba_layers=n_layers, mamba_cfg=mcfg
        )
        p = param_count(m)
        if best is None or abs(p - target) < abs(best[1] - target):
            best = (m, p, F)
        if abs(p - target) <= tol_frac * max(1, target):
            if verbose:
                print(f"[Prob-Mamba n={n_state}] target≈{target:,} | got={p:,} | d_feat={F} | "
                      f"layers={n_layers} | head-only≈{prob_head_params(F, n_state):,}")
            return m 

    # Fallback: return the closest we found
    m, p, F = best
    if verbose:
        print(f"[Prob-Mamba n={n_state}] (closest) target≈{target:,} | got={p:,} | d_feat={F} | "
              f"layers={n_layers} | head-only≈{prob_head_params(F, n_state):,}")
    return m 

def train_prob_mamba_with_history(model, train_loader, val_loader, device, epochs=100, lr=1e-3, print_every=10):
    """
    Trains Prob-Mamba using NLL and tracks training history for plotting.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    hist = {"train": [], "val": []}
    for ep in range(epochs):
        # --- train ---
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb, yb)         # expects y; returns dict with "nll"
            loss = out["nll"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * xb.size(0)
        tr = running / len(train_loader.dataset)
        hist["train"].append(tr)

        # --- val ---
        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                running += model(xb, yb)["nll"].item() * xb.size(0)
        va = running / len(val_loader.dataset)
        hist["val"].append(va)

        if (ep+1) % print_every == 0 or ep == 0:
            print(f"[ep {ep+1:03d}] train NLL {tr:.6f} | val NLL {va:.6f}")
    return hist

def plot_nll_history(hist, every=1, title="Prob-Mamba: NLL vs epoch"):
    """
    Function to plot training and validation NLL history.
    """
    import matplotlib.pyplot as plt
    idx = list(range(0, len(hist["val"]), every))
    plt.figure()
    plt.plot(hist["train"], label="Train NLL", alpha=0.7)
    plt.plot(hist["val"],   label="Val NLL",   alpha=0.9)
    if every > 1:
        plt.scatter(idx, [hist["val"][i] for i in idx], s=20, label=f"Val (every {every})")
    plt.xlabel("Epoch"); plt.ylabel("NLL"); plt.title(title); plt.legend(); plt.grid(True); plt.show()

def eval_prob_rmse_qlike_laststep(model, loader, device, eps=1e-12):
    """
    Evaluate Prob-Mamba on test set using RMSE and QLIKE on last-step-ahead predictions.
    """
    model.eval()
    nll_total = 0.0
    se_sum = 0.0
    n_pts = 0
    qlike_sum = 0.0
    nll_count = 0

    with torch.no_grad():
        for xb, yb in loader:                  # yb: (B, L, 1)
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb, yb)                # use y for filtered 1-step-ahead preds
            mu  = out["y_mean"][:, -1, :].cpu().numpy().ravel()       # (B,)
            var = out["y_var_diag"][:, -1, :].cpu().numpy().ravel()   # (B,)
            yt  = yb[:, -1, :].cpu().numpy().ravel()                  # (B,)

            var = np.maximum(var, eps)
            se_sum += np.sum((mu - yt)**2)
            qlike_sum += np.sum(((yt - mu)**2) / var + np.log(var))
            n_pts += yt.size

            if "nll" in out:
                nll_total += out["nll"].item() * xb.size(0)
                nll_count += xb.size(0)

    rmse = math.sqrt(se_sum / n_pts)
    qlike = qlike_sum / n_pts
    nll = nll_total / nll_count if nll_count > 0 else None
    return {"rmse": rmse, "qlike": qlike, "nll": nll}