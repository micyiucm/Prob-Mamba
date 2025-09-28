# Import necessary libraries
import sys
import time
import os
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from mamba_ssm import Mamba
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from torch.utils.data import DataLoader, TensorDataset

# Set seed for reproducibility
def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

# QLIKE metric for probabilistic models
def quasi_likelihood(residuals: np.ndarray, var_forecast: np.ndarray) -> float:
    """
    Compute QLIKE (quasi-likelihood) metric for probabilistic models.
    """
    if len(residuals) != len(var_forecast):
        raise ValueError("Residuals and variance arrays must have the same length.")
    # Avoid log(0) by adding a small constant to the variance
    var = np.clip(var_forecast, 1e-12, None)
    return float(np.mean((residuals**2) / var + np.log(var)))

# ARMA baseline
def clean_series(y):
    """
    Convert input to pandas series with float32 dtype,
    replace infinite values with Nan, and drop Nan values.
    """
    y = pd.Series(y, dtype = "float32").replace([np.inf, -np.inf], np.nan).dropna()
    if len(y) < 50:
        raise ValueError("Not enough valid data points after cleaning.")
    if y.std() < 1e-5:
        raise ValueError("Data has near-zero variance after cleaning.")
    return y

def arma_select_bic(y_train, p_max = 3, q_max = 3, trend = 'n'):
    """
    Select ARMA(p,q) model with lowest BIC over grid of (p,q) values.
    """
    y = clean_series(y_train)
    best = None
    best_fit = None
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            if p == 0 and q == 0:
                continue  # Skip meaningless model ARMA(0,0)
            try:
                model = ARIMA(y, order = (p, 0, q), trend = trend)
                fit = model.fit()
                bic = fit.bic
                if best is None or bic < best[0]:
                    best = (bic, (p, 0, q))
                    best_fit = fit
            except Exception as e:
                print(f"ARMA({p},0,{q}) failed: {e}", file=sys.stderr)
                continue
    if best is None:
        raise ValueError("No ARMA model could be fitted to the data.")
    return best[1], best_fit

def arma_fit_forecast(train_df, val_df, test_df, target_col = 'y_next',
                      p_max = 3, q_max = 3, trend = 'n', return_fit = False):
    """
    ARMA modelling pipeline: model selection, fitting and forecasting.
    """
    # Extract target series column and clean
    y_train = clean_series(train_df[target_col].values)
    y_val   = clean_series(val_df[target_col].values)
    y_test  = clean_series(test_df[target_col].values)

    # Model selection by BIC on training set
    order, best_fit = arma_select_bic(y_train, p_max = p_max, q_max = q_max, trend = trend)
    print(f"Selected ARMA order via BIC: {order}")

    # Refit on train_+val set and forecast on test set
    y_trainval = pd.concat([pd.Series(y_train), pd.Series(y_val)], axis=0).reset_index(drop=True)
    final = (ARIMA(y_trainval, order=order,
                   trend=trend,
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit())
    forecasts = final.forecast(steps=len(y_test))
    forecasts = pd.Series(forecasts, dtype="float32").reset_index(drop=True)
    y_test = pd.Series(y_test, dtype="float32").reset_index(drop=True)

    mask = np.isfinite(forecasts.values) & np.isfinite(y_test.values)
    if mask.sum() == 0:
        raise ValueError("No finite forecast/target pairs after alignment. Check NaNs in test data.")

    rmse = float(np.sqrt(np.mean((forecasts[mask] - y_test[mask]) ** 2)))
    if return_fit:
        return order, rmse, forecasts, final
    return order, rmse, forecasts

# GARCH baseline

def garch_fit_forecast(train_df, val_df, test_df, target_col = 'y_next',
                        dist= 'normal', scale = 100.0, mu_trval = None, mu_test = None):
    """
    Fit GARCH(1,1) on train+val residuals from ARMA model.
    """
    if mu_trval is None or mu_test is None:
        raise ValueError("Provide train_val mean and test mean from ARMA.")
    
    # Extract target series column
    y_train = pd.Series(train_df[target_col].values, dtype="float32")
    y_val   = pd.Series(val_df[target_col].values, dtype="float32")
    y_test  = pd.Series(test_df[target_col].values, dtype="float32")

    # Train + val residuals for volatility fit
    y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    mu_trval = pd.Series(mu_trval, dtype='float64').reset_index(drop=True)
    m = np.isfinite(y_trainval.values) & np.isfinite(mu_trval.values)
    resid_trval = (y_trainval.values[m] - mu_trval.values[m])

    # Fit zero-mean GARCH(1,1)
    am = arch_model(resid_trval * scale, mean='Zero', vol='GARCH', p=1, q=1, dist=dist)
    res = am.fit(disp='off')
    
    # Variance forecasting
    forecast = res.forecast(horizon=len(y_test), reindex=False)
    var_hat_scaled = forecast.variance.values[-1, :]

    # Unscaling
    var_hat = pd.Series(var_hat_scaled / (scale ** 2), dtype='float32').reset_index(drop=True)

    # Use ARMA mean forecasts for conditional mean
    mu_hat = pd.Series(mu_test, dtype='float64').reset_index(drop=True)
    y_t = y_test.reset_index(drop=True)
    n = min(len(y_t), len(mu_hat), len(var_hat))
    y_t, mu_hat, var_hat = y_t.iloc[:n], mu_hat.iloc[:n], var_hat.iloc[:n].clip(1e-12)

    # Metrics
    error = y_t - mu_hat
    rmse = float(np.sqrt(np.mean((error.values)**2)))
    qlike = quasi_likelihood(error.values, var_hat.values)
    nll = float(0.5*np.mean(np.log(2*np.pi*var_hat.values) + (error.values**2)/var_hat.values))
    return rmse, qlike, nll, mu_hat, var_hat

# Function to convert time-series data into input sequences
def create_sequences(data: pd.DataFrame, sequence_length: int, feature_columns: list, target_column: str):
    """
    Converts a time-series DataFrame into input sequences and corresponding target values.
    Args:
        data (pd.DataFrame): The pre-processed input DataFrame;
        sequence_length (int): The number of time steps in each input sequence;
        feature_columns (list): The list of the column names to be used as inout features;
        target_column (str): The name of the column to be predicted.
    Returns:
        tuple: A tuple containing the input sequences (X) and corresponding target values (y) as numpy arrays.
    """
    if target_column in feature_columns:
        raise ValueError("target_column should not be in feature_columns")
    
    # Define the features and target values
    features = data[feature_columns].values
    targets = data[target_column].values

    # Initialize lists to hold sequences and targets
    X, y = [], []
    N = len(data)
    L = sequence_length
    if N < L:
        raise ValueError(f"Data length {N} is shorter than sequence length {L}.")

    # Loop through the data to create sequences
    for i in range(0, N - L + 1):
        # The sequence of features
        seq_x = features[i:i + L]
        # The target value at the end of the sequence 
        seq_y = targets[i + L - 1]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    return X, y
# Deep Learning baselines

def rnn_param_count(I, H, L, O):
    """
    Calculate number of parameters in a standard RNN with L layers.
    I = input dim, H = hidden dim, L = num layers, O = output dim
    """
    return (2*L - 1) * H**2 + (I + 2*L + 0) * H + O


def train_for_val_epochs(model, train_loader, val_loader, loss_function, optimiser, epochs,
                         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Train model for a fixed number of epochs, return best validation loss and corresponding model state.
    """
    best_val = float('inf')
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb)
            loss = loss_function(pred, yb)
            loss.backward()
            optimiser.step()

        # One validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_function(pred, yb)
                val_loss += loss.item() * len(yb)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val:
            best_val = val_loss
        return best_val

def candidate_rnns_for_given_budget(input_dim, output_dim, target, n_layers = 1, window = 10, top_k = 6):
    """
    Build multiple RNN candidates around the closed-form H that matches the target by solving a quadratic equation.
    Returns a list of dicts with keys:
    factory: callable that returns the model instance
    hidden_dim: int, the hidden dimension
    L: int, number of RNN layers
    """
    a = 2*n_layers - 1
    b = input_dim + 2*n_layers + output_dim
    c = output_dim - target
    delta = b**2 - 4*a*c
    if delta < 0:
        raise ValueError("No real-valued hidden dimension can achieve the target parameter count.")
    # Solve the quadratic equation for H
    H0 = max(1, int((-b + math.sqrt(delta)) / (2*a)))
    # Explore hidden dims around H0
    candidates = []
    for H in range(max(1, H0 - window), H0 + window + 1):
        param_count = rnn_param_count(input_dim, H, n_layers, output_dim)
        # build a no-arg factory for fresh models
        def make(H_ = H):
            return SimpleRNN(input_dim, H_, n_layers, output_dim)
        candidates.append({
            "factory": make, "params": param_count, "H": H, "L": n_layers,
            "label": f"RNN(H={H},L={n_layers})"
        })
    # pick top k closest to target by absolute difference
    candidates.sort(key = lambda x: abs(x["params"] - target))
    return candidates[:top_k]

def candidate_mambas_for_budget(input_dim, output_dim, target,
                                n_layers = 1, d_state = 64, d_conv = 4,
                                expand = 2, d_model_min =64, d_model_max = 512,
                                top_k = 6):
    """
    Systematically trying different values of d_model and keep the closed top_k to target.
    """
    if "MambaModel" not in globals():
        raise ImportError("MambaModel not available.")
    candidates = []
    for d_model in range(d_model_min, d_model_max +1, step):
        model = MambaModel(input_dim, d_model = d_model, output_dim = output_dim,
                       n_layers = n_layers, d_state = d_state, d_conv = d_conv, expand = expand)
        param= param_count(m)
        def make (dm = d_model):
            return MambaModel(input_dim, d_model = dm, output_dim = output_dim,
                              n_layers = n_layers, d_state = d_state, d_conv = d_conv, expand = expand)
        candidates.append({
            "factory": make, "params": param, "d_model" : d_model, 
            "L": n_layers, "d_state": d_state, "d_conv": d_conv,
            "expand": expand, "label": f"Mamba(d_model={d_model}, L= {n_layers})"
        })
    candidates.sort(key = lambda x: abs(x["params"] - target))
    return candidates[:top_k]


