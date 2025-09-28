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
from prob_mamba.models import SimpleRNN, MambaModel

# Universal constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 1e-3
num_epochs = 100

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
# Deep Learning Model training and evaluation
# Training function
def train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs):
    """
    The main training loop for a PyTorch model.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        loss_function (nn.Module): The loss function (e.g., nn.MSELoss).
        optimizer (torch.optim.Optimizer): The optimizer (e.g., Adam).
        num_epochs (int): The total number of epochs to train for.
    """
    # Start the timer
    start_time = time.time()
    
    # Loop over the number of specified epochs
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()  # Set the model to training mode
        train_loss = 0.0
        for inputs, targets in train_loader:
            # Move data to the specified device (e.g., GPU)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 1. Clear previous gradients
            optimizer.zero_grad()
            
            # 2. Forward pass: compute predicted outputs
            outputs = model(inputs)
            
            # 3. Calculate loss
            loss = loss_function(outputs, targets)
            
            # 4. Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # 5. Call step() to update the weights
            optimizer.step()
            
            # Accumulate the training loss
            train_loss += loss.item() * inputs.size(0)
        
        # --- Validation Phase ---
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation
            for inputs, targets in val_loader:
                # Move data to the specified device
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                
                # Accumulate the validation loss
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate average losses for the epoch
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        # Print training progress (e.g., every 10 epochs)
        if (epoch + 1) % 10 == 0 or epoch == 0:
             print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
    
    # Print total training time
    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

# Evaluation function
def evaluation_function(model, test_loader, loss_function):
    """
    Evaluates the model on the test set and returns the average loss.
    Args:
        model (nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test set.
        loss_function (nn.Module): Loss function to use for evaluation.
    Returns:
        float: Final average loss on the test set.
    """
    model.to(device)  # Ensure model is on the correct device
    model.eval()  # Set model to evaluation mode
    # Initialisation
    total_loss = 0.0
    all_preds = []
    all_targets = []
    print("Evaluating model on test set...")
    start_time = time.time()
    with torch.no_grad():  # Disable gradient calculation for inference
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs) # Get model predictions
            loss = loss_function(outputs, targets) # Calculate loss
            total_loss += loss.item() * inputs.size(0)  # Accumulate loss

            # Store predictions and targets to analyse later if needed
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")

    # Calculate final average loss
    final_loss = total_loss / len(test_loader.dataset)
    print(f"Final Average Loss: {final_loss:.6f}")
    return final_loss

# Deep Learning baselines

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rnn_param_count(I, H, L, O):
    """
    Calculate number of parameters in a standard RNN with L layers without creating the model.
    I = input dim, H = hidden dim, L = num layers, O = output dim
    """
    return (2*L - 1) * H**2 + (I + 2*L + O) * H + O


def train_for_val_epochs(model, train_loader, val_loader, loss_function, optimiser, epochs,
                         device = device):
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
    H: int, the hidden dimension
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
        number_params = rnn_param_count(input_dim, H, n_layers, output_dim)
        # build a no-arg factory for fresh models
        def make(H_ = H):
            return SimpleRNN(input_dim, H_, n_layers, output_dim)
        candidates.append({
            "factory": make, "params": number_params, "H": H, "L": n_layers,
            "label": f"RNN(H={H},L={n_layers})"
        })
    # pick top k closest to target by absolute difference
    candidates.sort(key = lambda x: abs(x["params"] - target))
    return candidates[:top_k]

def candidate_mambas_for_budget(input_dim, output_dim, target,
                                n_layers = 1, d_state = 64, d_conv = 4,
                                expand = 2, d_model_min =64, d_model_max = 512,
                                step = 8, top_k = 6):
    """
    Systematically trying different values of d_model and keep the closest top_k to target.
    """
    if "MambaModel" not in globals():
        raise ImportError("MambaModel not available.")
    candidates = []
    for d_model in range(d_model_min, d_model_max +1, step):
        model = MambaModel(input_dim, d_model = d_model, output_dim = output_dim,
                       n_layers = n_layers, d_state = d_state, d_conv = d_conv, expand = expand)
        param= param_count(model)
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

def validate_param_distribution(candidates, train_loader, val_loader,
                               loss_function, seeds = (0,1,2), val_epochs = max(10, num_epochs//5),
                               lr = learning_rate, device = device,
                               verbose = True):
    """
    For each candidate, train briefly across multiple seeds and report the
    mean and standard deviation of best validation losses. Return the best
    candidate spec and a report dicionary for all.
    """
    report = []
    best_idx, best_mean = None, float("inf")

    for idx, spec in enumerate(candidates):
        seed_losses = []
        for s in seeds:
            set_seed(s)
            model = spec["factory"]().to(device)
            opt = optim.Adam(model.parameters(), lr = lr)
            best_val = train_for_val_epochs(model, train_loader, val_loader,
                                            loss_function, opt, val_epochs, 
                                            device = device)
            seed_losses.append(best_val)
        mean_v = float(np.mean(seed_losses))
        std_v = float(np.std(seed_losses))
        item = {
            "label": spec.get("label", f"candidate{idx}"),
            "params": int(spec["params"]),
            "val_mean": mean_v,
            "val_std": std_v,
            "seed_losses": seed_losses,
        }
        report.append(item)
        if verbose:
            print(f"[VAL] {item['label']}: params={item['params']:,}, "
                  f"val_mean={item['val_mean']:.4f} ±  val_std={item['val_std']:.4f} over {len(seeds)} seeds")
        if mean_v < best_mean:
            best_mean, best_idx = mean_v, idx
    return candidates[best_idx], report

def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch=batch_size):
    """
    Create DataLoader objects for training, validation, and test sets.
    """
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch, shuffle=False)
    return train_loader, val_loader, test_loader

def run_experiment_block_param_validated(dataset_name, all_processed_data,
                                         budgets = (100000, 300000),
                                         rnn_layers = 1, mamba_layers = 1,
                                         seeds = (0,1,2), val_epochs = max(10, num_epochs//5),
                                         tol_print = 6):
    """
    For each budget, build a distribution of candidates (RNN and Mamba),
    select by validation mean across seeds, then train the selected model fully and test.
    Returns a pandas DataFrame summarising choices and performance.
    """
    X_train = all_processed_data[dataset_name]["X_train"]
    y_train = all_processed_data[dataset_name]["y_train"]
    X_val = all_processed_data[dataset_name]["X_val"]
    y_val = all_processed_data[dataset_name]["y_val"]
    X_test = all_processed_data[dataset_name]["X_test"]
    y_test = all_processed_data[dataset_name]["y_test"]

    input_dim = X_train.shape[2]
    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch = batch_size
    )
    loss_function = nn.MSELoss()
    results = {}
    for T in budgets:
        print(f"\n{'='*20} Budget {T:,} params — candidate validation {'='*20}")

        # Build candidate sets
        rnn_candidates = candidate_rnns_for_given_budget(
            input_dim, output_dim = 1, target = T, n_layers = rnn_layers, window = 10, top_k = tol_print
        )
        mamba_candidates = candidate_mambas_for_budget(
            input_dim, output_dim = 1, target = T, n_layers = mamba_layers,
            d_state = 16, d_conv = 4, expand = 2,
            d_model_min = 64, d_model_max = 512, step = 8, top_k = tol_print
        )

        # Print the candidate parameter counts considered
        if rnn_candidates:
            print("RNN candidate params:", ", ".join(f"{c['params']:,}" for c in rnn_candidates))
        if mamba_candidates:
            print("Mamba candidate params:", ", ".join(f"{c['params']:,}" for c in mamba_candidates))

        # Validate across seeds to pick best candidate
        if rnn_candidates:
            best_rnn, rnn_report = validate_param_distribution(
                rnn_candidates, train_loader, val_loader,
                loss_function, seeds = seeds, val_epochs = val_epochs,
                lr = learning_rate, device = device,
                verbose = True
            )
        else:
            best_rnn, rnn_report = None, []
        if mamba_candidates:
            best_mamba, mamba_report = validate_param_distribution(
                mamba_candidates, train_loader, val_loader,
                loss_function, seeds = seeds, val_epochs = val_epochs,
                lr = learning_rate, device = device,
                verbose = True
            )
        else:
            best_mamba, mamba_report = None, []
        # Train full and test for the selected specs
        for label, best_spec in [("rnn", best_rnn), ("mamba", best_mamba)]:
            if best_spec is None:
                continue
            print(f"\n>>> Selected {label.upper()} for budget {T:,}: {best_spec.get('label')} "
                  f"(params={best_spec['params']:,})")
            model = best_spec["factory"]().to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            t0 = time.time()
            train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs)
            t1 = time.time()
            test_loss = evaluation_function(model, test_loader, loss_function)

            results[f"{label}_{T//1000}k"] = {
                "chosen_label": best_spec.get("label"),
                "parameters": int(best_spec["params"]),
                "val_selection_epochs": int(val_epochs),
                "val_seeds": len(seeds),
                "training_time_sec": float(t1 - t0),
                "test_rmse": float(np.sqrt(test_loss)),
                "val_report": (rnn_report if label=="rnn" else mamba_report),
            }

    df = pd.DataFrame(results).T[[
        "chosen_label","parameters","val_selection_epochs","val_seeds",
        "training_time_sec","test_rmse"
    ]]
    print("\n--- Param-Validated Final Results Summary:", dataset_name, "---")
    print(df.to_string(formatters={
        'parameters': '{:,.0f}'.format,
        'training_time_sec': '{:.2f}'.format,
        'test_rmse': '{:.6f}'.format
    }))
    return df, results

