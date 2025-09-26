import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress divide-by-zero warnings in VIF calculation
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

# Helper functions
def momentum_columns(df: pd.DataFrame):
    cols = [c for c in ['mom','mom1','mom2', 'mom3'] if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols

def split_by_dates(df: pd.DataFrame, date_col: str, train_end: str, val_end: str):
    """
    Splits the DataFrame into training, validation, and test sets based on date ranges.
    Args:
        df: DataFrame containing the dataset.
        date_col: Name of the date column to sort and split data.
        train_end: End date for the training set (inclusive).
        val_end: End date for the validation set (inclusive).
    """
    tr_end = pd.to_datetime(train_end)
    va_end = pd.to_datetime(val_end)
    tr_mask = df[date_col] <= tr_end
    va_mask = (df[date_col] > tr_end) & (df[date_col] <= va_end)
    te_mask = df[date_col] > va_end
    return tr_mask, va_mask, te_mask
def compute_vif(X:pd.DataFrame):
    """
    Computes Variance Inflation Factor (VIF) for each column in X.
    """
    cols = list(X.columns) # Convert features to list
    if X.shape[1] < 2:
        return pd.DataFrame({"feature": cols, "VIF": [np.nan]*len(cols)}) # VIF calculation requires at least 2 features
    # Drop constant columns to avoid zero-variance issues
    const_cols = X.columns[X.nunique()<=1].tolist()
    if const_cols:
        X = X.drop(columns=const_cols)
        cols = list(X.columns) # Update column list
        if X.shape[1] < 2:
            return pd.DataFrame({"feature": cols, "VIF": [np.nan]*len(cols)}) # VIF calculation requires at least 2 features
    Xc = sm.add_constant(X, has_constant='add') # Add constant column for intercept
    vifs = []
    for i, _ in enumerate(cols):
        try:
            vifs.append(variance_inflation_factor(Xc.values, i+1)) # i+1 to skip constant column at index 0
        except Exception as e:
            vifs.append(np.nan) # In case of error, append NaN
    return pd.DataFrame({"feature": cols, "VIF": vifs}).sort_values(
        "VIF", ascending=False, key = lambda s: s.fillna(-np.inf)) # Sort in descending order with NaNs at the bottom
def vif_iterative_prune(X:pd.DataFrame, threshold: float = 10.0):
    """
    Iteratively removes features with VIF above the specified threshold.
    Args:
        X: DataFrame containing the features.
        threshold: VIF threshold above which features are removed.
    Returns:
        X_reduced: DataFrame with reduced features after pruning.
        dropped_list: List of features that were dropped.
        final_vif_df: DataFrame of final VIF values for remaining features.
    """
    Xw = X.copy() # Work on a copy to avoid modifying original data
    # Remove exact constants up front
    const_cols = Xw.columns[Xw.nunique()<=1].tolist()
    dropped = []
    if const_cols:
        for c in const_cols:
            dropped.append((c, float('inf')))
        Xw = Xw.drop(columns=const_cols)
    while Xw.shape[1] >= 2: # Again, VIF requires at least 2 features
        vif_df = compute_vif(Xw)
        if vif_df.empty or vif_df["VIF"].isna().all():
            break # No valid VIFs to process
        max_row = vif_df.iloc[0]
        if pd.notna(max_row["VIF"]) and max_row["VIF"] > threshold:
            Xw = Xw.drop(columns=[max_row["feature"]]) # Drop the feature with highest VIF
            dropped.append((max_row["feature"], float(max_row["VIF"]))) # Record dropped feature
        else:
            break # All remaining features have acceptable VIF
    final_vif_df = compute_vif(Xw) if Xw.shape[1] >= 2 else pd.DataFrame({"feature": [], "VIF": []})
    return Xw, dropped, final_vif_df
def select_vif_features(
        df: pd.DataFrame,
        date_col: str = "Date",
        include_price: bool | None = None,
        feature_always_exclude: list | None = None,
        drop_momentum: bool | None = None,
        manual_drop: list | None = None
):
    """
    Selects features for VIF analysis, excluding specified features.
    Args:
        df: DataFrame containing the dataset.
        date_col: Name of the date column to sort data.
        include_price: If False, excludes the price column from features.
        feature_always_exclude: List of features to always exclude.
        drop_momentum: If True, excludes momentum features.
        manual_drop: List of additional features to manually exclude.
    """
    # If no input provided, use global variables
    if include_price is None:
        include_price = globals().get("Include_price", False)
    if feature_always_exclude is None:
        feature_always_exclude = globals().get("Feature_always_exclude", ["Date", "Price", "Name", "Weekday"])
    if drop_momentum is None:
        drop_momentum = globals().get("Drop_momentum", True)

    exclude = set(feature_always_exclude) | {date_col}
    if not include_price:
        exclude.add("Price")
    if drop_momentum:
        exclude |= set(momentum_columns(df))
    if manual_drop:
        exclude |= set(manual_drop)
    # Numeric columns
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    feature_cols = [c for c in numeric_cols if c.lower() not in exclude]
    excluded_cols = sorted(set(numeric_cols) - set(feature_cols))
    return feature_cols, excluded_cols

# Main pipeline
def run_vif(
        feather_path: str,
        train_end: str,
        val_end: str,
        vif_threshold: float = 10.0,
        date_col: str = "Date",
        include_price: bool | None = None,
        feature_always_exclude: list | None = None,
        drop_momentum: bool | None = None,
        manual_drop: list | None = None,
        save_artifacts: bool | None = None
):
    """
    VIF pipeline: load data, select features, split by dates,
    casual imputation (forward-fill), drop leading incomplete rows,
    split by dates, run VIF on trading set and apply results to validation and test sets.
    """
    # If no input provided, use global variables
    if save_artifacts is None:
        save_artifacts = globals().get("Save_artifacts", True)
    if include_price is None:
        include_price = globals().get("Include_price", False)
    if feature_always_exclude is None:
        feature_always_exclude = globals().get("Feature_always_exclude", ["Date", "Price", "Name", "Weekday"])
    if drop_momentum is None:
        drop_momentum = globals().get("Drop_momentum", True)

    # Load dataset
    if not os.path.isfile(feather_path):
        print(f"File not found: {feather_path}")
        return  None, None, None
    print(f"\n VIF analysis for dataset: {feather_path}")
    df = pd.read_feather(feather_path)

    # Ensure date column is datetime, drop duplicates and sort
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in {feather_path}")
    df[date_col] = pd.to_datetime(df[date_col], errors = "raise")
    df = df.drop_duplicates()
    df = df.sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True)

    # Build exclusion set
    exclude = set(feature_always_exclude) | {date_col}
    if not include_price:
        exclude.add("Price")
    if drop_momentum:
        exclude |= set(momentum_columns(df))
    if manual_drop:
        exclude |= set(manual_drop)

    # Numeric features only
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    feature_cols = [c for c in numeric_cols if c not in exclude]
    excluded_cols = sorted(set(numeric_cols) - set(feature_cols))

    if len(feature_cols) < 2:
        print("Not enough numeric features for VIF analysis after exclusions.")
        return feature_cols, pd.DataFrame({"feature": feature_cols, "VIF": np.nan}), []

    # Causal Imputation: forward-fill only
    # 1. Forward-fill features only
    df[feature_cols] = df[feature_cols].ffill()
    # 2. Drop leading rows with any NaNs in feature columns
    complete_mask = df[feature_cols].notna().all(axis=1) # Boolean mask for rows with all valid features
    if not complete_mask.any():
        print("No complete rows after forward-fill imputation.")
        return feature_cols, pd.DataFrame({"feature": feature_cols, "VIF": np.nan}), []
    first_complete_index = complete_mask.idxmax() # First index where all features are valid
    if isinstance(first_complete_index, (np.integer, int)):
        df = df.loc[first_complete_index:].reset_index(drop=True) # Drop leading incomplete rows
    else:
        first_pos = np.argmax(complete_mask.values) # Fallback if non-integer index is returned, edge case
        df = df.iloc[first_pos:].reset_index(drop=True) # Drop leading incomplete rows

    # Splitting into training, validation and test sets based on dates
    train_mask, val_mask, test_mask = split_by_dates(df, date_col, train_end, val_end)
    X_train = df.loc[train_mask, feature_cols].copy()
    X_val = df.loc[val_mask, feature_cols].copy()
    X_test = df.loc[test_mask, feature_cols].copy()

    # Ensure no NaNs remain after imputation
    if X_train.isna().any().any() or X_val.isna().any().any() or X_test.isna().any().any():
        X_train = X_train.dropna()
        X_val = X_val.dropna()
        X_test = X_test.dropna()
    
    # VIF on training set
    X_train_reduced, dropped_features, final_vif_df = vif_iterative_prune(X_train, threshold=vif_threshold)
    retained_features = list(X_train_reduced.columns)

    # Apply same feature selection to validation and test sets
    X_val_reduced = X_val[retained_features].copy()
    X_test_reduced = X_test[retained_features].copy()

    # Summary
    print(f"Excluded features before VIF: {excluded_cols}")
    print(f"Rows in training set after imputation: {X_train.shape[0]}")
    print(f"Rows in validation set after imputation: {X_val.shape[0]}")
    print(f"Rows in test set after imputation: {X_test.shape[0]}")
    print(f"Features retained after VIF pruning: {retained_features}")
    if dropped_features:
        print(f"Dropped features due to high VIF (> {vif_threshold}): {[f for f, v in dropped_features]}")