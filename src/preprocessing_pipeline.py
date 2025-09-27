import numpy as np
import pandas as pd
import pyarrow.feather as feather
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Function to clean the equities datasets and select features
def clean_and_select_features_equities(df: pd.DataFrame, redundant_features: list) -> pd.DataFrame:
    """
    Takes a raw equities dataframe (with date on YYYY-MM-DD format) and returns a cleaned feature set after handling look-ahead bias and multicollinearity.
    Args:
        df: The raw input dataframe
        redundant_features:  list of redundant column names to drop
    Returns:
        A dataframe containing only the cleaned and selected features.
    """

    # Sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Identify and shift "-F" columns to avoid look-ahead bias
    future_cols = [c for c in df.columns if '-F' in c]
    intl_indices = ['FTSE', 'GDAXI', 'FCHI', 'HSI', 'SSEC']
    col_to_shift = future_cols + [c for c in intl_indices if c in df.columns]
    print(f"\nShifting {len(col_to_shift)} columns by +1 day")

    for col in col_to_shift:
        df[col] = df[col].shift(1)

    # Define leaky feature columns to be dropped 
    leaky_momentum = ['mom', 'mom1', 'mom2', 'mom3']

     # Drop name of index (constant across all observations)
    name= ['Name']

    cols_to_drop = leaky_momentum + name + redundant_features
    # Combine all lists of columns to drop 
    cleaned_df = df.drop(columns=cols_to_drop, errors='ignore')

    print(f"Number of columns dropped: {len(set(cols_to_drop) & set(df.columns))}")

    return cleaned_df

# Function to pre-process equities data
def preprocess_data_equities(feather_filepath: str, train_end: str = '2019-12-31', val_end: str = '2021-12-31'):
    """
    Loads data with date in YYYY-MM-DD format from an arrow file and performs a full pre-processing pipeline:
    1. Handles missing values using forward fill (no backward fill to avoid look-ahead bias);
    2. Drop any leading rows that remain NaNs after forward fill;
    3. Splits data chronologically into training, validation, and test sets;
    4. Normalise the data using StandardScaler fitted on the training set.
    Args:
        feather_filepath (str): The path to the Arrow file to be processed.
        train_end (str): The end date for the training set
        val_end (str): The end date for the validation set
    Returns:
        tuple: (train_df, val_df, test_df, scaler)
    """
    print(f"Beginning pre-processing of {feather_filepath}:")

    # Step 1: Load and sort the data
    df = feather.read_feather(feather_filepath)
    date_col = "Date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    print(f"Loaded data from {feather_filepath} with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Step 2: Engineer target column: log-returns
    price_col = "Price" if "Price" in df.columns else "close"
    df["_log_price"] = np.log(df[price_col])
    df["ret_t"] = df["_log_price"].diff()
    df["y_next"]     = df["_log_price"].shift(-1) - df["_log_price"]
    df = df.dropna(subset=["y_next"]).reset_index(drop=True)

    # Step 3: Causal imputation (forward-fill only) and drop leading incomplete rows
    # Forward-fill all columns except data, target and log-price columns
    protect_cols = {date_col, "y_next", "_log_price"}
    ffill_cols = [c for c in df.columns if c not in protect_cols]
    df.loc[:, ffill_cols] = df[ffill_cols].ffill()

    # Build numeric feature set excluding target and log_price
    numeric_feature_cols = (
        df.select_dtypes(include=np.number).columns.difference(["y_next", "_log_price"])
    )
    # Drop columns that are entirely NaN (prevents false "no complete rows")
    all_nan_cols = [c for c in numeric_feature_cols if df[c].isna().all()]
    if all_nan_cols:
        print(f"Dropping {len(all_nan_cols)} all-NaN columns (e.g., {all_nan_cols[:5]})")
        df = df.drop(columns=all_nan_cols)
        numeric_feature_cols = [c for c in numeric_feature_cols if c not in all_nan_cols]

    if len(numeric_feature_cols) == 0:
        raise ValueError("No numeric feature columns remain after exclusions.")

    complete_mask = df[numeric_feature_cols].notna().all(axis=1)
    if not complete_mask.any():
        na_counts = df[numeric_feature_cols].isna().sum().sort_values(ascending=False).head(10)
        raise ValueError(f"After forward-fill, no rows have complete numeric features. Top NA columns:\n{na_counts}")
    first_complete_idx = complete_mask.idxmax()
    if first_complete_idx > 0:
        print(f"Dropping {first_complete_idx} leading rows with unresolved NaNs.")
    df = df.loc[first_complete_idx:].reset_index(drop=True)
    
    print("Missing values handled.")

    # Step 4: Chronological split using the provided end dates
    # Random splitting ignores the temporal order of the data, which is crucial for time series.
    # Here we split the data chronologically into training, validation, and test sets.
    train_end_date = pd.Timestamp(train_end)
    val_end_date = pd.Timestamp(val_end)
    train_df = df[df[date_col] <= train_end_date].copy()
    val_df = df[(df[date_col] > train_end_date) & (df[date_col] <= val_end_date)].copy()
    test_df = df[df[date_col] > val_end_date].copy()
    print("Split data chronologically:")
    print(f"Training set: {train_df.shape[0]} rows (<= {train_end_date.date()})")
    print(f"Validation set: {val_df.shape[0]} rows (until {val_end_date.date()})")
    print(f"Test set: {test_df.shape[0]} rows (after {val_end_date.date()})")

    # Step 5: Normalisation
    # Exclude target and log_price from scaling
    numeric_cols = df.select_dtypes(include=np.number).columns.difference(["y_next", "_log_price"])
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    print("Scaler fitted on training data.")
    # Transform the training, validation, and test sets
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    return train_df, val_df, test_df, scaler

def preprocess_data_crypto(feather_filepath: str, train_end: str = '2021-08-13 23:59:59', val_end: str = '2021-11-07 23:59:59'):
    """
    Loads crypto data with UTC datetime from an arrow file and performs a full pre-processing pipeline:
    1. Handles missing values using forward fill (no backward fill to avoid look-ahead bias);
    2. Drop any leading rows that remain NaNs after forward fill;
    3. Splits data chronologically into training, validation, and test sets;
    4. Normalise the data using StandardScaler fitted on the training set.
    Args:
        feather_filepath (str): The path to the Arrow file to be processed.
        train_end (str): The end date for the training set
        val_end (str): The end date for the validation set
    Returns:
        tuple: (train_df, val_df, test_df, scaler)
    """
    print(f"Beginning pre-processing of {feather_filepath}:")

    # Step 1: Load and sort the data
    df = feather.read_feather(feather_filepath)
    date_col = "date" if "date" in df.columns else "Date"

    # Handle UTC datetime - convert to timezone-naive if needed
    df[date_col] = pd.to_datetime(df[date_col])
    if df[date_col].dt.tz is not None:
        print(f"Converting timezone-aware datetime ({df[date_col].dt.tz}) to UTC timezone-naive")
        df[date_col] = df[date_col].dt.tz_convert('UTC').dt.tz_localize(None)
    
    df = df.sort_values(date_col).reset_index(drop=True)
    print(f"Loaded data from {feather_filepath} with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Step 2: Engineer target column: log-returns
    price_col = "close"
    df["_log_price"] = np.log(df[price_col])
    df["ret_t"] = df["_log_price"].diff()
    df["y_next"] = df["_log_price"].shift(-1) - df["_log_price"]
    df = df.dropna(subset=["y_next"]).reset_index(drop=True)

    # Step 3: Causal imputation (forward-fill only) and drop leading incomplete rows
    # Forward-fill all columns except date, target and log-price columns
    protect_cols = {date_col, "y_next", "_log_price"}
    ffill_cols = [c for c in df.columns if c not in protect_cols]
    df.loc[:, ffill_cols] = df[ffill_cols].ffill()

    # Build numeric feature set excluding target and log_price
    numeric_feature_cols = (
        df.select_dtypes(include=np.number).columns.difference(["y_next", "_log_price"])
    )
    # Drop columns that are entirely NaN (prevents false "no complete rows")
    all_nan_cols = [c for c in numeric_feature_cols if df[c].isna().all()]
    if all_nan_cols:
        print(f"Dropping {len(all_nan_cols)} all-NaN columns (e.g., {all_nan_cols[:5]})")
        df = df.drop(columns=all_nan_cols)
        numeric_feature_cols = [c for c in numeric_feature_cols if c not in all_nan_cols]

    if len(numeric_feature_cols) == 0:
        raise ValueError("No numeric feature columns remain after exclusions.")

    complete_mask = df[numeric_feature_cols].notna().all(axis=1)
    if not complete_mask.any():
        na_counts = df[numeric_feature_cols].isna().sum().sort_values(ascending=False).head(10)
        raise ValueError(f"After forward-fill, no rows have complete numeric features. Top NA columns:\n{na_counts}")
    first_complete_idx = complete_mask.idxmax()
    if first_complete_idx > 0:
        print(f"Dropping {first_complete_idx} leading rows with unresolved NaNs.")
    df = df.loc[first_complete_idx:].reset_index(drop=True)
    
    print("Missing values handled.")

    # Step 4: Chronological split using the provided end dates
    # Handle timezone-naive timestamps for comparison
    train_end_date = pd.Timestamp(train_end)
    val_end_date = pd.Timestamp(val_end)
    
    # Ensure comparison timestamps are timezone-naive
    if train_end_date.tz is not None:
        train_end_date = train_end_date.tz_localize(None)
    if val_end_date.tz is not None:
        val_end_date = val_end_date.tz_localize(None)
    
    train_df = df[df[date_col] <= train_end_date].copy()
    val_df = df[(df[date_col] > train_end_date) & (df[date_col] <= val_end_date)].copy()
    test_df = df[df[date_col] > val_end_date].copy()
    
    print("Split data chronologically:")
    print(f"Training set: {train_df.shape[0]} rows (<= {train_end_date.date()})")
    print(f"Validation set: {val_df.shape[0]} rows (until {val_end_date.date()})")
    print(f"Test set: {test_df.shape[0]} rows (after {val_end_date.date()})")

    # Step 5: Normalisation
    # Exclude target and log_price from scaling
    numeric_cols = df.select_dtypes(include=np.number).columns.difference(["y_next", "_log_price"])
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    print("Scaler fitted on training data.")
    # Transform the training, validation, and test sets
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    return train_df, val_df, test_df, scaler