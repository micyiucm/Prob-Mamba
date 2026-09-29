"""Deprecated compatibility layer; import from :mod:`prob_mamba.data` instead."""

import pandas as pd

from prob_mamba.data import clean_equity_features, preprocess_feather, preprocess_frame


def clean_and_select_features_equities(df, redundant_features):
    return clean_equity_features(df, redundant_features)


def preprocess_data_equities(feather_filepath, train_end="2019-12-31", val_end="2021-12-31"):
    split, scaler = preprocess_feather(feather_filepath, train_end, val_end, "Date", "Price")
    return split.train, split.validation, split.test, scaler


def preprocess_data_crypto(
    feather_filepath, train_end="2021-08-13 23:59:59", val_end="2021-11-07 23:59:59"
):
    frame = pd.read_feather(feather_filepath)
    date_column = "date" if "date" in frame else "Date"
    split, scaler = preprocess_frame(frame, train_end, val_end, date_column, "close")
    return split.train, split.validation, split.test, scaler
