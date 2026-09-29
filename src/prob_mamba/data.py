"""Leakage-aware preprocessing for financial time-series data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TimeSplit:
    """Chronological train, validation, and test data frames."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def chronological(self, origin_timestamp_column: str = "origin_timestamp") -> pd.DataFrame:
        """Return all partitions in time order for context-aware window construction."""
        combined = pd.concat(
            [self.train, self.validation, self.test], axis=0, ignore_index=True
        )
        if origin_timestamp_column not in combined:
            raise KeyError(f"missing origin timestamp column: {origin_timestamp_column}")
        if combined[origin_timestamp_column].duplicated().any():
            raise ValueError("origin timestamps must be unique across partitions")
        return combined.sort_values(origin_timestamp_column).reset_index(drop=True)


def clean_equity_features(
    frame: pd.DataFrame,
    redundant_features: Iterable[str] = (),
    date_column: str = "Date",
    international_indices: Iterable[str] = ("FTSE", "GDAXI", "FCHI", "HSI", "SSEC"),
) -> pd.DataFrame:
    """Sort equities data, shift known look-ahead columns, and drop redundant fields."""
    if date_column not in frame:
        raise KeyError(f"missing date column: {date_column}")
    cleaned = frame.copy()
    cleaned[date_column] = pd.to_datetime(cleaned[date_column], errors="raise")
    cleaned = cleaned.sort_values(date_column).reset_index(drop=True)
    shifted = [column for column in cleaned if "-F" in column]
    shifted.extend(column for column in international_indices if column in cleaned)
    if shifted:
        cleaned.loc[:, sorted(set(shifted))] = cleaned.loc[:, sorted(set(shifted))].shift(1)
    return cleaned.drop(
        columns=["mom", "mom1", "mom2", "mom3", "Name", *redundant_features],
        errors="ignore",
    )


def preprocess_frame(
    frame: pd.DataFrame,
    train_end: str,
    validation_end: str,
    date_column: str,
    price_column: str,
    target_column: str = "y_next",
    origin_timestamp_column: str = "origin_timestamp",
    target_timestamp_column: str = "target_timestamp",
    label_available_timestamp_column: str = "label_available_timestamp",
) -> tuple[TimeSplit, StandardScaler]:
    """Create timestamped next-return targets and split by when labels become known.

    A row is a forecast made at ``origin_timestamp`` for ``target_timestamp``.  Its
    return label only becomes usable at ``label_available_timestamp`` (the target
    close for this one-step task).  Partitions are therefore assigned by target
    timestamp, preventing the last label of one partition from using a price in
    the following partition.
    """
    if date_column not in frame or price_column not in frame:
        raise KeyError(f"expected {date_column!r} and {price_column!r} columns")
    prepared = frame.copy()
    prepared[date_column] = pd.to_datetime(prepared[date_column], errors="raise")
    if prepared[date_column].isna().any():
        raise ValueError(f"{date_column!r} must not contain missing timestamps")
    if prepared[date_column].dt.tz is not None:
        prepared[date_column] = prepared[date_column].dt.tz_convert("UTC").dt.tz_localize(None)
    prepared = prepared.sort_values(date_column).reset_index(drop=True)
    if prepared[date_column].duplicated().any():
        raise ValueError(f"{date_column!r} must contain unique timestamps")
    prepared[price_column] = pd.to_numeric(prepared[price_column], errors="raise")
    if not np.isfinite(prepared[price_column]).all() or (prepared[price_column] <= 0).any():
        raise ValueError(f"{price_column!r} must contain finite, strictly positive prices")
    log_price = np.log(prepared[price_column])
    prepared[origin_timestamp_column] = prepared[date_column]
    prepared[target_timestamp_column] = prepared[date_column].shift(-1)
    prepared[label_available_timestamp_column] = prepared[target_timestamp_column]
    prepared["ret_t"] = log_price.diff()
    prepared[target_column] = log_price.shift(-1) - log_price
    prepared = prepared.dropna(
        subset=[target_column, target_timestamp_column, label_available_timestamp_column]
    ).reset_index(drop=True)

    protected = {
        date_column,
        origin_timestamp_column,
        target_timestamp_column,
        label_available_timestamp_column,
        target_column,
    }
    imputed_columns = [column for column in prepared if column not in protected]
    prepared.loc[:, imputed_columns] = prepared.loc[:, imputed_columns].ffill()
    feature_columns = prepared.select_dtypes(include=np.number).columns.difference([target_column])
    empty_columns = [column for column in feature_columns if prepared[column].isna().all()]
    prepared = prepared.drop(columns=empty_columns)
    feature_columns = feature_columns.drop(empty_columns)
    if feature_columns.empty:
        raise ValueError("no numeric feature columns remain after preprocessing")
    complete = prepared.loc[:, feature_columns].notna().all(axis=1)
    if not complete.any():
        raise ValueError("causal imputation left no complete observations")
    prepared = prepared.loc[complete.idxmax() :].reset_index(drop=True)

    train_boundary = pd.to_datetime(train_end, utc=True).tz_localize(None)
    validation_boundary = pd.to_datetime(validation_end, utc=True).tz_localize(None)
    if train_boundary >= validation_boundary:
        raise ValueError("train_end must be before validation_end")
    train = prepared.loc[prepared[target_timestamp_column] <= train_boundary].copy()
    validation = prepared.loc[
        (prepared[target_timestamp_column] > train_boundary)
        & (prepared[target_timestamp_column] <= validation_boundary)
    ].copy()
    test = prepared.loc[prepared[target_timestamp_column] > validation_boundary].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("chronological split produced an empty partition")

    feature_types = {column: "float64" for column in feature_columns}
    train = train.astype(feature_types)
    validation = validation.astype(feature_types)
    test = test.astype(feature_types)
    scaler = StandardScaler().fit(train.loc[:, feature_columns])
    for partition in (train, validation, test):
        partition.loc[:, feature_columns] = scaler.transform(partition.loc[:, feature_columns])
    if (train[label_available_timestamp_column] > train_boundary).any():
        raise AssertionError("training contains a label unavailable at the training cutoff")
    if (validation[label_available_timestamp_column] > validation_boundary).any():
        raise AssertionError("validation contains a label unavailable at the validation cutoff")
    return TimeSplit(train=train, validation=validation, test=test), scaler


def preprocess_feather(
    path: str | Path,
    train_end: str,
    validation_end: str,
    date_column: str,
    price_column: str,
) -> tuple[TimeSplit, StandardScaler]:
    """Read a Feather data set and apply :func:`preprocess_frame`."""
    return preprocess_frame(
        pd.read_feather(path), train_end, validation_end, date_column, price_column
    )
