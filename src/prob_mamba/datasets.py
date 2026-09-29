"""Time-series sequence construction and DataLoader factories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class CausalWindows:
    """Windows and timestamps for an auditable one-step forecasting sample."""

    inputs: np.ndarray
    labels: np.ndarray
    target_sequences: np.ndarray
    observed_mask: np.ndarray
    origin_timestamps: np.ndarray
    target_timestamps: np.ndarray
    label_available_timestamps: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def create_causal_windows(
    frame: pd.DataFrame,
    sequence_length: int,
    feature_columns: Sequence[str],
    target_column: str = "y_next",
    origin_timestamp_column: str = "origin_timestamp",
    target_timestamp_column: str = "target_timestamp",
    label_available_timestamp_column: str = "label_available_timestamp",
    score_start: str | pd.Timestamp | None = None,
    score_end: str | pd.Timestamp | None = None,
    score_target_timestamps: Sequence[object] | None = None,
    require_all: bool = True,
) -> CausalWindows:
    """Build windows from full history while selecting rows by target timestamp.

    The caller should pass the concatenated chronological train/validation/test
    frame.  Selection affects only which window ends are scored, so validation
    and test windows retain context from earlier partitions.  Historical labels
    inside a window must be available by its forecast origin; the final label
    must still be in the future.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if target_column in feature_columns:
        raise ValueError("target_column cannot also be an input feature")
    required = {
        *feature_columns,
        target_column,
        origin_timestamp_column,
        target_timestamp_column,
        label_available_timestamp_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing columns: {missing}")
    if score_target_timestamps is not None and (score_start is not None or score_end is not None):
        raise ValueError("choose explicit score_target_timestamps or score_start/score_end")

    ordered = frame.copy()
    for column in (
        origin_timestamp_column,
        target_timestamp_column,
        label_available_timestamp_column,
    ):
        ordered[column] = pd.to_datetime(ordered[column], errors="raise", utc=True).dt.tz_localize(None)
    ordered = ordered.sort_values(origin_timestamp_column).reset_index(drop=True)
    origins = ordered[origin_timestamp_column]
    targets_at = ordered[target_timestamp_column]
    available_at = ordered[label_available_timestamp_column]
    if origins.duplicated().any():
        raise ValueError("origin timestamps must be unique")
    if targets_at.duplicated().any():
        raise ValueError("target timestamps must be unique for one-step windows")
    if not (targets_at > origins).all():
        raise ValueError("every target timestamp must be after its forecast origin")
    if not available_at.equals(targets_at):
        if (available_at < targets_at).any():
            raise ValueError("a label cannot be available before its target timestamp")

    selected = pd.Series(True, index=ordered.index)
    requested: pd.DatetimeIndex | None = None
    if score_target_timestamps is not None:
        requested = pd.DatetimeIndex(pd.to_datetime(list(score_target_timestamps), utc=True)).tz_localize(None)
        if requested.has_duplicates:
            raise ValueError("score_target_timestamps must be unique")
        selected = targets_at.isin(requested)
        missing_targets = requested.difference(pd.DatetimeIndex(targets_at))
        if len(missing_targets):
            raise ValueError(f"requested target timestamps are absent: {missing_targets.tolist()}")
    else:
        if score_start is not None:
            selected &= targets_at >= pd.Timestamp(score_start).tz_localize(None)
        if score_end is not None:
            selected &= targets_at <= pd.Timestamp(score_end).tz_localize(None)

    selected_positions = np.flatnonzero(selected.to_numpy())
    if not len(selected_positions):
        raise ValueError("the requested scoring interval contains no targets")
    insufficient = selected_positions[selected_positions < sequence_length - 1]
    if len(insufficient) and require_all:
        missing_dates = targets_at.iloc[insufficient].tolist()
        raise ValueError(
            "insufficient pre-period history for requested targets: " f"{missing_dates}"
        )
    selected_positions = selected_positions[selected_positions >= sequence_length - 1]
    if not len(selected_positions):
        raise ValueError("no requested target has enough history for one complete window")

    feature_values = ordered.loc[:, feature_columns].to_numpy(dtype=np.float32)
    target_values = ordered[target_column].to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all() or not np.isfinite(target_values).all():
        raise ValueError("features and targets must be finite; refusing to drop rows implicitly")

    inputs: list[np.ndarray] = []
    target_sequences: list[np.ndarray] = []
    observed_masks: list[np.ndarray] = []
    for end in selected_positions:
        start = end - sequence_length + 1
        observed = (available_at.iloc[start : end + 1] <= origins.iloc[end]).to_numpy()
        if not observed[:-1].all():
            raise ValueError("a window uses a historical label unavailable at its forecast origin")
        if observed[-1]:
            raise ValueError("the forecast target is already available at its origin")
        inputs.append(feature_values[start : end + 1])
        target_sequences.append(target_values[start : end + 1])
        observed_masks.append(observed)

    return CausalWindows(
        inputs=np.stack(inputs),
        labels=target_values[selected_positions],
        target_sequences=np.stack(target_sequences).astype(np.float32)[..., None],
        observed_mask=np.stack(observed_masks),
        origin_timestamps=origins.iloc[selected_positions].to_numpy(dtype="datetime64[ns]"),
        target_timestamps=targets_at.iloc[selected_positions].to_numpy(dtype="datetime64[ns]"),
        label_available_timestamps=available_at.iloc[selected_positions].to_numpy(
            dtype="datetime64[ns]"
        ),
    )


def create_sequences(
    frame: pd.DataFrame,
    sequence_length: int,
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping input windows and their final-step targets."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if target_column in feature_columns:
        raise ValueError("target_column cannot also be an input feature")
    if len(frame) < sequence_length:
        raise ValueError("data is shorter than sequence_length")

    features = frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
    targets = frame.loc[:, target_column].to_numpy(dtype=np.float32)
    window_count = len(frame) - sequence_length + 1
    inputs = np.stack([features[index : index + sequence_length] for index in range(window_count)])
    labels = targets[sequence_length - 1 :]
    return inputs, labels


def create_seq2seq_targets(
    frame: pd.DataFrame, sequence_length: int, target_column: str = "y_next"
) -> np.ndarray:
    """Create aligned target windows with shape ``(samples, steps, 1)``."""
    if len(frame) < sequence_length:
        raise ValueError("data is shorter than sequence_length")
    target = frame.loc[:, target_column].to_numpy(dtype=np.float32)
    windows = [target[index : index + sequence_length] for index in range(len(target) - sequence_length + 1)]
    return np.stack(windows)[..., None]


def make_loaders(
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    batch_size: int = 64,
    sequence_targets: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test loaders with explicit shuffle policy."""
    def as_loader(split: tuple[np.ndarray, np.ndarray], shuffle: bool) -> DataLoader:
        features, targets = split
        target_tensor = torch.as_tensor(targets, dtype=torch.float32)
        if not sequence_targets:
            target_tensor = target_tensor.reshape(-1, 1)
        return DataLoader(
            TensorDataset(torch.as_tensor(features, dtype=torch.float32), target_tensor),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return as_loader(train, True), as_loader(validation, False), as_loader(test, False)
