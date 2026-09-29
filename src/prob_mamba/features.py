"""Feature selection helpers that avoid temporal leakage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VIFSelection:
    """Result of VIF pruning fitted exclusively on the training partition."""

    retained_features: list[str]
    dropped_features: list[tuple[str, float]]
    vif: pd.DataFrame


def momentum_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric momentum columns that should not be used as predictors."""
    return [
        column
        for column in ("mom", "mom1", "mom2", "mom3")
        if column in frame and pd.api.types.is_numeric_dtype(frame[column])
    ]


def split_by_dates(
    frame: pd.DataFrame, date_column: str, train_end: str, validation_end: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return chronological partition masks with inclusive train/validation boundaries."""
    dates = pd.to_datetime(frame[date_column], errors="raise")
    train_boundary, validation_boundary = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    return (
        dates <= train_boundary,
        (dates > train_boundary) & (dates <= validation_boundary),
        dates > validation_boundary,
    )


def shift_leak_prone_columns(
    frame: pd.DataFrame, extra_columns: Iterable[str] = ()
) -> pd.DataFrame:
    """Shift futures and supplied columns by one timestep without mutating ``frame``."""
    shifted = frame.copy()
    columns = {column for column in shifted if "-F" in column}
    columns.update(column for column in extra_columns if column in shifted)
    if columns:
        ordered = sorted(columns)
        shifted.loc[:, ordered] = shifted.loc[:, ordered].shift(1)
    return shifted


def select_vif_features(
    frame: pd.DataFrame,
    date_column: str = "Date",
    include_price: bool = False,
    excluded_columns: Iterable[str] = ("Date", "Price", "Name", "Weekday"),
    drop_momentum: bool = True,
    manual_drop: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """Choose numeric candidate features using explicit configuration only."""
    excluded = set(excluded_columns) | {date_column} | set(manual_drop)
    if not include_price:
        excluded.add("Price")
    if drop_momentum:
        excluded.update(momentum_columns(frame))
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    selected = [column for column in numeric if column not in excluded]
    return selected, sorted(set(numeric) - set(selected))


def compute_vif(features: pd.DataFrame) -> pd.DataFrame:
    """Calculate variance inflation factors, importing statsmodels only when needed."""
    if features.shape[1] < 2:
        return pd.DataFrame({"feature": features.columns, "VIF": np.nan})
    try:
        import statsmodels.api as statsmodels
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError as error:
        raise ImportError("VIF selection requires `pip install -e '.[baselines]'`.") from error
    usable = features.loc[:, features.nunique() > 1]
    if usable.shape[1] < 2:
        return pd.DataFrame({"feature": usable.columns, "VIF": np.nan})
    with_constant = statsmodels.add_constant(usable, has_constant="add")
    values = []
    for index, _ in enumerate(usable.columns):
        try:
            values.append(variance_inflation_factor(with_constant.values, index + 1))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            values.append(np.nan)
    return pd.DataFrame({"feature": usable.columns, "VIF": values}).sort_values(
        "VIF", ascending=False, key=lambda series: series.fillna(-np.inf)
    )


def vif_iterative_prune(
    features: pd.DataFrame, threshold: float = 10.0
) -> tuple[pd.DataFrame, list[tuple[str, float]], pd.DataFrame]:
    """Fit VIF pruning on one training feature frame."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    reduced = features.copy()
    dropped = [(column, float("inf")) for column in reduced.columns[reduced.nunique() <= 1]]
    reduced = reduced.drop(columns=[column for column, _ in dropped])
    while reduced.shape[1] >= 2:
        vif = compute_vif(reduced)
        if vif.empty or vif["VIF"].isna().all() or vif.iloc[0]["VIF"] <= threshold:
            break
        row = vif.iloc[0]
        reduced = reduced.drop(columns=[row["feature"]])
        dropped.append((str(row["feature"]), float(row["VIF"])))
    final_vif = compute_vif(reduced) if reduced.shape[1] >= 2 else pd.DataFrame(
        {"feature": reduced.columns, "VIF": np.nan}
    )
    return reduced, dropped, final_vif
