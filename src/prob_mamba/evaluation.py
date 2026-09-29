"""Timestamped prediction tables and comparable one-step scoring."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .datasets import CausalWindows


PREDICTION_COLUMNS = (
    "dataset",
    "model",
    "split",
    "seed",
    "horizon",
    "origin_timestamp",
    "target_timestamp",
    "y_true",
    "mean",
    "variance",
)


def predict_causal_windows(
    model,
    windows: "CausalWindows",
    *,
    dataset: str,
    model_name: str,
    seed: int | None = None,
    split: str = "test",
    batch_size: int = 64,
    device: str = "cpu",
) -> pd.DataFrame:
    """Predict scalar window endpoints with the unknown final target withheld.

    Each window starts from the model prior; earlier outcomes are assimilated.
    Timestamps and unrounded targets come from the same CausalWindows object.
    """
    import torch

    if batch_size <= 0 or not len(windows):
        raise ValueError("batch_size and number of windows must be positive")
    if windows.observed_mask[:, -1].any():
        raise ValueError("forecast endpoints must be marked unobserved")
    model.to(device).eval()
    means, variances = [], []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            stop = start + batch_size
            inputs = torch.as_tensor(windows.inputs[start:stop], device=device)
            targets = torch.as_tensor(windows.target_sequences[start:stop], device=device).clone()
            observed = torch.as_tensor(windows.observed_mask[start:stop], device=device)
            targets[~observed] = float("nan")
            output = model(inputs, targets, observed_mask=observed)
            if output["y_mean"].size(-1) != 1:
                raise ValueError("timestamped prediction tables currently support scalar targets")
            means.extend(output["y_mean"][:, -1, 0].cpu().tolist())
            variances.extend(output["y_var_diag"][:, -1, 0].cpu().tolist())
    return make_prediction_frame(
        dataset=dataset, model=model_name, seed=seed, split=split,
        origin_timestamps=windows.origin_timestamps,
        target_timestamps=windows.target_timestamps,
        y_true=windows.labels, mean=means, variance=variances,
    )


def make_prediction_frame(
    *,
    dataset: str,
    model: str,
    origin_timestamps: Sequence[object],
    target_timestamps: Sequence[object],
    y_true: Sequence[float],
    mean: Sequence[float],
    variance: Sequence[float] | None = None,
    split: str = "test",
    seed: int | None = None,
    horizon: int = 1,
) -> pd.DataFrame:
    """Create a strict prediction table without truncating or joining observations."""
    arrays = {
        "origin_timestamp": list(origin_timestamps),
        "target_timestamp": list(target_timestamps),
        "y_true": list(y_true),
        "mean": list(mean),
    }
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"prediction fields have unequal lengths: {lengths}")
    count = next(iter(lengths.values()))
    if variance is None:
        variance_values = np.full(count, np.nan)
    else:
        variance_values = list(variance)
        if len(variance_values) != count:
            raise ValueError(
                f"variance has length {len(variance_values)} but predictions have length {count}"
            )
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "model": model,
            "split": split,
            "seed": seed,
            "horizon": horizon,
            **arrays,
            "variance": variance_values,
        }
    )
    return validate_prediction_frame(frame)


def validate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the common one-step prediction-table contract."""
    missing = sorted(set(PREDICTION_COLUMNS).difference(frame.columns))
    if missing:
        raise KeyError(f"prediction table is missing columns: {missing}")
    if frame.empty:
        raise ValueError("prediction table must contain at least one row")
    result = frame.loc[:, PREDICTION_COLUMNS].copy()
    for column in ("origin_timestamp", "target_timestamp"):
        result[column] = pd.to_datetime(result[column], errors="raise", utc=True).dt.tz_localize(None)
    if not (result["target_timestamp"] > result["origin_timestamp"]).all():
        raise ValueError("target_timestamp must be after origin_timestamp")
    for column in ("y_true", "mean", "variance"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not np.isfinite(result[["y_true", "mean"]].to_numpy()).all():
        raise ValueError("y_true and mean must be finite")
    has_variance = result["variance"].notna()
    if has_variance.any() and not has_variance.all():
        raise ValueError("variance must be supplied for every row of a probabilistic forecast")
    if has_variance.all():
        values = result["variance"].to_numpy()
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError("variance forecasts must be finite and strictly positive")
    result["horizon"] = pd.to_numeric(result["horizon"], errors="raise")
    if not (result["horizon"] == 1).all():
        raise ValueError("this evaluation contract currently supports horizon=1 only")
    result["horizon"] = result["horizon"].astype(int)
    if result[["dataset", "model", "split"]].isna().any().any():
        raise ValueError("dataset, model, and split must be present")
    key = ["dataset", "model", "split", "seed", "horizon", "target_timestamp"]
    if result.duplicated(key).any():
        raise ValueError(f"prediction rows must be unique on {key}")
    return result.sort_values(["target_timestamp", "model"], kind="stable").reset_index(drop=True)


def assert_common_scoring_support(frames: Iterable[pd.DataFrame]) -> list[pd.DataFrame]:
    """Require models to contain the same targets and observations in the same order."""
    validated = [validate_prediction_frame(frame) for frame in frames]
    if len(validated) < 2:
        return validated
    reference = validated[0].sort_values("target_timestamp").reset_index(drop=True)
    reference_times = reference["target_timestamp"].to_numpy()
    reference_values = reference["y_true"].to_numpy()
    for candidate in validated[1:]:
        ordered = candidate.sort_values("target_timestamp").reset_index(drop=True)
        if not np.array_equal(ordered["target_timestamp"].to_numpy(), reference_times):
            raise ValueError("models do not have identical target-timestamp support")
        for column in ("dataset", "split", "horizon", "origin_timestamp"):
            if not np.array_equal(ordered[column].to_numpy(), reference[column].to_numpy()):
                raise ValueError(f"models do not have identical {column} support")
        if not np.allclose(ordered["y_true"].to_numpy(), reference_values, rtol=0.0, atol=1e-12):
            raise ValueError("models disagree on y_true for a shared target timestamp")
    return validated


def score_prediction_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    """Score one model/seed table, including simple Gaussian calibration diagnostics."""
    evaluated = validate_prediction_frame(frame)
    if evaluated[["dataset", "model", "split", "seed"]].drop_duplicates().shape[0] != 1:
        raise ValueError("score one dataset/model/split/seed at a time")
    residual = evaluated["y_true"].to_numpy() - evaluated["mean"].to_numpy()
    scores: dict[str, float | int] = {
        "n_obs": len(evaluated),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
    }
    if evaluated["variance"].isna().all():
        return scores
    variance = evaluated["variance"].to_numpy()
    qlike_terms = np.square(residual) / variance + np.log(variance)
    zero_mean_qlike_terms = np.square(evaluated["y_true"].to_numpy()) / variance + np.log(
        variance
    )
    standardized = residual / np.sqrt(variance)
    pit = np.asarray(
        [0.5 * (1.0 + math.erf(value / math.sqrt(2.0))) for value in standardized]
    )
    scores.update(
        {
            "qlike_residual": float(np.mean(qlike_terms)),
            "qlike_zero_mean": float(np.mean(zero_mean_qlike_terms)),
            "gaussian_nll": float(0.5 * (np.mean(qlike_terms) + math.log(2.0 * math.pi))),
            "standardized_innovation_mean": float(np.mean(standardized)),
            "standardized_innovation_variance": float(np.var(standardized)),
            "pit_mean": float(np.mean(pit)),
            "pit_variance": float(np.var(pit)),
        }
    )
    normal = NormalDist()
    for level in (0.50, 0.80, 0.95):
        critical = normal.inv_cdf((1.0 + level) / 2.0)
        scores[f"coverage_{int(level * 100)}"] = float(
            np.mean(np.abs(standardized) <= critical)
        )
        scores[f"mean_interval_width_{int(level * 100)}"] = float(
            2.0 * critical * np.mean(np.sqrt(variance))
        )
    return scores
