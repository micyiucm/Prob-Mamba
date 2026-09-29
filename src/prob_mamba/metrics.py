"""Metrics used to evaluate point and probabilistic forecasts."""

from __future__ import annotations

import numpy as np


def quasi_likelihood(
    residuals: np.ndarray,
    variances: np.ndarray,
    *,
    min_variance: float | None = None,
) -> float:
    """Compute QLIKE with an optional caller-specified variance floor.

    Model-produced variances should already satisfy their scale-aware floor. The
    default only protects the logarithm at floating-point zero and does not impose
    an absolute floor in the units of a particular dataset.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    if residuals.shape != variances.shape:
        raise ValueError("residuals and variances must have the same shape")
    if not np.isfinite(residuals).all() or not np.isfinite(variances).all():
        raise ValueError("residuals and variances must be finite")
    if (variances < 0.0).any():
        raise ValueError("variances must be non-negative")
    if min_variance is None:
        min_variance = np.finfo(np.float64).tiny
    if not np.isfinite(min_variance) or min_variance <= 0.0:
        raise ValueError("min_variance must be finite and strictly positive")
    bounded_variances = np.maximum(variances, min_variance)
    return float(np.mean(np.square(residuals) / bounded_variances + np.log(bounded_variances)))
