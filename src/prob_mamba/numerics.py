"""Numerically stable primitives used by the LGSSM implementation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def softplus_pos(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return strictly positive values while retaining useful gradients."""
    return functional.softplus(values) + eps


def inverse_softplus(value: float) -> float:
    """Return the finite scalar whose softplus is ``value``.

    This is used to initialise positive scale parameters in the units supplied by
    the caller, rather than relying on a zero bias whose softplus is about 0.693.
    """
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("value must be finite and strictly positive")
    return value + math.log(-math.expm1(-value))


def gamma(values: torch.Tensor) -> torch.Tensor:
    """Evaluate ``expm1(z) / z`` accurately around zero."""
    tolerance = torch.finfo(values.dtype).eps**0.5
    large = values.abs() >= tolerance
    result = torch.empty_like(values)
    result[large] = torch.expm1(values[large]) / values[large]
    small_values = values[~large]
    result[~large] = 1.0 + 0.5 * small_values + small_values.square() / 6.0
    return result


def rho(values: torch.Tensor) -> torch.Tensor:
    """Evaluate ``expm1(2z) / (2z)`` accurately around zero."""
    tolerance = torch.finfo(values.dtype).eps**0.5
    large = values.abs() >= tolerance
    result = torch.empty_like(values)
    result[large] = torch.expm1(2.0 * values[large]) / (2.0 * values[large])
    small_values = values[~large]
    result[~large] = 1.0 + small_values + 2.0 * small_values.square() / 3.0
    return result


def safe_cholesky(
    matrix: torch.Tensor, jitter: float = 1e-6, max_tries: int = 5
) -> tuple[torch.Tensor, float]:
    """Compute a batched Cholesky decomposition with bounded diagonal jitter."""
    factor, info = torch.linalg.cholesky_ex(matrix, upper=False)
    if (info == 0).all():
        return factor, 0.0

    identity = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)
    current_jitter = jitter
    for _ in range(max_tries):
        factor, info = torch.linalg.cholesky_ex(
            matrix + current_jitter * identity, upper=False
        )
        if (info == 0).all():
            return factor, current_jitter
        current_jitter *= 10.0
    raise RuntimeError("Innovation covariance is not positive definite after jitter.")
