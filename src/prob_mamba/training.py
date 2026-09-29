"""Small, explicit training loops shared by notebook and script entry points."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve a device and reject explicit CUDA requests on CPU-only hosts."""
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def train_probabilistic_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float = 1e-3,
    device: str | torch.device | None = None,
    gradient_clip_norm: float | None = 1.0,
    restore_best: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_metadata: Mapping[str, object] | None = None,
    loss_scope: str = "last_step",
) -> dict[str, list[float] | float | int]:
    """Optimise endpoint NLL and retain the best validation state.

    By default only the final prediction of each window contributes to the loss;
    earlier targets condition the filter. This prevents training-context rows
    inside validation windows from contributing to checkpoint selection. Set
    ``loss_scope='sequence'`` explicitly for a reset-window composite objective.

    When ``checkpoint_path`` is supplied, the saved payload contains the best
    model and optimiser states, its zero-based epoch, validation NLL, and the
    caller-provided experiment metadata. The best model is restored by default.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if loss_scope not in {"last_step", "sequence"}:
        raise ValueError("loss_scope must be 'last_step' or 'sequence'")

    def loss_totals(output):
        if loss_scope == "last_step":
            if "log_likelihood" not in output:
                raise ValueError("endpoint training requires per-step log_likelihood")
            endpoint_nll = -output["log_likelihood"][:, -1]
            return endpoint_nll.sum(), endpoint_nll.numel()
        return output["nll_sum"], int(output["scored_steps"].detach().item())
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_nll": [], "validation_nll": []}
    best_validation_nll = float("inf")
    best_epoch = -1
    best_model_state = None
    best_optimizer_state = None
    for epoch in range(epochs):
        model.train()
        train_nll_sum, train_scored_steps = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(resolved_device), targets.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs, targets)
            if "nll" not in output or "nll_sum" not in output or "scored_steps" not in output:
                raise ValueError("model must return nll, nll_sum, and scored_steps during training")
            nll_sum, scored_steps = loss_totals(output)
            loss = nll_sum / scored_steps
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            train_nll_sum += nll_sum.detach().item()
            train_scored_steps += scored_steps
        if train_scored_steps == 0:
            raise ValueError("training loader contains no scored observations")
        history["train_nll"].append(train_nll_sum / train_scored_steps)

        model.eval()
        validation_nll_sum, validation_scored_steps = 0.0, 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                output = model(inputs.to(resolved_device), targets.to(resolved_device))
                if "nll_sum" not in output or "scored_steps" not in output:
                    raise ValueError("model must return likelihood totals during validation")
                nll_sum, scored_steps = loss_totals(output)
                validation_nll_sum += nll_sum.item()
                validation_scored_steps += scored_steps
        if validation_scored_steps == 0:
            raise ValueError("validation loader contains no scored observations")
        validation_nll = validation_nll_sum / validation_scored_steps
        history["validation_nll"].append(validation_nll)
        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    if best_model_state is None or best_optimizer_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    if checkpoint_path is not None:
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_model_state,
                "optimizer_state_dict": best_optimizer_state,
                "epoch": best_epoch,
                "validation_nll_per_scored_step": best_validation_nll,
                "loss_scope": loss_scope,
                "metadata": dict(checkpoint_metadata or {}),
            },
            destination,
        )
    if restore_best:
        model.load_state_dict(best_model_state)
    history["best_epoch"] = best_epoch
    history["best_validation_nll"] = best_validation_nll
    return history


def evaluate_last_step_probabilistic(
    model: nn.Module, loader: DataLoader, device: str | torch.device | None = None
) -> dict[str, float]:
    """Evaluate RMSE, QLIKE, and joint Gaussian NLL at window endpoints only."""
    from .metrics import quasi_likelihood

    resolved_device = resolve_device(device)
    model.to(resolved_device).eval()
    squared_error = 0.0
    residuals: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    nll_total, nll_count, count = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(resolved_device), targets.to(resolved_device)
            observed_mask = torch.ones(targets.shape[:2], dtype=torch.bool, device=resolved_device)
            observed_mask[:, -1] = False
            output = model(inputs, targets, observed_mask=observed_mask)
            predictions = output["y_mean"][:, -1, :]
            predicted_variances = output["y_var_diag"][:, -1, :]
            observed = targets[:, -1, :]
            squared_error += (predictions - observed).square().sum().item()
            residuals.append((observed - predictions).cpu().reshape(-1))
            variances.append(predicted_variances.cpu().reshape(-1))
            distribution = torch.distributions.MultivariateNormal(
                predictions, covariance_matrix=output["y_covariance"][:, -1]
            )
            nll_total -= distribution.log_prob(observed).sum().item()
            nll_count += observed.size(0)
            count += observed.numel()
    if count == 0:
        raise ValueError("evaluation loader contains no observations")
    residual_array = torch.cat(residuals).numpy()
    variance_array = torch.cat(variances).numpy()
    return {
        "rmse": (squared_error / count) ** 0.5,
        "qlike": quasi_likelihood(residual_array, variance_array),
        "nll": nll_total / nll_count,
    }
