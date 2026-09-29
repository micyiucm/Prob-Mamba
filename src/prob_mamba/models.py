"""Neural and probabilistic forecasting models.

The Mamba dependency is deliberately imported only when a Mamba-backed model is
constructed.  The LGSSM head therefore remains usable for CPU-only analysis,
testing, and environments where the optional CUDA extension is unavailable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Optional

import torch
from torch import nn

from .numerics import gamma, inverse_softplus, rho, safe_cholesky, softplus_pos


def require_mamba_block():
    """Return the optional Mamba block or provide an actionable installation error."""
    try:
        from mamba_ssm import Mamba
    except ImportError as error:
        raise ImportError(
            "Mamba models require the optional dependency mamba-ssm and its GPU backend. "
            "Follow the README's "
            "Mamba setup instructions in a supported CUDA environment."
        ) from error
    return Mamba


class SimpleRNN(nn.Module):
    """Last-step regression baseline built on :class:`torch.nn.RNN`."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        initial_state = inputs.new_zeros(self.n_layers, inputs.size(0), self.hidden_dim)
        outputs, _ = self.rnn(inputs, initial_state)
        return self.output(outputs[:, -1, :])


class MambaModel(nn.Module):
    """Residual Mamba stack for deterministic last-step regression."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        output_dim: int = 1,
        n_layers: int = 1,
    ):
        super().__init__()
        mamba = require_mamba_block()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        )
        self.normalization = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.input_projection(inputs)
        for block in self.blocks:
            features = features + block(features)
        return self.output(self.normalization(features)[:, -1, :])


class FeatureNet(nn.Module):
    """Mamba feature extractor mapping ``(batch, steps, inputs)`` to features."""

    def __init__(
        self,
        d_in: int,
        d_feat: int,
        n_layers: int = 1,
        mamba_cfg: Optional[Mapping[str, int]] = None,
    ):
        super().__init__()
        mamba = require_mamba_block()
        config = mamba_cfg or {}
        self.input_projection = nn.Linear(d_in, d_feat)
        self.blocks = nn.ModuleList(
            mamba(
                d_model=d_feat,
                d_state=config.get("d_state", 16),
                d_conv=config.get("d_conv", 4),
                expand=config.get("expand", 2),
            )
            for _ in range(n_layers)
        )
        self.normalization = nn.LayerNorm(d_feat)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.input_projection(inputs)
        for block in self.blocks:
            features = features + block(features)
        return self.normalization(features)


class ProbMambaHead(nn.Module):
    """Input-conditioned linear-Gaussian state-space model with Kalman filtering."""

    def __init__(
        self,
        d_feat: int,
        d_y: int,
        n_state: int,
        eps_pos: float = 1e-6,
        *,
        target_scale: float = 1.0,
        initial_process_std_ratio: float = 0.1,
        initial_observation_variance_ratio: float = 1.0,
        initial_state_variance_ratio: float = 1.0,
        variance_floor_ratio: float = 1e-8,
    ):
        super().__init__()
        if d_feat <= 0 or d_y <= 0 or n_state <= 0:
            raise ValueError("d_feat, d_y, and n_state must all be positive")
        scale_arguments = {
            "target_scale": target_scale,
            "initial_process_std_ratio": initial_process_std_ratio,
            "initial_observation_variance_ratio": initial_observation_variance_ratio,
            "initial_state_variance_ratio": initial_state_variance_ratio,
            "variance_floor_ratio": variance_floor_ratio,
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in scale_arguments.values()):
            raise ValueError("target scale, initial scale ratios, and variance floor ratio must be positive")
        self.d_feat, self.d_y, self.n_state, self.eps_pos = d_feat, d_y, n_state, eps_pos
        self.target_scale = float(target_scale)
        self.variance_floor = float(target_scale**2 * variance_floor_ratio)
        self.process_std_floor = self.variance_floor**0.5
        self.a_raw = nn.Parameter(torch.randn(n_state) * 0.02)
        self.gate = nn.Linear(d_feat, 1)
        self.map_b = nn.Linear(d_feat, n_state * d_feat)
        self.map_c = nn.Linear(d_feat, d_y * n_state)
        self.map_sigma = nn.Linear(d_feat, n_state)
        self.map_r = nn.Linear(d_feat, d_y)
        initial_process_std = target_scale * initial_process_std_ratio
        initial_observation_variance = target_scale**2 * initial_observation_variance_ratio
        initial_state_variance = target_scale**2 * initial_state_variance_ratio
        self.p0_raw = nn.Parameter(
            torch.full((n_state,), inverse_softplus(initial_state_variance))
        )
        for layer in (self.gate, self.map_b, self.map_c):
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.map_sigma.bias, inverse_softplus(initial_process_std))
        nn.init.constant_(self.map_r.bias, inverse_softplus(initial_observation_variance))

        self.delta_min, self.delta_max = 1e-3, 1.0

    def initial_state(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create the initial filtering state in the feature tensor's device and dtype."""
        batch_size = features.size(0)
        mean = features.new_zeros(batch_size, self.n_state)
        covariance = torch.diag_embed(
            torch.nn.functional.softplus(self.p0_raw)
            .clamp_min(self.variance_floor)
            .expand(batch_size, -1)
        )
        return mean, covariance

    @staticmethod
    def predict_step(
        state_mean: torch.Tensor,
        state_covariance: torch.Tensor,
        transition: torch.Tensor,
        forcing: torch.Tensor,
        process_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one LGSSM prediction step without conditioning on an observation."""
        predicted_mean = transition * state_mean + forcing
        predicted_covariance = (
            transition.unsqueeze(-1) * state_covariance * transition.unsqueeze(-2)
            + torch.diag_embed(process_variance)
        )
        return predicted_mean, predicted_covariance

    def update_step(
        self,
        predicted_mean: torch.Tensor,
        predicted_covariance: torch.Tensor,
        observation: torch.Tensor,
        measurement_variance: torch.Tensor,
        target: torch.Tensor,
        observed: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        torch.Tensor,
    ]:
        """Condition one predicted state, leaving unobserved batch rows unchanged.

        ``observed`` has one Boolean value per batch row. A row marked false
        contributes no likelihood and its posterior is exactly its prediction.
        """
        predicted_observation = torch.bmm(
            observation, predicted_mean.unsqueeze(-1)
        ).squeeze(-1)
        observed = observed.to(dtype=torch.bool, device=predicted_mean.device)
        safe_target = torch.where(observed.unsqueeze(-1), target, predicted_observation)
        innovation = safe_target - predicted_observation

        if self.d_y == 1:
            covariance_times_observation = torch.bmm(
                predicted_covariance, observation.transpose(1, 2)
            )
            innovation_covariance = (
                torch.bmm(observation, covariance_times_observation).squeeze(-1).squeeze(-1)
                + measurement_variance[:, 0]
            ).clamp_min(self.variance_floor)
            gain = covariance_times_observation.squeeze(-1) / innovation_covariance.unsqueeze(-1)
            candidate_mean = predicted_mean + gain * innovation[:, 0].unsqueeze(-1)
            identity = torch.eye(
                self.n_state, dtype=predicted_mean.dtype, device=predicted_mean.device
            ).expand(predicted_mean.size(0), -1, -1)
            residual_projection = identity - gain.unsqueeze(-1) @ observation
            # Retain the Joseph form for numerical stability. Optimizing the
            # covariance algebra requires separate conditioning/PSD benchmarks.
            candidate_covariance = (
                residual_projection @ predicted_covariance @ residual_projection.transpose(1, 2)
                + measurement_variance[:, 0, None, None]
                * (gain.unsqueeze(-1) @ gain.unsqueeze(1))
            )
            log_likelihood = -0.5 * (
                torch.log(innovation_covariance)
                + innovation[:, 0].square() / innovation_covariance
                + math.log(2.0 * math.pi)
            )
            full_innovation_covariance = innovation_covariance[:, None, None]
            used_jitter = 0.0
        else:
            innovation_covariance = (
                observation @ predicted_covariance @ observation.transpose(1, 2)
                + torch.diag_embed(measurement_variance)
            )
            factor, used_jitter = safe_cholesky(
                innovation_covariance, jitter=self.variance_floor
            )
            effective_measurement_variance = measurement_variance + used_jitter
            if used_jitter:
                innovation_covariance = innovation_covariance + used_jitter * torch.eye(
                    self.d_y, dtype=predicted_mean.dtype, device=predicted_mean.device
                )
            gain = torch.cholesky_solve(
                observation @ predicted_covariance, factor
            ).transpose(1, 2)
            candidate_mean = predicted_mean + torch.bmm(
                gain, innovation.unsqueeze(-1)
            ).squeeze(-1)
            identity = torch.eye(
                self.n_state, dtype=predicted_mean.dtype, device=predicted_mean.device
            ).expand(predicted_mean.size(0), -1, -1)
            residual_projection = identity - gain @ observation
            candidate_covariance = (
                residual_projection @ predicted_covariance @ residual_projection.transpose(1, 2)
                + gain @ torch.diag_embed(effective_measurement_variance) @ gain.transpose(1, 2)
            )
            log_determinant = 2.0 * torch.log(
                torch.diagonal(factor, dim1=-2, dim2=-1)
            ).sum(-1)
            mahalanobis = torch.bmm(
                innovation.unsqueeze(1),
                torch.cholesky_solve(innovation.unsqueeze(-1), factor),
            ).squeeze(-1).squeeze(-1)
            log_likelihood = -0.5 * (
                log_determinant + mahalanobis + self.d_y * math.log(2.0 * math.pi)
            )
            full_innovation_covariance = innovation_covariance

        state_mean = torch.where(observed.unsqueeze(-1), candidate_mean, predicted_mean)
        state_covariance = torch.where(
            observed[:, None, None], candidate_covariance, predicted_covariance
        )
        log_likelihood = torch.where(observed, log_likelihood, torch.zeros_like(log_likelihood))
        return (
            state_mean,
            state_covariance,
            predicted_observation,
            full_innovation_covariance,
            used_jitter,
            log_likelihood,
        )

    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
        initial_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Filter a sequence and return predictive moments before each update.

        Targets condition only their own update; each returned prediction at time
        ``t`` depends on targets strictly before ``t``. ``observed_mask`` marks
        whole output vectors as observed. Passing no targets performs prediction
        only and never reduces covariance.
        """
        if features.ndim != 3 or features.size(-1) != self.d_feat:
            raise ValueError(f"features must have shape (batch, steps, {self.d_feat})")
        if targets is not None and targets.shape != (*features.shape[:2], self.d_y):
            raise ValueError(f"targets must have shape (batch, steps, {self.d_y})")

        batch_size, steps, _ = features.shape
        if batch_size == 0 or steps == 0:
            raise ValueError("features must contain at least one batch row and time step")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        if observed_mask is None:
            observed_mask = torch.full(
                (batch_size, steps), targets is not None, dtype=torch.bool, device=features.device
            )
        elif observed_mask.shape != (batch_size, steps):
            raise ValueError("observed_mask must have shape (batch, steps)")
        else:
            observed_mask = observed_mask.to(dtype=torch.bool, device=features.device)
        if targets is None and observed_mask.any():
            raise ValueError("targets are required wherever observed_mask is true")
        if targets is not None and not torch.isfinite(targets[observed_mask]).all():
            raise ValueError("observed targets must be finite")

        flattened = features.reshape(batch_size * steps, self.d_feat)
        delta = softplus_pos(self.gate(flattened), self.eps_pos).clamp(
            self.delta_min, self.delta_max
        ).view(batch_size, steps, 1)
        b_matrix = self.map_b(flattened).view(batch_size, steps, self.n_state, self.d_feat)
        c_matrix = self.map_c(flattened).view(batch_size, steps, self.d_y, self.n_state)
        sigma = torch.nn.functional.softplus(self.map_sigma(flattened)).clamp_min(
            self.process_std_floor
        ).view(batch_size, steps, self.n_state)
        measurement_noise = torch.nn.functional.softplus(self.map_r(flattened)).clamp_min(
            self.variance_floor
        ).view(batch_size, steps, self.d_y)

        rates = -softplus_pos(self.a_raw, self.eps_pos)
        # Rates are negative, so exp/expm1 do not overflow. Clipping this product
        # would change the stated continuous-time drift and its discretization.
        z_values = delta * rates.view(1, 1, -1)
        transitions = torch.exp(z_values)
        gamma_values, rho_values = gamma(z_values), rho(z_values)
        if initial_state is None:
            state_mean, state_covariance = self.initial_state(features)
        else:
            state_mean, state_covariance = initial_state
            if state_mean.shape != (batch_size, self.n_state):
                raise ValueError("initial state mean has the wrong shape")
            if state_covariance.shape != (batch_size, self.n_state, self.n_state):
                raise ValueError("initial state covariance has the wrong shape")
            state_mean = state_mean.to(dtype=features.dtype, device=features.device)
            state_covariance = state_covariance.to(dtype=features.dtype, device=features.device)
        predicted_means: list[torch.Tensor] = []
        predicted_variances: list[torch.Tensor] = []
        predicted_covariances: list[torch.Tensor] = []
        log_likelihoods: list[torch.Tensor] = []
        used_jitters: list[float] = []

        for step in range(steps):
            transition = transitions[:, step, :]
            observation = c_matrix[:, step, :, :]
            b_bar = (gamma_values[:, step, :] * delta[:, step, :]).unsqueeze(-1) * b_matrix[:, step]
            forcing = torch.bmm(b_bar, features[:, step, :].unsqueeze(-1)).squeeze(-1)
            process_variance = sigma[:, step].square() * rho_values[:, step] * delta[:, step]
            predicted_state, predicted_covariance = self.predict_step(
                state_mean, state_covariance, transition, forcing, process_variance
            )
            target = (
                targets[:, step]
                if targets is not None
                else features.new_zeros(batch_size, self.d_y)
            )
            (
                state_mean,
                state_covariance,
                predicted_observation,
                innovation_covariance,
                used_jitter,
                log_likelihood,
            ) = self.update_step(
                predicted_state,
                predicted_covariance,
                observation,
                measurement_noise[:, step],
                target,
                observed_mask[:, step],
            )
            predicted_means.append(predicted_observation)
            predicted_covariances.append(innovation_covariance)
            predicted_variances.append(
                torch.diagonal(innovation_covariance, dim1=-2, dim2=-1)
            )
            log_likelihoods.append(log_likelihood)
            used_jitters.append(used_jitter)

        result = {
            "y_mean": torch.stack(predicted_means, dim=1),
            "y_var_diag": torch.stack(predicted_variances, dim=1),
            "y_covariance": torch.stack(predicted_covariances, dim=1),
            "a_continuous": rates.detach(),
            "used_jitter": features.new_tensor(max(used_jitters, default=0.0)),
            "final_state_mean": state_mean,
            "final_state_covariance": state_covariance,
        }
        if targets is not None and observed_mask.any():
            log_likelihood = torch.stack(log_likelihoods, dim=1)
            scored_steps = observed_mask.sum()
            nll_sum = -log_likelihood.sum()
            result["log_likelihood"] = log_likelihood
            result["nll_sum"] = nll_sum
            result["scored_steps"] = scored_steps
            result["nll"] = nll_sum / scored_steps
            result["avg_ll_per_step"] = log_likelihood.sum() / scored_steps
        return result


class ProbabilisticMamba(nn.Module):
    """Mamba feature extractor followed by the probabilistic LGSSM head."""

    def __init__(
        self,
        d_in: int,
        d_feat: int,
        d_y: int,
        n_state: int,
        n_mamba_layers: int = 1,
        mamba_cfg: Optional[Mapping[str, int]] = None,
        head_cfg: Optional[Mapping[str, float]] = None,
    ):
        super().__init__()
        self.features = FeatureNet(d_in, d_feat, n_mamba_layers, mamba_cfg)
        self.lgssm = ProbMambaHead(d_feat, d_y, n_state, **(head_cfg or {}))

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
        initial_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        return self.lgssm(
            self.features(inputs),
            targets,
            observed_mask=observed_mask,
            initial_state=initial_state,
        )
