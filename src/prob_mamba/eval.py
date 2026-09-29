"""Evaluation and model-budget helpers retained for notebook migration."""

from __future__ import annotations

from .training import evaluate_last_step_probabilistic, train_probabilistic_model
from .utils import param_count


def prob_head_params(d_feat: int, n_state: int, d_y: int = 1) -> int:
    """Return the exact trainable parameter count of :class:`ProbMambaHead`."""
    from .models import ProbMambaHead

    return param_count(ProbMambaHead(d_feat=d_feat, d_y=d_y, n_state=n_state))


def build_prob_mamba_budget_fixed_n(
    input_dim: int,
    target: int = 100_000,
    n_state: int = 16,
    n_layers: int = 1,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    d_feat_min: int = 32,
    d_feat_max: int = 192,
    d_feat_step: int = 8,
):
    """Build the Mamba model whose trainable parameter count is closest to target."""
    from .models import ProbabilisticMamba

    best_model, best_distance = None, float("inf")
    config = {"d_state": d_state, "d_conv": d_conv, "expand": expand}
    for d_feat in range(d_feat_min, d_feat_max + 1, d_feat_step):
        model = ProbabilisticMamba(input_dim, d_feat, 1, n_state, n_layers, config)
        distance = abs(param_count(model) - target)
        if distance < best_distance:
            best_model, best_distance = model, distance
    if best_model is None:
        raise ValueError("feature range does not contain a candidate")
    return best_model


train_prob_mamba_with_history = train_probabilistic_model
eval_prob_rmse_qlike_laststep = evaluate_last_step_probabilistic
