import importlib.util

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prob_mamba.models import MambaModel, ProbMambaHead
from prob_mamba.training import evaluate_last_step_probabilistic, train_probabilistic_model


def test_probabilistic_head_runs_forward_and_backward_on_cpu():
    torch.manual_seed(4)
    head = ProbMambaHead(d_feat=3, d_y=1, n_state=2)
    features = torch.randn(2, 5, 3, requires_grad=True)
    targets = torch.randn(2, 5, 1)
    output = head(features, targets)
    assert output["y_mean"].shape == (2, 5, 1)
    assert output["y_var_diag"].shape == (2, 5, 1)
    assert output["y_covariance"].shape == (2, 5, 1, 1)
    assert torch.isfinite(output["nll"])
    torch.testing.assert_close(
        output["nll"], output["nll_sum"] / output["scored_steps"]
    )
    output["nll"].backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in head.parameters() if parameter.grad is not None)


def test_multivariate_head_and_prediction_only_mode():
    head = ProbMambaHead(d_feat=2, d_y=2, n_state=3)
    features = torch.randn(2, 4, 2)
    prediction = head(features)
    fitted = head(features, torch.randn(2, 4, 2))
    assert "nll" not in prediction
    assert prediction["y_var_diag"].shape == (2, 4, 2)
    assert prediction["y_covariance"].shape == (2, 4, 2, 2)
    torch.testing.assert_close(
        prediction["y_covariance"], prediction["y_covariance"].transpose(-1, -2)
    )
    torch.testing.assert_close(
        prediction["y_var_diag"],
        torch.diagonal(prediction["y_covariance"], dim1=-2, dim2=-1),
    )
    assert torch.isfinite(fitted["nll"])


def test_scalar_prediction_update_matches_reference_values():
    head = ProbMambaHead(d_feat=1, d_y=1, n_state=1).double()
    state_mean = torch.tensor([[0.1]], dtype=torch.float64)
    state_covariance = torch.tensor([[[0.09]]], dtype=torch.float64)
    rate, delta = -0.7, 0.4
    z = torch.tensor([[rate * delta]], dtype=torch.float64)
    transition = torch.exp(z)
    forcing = ((torch.expm1(z) / z) * delta * 0.2)
    process_variance = torch.tensor([[0.027565131609719042]], dtype=torch.float64)
    predicted_mean, predicted_covariance = head.predict_step(
        state_mean, state_covariance, transition, forcing, process_variance
    )
    (
        posterior_mean,
        posterior_covariance,
        predicted_observation,
        innovation_covariance,
        _,
        log_likelihood,
    ) = head.update_step(
        predicted_mean,
        predicted_covariance,
        torch.tensor([[[1.2]]], dtype=torch.float64),
        torch.tensor([[0.04]], dtype=torch.float64),
        torch.tensor([[0.05]], dtype=torch.float64),
        torch.tensor([True]),
    )
    torch.testing.assert_close(transition, torch.tensor([[0.7557837414557255]], dtype=torch.float64))
    torch.testing.assert_close(forcing, torch.tensor([[0.06977607386979272]], dtype=torch.float64))
    torch.testing.assert_close(predicted_mean, torch.tensor([[0.1453544480153653]], dtype=torch.float64))
    torch.testing.assert_close(predicted_covariance, torch.tensor([[[0.07897394735611238]]], dtype=torch.float64))
    torch.testing.assert_close(predicted_observation, torch.tensor([[0.17442533761843834]], dtype=torch.float64))
    torch.testing.assert_close(innovation_covariance, torch.tensor([[[0.15372248419280182]]], dtype=torch.float64))
    torch.testing.assert_close(posterior_mean, torch.tensor([[0.06864717817561454]], dtype=torch.float64))
    torch.testing.assert_close(posterior_covariance, torch.tensor([[[0.020549745281779774]]], dtype=torch.float64))
    torch.testing.assert_close(-log_likelihood, torch.tensor([0.03299124555333387], dtype=torch.float64))
    skipped_mean, skipped_covariance, *_, skipped_log_likelihood = head.update_step(
        predicted_mean,
        predicted_covariance,
        torch.tensor([[[1.2]]], dtype=torch.float64),
        torch.tensor([[0.04]], dtype=torch.float64),
        torch.tensor([[999.0]], dtype=torch.float64),
        torch.tensor([False]),
    )
    torch.testing.assert_close(skipped_mean, predicted_mean)
    torch.testing.assert_close(skipped_covariance, predicted_covariance)
    torch.testing.assert_close(skipped_log_likelihood, torch.zeros(1, dtype=torch.float64))


def test_missing_observation_is_identical_to_prediction_only():
    torch.manual_seed(12)
    head = ProbMambaHead(d_feat=2, d_y=1, n_state=3)
    features = torch.randn(2, 4, 2)
    targets = torch.full((2, 4, 1), float("nan"))
    skipped = head(features, targets, observed_mask=torch.zeros(2, 4, dtype=torch.bool))
    prediction_only = head(features)
    assert "nll" not in skipped
    torch.testing.assert_close(skipped["y_mean"], prediction_only["y_mean"])
    torch.testing.assert_close(skipped["y_covariance"], prediction_only["y_covariance"])
    torch.testing.assert_close(skipped["final_state_mean"], prediction_only["final_state_mean"])
    torch.testing.assert_close(
        skipped["final_state_covariance"], prediction_only["final_state_covariance"]
    )


def test_complete_head_discretization_matches_scalar_reference():
    from prob_mamba.numerics import inverse_softplus
    head = ProbMambaHead(d_feat=1, d_y=1, n_state=1).double()
    with torch.no_grad():
        head.a_raw.fill_(inverse_softplus(0.7 - head.eps_pos))
        for layer in (head.gate, head.map_b, head.map_c, head.map_sigma, head.map_r):
            layer.weight.zero_()
        head.gate.bias.fill_(inverse_softplus(0.4 - head.eps_pos))
        head.map_b.bias.fill_(0.2)
        head.map_c.bias.fill_(1.2)
        head.map_sigma.bias.fill_(inverse_softplus(0.3))
        head.map_r.bias.fill_(inverse_softplus(0.04))
    output = head(
        torch.ones(1, 1, 1, dtype=torch.float64),
        torch.tensor([[[0.05]]], dtype=torch.float64),
        initial_state=(torch.tensor([[0.1]], dtype=torch.float64),
                       torch.tensor([[[0.09]]], dtype=torch.float64)),
    )
    assert output["y_mean"].item() == pytest.approx(0.17442533761843834, abs=1e-12)
    assert output["y_var_diag"].item() == pytest.approx(0.15372248419280182, abs=1e-12)
    assert output["final_state_covariance"].item() == pytest.approx(0.020549745281779774, abs=1e-12)
    assert output["nll"].item() == pytest.approx(0.03299124555333387, abs=1e-12)


def test_prediction_at_t_depends_only_on_targets_before_t():
    torch.manual_seed(21)
    head = ProbMambaHead(d_feat=2, d_y=1, n_state=2)
    features = torch.randn(1, 4, 2)
    first_targets = torch.zeros(1, 4, 1)
    second_targets = first_targets.clone()
    second_targets[:, 2] = 10.0
    first = head(features, first_targets)
    second = head(features, second_targets)
    torch.testing.assert_close(first["y_mean"][:, :3], second["y_mean"][:, :3])
    assert not torch.allclose(first["y_mean"][:, 3], second["y_mean"][:, 3])


def test_filter_state_continuation_matches_one_full_sequence():
    torch.manual_seed(8)
    head = ProbMambaHead(d_feat=2, d_y=1, n_state=2)
    features = torch.randn(2, 5, 2)
    targets = torch.randn(2, 5, 1)
    full = head(features, targets)
    prefix = head(features[:, :2], targets[:, :2])
    suffix = head(
        features[:, 2:],
        targets[:, 2:],
        initial_state=(prefix["final_state_mean"], prefix["final_state_covariance"]),
    )
    torch.testing.assert_close(suffix["y_mean"], full["y_mean"][:, 2:])
    torch.testing.assert_close(suffix["y_covariance"], full["y_covariance"][:, 2:])
    torch.testing.assert_close(suffix["final_state_mean"], full["final_state_mean"])
    torch.testing.assert_close(
        suffix["final_state_covariance"], full["final_state_covariance"]
    )


def test_variance_initialization_uses_the_declared_target_scale():
    head = ProbMambaHead(
        d_feat=1,
        d_y=1,
        n_state=1,
        target_scale=0.01,
        initial_process_std_ratio=0.1,
        initial_observation_variance_ratio=1.0,
        variance_floor_ratio=1e-8,
    )
    prediction = head(torch.zeros(1, 1, 1))
    torch.testing.assert_close(
        prediction["y_var_diag"], torch.tensor([[[1e-4]]]), rtol=1e-5, atol=1e-9
    )
    torch.testing.assert_close(
        torch.nn.functional.softplus(head.map_sigma.bias),
        torch.tensor([1e-3]),
        rtol=1e-5,
        atol=1e-9,
    )
    assert head.variance_floor == pytest.approx(1e-12)


def test_mamba_dependency_is_optional():
    if importlib.util.find_spec("mamba_ssm") is None:
        with pytest.raises(ImportError, match="optional dependency"):
            MambaModel(input_dim=2)


def test_training_and_evaluation_helpers_run_on_cpu():
    inputs = torch.randn(6, 3, 2)
    targets = torch.randn(6, 3, 1)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    model = ProbMambaHead(d_feat=2, d_y=1, n_state=2)
    history = train_probabilistic_model(model, loader, loader, epochs=1, device="cpu")
    metrics = evaluate_last_step_probabilistic(model, loader, device="cpu")
    assert len(history["train_nll"]) == len(history["validation_nll"]) == 1
    assert history["best_epoch"] == 0
    assert set(metrics) == {"rmse", "qlike", "nll"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    import math
    assert metrics["nll"] == pytest.approx(
        0.5 * (metrics["qlike"] + math.log(2 * math.pi)), abs=1e-5
    )


def test_endpoint_validation_excludes_context_losses():
    torch.manual_seed(49)
    model = ProbMambaHead(d_feat=1, d_y=1, n_state=2)
    inputs = torch.randn(2, 4, 1)
    targets = torch.randn(2, 4, 1)
    targets[:, :-1] += 10
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    expected = model(inputs, targets)["log_likelihood"].detach()
    history = train_probabilistic_model(
        model, loader, loader, epochs=1, learning_rate=0, device="cpu"
    )
    assert history["validation_nll"][0] == pytest.approx(-expected[:, -1].mean().item())
    assert history["validation_nll"][0] != pytest.approx(-expected.mean().item())


class _OpposedTrainValidationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs, targets):
        scored_steps = torch.tensor(targets.numel(), device=targets.device)
        nll = (self.weight - targets.mean()).square()
        return {"nll": nll, "nll_sum": nll * scored_steps, "scored_steps": scored_steps,
                "log_likelihood": -nll.expand(targets.shape[:2])}


def test_training_restores_and_saves_best_validation_checkpoint(tmp_path):
    inputs = torch.zeros(1, 1, 1)
    train_loader = DataLoader(TensorDataset(inputs, torch.ones_like(inputs)), batch_size=1)
    validation_loader = DataLoader(TensorDataset(inputs, -torch.ones_like(inputs)), batch_size=1)
    model = _OpposedTrainValidationModel()
    checkpoint_path = tmp_path / "best.pt"
    history = train_probabilistic_model(
        model,
        train_loader,
        validation_loader,
        epochs=3,
        learning_rate=0.1,
        device="cpu",
        checkpoint_path=checkpoint_path,
        checkpoint_metadata={"dataset": "synthetic"},
    )
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert history["best_epoch"] == checkpoint["epoch"] == 0
    assert checkpoint["metadata"] == {"dataset": "synthetic"}
    torch.testing.assert_close(model.weight, checkpoint["model_state_dict"]["weight"])
    assert model.weight.item() < 0.2
