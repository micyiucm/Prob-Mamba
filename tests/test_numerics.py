import torch

from prob_mamba.numerics import gamma, inverse_softplus, rho, safe_cholesky


def test_discretization_helpers_are_accurate_near_zero():
    values = torch.tensor([-1e-8, 0.0, 1e-8], dtype=torch.float64)
    torch.testing.assert_close(gamma(values), torch.ones_like(values), atol=1e-8, rtol=0)
    torch.testing.assert_close(rho(values), torch.ones_like(values), atol=1e-8, rtol=0)


def test_safe_cholesky_returns_factor_for_spd_batch():
    matrix = torch.tensor([[[2.0, 0.5], [0.5, 1.0]]])
    factor, jitter = safe_cholesky(matrix)
    torch.testing.assert_close(factor @ factor.transpose(-1, -2), matrix)
    assert jitter == 0.0


def test_inverse_softplus_round_trips_small_domain_scales():
    expected = torch.tensor([1e-6, 1e-4, 1e-2, 1.0], dtype=torch.float64)
    raw = torch.tensor(
        [inverse_softplus(value.item()) for value in expected], dtype=torch.float64
    )
    torch.testing.assert_close(torch.nn.functional.softplus(raw), expected)
