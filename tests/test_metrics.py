import numpy as np
import pytest

from prob_mamba.metrics import quasi_likelihood


def test_qlike_matches_scalar_gaussian_variance_score_without_hidden_floor():
    residual = np.array([0.05 - 0.17442533761843834])
    variance = np.array([0.15372248419280182])
    expected = residual.item() ** 2 / variance.item() + np.log(variance.item())
    assert quasi_likelihood(residual, variance) == pytest.approx(expected)


def test_qlike_floor_is_explicit_and_rejects_negative_variance():
    assert quasi_likelihood(np.array([0.0]), np.array([0.0]), min_variance=1e-10) == pytest.approx(
        np.log(1e-10)
    )
    with pytest.raises(ValueError, match="non-negative"):
        quasi_likelihood(np.array([0.0]), np.array([-1.0]))
