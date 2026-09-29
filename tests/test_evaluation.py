import math

import numpy as np
import pandas as pd
import pytest

from prob_mamba.evaluation import (
    assert_common_scoring_support,
    make_prediction_frame,
    score_prediction_frame,
)


def prediction_frame(model="model", targets=None, observations=None):
    targets = pd.DatetimeIndex(targets or pd.date_range("2020-01-02", periods=3))
    observations = observations or [0.1, -0.2, 0.3]
    return make_prediction_frame(
        dataset="example",
        model=model,
        origin_timestamps=targets - pd.Timedelta(days=1),
        target_timestamps=targets,
        y_true=observations,
        mean=[0.0, 0.0, 0.0],
        variance=[0.04, 0.04, 0.04],
        seed=0,
    )


def test_prediction_table_rejects_unequal_lengths_and_invalid_variance():
    with pytest.raises(ValueError, match="unequal lengths"):
        make_prediction_frame(
            dataset="x",
            model="x",
            origin_timestamps=pd.date_range("2020-01-01", periods=2),
            target_timestamps=pd.date_range("2020-01-02", periods=3),
            y_true=[1.0, 2.0],
            mean=[1.0, 2.0],
        )
    frame = prediction_frame()
    frame.loc[0, "variance"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        score_prediction_frame(frame)


def test_common_support_rejects_missing_target_instead_of_inner_joining():
    first = prediction_frame("first")
    second = prediction_frame("second").iloc[1:].copy()
    with pytest.raises(ValueError, match="identical target-timestamp support"):
        assert_common_scoring_support([first, second])


def test_common_support_rejects_different_forecast_origins():
    first = prediction_frame("first")
    second = prediction_frame("second")
    second.loc[0, "origin_timestamp"] -= pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="origin_timestamp"):
        assert_common_scoring_support([first, second])


def test_fractional_horizon_is_not_silently_truncated():
    frame = prediction_frame()
    frame["horizon"] = 1.5
    with pytest.raises(ValueError, match="horizon=1"):
        score_prediction_frame(frame)


def test_gaussian_nll_is_affine_transform_of_residual_qlike():
    scores = score_prediction_frame(prediction_frame())
    assert scores["n_obs"] == 3
    assert scores["gaussian_nll"] == pytest.approx(
        0.5 * (scores["qlike_residual"] + math.log(2.0 * math.pi))
    )
    assert np.isfinite(scores["qlike_zero_mean"])
    assert np.isfinite(scores["pit_mean"])
    assert {"coverage_50", "coverage_80", "coverage_95"}.issubset(scores)
