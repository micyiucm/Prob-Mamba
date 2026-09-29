import numpy as np
import pandas as pd
import pytest

from prob_mamba.baselines import rolling_arma_forecast, rolling_ewma_forecast


def timestamped_frame(start, values):
    origins = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "origin_timestamp": origins,
            "target_timestamp": origins + pd.Timedelta(days=1),
            "label_available_timestamp": origins + pd.Timedelta(days=1),
            "y_next": values,
        }
    )


def test_rolling_ewma_updates_with_each_newly_observed_test_return():
    history = timestamped_frame("2020-01-01", [0.1, -0.2, 0.3, -0.4])
    test = timestamped_frame("2020-01-05", [0.5, -0.6, 0.7])
    result = rolling_ewma_forecast(history, test, decay=0.5, dataset="example")

    variance = np.mean(np.square([0.1, -0.2, 0.3, -0.4]))
    for observed in [0.1, -0.2, 0.3, -0.4]:
        variance = 0.5 * variance + 0.5 * observed**2
    expected = [variance]
    expected.append(0.5 * expected[-1] + 0.5 * 0.5**2)
    expected.append(0.5 * expected[-1] + 0.5 * (-0.6) ** 2)
    np.testing.assert_allclose(result["variance"], expected)
    np.testing.assert_array_equal(result["target_timestamp"], test["target_timestamp"])
    np.testing.assert_array_equal(result["mean"], 0.0)


def test_rolling_baseline_refuses_outcomes_unavailable_at_next_origin():
    history = timestamped_frame("2020-01-01", [0.1, -0.2, 0.3, -0.4])
    test = timestamped_frame("2020-01-05", [0.5, -0.6])
    test.loc[0, "label_available_timestamp"] = pd.Timestamp("2020-01-07")
    with pytest.raises(ValueError, match="overlapping outcomes"):
        rolling_ewma_forecast(history, test)


def test_rolling_baseline_rejects_silently_skipped_test_observations():
    history = timestamped_frame("2020-01-01", [0.1, -0.2, 0.3, -0.4])
    test = timestamped_frame("2020-01-05", [0.5, -0.6, 0.7]).iloc[[0, 2]]
    with pytest.raises(ValueError, match="intervening"):
        rolling_ewma_forecast(history, test)


def test_rolling_arma_scores_every_test_timestamp_and_updates_without_refit():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(4)
    values = np.empty(100)
    values[0] = 0.1
    for index in range(1, len(values)):
        values[index] = 0.7 * values[index - 1] + rng.normal(scale=0.05)
    train = timestamped_frame("2020-01-01", values[:70])
    validation = timestamped_frame("2020-03-11", values[70:90])
    test = timestamped_frame("2020-03-31", values[90:])
    order, predictions, final_state = rolling_arma_forecast(
        train,
        validation,
        test,
        max_ar=1,
        max_ma=0,
        dataset="synthetic",
    )
    assert order == (1, 0, 0)
    assert len(predictions) == len(test)
    np.testing.assert_array_equal(predictions["target_timestamp"], test["target_timestamp"])
    assert len(final_state.model.endog) == 99


@pytest.mark.parametrize("with_arma", [False, True])
def test_garch_first_forecast_matches_library_and_next_uses_realized_shock(with_arma):
    pytest.importorskip("statsmodels")
    pytest.importorskip("arch")
    from prob_mamba.baselines import rolling_arma_garch_forecast, rolling_zero_mean_garch_forecast
    rng = np.random.default_rng(5)
    values = rng.normal(scale=0.01, size=124)
    full = timestamped_frame("2020-01-01", values)
    train, validation, test = full.iloc[:100], full.iloc[100:120], full.iloc[120:]
    if with_arma:
        _, predictions, _, fit = rolling_arma_garch_forecast(
            train, validation, test, max_ar=1, max_ma=0, scale=100,
        )
    else:
        predictions, fit = rolling_zero_mean_garch_forecast(full.iloc[:120], test, scale=100)
    initial = fit.forecast(horizon=1, reindex=False).variance.values[-1, 0] / 100**2
    assert predictions["variance"].iloc[0] == pytest.approx(initial)
    error = (test["y_next"].iloc[0] - predictions["mean"].iloc[0]) * 100
    expected_second = (
        fit.params["omega"] + fit.params["alpha[1]"] * error**2
        + fit.params["beta[1]"] * initial * 100**2
    ) / 100**2
    assert predictions["variance"].iloc[1] == pytest.approx(expected_second)
    np.testing.assert_array_equal(predictions["target_timestamp"], test["target_timestamp"])
