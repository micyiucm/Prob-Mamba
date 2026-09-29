"""Optional classical ARMA and GARCH forecasting baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

from .evaluation import make_prediction_frame
from .metrics import quasi_likelihood


def _timestamped_targets(
    frame: pd.DataFrame,
    *,
    target_column: str,
    origin_timestamp_column: str,
    target_timestamp_column: str,
    label_available_timestamp_column: str,
) -> pd.DataFrame:
    """Validate a forecasting frame without dropping or realigning rows."""
    columns = [
        origin_timestamp_column,
        target_timestamp_column,
        label_available_timestamp_column,
        target_column,
    ]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"missing columns: {missing}")
    result = frame.loc[:, columns].copy()
    if result.empty:
        raise ValueError("forecasting frames must not be empty")
    for column in columns[:3]:
        result[column] = pd.to_datetime(result[column], errors="raise", utc=True).dt.tz_localize(None)
    if result[columns[:3]].isna().any().any():
        raise ValueError("forecast timestamps must not be missing")
    result[target_column] = pd.to_numeric(result[target_column], errors="raise")
    if not np.isfinite(result[target_column].to_numpy()).all():
        raise ValueError("target observations must be finite; refusing to drop rows implicitly")
    result = result.sort_values(origin_timestamp_column).reset_index(drop=True)
    if result[origin_timestamp_column].duplicated().any():
        raise ValueError("origin timestamps must be unique")
    if result[target_timestamp_column].duplicated().any():
        raise ValueError("target timestamps must be unique")
    if not (result[target_timestamp_column] > result[origin_timestamp_column]).all():
        raise ValueError("target timestamps must be after forecast origins")
    if (result[label_available_timestamp_column] < result[target_timestamp_column]).any():
        raise ValueError("labels cannot be available before their target timestamps")
    return result


def _known_values(
    frames: list[pd.DataFrame],
    cutoff: pd.Timestamp,
    *,
    target_column: str,
    label_available_timestamp_column: str,
) -> pd.Series:
    combined = pd.concat(frames, axis=0, ignore_index=True)
    known = combined.loc[
        combined[label_available_timestamp_column] <= cutoff
    ].sort_values(label_available_timestamp_column)
    if known.empty:
        raise ValueError(f"no labels are available by fit cutoff {cutoff}")
    return known[target_column].reset_index(drop=True).astype("float64")


def _check_rolling_schedule(
    test: pd.DataFrame,
    *,
    origin_timestamp_column: str,
    target_timestamp_column: str = "target_timestamp",
    label_available_timestamp_column: str,
) -> None:
    """Ensure each prior test outcome is known before the next one-step origin."""
    if len(test) < 2:
        return
    prior_available = test[label_available_timestamp_column].iloc[:-1].to_numpy()
    next_origins = test[origin_timestamp_column].iloc[1:].to_numpy()
    if not (prior_available <= next_origins).all():
        raise ValueError("test schedule has overlapping outcomes; one-step rolling updates are undefined")
    prior_targets = test[target_timestamp_column].iloc[:-1].to_numpy()
    if not (prior_targets == next_origins).all():
        raise ValueError("test schedule must include every intervening one-step observation")


def clean_series(values, min_observations: int = 50) -> pd.Series:
    """Return finite, non-degenerate observations suitable for classical models."""
    series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < min_observations:
        raise ValueError(f"need at least {min_observations} valid observations")
    if series.std() < 1e-5:
        raise ValueError("series has near-zero variance")
    return series


def select_arma_order(
    train_values, max_ar: int = 3, max_ma: int = 3, trend: str = "n"
) -> tuple[tuple[int, int, int], object]:
    """Select the lowest-BIC ARMA order using only training observations."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as error:
        raise ImportError("ARMA baselines require `pip install -e '.[baselines]'`.") from error
    train = clean_series(train_values)
    best: tuple[float, tuple[int, int, int], object] | None = None
    for ar_order in range(max_ar + 1):
        for ma_order in range(max_ma + 1):
            if ar_order == ma_order == 0:
                continue
            try:
                fitted = ARIMA(train, order=(ar_order, 0, ma_order), trend=trend).fit()
            except (ValueError, np.linalg.LinAlgError):
                continue
            candidate = (float(fitted.bic), (ar_order, 0, ma_order), fitted)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise ValueError("no ARMA specification could be fitted")
    return best[1], best[2]


def fit_arma_forecast(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str = "y_next",
    max_ar: int = 3,
    max_ma: int = 3,
    trend: str = "n",
) -> tuple[tuple[int, int, int], float, pd.Series, object]:
    """Compatibility return format for the timestamped rolling ARMA evaluator.

    Inputs now require explicit forecast/target/availability timestamps. Use
    ``rolling_arma_forecast`` directly to retain timestamps in the output.
    """
    order, predictions, fitted = rolling_arma_forecast(
        train, validation, test, target_column=target_column,
        max_ar=max_ar, max_ma=max_ma, trend=trend,
    )
    rmse = float(np.sqrt(np.mean((predictions["mean"] - predictions["y_true"]) ** 2)))
    return order, rmse, predictions["mean"], fitted


def fit_garch_forecast(
    train_validation_residuals,
    test_residuals,
    scale: float = 100.0,
) -> tuple[float, float, pd.Series]:
    """Legacy fixed-origin multi-step forecast; not a rolling evaluation.

    Retained only for explicit reproduction of the historical protocol. New
    experiments must use the timestamped rolling functions below.
    """
    warnings.warn(
        "fit_garch_forecast is the historical fixed-origin multi-step protocol; "
        "use rolling_arma_garch_forecast or rolling_zero_mean_garch_forecast",
        DeprecationWarning, stacklevel=2,
    )
    try:
        from arch import arch_model
    except ImportError as error:
        raise ImportError("GARCH baselines require `pip install -e '.[baselines]'`.") from error
    training = clean_series(train_validation_residuals)
    observed = pd.Series(test_residuals, dtype="float64").reset_index(drop=True)
    if observed.empty or not np.isfinite(observed).all():
        raise ValueError("test residuals must be nonempty and finite; no implicit row dropping")
    fitted = arch_model(training * scale, mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off")
    variances = pd.Series(
        fitted.forecast(horizon=len(observed), reindex=False).variance.values[-1] / scale**2
    ).clip(lower=1e-12)
    residuals = observed.to_numpy()
    variance_values = variances.to_numpy()
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return rmse, quasi_likelihood(residuals, variance_values), variances


def rolling_arma_forecast(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str = "y_next",
    origin_timestamp_column: str = "origin_timestamp",
    target_timestamp_column: str = "target_timestamp",
    label_available_timestamp_column: str = "label_available_timestamp",
    max_ar: int = 3,
    max_ma: int = 3,
    trend: str = "n",
    dataset: str = "dataset",
    model_name: str = "ARMA",
) -> tuple[tuple[int, int, int], pd.DataFrame, object]:
    """Fit once, then issue one-step ARMA forecasts with state-only test updates.

    Order selection uses the training partition.  Coefficients are fitted to all
    training/validation labels available at the first test origin and remain
    fixed.  After each target becomes observable it is appended with
    ``refit=False`` before the next one-step forecast.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as error:
        raise ImportError("ARMA baselines require `pip install -e '.[baselines]'`.") from error
    arguments = {
        "target_column": target_column,
        "origin_timestamp_column": origin_timestamp_column,
        "target_timestamp_column": target_timestamp_column,
        "label_available_timestamp_column": label_available_timestamp_column,
    }
    train_checked = _timestamped_targets(train, **arguments)
    validation_checked = _timestamped_targets(validation, **arguments)
    test_checked = _timestamped_targets(test, **arguments)
    _check_rolling_schedule(
        test_checked,
        origin_timestamp_column=origin_timestamp_column,
        target_timestamp_column=target_timestamp_column,
        label_available_timestamp_column=label_available_timestamp_column,
    )
    first_origin = test_checked[origin_timestamp_column].iloc[0]
    selection_cutoff = validation_checked[origin_timestamp_column].iloc[0]
    train_known = _known_values(
        [train_checked],
        selection_cutoff,
        target_column=target_column,
        label_available_timestamp_column=label_available_timestamp_column,
    )
    order, _ = select_arma_order(train_known, max_ar, max_ma, trend)
    fit_values = _known_values(
        [train_checked, validation_checked],
        first_origin,
        target_column=target_column,
        label_available_timestamp_column=label_available_timestamp_column,
    )
    state = ARIMA(
        fit_values,
        order=order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()

    means: list[float] = []
    for position, row in test_checked.iterrows():
        if position:
            state = state.append(
                [float(test_checked.loc[position - 1, target_column])], refit=False
            )
        means.append(float(np.asarray(state.forecast(steps=1)).reshape(-1)[0]))
    predictions = make_prediction_frame(
        dataset=dataset,
        model=model_name,
        origin_timestamps=test_checked[origin_timestamp_column],
        target_timestamps=test_checked[target_timestamp_column],
        y_true=test_checked[target_column],
        mean=means,
    )
    return order, predictions, state


def rolling_arma_garch_forecast(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str = "y_next",
    origin_timestamp_column: str = "origin_timestamp",
    target_timestamp_column: str = "target_timestamp",
    label_available_timestamp_column: str = "label_available_timestamp",
    max_ar: int = 3,
    max_ma: int = 3,
    trend: str = "n",
    scale: float = 100.0,
    dataset: str = "dataset",
    model_name: str = "ARMA-GARCH",
) -> tuple[tuple[int, int, int], pd.DataFrame, object, object]:
    """Rolling one-step ARMA-GARCH forecasts with frozen fitted parameters."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    try:
        from arch import arch_model
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as error:
        raise ImportError(
            "ARMA-GARCH baselines require `pip install -e '.[baselines]'`."
        ) from error
    arguments = {
        "target_column": target_column,
        "origin_timestamp_column": origin_timestamp_column,
        "target_timestamp_column": target_timestamp_column,
        "label_available_timestamp_column": label_available_timestamp_column,
    }
    train_checked = _timestamped_targets(train, **arguments)
    validation_checked = _timestamped_targets(validation, **arguments)
    test_checked = _timestamped_targets(test, **arguments)
    _check_rolling_schedule(
        test_checked,
        origin_timestamp_column=origin_timestamp_column,
        target_timestamp_column=target_timestamp_column,
        label_available_timestamp_column=label_available_timestamp_column,
    )
    first_origin = test_checked[origin_timestamp_column].iloc[0]
    selection_cutoff = validation_checked[origin_timestamp_column].iloc[0]
    train_known = _known_values(
        [train_checked],
        selection_cutoff,
        target_column=target_column,
        label_available_timestamp_column=label_available_timestamp_column,
    )
    order, _ = select_arma_order(train_known, max_ar, max_ma, trend)
    fit_values = _known_values(
        [train_checked, validation_checked],
        first_origin,
        target_column=target_column,
        label_available_timestamp_column=label_available_timestamp_column,
    )
    arma_state = ARIMA(
        fit_values,
        order=order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    residuals = np.asarray(arma_state.resid, dtype="float64")
    if not np.isfinite(residuals).all():
        raise ValueError("ARMA fit produced non-finite residuals")
    garch_fit = arch_model(
        residuals * scale, mean="Zero", vol="GARCH", p=1, q=1, dist="normal"
    ).fit(disp="off")
    parameters = garch_fit.params
    omega = float(parameters["omega"])
    alpha = float(parameters["alpha[1]"])
    beta = float(parameters["beta[1]"])
    last_variance = float(np.asarray(garch_fit.conditional_volatility)[-1] ** 2)
    last_residual = float(residuals[-1] * scale)
    next_variance = omega + alpha * last_residual**2 + beta * last_variance

    means: list[float] = []
    variances: list[float] = []
    for position, row in test_checked.iterrows():
        if position:
            prior_observed = float(test_checked.loc[position - 1, target_column])
            prior_residual = (prior_observed - means[-1]) * scale
            next_variance = omega + alpha * prior_residual**2 + beta * next_variance
            arma_state = arma_state.append([prior_observed], refit=False)
        mean = float(np.asarray(arma_state.forecast(steps=1)).reshape(-1)[0])
        means.append(mean)
        variances.append(max(next_variance / scale**2, np.finfo(float).tiny))

    predictions = make_prediction_frame(
        dataset=dataset,
        model=model_name,
        origin_timestamps=test_checked[origin_timestamp_column],
        target_timestamps=test_checked[target_timestamp_column],
        y_true=test_checked[target_column],
        mean=means,
        variance=variances,
    )
    return order, predictions, arma_state, garch_fit


def rolling_zero_mean_garch_forecast(
    history: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str = "y_next",
    origin_timestamp_column: str = "origin_timestamp",
    target_timestamp_column: str = "target_timestamp",
    label_available_timestamp_column: str = "label_available_timestamp",
    scale: float = 100.0,
    dataset: str = "dataset",
    model_name: str = "GARCH-zero-mean",
) -> tuple[pd.DataFrame, object]:
    """Fit zero-mean GARCH once and roll its variance state through test."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    try:
        from arch import arch_model
    except ImportError as error:
        raise ImportError("GARCH baselines require `pip install -e '.[baselines]'`.") from error
    arguments = {
        "target_column": target_column,
        "origin_timestamp_column": origin_timestamp_column,
        "target_timestamp_column": target_timestamp_column,
        "label_available_timestamp_column": label_available_timestamp_column,
    }
    history_checked = _timestamped_targets(history, **arguments)
    test_checked = _timestamped_targets(test, **arguments)
    _check_rolling_schedule(
        test_checked,
        origin_timestamp_column=origin_timestamp_column,
        target_timestamp_column=target_timestamp_column,
        label_available_timestamp_column=label_available_timestamp_column,
    )
    first_origin = test_checked[origin_timestamp_column].iloc[0]
    known = _known_values(
        [history_checked],
        first_origin,
        target_column=target_column,
        label_available_timestamp_column=label_available_timestamp_column,
    ).to_numpy()
    fitted = arch_model(
        known * scale, mean="Zero", vol="GARCH", p=1, q=1, dist="normal"
    ).fit(disp="off")
    parameters = fitted.params
    omega = float(parameters["omega"])
    alpha = float(parameters["alpha[1]"])
    beta = float(parameters["beta[1]"])
    last_variance = float(np.asarray(fitted.conditional_volatility)[-1] ** 2)
    next_variance = omega + alpha * float(known[-1] * scale) ** 2 + beta * last_variance
    variances: list[float] = []
    for position, row in test_checked.iterrows():
        if position:
            prior = float(test_checked.loc[position - 1, target_column]) * scale
            next_variance = omega + alpha * prior**2 + beta * next_variance
        variances.append(max(next_variance / scale**2, np.finfo(float).tiny))
    predictions = make_prediction_frame(
        dataset=dataset,
        model=model_name,
        origin_timestamps=test_checked[origin_timestamp_column],
        target_timestamps=test_checked[target_timestamp_column],
        y_true=test_checked[target_column],
        mean=np.zeros(len(test_checked)),
        variance=variances,
    )
    return predictions, fitted


def rolling_ewma_forecast(
    history: pd.DataFrame,
    test: pd.DataFrame,
    *,
    decay: float = 0.94,
    target_column: str = "y_next",
    origin_timestamp_column: str = "origin_timestamp",
    target_timestamp_column: str = "target_timestamp",
    label_available_timestamp_column: str = "label_available_timestamp",
    dataset: str = "dataset",
    model_name: str = "EWMA-zero-mean",
) -> pd.DataFrame:
    """Issue zero-mean EWMA variance forecasts on the same rolling origins."""
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must lie strictly between zero and one")
    arguments = {
        "target_column": target_column,
        "origin_timestamp_column": origin_timestamp_column,
        "target_timestamp_column": target_timestamp_column,
        "label_available_timestamp_column": label_available_timestamp_column,
    }
    history_checked = _timestamped_targets(history, **arguments)
    test_checked = _timestamped_targets(test, **arguments)
    _check_rolling_schedule(
        test_checked,
        origin_timestamp_column=origin_timestamp_column,
        target_timestamp_column=target_timestamp_column,
        label_available_timestamp_column=label_available_timestamp_column,
    )
    first_origin = test_checked[origin_timestamp_column].iloc[0]
    known = _known_values(
        [history_checked],
        first_origin,
        target_column=target_column,
        label_available_timestamp_column=label_available_timestamp_column,
    ).to_numpy()
    variance = float(np.mean(np.square(known[: min(20, len(known))])))
    variance = max(variance, np.finfo(float).tiny)
    for observed in known:
        variance = decay * variance + (1.0 - decay) * float(observed) ** 2

    variances: list[float] = []
    for position, row in test_checked.iterrows():
        if position:
            prior = float(test_checked.loc[position - 1, target_column])
            variance = decay * variance + (1.0 - decay) * prior**2
        variances.append(max(variance, np.finfo(float).tiny))
    return make_prediction_frame(
        dataset=dataset,
        model=model_name,
        origin_timestamps=test_checked[origin_timestamp_column],
        target_timestamps=test_checked[target_timestamp_column],
        y_true=test_checked[target_column],
        mean=np.zeros(len(test_checked)),
        variance=variances,
    )
