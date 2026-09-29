import numpy as np
import pandas as pd
import pytest

from prob_mamba.data import clean_equity_features, preprocess_frame
from prob_mamba.datasets import (
    create_causal_windows,
    create_seq2seq_targets,
    create_sequences,
    make_loaders,
)
from prob_mamba.features import select_vif_features, shift_leak_prone_columns


def test_equity_cleaning_shifts_leaky_columns_without_mutating_input():
    raw = pd.DataFrame(
        {
            "Date": ["2020-01-02", "2020-01-01"],
            "ABC-F": [20.0, 10.0],
            "FTSE": [40.0, 30.0],
            "Name": ["x", "x"],
            "mom": [1.0, 2.0],
        }
    )
    cleaned = clean_equity_features(raw)
    assert "Name" not in cleaned and "mom" not in cleaned
    assert pd.isna(cleaned.loc[0, "ABC-F"])
    assert raw.loc[0, "ABC-F"] == 20.0


def test_preprocess_frame_is_causal_and_scales_using_training_only():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=14),
            "Price": np.linspace(10.0, 23.0, 14),
            "signal": [np.nan, *range(1, 14)],
            "unused": np.nan,
        }
    )
    split, _ = preprocess_frame(frame, "2020-01-07", "2020-01-10", "Date", "Price")
    assert split.train["signal"].mean() == pytest.approx(0.0)
    assert "unused" not in split.train
    assert len(split.validation) == 3
    assert len(split.test) == 4
    assert split.train["target_timestamp"].max() <= pd.Timestamp("2020-01-07")
    assert split.validation["label_available_timestamp"].max() <= pd.Timestamp("2020-01-10")
    assert split.test["target_timestamp"].min() > pd.Timestamp("2020-01-10")
    assert len(split.chronological()) == 12


def test_sequence_builders_preserve_time_alignment():
    frame = pd.DataFrame({"feature": range(5), "target": range(10, 15)})
    inputs, labels = create_sequences(frame, 3, ["feature"], "target")
    assert inputs.shape == (3, 3, 1)
    np.testing.assert_array_equal(labels, [12, 13, 14])
    targets = create_seq2seq_targets(frame.rename(columns={"target": "y_next"}), 3)
    assert targets.shape == (3, 3, 1)
    train, validation, test = make_loaders((inputs, labels), (inputs, labels), (inputs, labels), 2)
    assert len(train.dataset) == len(validation.dataset) == len(test.dataset) == 3


def test_feature_selection_is_explicit_and_case_sensitive():
    frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=2), "Price": [1, 2], "mom": [1, 2], "x": [3, 4]})
    selected, excluded = select_vif_features(frame)
    assert selected == ["x"]
    assert set(excluded) == {"Price", "mom"}
    shifted = shift_leak_prone_columns(frame.assign(**{"S&P-F": [1.0, 2.0]}))
    assert pd.isna(shifted.loc[0, "S&P-F"])


def test_causal_windows_use_preperiod_context_and_score_every_requested_target():
    origins = pd.date_range("2020-01-01", periods=8)
    frame = pd.DataFrame(
        {
            "origin_timestamp": origins,
            "target_timestamp": origins + pd.Timedelta(days=1),
            "label_available_timestamp": origins + pd.Timedelta(days=1),
            "feature": np.arange(8, dtype=float),
            "y_next": np.arange(10, 18, dtype=float),
        }
    )
    windows = create_causal_windows(
        frame,
        3,
        ["feature"],
        score_start="2020-01-06",
        score_end="2020-01-09",
    )
    assert len(windows) == 4
    np.testing.assert_array_equal(windows.inputs[0, :, 0], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(windows.labels, [14.0, 15.0, 16.0, 17.0])
    np.testing.assert_array_equal(windows.target_sequences[:, -1, 0], windows.labels)
    np.testing.assert_array_equal(windows.observed_mask[0], [True, True, False])
    assert windows.origin_timestamps[0] == np.datetime64("2020-01-05")
    assert windows.target_timestamps[0] == np.datetime64("2020-01-06")


def test_causal_windows_refuse_to_silently_drop_requested_targets():
    origins = pd.date_range("2020-01-01", periods=4)
    frame = pd.DataFrame(
        {
            "origin_timestamp": origins,
            "target_timestamp": origins + pd.Timedelta(days=1),
            "label_available_timestamp": origins + pd.Timedelta(days=1),
            "feature": range(4),
            "y_next": range(4),
        }
    )
    with pytest.raises(ValueError, match="insufficient pre-period history"):
        create_causal_windows(frame, 3, ["feature"], score_start="2020-01-02")
