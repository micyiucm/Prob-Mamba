"""Deprecated compatibility layer; import from :mod:`prob_mamba.features` instead."""

from prob_mamba.features import compute_vif, momentum_columns, select_vif_features, shift_leak_prone_columns, split_by_dates, vif_iterative_prune


def shift_series(df, date_col="Date", extra_shift=None):
    return shift_leak_prone_columns(df, extra_shift or ())
