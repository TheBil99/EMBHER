"""Shared table reading for the embher commands."""
from __future__ import annotations

import pandas as pd


def read_table(path):
    """A donor-indexed table: parquet, .csv or tab-separated text; first column (or the parquet
    index) is the donor id, read as str."""
    path = str(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        if df.index.dtype == "int64" and df.index.name is None and not df.columns.empty:
            df = df.set_index(df.columns[0])
    else:
        df = pd.read_csv(path, sep="," if path.endswith(".csv") else "\t")
        df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    return df[~df.index.duplicated(keep="first")]
