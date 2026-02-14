# -*- coding: utf-8 -*-
"""Data package utilities."""

import pandas as pd


def get_close_col(df: pd.DataFrame) -> str:
    """Return the best close price column name available in the DataFrame."""
    return "Adj Close" if "Adj Close" in df.columns else "Close"