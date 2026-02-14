# -*- coding: utf-8 -*-
"""Technical indicators: RSI, MACD, Moving Averages, Bollinger Bands."""

import numpy as np
import pandas as pd

from src.config import MA_WINDOWS, LAG_PERIODS, ROLLING_WINDOWS


# ==================== Basic Technical Indicators ====================

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)."""
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    delta = close.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    gains = up.ewm(com=period - 1, min_periods=period).mean()
    losses = down.abs().ewm(com=period - 1, min_periods=period).mean()
    rs = gains / losses
    return pd.Series(100 - (100 / (1 + rs)), name="rsi", index=df.index)


def calculate_macd(df: pd.DataFrame, short: int = 12, long: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD, signal, and histogram."""
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "signal": sig, "macdhist": hist}, index=df.index)


def calculate_moving_averages(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Calculate price/MA ratio for each window."""
    if windows is None:
        windows = MA_WINDOWS
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    result = pd.DataFrame(index=df.index)
    for w in windows:
        result[f"MA{w}"] = close / close.rolling(window=w).mean()
    return result


def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """Calculate Bollinger Band position (0-1 scale)."""
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    bb_position = (close - lower) / (upper - lower)
    bb_width = (upper - lower) / ma
    return pd.DataFrame({
        "bb_position": bb_position,
        "bb_width": bb_width,
    }, index=df.index)


# ==================== Feature Helpers ====================

def calculate_lag_features(series: pd.Series, name: str, lags: list[int] = None) -> pd.DataFrame:
    """Create lagged versions of a series."""
    if lags is None:
        lags = LAG_PERIODS
    result = pd.DataFrame(index=series.index)
    for lag in lags:
        result[f"{name}_lag{lag}"] = series.shift(lag)
    return result


def calculate_rolling_stats(series: pd.Series, name: str, windows: list[int] = None) -> pd.DataFrame:
    """Calculate rolling mean, std for a series."""
    if windows is None:
        windows = ROLLING_WINDOWS
    result = pd.DataFrame(index=series.index)
    for w in windows:
        result[f"{name}_rollmean{w}"] = series.rolling(w).mean()
        result[f"{name}_rollstd{w}"] = series.rolling(w).std()
    return result


def calculate_momentum(df: pd.DataFrame, periods: list[int] = None) -> pd.DataFrame:
    """Calculate price momentum (return over N days)."""
    if periods is None:
        periods = [1, 5, 10, 20]
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    result = pd.DataFrame(index=df.index)
    for p in periods:
        result[f"momentum_{p}d"] = close / close.shift(p) - 1
    return result
