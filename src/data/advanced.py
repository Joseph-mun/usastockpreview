# -*- coding: utf-8 -*-
"""Advanced technical indicators: ATR, Stochastic, ADX, OBV, ROC, Volatility, etc."""

import numpy as np
import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (volatility measure)."""
    high = df["High"]
    low = df["Low"]
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return pd.Series(atr, name="atr", index=df.index)


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator (%K, %D)."""
    high = df["High"]
    low = df["Low"]
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d}, index=df.index)


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (trend strength)."""
    high = df["High"]
    low = df["Low"]
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return pd.Series(adx, name="adx", index=df.index)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume."""
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    volume = df["Volume"]
    direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    obv = (volume * direction).cumsum()
    return pd.Series(obv, name="obv", index=df.index)


def calculate_roc(df: pd.DataFrame, periods: list[int] = None) -> pd.DataFrame:
    """Calculate Rate of Change."""
    if periods is None:
        periods = [5, 10, 20]
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    result = pd.DataFrame(index=df.index)
    for p in periods:
        result[f"roc_{p}d"] = 100 * (close - close.shift(p)) / close.shift(p)
    return result


def calculate_volatility(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Calculate historical volatility (annualized std of log returns)."""
    if windows is None:
        windows = [10, 20, 60]
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    log_ret = np.log(close / close.shift(1))
    result = pd.DataFrame(index=df.index)
    for w in windows:
        result[f"hvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)
    return result


def calculate_cross_asset_features(spy_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cross-asset derived features from already-joined data."""
    result = pd.DataFrame(index=spy_df.index)
    close = spy_df["Adj Close"] if "Adj Close" in spy_df.columns else spy_df["Close"]

    if "vix" in spy_df.columns:
        result["vix_price_ratio"] = spy_df["vix"] / (close / close.iloc[0] * 100)
        vix_ma20 = spy_df["vix"].rolling(20).mean()
        result["vix_vs_ma20"] = spy_df["vix"] / vix_ma20.replace(0, np.nan)

    if "T10Y2Y" in spy_df.columns:
        result["yield_spread_chg5"] = spy_df["T10Y2Y"] - spy_df["T10Y2Y"].shift(5)
        result["yield_spread_chg20"] = spy_df["T10Y2Y"] - spy_df["T10Y2Y"].shift(20)

    return result


def calculate_mean_reversion(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Calculate mean-reversion Z-score signals."""
    if windows is None:
        windows = [20, 60]
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    result = pd.DataFrame(index=df.index)
    for w in windows:
        ma = close.rolling(w).mean()
        std = close.rolling(w).std()
        result[f"zscore_{w}d"] = (close - ma) / std.replace(0, np.nan)
    return result
