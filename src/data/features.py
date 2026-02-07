# -*- coding: utf-8 -*-
"""Feature engineering: technical indicators, lag features, rolling stats."""

import numpy as np
import pandas as pd

from src.config import (
    DATA_START_DATE, MA_WINDOWS, LAG_PERIODS, ROLLING_WINDOWS,
    TARGET_LOOKAHEAD_DAYS, TARGET_UP_THRESHOLD, TARGET_DOWN_THRESHOLD,
    LEAK_COLUMNS, BASE_PRICE_COLUMNS,
)
from src.data.collectors import (
    get_index_data, get_bond_data, get_vix_data, get_yield_spread,
)


# ==================== Technical Indicators ====================
# Reused from stock_analysis_refactored.py L176-207

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


# ==================== New Features ====================

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


def calculate_lag_features(series: pd.Series, name: str, lags: list[int] = None) -> pd.DataFrame:
    """Create lagged versions of a series."""
    if lags is None:
        lags = LAG_PERIODS
    result = pd.DataFrame(index=series.index)
    for lag in lags:
        result[f"{name}_lag{lag}"] = series.shift(lag)
    return result


def calculate_rolling_stats(series: pd.Series, name: str, windows: list[int] = None) -> pd.DataFrame:
    """Calculate rolling mean, std, skew for a series."""
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


# ==================== Target Variable ====================

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build target variables with unified lookahead.
    Fixed: previously after=shift(-15) and after2=rolling(20) were inconsistent.
    Now both use TARGET_LOOKAHEAD_DAYS=20.
    """
    spy = df.copy()
    close = spy["Adj Close"] if "Adj Close" in spy.columns else spy["Close"]

    # Future max/min over lookahead window (reverse-rolling trick)
    spy_rev = spy.sort_index(ascending=False)
    close_rev = close.sort_index(ascending=False)
    spy_rev["after2"] = close_rev.rolling(window=TARGET_LOOKAHEAD_DAYS).max()
    spy_rev["after2_low"] = close_rev.rolling(window=TARGET_LOOKAHEAD_DAYS).min()
    spy = spy_rev.sort_index(ascending=True)

    # After price (unified to lookahead days)
    spy["after"] = close.shift(-TARGET_LOOKAHEAD_DAYS)

    # Binary targets
    spy["Target"] = np.where(spy["after2"] >= TARGET_UP_THRESHOLD * close, 1, 0)
    spy["TargetDown"] = np.where(spy["after2_low"] <= TARGET_DOWN_THRESHOLD * close, 1, 0)

    # Return rate
    spy["suik_rate"] = 100 * (spy["after"] - close) / close

    return spy


# ==================== Full Dataset Builder ====================

class DatasetBuilder:
    """Build complete feature matrix for a given index."""

    def __init__(self, sma_ratios: dict[str, pd.Series] = None):
        """
        sma_ratios: pre-computed SMA ratios from SMACollector.compute_ratios()
                    {"ratio_sma15": Series, "ratio_sma30": Series, "ratio_sma50": Series}
        """
        self.sma_ratios = sma_ratios or {}

    def build(
        self,
        index_ticker: str,
        for_prediction: bool = False,
        start_date: str = DATA_START_DATE,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Build full dataset for one index.

        Returns:
            X: feature matrix (no leakage columns)
            spy: full dataframe with all columns
            y: target series (None if for_prediction)
        """
        end_date = None  # defaults to today+1 in get_index_data

        # 1. Get index price data (SINGLE fetch, reused below)
        raw_df = get_index_data(index_ticker, start_date, end_date)
        raw_df.index = pd.to_datetime(raw_df.index)
        raw_df = raw_df.sort_index()

        spy = raw_df.copy()
        close_col = "Adj Close" if "Adj Close" in spy.columns else "Close"
        spy["Close"] = spy[close_col]
        if "Adj Close" not in spy.columns:
            spy["Adj Close"] = spy["Close"]
        base_index = spy.index
        join_how = "left" if for_prediction else "inner"

        # 2. Build target
        spy = build_target(spy)
        if not for_prediction:
            spy = spy[~spy["after"].isnull()]

        # 3. Change20day (20-day cumulative return) - reuse raw_df
        k1 = raw_df.copy()
        k1_close = k1["Adj Close"] if "Adj Close" in k1.columns else k1["Close"]
        k1["Change"] = k1_close.pct_change()
        k1 = k1[~k1["Change"].isnull()]
        k2 = np.log10(1 + k1["Change"]).rolling(window=20).sum()
        k1["Change20day"] = (pow(10, k2) - 1) * 100
        spy = spy.join(k1[["Change20day"]], how=join_how)
        if for_prediction:
            spy["Change20day"] = spy["Change20day"].ffill().bfill()
        else:
            spy = spy[~spy["Change20day"].isnull()]

        # 4. SMA ratios
        for col_name, ratio_series in self.sma_ratios.items():
            if ratio_series is not None and not ratio_series.empty:
                ratio_aligned = ratio_series.reindex(spy.index)
                if for_prediction:
                    ratio_aligned = ratio_aligned.ffill().bfill()
                spy[col_name] = ratio_aligned
                if not for_prediction:
                    spy = spy[~spy[col_name].isnull()]
            else:
                spy[col_name] = 0.5  # neutral default

        # 5. Bond data
        for bond_code in ["DGS2", "DGS10", "DGS30"]:
            bond_df = get_bond_data(bond_code)
            bond_df.index = pd.to_datetime(bond_df.index)
            spy = spy.join(bond_df, how=join_how)
            if for_prediction:
                for c in bond_df.columns:
                    spy[c] = spy[c].ffill().bfill()

        # 6. VIX data
        vix_df = get_vix_data(start_date)
        vix_df.index = pd.to_datetime(vix_df.index)
        spy = spy.join(vix_df, how=join_how)
        if for_prediction:
            for c in vix_df.columns:
                spy[c] = spy[c].ffill().bfill()

        # 7. Technical indicators from index (reuse raw_df)
        index_df = raw_df.copy()

        # MACD
        macd_df = calculate_macd(index_df)
        spy = spy.join(macd_df, how=join_how)
        if for_prediction:
            for c in macd_df.columns:
                spy[c] = spy[c].ffill().bfill()

        # RSI
        rsi = calculate_rsi(index_df)
        spy["rsi"] = rsi.reindex(spy.index)
        if for_prediction:
            spy["rsi"] = spy["rsi"].ffill().bfill()
        else:
            spy = spy[~spy["rsi"].isnull()]

        # Moving Averages
        ma_df = calculate_moving_averages(index_df)
        spy = spy.join(ma_df, how=join_how)
        if for_prediction:
            for c in ma_df.columns:
                spy[c] = spy[c].ffill().bfill()

        # Yield spread
        ys_df = get_yield_spread()
        if not ys_df.empty:
            spy = spy.join(ys_df, how=join_how)
            if for_prediction:
                for c in ys_df.columns:
                    spy[c] = spy[c].ffill().bfill()

        # 8. Bollinger Bands (NEW)
        bb_df = calculate_bollinger_bands(index_df)
        spy = spy.join(bb_df, how=join_how)
        if for_prediction:
            for c in bb_df.columns:
                spy[c] = spy[c].ffill().bfill()

        # 9. Momentum (NEW)
        mom_df = calculate_momentum(index_df)
        spy = spy.join(mom_df, how=join_how)
        if for_prediction:
            for c in mom_df.columns:
                spy[c] = spy[c].ffill().bfill()

        # 10. Lag features (NEW) - for key indicators
        for col in ["rsi", "vix", "Change20day"]:
            if col in spy.columns:
                lag_df = calculate_lag_features(spy[col], col)
                for lc in lag_df.columns:
                    spy[lc] = lag_df[lc]

        # 11. Rolling stats (NEW) - for returns
        close = spy["Adj Close"] if "Adj Close" in spy.columns else spy["Close"]
        returns = close.pct_change()
        roll_df = calculate_rolling_stats(returns, "ret")
        for rc in roll_df.columns:
            spy[rc] = roll_df[rc]

        # 12. SMA ratio lag features (NEW)
        for col in ["ratio_sma15", "ratio_sma30", "ratio_sma50"]:
            if col in spy.columns:
                lag_df = calculate_lag_features(spy[col], col, lags=[1, 5, 10])
                for lc in lag_df.columns:
                    spy[lc] = lag_df[lc]

        # Final: fill remaining NaN for prediction
        if for_prediction:
            fill_exclude = {"after", "after2", "after2_low", "Target", "TargetDown", "suik_rate"}
            fill_cols = [c for c in spy.columns if c not in fill_exclude]
            spy.loc[:, fill_cols] = spy.loc[:, fill_cols].ffill().bfill()

        # Build feature matrix (no leakage)
        X = _build_feature_matrix(spy)
        y = spy["Target"] if not for_prediction else None

        return X, spy, y


def _build_feature_matrix(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Extract feature columns, removing leakage and base price columns.
    Adapted from stock_analysis_refactored.py L85-124.
    """
    X = spy.copy()
    drop_cols = [c for c in X.columns if c in LEAK_COLUMNS or c in BASE_PRICE_COLUMNS]
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

    for c in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.fillna(0)
    # Final safety check
    X = X.drop(columns=[c for c in LEAK_COLUMNS if c in X.columns], errors="ignore")
    return X
