# -*- coding: utf-8 -*-
"""Feature engineering: dataset builder and target variable construction.

Indicator functions are split into:
  - indicators.py: basic indicators (RSI, MACD, MA, Bollinger, momentum, lag, rolling)
  - advanced.py: advanced indicators (ATR, Stochastic, ADX, OBV, ROC, volatility, cross-asset)
"""

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from src.config import (
    DATA_START_DATE,
    TARGET_LOOKAHEAD_DAYS, TARGET_UP_THRESHOLD, TARGET_DOWN_THRESHOLD,
    LEAK_COLUMNS, BASE_PRICE_COLUMNS, TRAIN_WINDOW_YEARS,
    TARGET_MODE, TARGET_ROLLING_MEDIAN_WINDOW,
    FEAR_GREED_ENABLED, SECTOR_ROTATION_ENABLED, ECON_CALENDAR_ENABLED,
)
from src.data.collectors import (
    get_index_data, get_bond_data, get_vix_data, get_yield_spread,
    get_fear_greed_index, get_sector_rotation, get_economic_calendar,
    get_put_call_proxy,
)

# Re-export indicator functions for backward compatibility
from src.data.indicators import (  # noqa: F401
    calculate_rsi,
    calculate_macd,
    calculate_moving_averages,
    calculate_bollinger_bands,
    calculate_lag_features,
    calculate_rolling_stats,
    calculate_momentum,
)
from src.data import get_close_col
from src.data.advanced import (  # noqa: F401
    calculate_atr,
    calculate_stochastic,
    calculate_adx,
    calculate_obv,
    calculate_roc,
    calculate_volatility,
    calculate_cross_asset_features,
    calculate_mean_reversion,
)


# ==================== Target Variable ====================

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build target variables with unified lookahead.

    TARGET_MODE controls target definition:
      - "raw": original binary (price up in 20d, base rate ~67%)
      - "excess_return": above rolling median 20d return (base rate ~50%)
    """
    spy = df.copy()
    close = spy[get_close_col(spy)]

    # Future max/min over lookahead window (reverse-rolling trick)
    spy_rev = spy.sort_index(ascending=False)
    close_rev = close.sort_index(ascending=False)
    spy_rev["after2"] = close_rev.rolling(window=TARGET_LOOKAHEAD_DAYS).max()
    spy_rev["after2_low"] = close_rev.rolling(window=TARGET_LOOKAHEAD_DAYS).min()
    spy = spy_rev.sort_index(ascending=True)

    # After price (unified to lookahead days)
    spy["after"] = close.shift(-TARGET_LOOKAHEAD_DAYS)

    # Return rate
    spy["suik_rate"] = 100 * (spy["after"] - close) / close

    # Binary targets
    if TARGET_MODE == "excess_return":
        future_ret = close.shift(-TARGET_LOOKAHEAD_DAYS) / close - 1
        past_ret = close.pct_change(TARGET_LOOKAHEAD_DAYS)
        rolling_med = past_ret.rolling(TARGET_ROLLING_MEDIAN_WINDOW).median()
        spy["Target"] = np.where(future_ret > rolling_med, 1, 0)
    else:
        spy["Target"] = np.where(
            spy["after"] > TARGET_UP_THRESHOLD * close, 1, 0
        )

    spy["TargetDown"] = np.where(
        spy["after2_low"] < TARGET_DOWN_THRESHOLD * close, 1, 0
    )

    return spy


# ==================== Full Dataset Builder ====================

class DatasetBuilder:
    """Build complete feature matrix for a given index."""

    def __init__(self, sma_ratios: dict[str, pd.Series] = None):
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
        end_date = None

        # 1. Get index price data
        raw_df = get_index_data(index_ticker, start_date, end_date)
        raw_df.index = pd.to_datetime(raw_df.index)
        raw_df = raw_df.sort_index()

        spy = raw_df.copy()
        close_col = get_close_col(spy)
        spy["Close"] = spy[close_col]
        if "Adj Close" not in spy.columns:
            spy["Adj Close"] = spy["Close"]
        join_how = "left" if for_prediction else "inner"

        # 2. Build target
        spy = build_target(spy)
        if not for_prediction:
            spy = spy[~spy["after"].isnull()]

        # 3. Change20day
        k1 = raw_df.copy()
        k1_close = k1[get_close_col(k1)]
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
                spy[col_name] = 0.5

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

        # 7. Technical indicators
        index_df = raw_df.copy()
        spy = self._add_indicators(spy, index_df, join_how, for_prediction)

        # 8. Yield spread
        ys_df = get_yield_spread()
        if not ys_df.empty:
            spy = spy.join(ys_df, how=join_how)
            if for_prediction:
                for c in ys_df.columns:
                    spy[c] = spy[c].ffill().bfill()

        # 9. Fear & Greed Index
        if FEAR_GREED_ENABLED:
            fg_df = get_fear_greed_index(start_date)
            if not fg_df.empty:
                spy = spy.join(fg_df, how=join_how)
                if for_prediction:
                    for c in fg_df.columns:
                        spy[c] = spy[c].ffill().bfill()

        # 10. Sector Rotation
        if SECTOR_ROTATION_ENABLED:
            sector_df = get_sector_rotation(start_date)
            if not sector_df.empty:
                spy = spy.join(sector_df, how=join_how)
                if for_prediction:
                    for c in sector_df.columns:
                        spy[c] = spy[c].ffill().bfill()

        # 11. Economic Calendar
        if ECON_CALENDAR_ENABLED:
            econ_df = get_economic_calendar(spy.index)
            if not econ_df.empty:
                spy = spy.join(econ_df, how=join_how)

        # 12. Put/Call Ratio Proxy (derived from VIX already in spy)
        if "vix" in spy.columns:
            pc_df = get_put_call_proxy(spy["vix"])
            if not pc_df.empty:
                spy = spy.join(pc_df, how=join_how)

        # 13. Derived features
        spy = self._add_derived_features(spy, index_df, join_how, for_prediction)

        # Final NaN fill for prediction
        if for_prediction:
            fill_exclude = {"after", "after2", "after2_low", "Target", "TargetDown", "suik_rate"}
            fill_cols = [c for c in spy.columns if c not in fill_exclude]
            spy.loc[:, fill_cols] = spy.loc[:, fill_cols].ffill().bfill()

        # Build feature matrix
        X = _build_feature_matrix(spy)
        y = spy["Target"] if not for_prediction else None

        # Training window filter
        if not for_prediction and TRAIN_WINDOW_YEARS:
            cutoff = datetime.now() - timedelta(days=TRAIN_WINDOW_YEARS * 365)
            mask = X.index >= cutoff
            X = X[mask]
            if y is not None:
                y = y[mask]
            spy = spy.loc[X.index]

        return X, spy, y

    def _add_indicators(self, spy, index_df, join_how, for_prediction):
        """Add technical indicator features."""
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

        # Bollinger Bands
        bb_df = calculate_bollinger_bands(index_df)
        spy = spy.join(bb_df, how=join_how)
        if for_prediction:
            for c in bb_df.columns:
                spy[c] = spy[c].ffill().bfill()

        # Momentum
        mom_df = calculate_momentum(index_df)
        spy = spy.join(mom_df, how=join_how)
        if for_prediction:
            for c in mom_df.columns:
                spy[c] = spy[c].ffill().bfill()

        return spy

    def _add_derived_features(self, spy, index_df, join_how, for_prediction):
        """Add advanced and derived features."""
        # ATR
        atr = calculate_atr(index_df)
        spy["atr"] = atr.reindex(spy.index)
        close_for_norm = spy[get_close_col(spy)]
        spy["atr_pct"] = spy["atr"] / close_for_norm
        if for_prediction:
            spy["atr"] = spy["atr"].ffill().bfill()
            spy["atr_pct"] = spy["atr_pct"].ffill().bfill()

        # Stochastic
        stoch_df = calculate_stochastic(index_df)
        spy = spy.join(stoch_df, how=join_how)
        if for_prediction:
            for c in stoch_df.columns:
                spy[c] = spy[c].ffill().bfill()

        # ADX
        adx = calculate_adx(index_df)
        spy["adx"] = adx.reindex(spy.index)
        if for_prediction:
            spy["adx"] = spy["adx"].ffill().bfill()

        # OBV
        obv = calculate_obv(index_df)
        spy["obv_roc5"] = obv.pct_change(5).reindex(spy.index)
        spy["obv_roc20"] = obv.pct_change(20).reindex(spy.index)
        if for_prediction:
            spy["obv_roc5"] = spy["obv_roc5"].ffill().bfill()
            spy["obv_roc20"] = spy["obv_roc20"].ffill().bfill()

        # Rate of Change
        roc_df = calculate_roc(index_df)
        spy = spy.join(roc_df, how=join_how)
        if for_prediction:
            for c in roc_df.columns:
                spy[c] = spy[c].ffill().bfill()

        # Historical Volatility
        vol_df = calculate_volatility(index_df)
        spy = spy.join(vol_df, how=join_how)
        if for_prediction:
            for c in vol_df.columns:
                spy[c] = spy[c].ffill().bfill()

        # Mean Reversion Z-score
        zs_df = calculate_mean_reversion(index_df)
        spy = spy.join(zs_df, how=join_how)
        if for_prediction:
            for c in zs_df.columns:
                spy[c] = spy[c].ffill().bfill()

        # Cross-asset features
        cross_df = calculate_cross_asset_features(spy)
        for c in cross_df.columns:
            spy[c] = cross_df[c]
        if for_prediction:
            for c in cross_df.columns:
                spy[c] = spy[c].ffill().bfill()

        # Lag features
        for col in ["rsi", "vix", "Change20day", "adx", "atr_pct"]:
            if col in spy.columns:
                lag_df = calculate_lag_features(spy[col], col)
                for lc in lag_df.columns:
                    spy[lc] = lag_df[lc]

        # Rolling stats
        close = spy[get_close_col(spy)]
        returns = close.pct_change()
        roll_df = calculate_rolling_stats(returns, "ret")
        for rc in roll_df.columns:
            spy[rc] = roll_df[rc]

        # SMA ratio lags
        for col in ["ratio_sma15", "ratio_sma30", "ratio_sma50"]:
            if col in spy.columns:
                lag_df = calculate_lag_features(spy[col], col, lags=[1, 5, 10])
                for lc in lag_df.columns:
                    spy[lc] = lag_df[lc]

        # Volume features
        if "Volume" in spy.columns:
            vol_series = spy["Volume"].astype(float)
            spy["vol_roc5"] = vol_series / vol_series.rolling(5).mean()
            spy["vol_roc20"] = vol_series / vol_series.rolling(20).mean()
            if for_prediction:
                spy["vol_roc5"] = spy["vol_roc5"].ffill().bfill()
                spy["vol_roc20"] = spy["vol_roc20"].ffill().bfill()

        # Momentum percentile
        close_feat = spy[get_close_col(spy)]
        for p in [10, 20]:
            mom_col = f"momentum_{p}d"
            if mom_col in spy.columns:
                spy[f"mom{p}_pctile"] = spy[mom_col].rolling(252, min_periods=60).rank(pct=True)
                if for_prediction:
                    spy[f"mom{p}_pctile"] = spy[f"mom{p}_pctile"].ffill().bfill()

        # Rolling median return
        past_ret_feat = close_feat.pct_change(TARGET_LOOKAHEAD_DAYS)
        spy["rolling_med_ret"] = past_ret_feat.rolling(TARGET_ROLLING_MEDIAN_WINDOW, min_periods=60).median()
        if for_prediction:
            spy["rolling_med_ret"] = spy["rolling_med_ret"].ffill().bfill()

        return spy


def _build_feature_matrix(spy: pd.DataFrame) -> pd.DataFrame:
    """Extract feature columns, removing leakage and base price columns."""
    X = spy.copy()
    drop_cols = [c for c in X.columns if c in LEAK_COLUMNS or c in BASE_PRICE_COLUMNS]
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

    for c in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.fillna(0)
    X = X.drop(columns=[c for c in LEAK_COLUMNS if c in X.columns], errors="ignore")
    return X
