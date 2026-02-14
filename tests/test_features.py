# -*- coding: utf-8 -*-
"""Tests for feature engineering functions."""

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    calculate_rsi,
    calculate_macd,
    calculate_moving_averages,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_momentum,
    calculate_lag_features,
    calculate_rolling_stats,
    build_target,
)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.2,
        "High": close + abs(np.random.randn(n) * 0.5),
        "Low": close - abs(np.random.randn(n) * 0.5),
        "Close": close,
        "Adj Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


class TestRSI:
    def test_rsi_range(self, sample_ohlcv):
        rsi = calculate_rsi(sample_ohlcv)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_length(self, sample_ohlcv):
        rsi = calculate_rsi(sample_ohlcv)
        assert len(rsi) == len(sample_ohlcv)


class TestMACD:
    def test_macd_columns(self, sample_ohlcv):
        result = calculate_macd(sample_ohlcv)
        assert set(result.columns) == {"macd", "signal", "macdhist"}

    def test_histogram_equals_diff(self, sample_ohlcv):
        result = calculate_macd(sample_ohlcv)
        valid = result.dropna()
        diff = (valid["macd"] - valid["signal"]).round(10)
        hist = valid["macdhist"].round(10)
        assert np.allclose(diff, hist, atol=1e-8)


class TestMovingAverages:
    def test_default_windows(self, sample_ohlcv):
        result = calculate_moving_averages(sample_ohlcv)
        assert all(f"MA{w}" in result.columns for w in [5, 20, 60, 120, 200])

    def test_custom_windows(self, sample_ohlcv):
        result = calculate_moving_averages(sample_ohlcv, windows=[10, 50])
        assert set(result.columns) == {"MA10", "MA50"}


class TestBollingerBands:
    def test_columns(self, sample_ohlcv):
        result = calculate_bollinger_bands(sample_ohlcv)
        assert set(result.columns) == {"bb_position", "bb_width"}

    def test_bb_width_positive(self, sample_ohlcv):
        result = calculate_bollinger_bands(sample_ohlcv)
        valid = result["bb_width"].dropna()
        assert (valid >= 0).all()


class TestATR:
    def test_atr_positive(self, sample_ohlcv):
        atr = calculate_atr(sample_ohlcv)
        valid = atr.dropna()
        assert (valid >= 0).all()


class TestMomentum:
    def test_columns(self, sample_ohlcv):
        result = calculate_momentum(sample_ohlcv)
        assert all(f"momentum_{p}d" in result.columns for p in [1, 5, 10, 20])


class TestLagFeatures:
    def test_lag_columns(self):
        series = pd.Series(range(20), name="test")
        result = calculate_lag_features(series, "test", lags=[1, 5])
        assert "test_lag1" in result.columns
        assert "test_lag5" in result.columns

    def test_lag_values(self):
        series = pd.Series([10, 20, 30, 40, 50])
        result = calculate_lag_features(series, "x", lags=[1])
        assert result["x_lag1"].iloc[1] == 10
        assert pd.isna(result["x_lag1"].iloc[0])


class TestRollingStats:
    def test_columns(self):
        series = pd.Series(range(100))
        result = calculate_rolling_stats(series, "ret", windows=[5, 10])
        expected = {"ret_rollmean5", "ret_rollstd5", "ret_rollmean10", "ret_rollstd10"}
        assert expected == set(result.columns)


class TestBuildTarget:
    def test_target_columns(self, sample_ohlcv):
        result = build_target(sample_ohlcv)
        assert "Target" in result.columns
        assert "TargetDown" in result.columns

    def test_target_binary(self, sample_ohlcv):
        result = build_target(sample_ohlcv)
        valid = result["Target"].dropna()
        assert set(valid.unique()).issubset({0, 1})
