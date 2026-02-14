# -*- coding: utf-8 -*-
"""Tests for SMA cache management."""

import io
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.cache import SMACache, MAX_ZIP_SIZE_BYTES, MAX_ENTRIES


@pytest.fixture
def tmp_cache_dir(tmp_path):
    return str(tmp_path / "cache")


@pytest.fixture
def sample_dataframes():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return {
        "sma15stock_df": pd.DataFrame({
            "Date": dates,
            "Adj Close": range(100, 110),
            "SMA15": range(95, 105),
            "SMA15_YN": [1] * 10,
            "Code": ["AAPL"] * 10,
        }),
        "sma30stock_df": pd.DataFrame({
            "Date": dates,
            "Adj Close": range(100, 110),
            "SMA30": range(90, 100),
            "SMA30_YN": [1] * 10,
            "Code": ["AAPL"] * 10,
        }),
    }


class TestSMACache:
    def test_save_and_load_roundtrip(self, tmp_cache_dir, sample_dataframes):
        cache = SMACache(cache_dir=tmp_cache_dir)
        path = cache.save(sample_dataframes, meta={"test": True})
        assert Path(path).exists()

        loaded_sma, loaded_meta = cache.load(path)
        assert "test" in loaded_meta
        assert "sma15stock_df" in loaded_sma
        assert "sma30stock_df" in loaded_sma
        assert len(loaded_sma["sma15stock_df"]) == 10

    def test_load_latest(self, tmp_cache_dir, sample_dataframes):
        cache = SMACache(cache_dir=tmp_cache_dir)
        cache.save(sample_dataframes)
        loaded_sma, _ = cache.load()
        assert len(loaded_sma) > 0

    def test_load_empty_cache(self, tmp_cache_dir):
        cache = SMACache(cache_dir=tmp_cache_dir)
        sma, meta = cache.load()
        assert sma == {}
        assert meta == {}

    def test_is_fresh(self, tmp_cache_dir, sample_dataframes):
        cache = SMACache(cache_dir=tmp_cache_dir)
        cache.save(sample_dataframes)
        assert cache.is_fresh(max_age_days=1) is True
        assert cache.is_fresh(max_age_days=0) is True

    def test_rejects_too_many_entries(self, tmp_cache_dir):
        """ZIP with too many entries should be rejected."""
        zip_path = Path(tmp_cache_dir)
        zip_path.mkdir(parents=True, exist_ok=True)
        bad_zip = zip_path / "sma_cache_bad.zip"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for i in range(MAX_ENTRIES + 1):
                zf.writestr(f"file_{i}.csv", "a,b\n1,2")
        bad_zip.write_bytes(buf.getvalue())

        cache = SMACache(cache_dir=tmp_cache_dir)
        with pytest.raises(ValueError, match="too many entries"):
            cache.load(str(bad_zip))
