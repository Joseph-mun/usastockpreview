# -*- coding: utf-8 -*-
"""Market data collection from FinanceDataReader, FRED, Wikipedia."""

import os
import io
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import requests

from src.config import (
    DATA_START_DATE, SMA_WINDOWS, BOND_CODES, BOND_CHANGE_PERIODS,
    VIX_CHANGE_PERIODS, INDEX_CONFIGS, get_logger,
    FEAR_GREED_ENABLED, FEAR_GREED_API_URL, FEAR_GREED_CHANGE_PERIODS,
    SECTOR_ROTATION_ENABLED, SECTOR_ETFS, SECTOR_LOOKBACK,
    ECON_CALENDAR_ENABLED, ECON_EVENT_WINDOW_DAYS,
)

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

# ==================== Proxy workaround ====================
# Adapted from stock_analysis_refactored.py L24-81
_BAD_PROXY_MARKERS = ("127.0.0.1:9", "localhost:9")


def _disable_bad_local_proxy():
    keys = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    )
    removed = False
    for k in keys:
        v = os.environ.get(k)
        if v and any(m in str(v) for m in _BAD_PROXY_MARKERS):
            os.environ.pop(k, None)
            removed = True
    if removed:
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"


_disable_bad_local_proxy()


# ==================== S&P 500 Tickers ====================
# Reused from stock_analysis_refactored.py L126-157

def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 ticker list from Wikipedia (fallback: FDR)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        for table in tables:
            if "Symbol" in table.columns and "Security" in table.columns:
                df = table.rename(columns={"Security": "Name"})
                df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
                return list(df.Symbol[:500])
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("Wikipedia S&P500 수집 실패, FDR로 대체 시도: %s", e)

    try:
        listing = fdr.StockListing("S&P500")
        if listing is not None and not listing.empty and "Symbol" in listing.columns:
            syms = listing["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
            return syms[:500]
    except (ValueError, KeyError, RuntimeError) as e:
        logger.warning("FDR S&P500 수집 실패: %s", e)

    raise ValueError("Could not load S&P 500 tickers (Wikipedia/FDR both failed)")


# ==================== Bond Data ====================
# Reused from stock_analysis_refactored.py L159-174

def get_bond_data(bond_code: str, start_date: str = DATA_START_DATE) -> pd.DataFrame:
    """Fetch bond yield data and compute change rates."""
    df = fdr.DataReader(f"FRED:{bond_code}", start=start_date, data_source="fred")
    df = df.reset_index().rename(columns={"DATE": "Date"})
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    for period in BOND_CHANGE_PERIODS:
        df[f"{bond_code}_{period}up"] = (
            (df[bond_code] - df[bond_code].shift(period)) / df[bond_code].shift(period)
        )

    for period in BOND_CHANGE_PERIODS:
        df = df[~df[f"{bond_code}_{period}up"].isnull()]

    return df[[f"{bond_code}_{period}up" for period in BOND_CHANGE_PERIODS]]


# ==================== VIX Data ====================
# Reused from stock_analysis_refactored.py L209-228

def get_vix_data(start_date: str = DATA_START_DATE, end_date: str | None = None) -> pd.DataFrame:
    """Fetch VIX data and compute change rates."""
    if end_date is None:
        end_date = datetime.now().date() + timedelta(days=1)

    df = fdr.DataReader("VIX", start_date, end_date)
    df = df.reset_index().rename(columns={"index": "Date"})
    df["Close"] = df["Adj Close"]

    for period in VIX_CHANGE_PERIODS:
        df[f"vix_flunc{period}"] = (
            100 * (df["Close"] - df["Close"].shift(period)) / df["Close"].shift(period)
        )

    df = df[["Date", "Close"] + [f"vix_flunc{period}" for period in VIX_CHANGE_PERIODS]]
    df = df.rename(columns={"Close": "vix"})
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df.ffill().bfill()

    return df[["vix"] + [f"vix_flunc{period}" for period in VIX_CHANGE_PERIODS]]


# ==================== Yield Spread ====================

def get_yield_spread(start_date: str = DATA_START_DATE) -> pd.DataFrame:
    """Fetch 10Y-2Y yield spread (T10Y2Y)."""
    try:
        k9 = fdr.DataReader("FRED:T10Y2Y", start=start_date, data_source="fred")
        df = k9.reset_index()
        if "DATE" in df.columns:
            df = df.rename(columns={"DATE": "Date"})
        df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
        df = df.ffill()
        return df
    except Exception as e:
        logger.warning("T10Y2Y 데이터 수집 실패: %s", e)
        return pd.DataFrame()


# ==================== Index Data ====================

def get_index_data(ticker: str, start_date: str = DATA_START_DATE, end_date: str | None = None) -> pd.DataFrame:
    """Fetch index OHLCV data. Tries multiple symbol candidates."""
    if end_date is None:
        end_date = datetime.now().date() + timedelta(days=1)

    # If ticker matches a known index config, try candidates
    candidates = [ticker]
    for cfg in INDEX_CONFIGS.values():
        if ticker in cfg["fdr_candidates"]:
            candidates = cfg["fdr_candidates"]
            break

    last_err = None
    for sym in candidates:
        try:
            df = fdr.DataReader(sym, start_date, end_date)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to fetch index data: {candidates}, error={last_err}")


# ==================== SPY ETF Data ====================

def get_spy_data(start_date: str = DATA_START_DATE, end_date: str | None = None) -> pd.DataFrame:
    """Fetch SPY ETF price data for portfolio backtest."""
    if end_date is None:
        end_date = datetime.now().date() + timedelta(days=1)

    df = fdr.DataReader("SPY", start_date, end_date)
    if df is None or df.empty:
        raise RuntimeError("Failed to fetch SPY data")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# ==================== SMA Data Collection ====================
# Adapted from stock_analysis_refactored.py L256-356

class SMACollector:
    """Collect SMA data for S&P 500 stocks and compute ratio time series."""

    def __init__(self, start_date: str = DATA_START_DATE, end_date: str | None = None):
        self.start_date = start_date
        self.end_date = end_date or (datetime.now().date() + timedelta(days=1))
        self.raw_dataframes: dict[str, pd.DataFrame] = {}

    def collect(
        self,
        ticker_list: list[str],
        max_tickers: int = 500,
        progress_callback=None,
        status_callback=None,
    ) -> dict[str, pd.DataFrame]:
        """
        Collect SMA data for given tickers.
        Returns raw dataframes dict (sma15stock_df, sma30stock_df, sma50stock_df).
        """
        if status_callback:
            status_callback("S&P500 SMA 데이터 수집 중...")

        window_rows: dict[int, list] = {k: [] for k in SMA_WINDOWS}
        total = min(len(ticker_list), max_tickers)

        for idx, code in enumerate(ticker_list[:max_tickers]):
            if progress_callback:
                progress_callback((idx + 1) / total)
            if status_callback:
                status_callback(f"수집 중: {code} ({idx + 1}/{total})")

            try:
                stock_df = fdr.DataReader(code, self.start_date, self.end_date)
                if stock_df is None or stock_df.empty:
                    continue
                stock_df = stock_df.sort_index()
                stock_df["Code"] = code

                for b in SMA_WINDOWS:
                    temp = stock_df[["Adj Close", "Code"]].copy()
                    temp[f"SMA{b}"] = temp["Adj Close"].rolling(window=b).mean()
                    temp[f"SMA{b}_YN"] = np.where(temp["Adj Close"] > temp[f"SMA{b}"], 1, 0)
                    temp = temp[~temp[f"SMA{b}"].isnull()]
                    if temp.empty:
                        continue
                    temp = temp.reset_index().rename(columns={"index": "Date"})
                    window_rows[b].append(temp)

            except Exception as e:
                if status_callback:
                    status_callback(f"  {code} 실패: {str(e)[:40]}")
                continue

        for k in SMA_WINDOWS:
            if window_rows[k]:
                self.raw_dataframes[f"sma{k}stock_df"] = pd.concat(window_rows[k], ignore_index=True)
            else:
                self.raw_dataframes[f"sma{k}stock_df"] = pd.DataFrame()

        if status_callback:
            status_callback("SMA 데이터 수집 완료")

        return self.raw_dataframes

    def compute_ratios(self, base_index: pd.DatetimeIndex = None) -> dict[str, pd.Series]:
        """
        Compute daily SMA ratio (fraction of stocks above SMA) from raw data.
        Returns dict: {"ratio_sma15": Series, "ratio_sma30": Series, "ratio_sma50": Series}
        """
        ratios = {}
        for b in SMA_WINDOWS:
            key = f"sma{b}stock_df"
            df = self.raw_dataframes.get(key)
            if df is None or df.empty:
                ratios[f"ratio_sma{b}"] = pd.Series(dtype=float)
                continue

            temp = df.copy()
            if "Date" in temp.columns:
                temp["Date"] = pd.to_datetime(temp["Date"])
                temp = temp.set_index("Date")
            else:
                temp.index = pd.to_datetime(temp.index)
            temp = temp.sort_index()

            ratio = (
                temp.groupby(level=0)[f"SMA{b}_YN"].sum()
                / temp.groupby(level=0)[f"SMA{b}_YN"].count()
            )
            ratio = ratio.sort_index()

            if base_index is not None:
                ratio = ratio.reindex(base_index).ffill().bfill()

            ratios[f"ratio_sma{b}"] = ratio

        return ratios


# ==================== Fear & Greed Index ====================

def get_fear_greed_index(start_date: str = DATA_START_DATE) -> pd.DataFrame:
    """Fetch Fear & Greed Index from Alternative.me (crypto-based market sentiment).

    Returns DataFrame with columns: fear_greed, fg_change5d, fg_change10d
    Highly correlated with overall market sentiment (0.7+).
    Returns empty DataFrame on error (no pipeline disruption).
    """
    try:
        resp = requests.get(FEAR_GREED_API_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = []
        for item in data.get("data", []):
            ts = int(item["timestamp"])
            date = pd.to_datetime(ts, unit="s").normalize()
            value = float(item["value"])
            records.append({"Date": date, "fear_greed": value})

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).drop_duplicates(subset="Date").set_index("Date").sort_index()
        df = df[df.index >= pd.to_datetime(start_date)]

        # Resample to business days and forward-fill gaps
        df = df.resample("B").ffill()

        # Compute change rates
        for period in FEAR_GREED_CHANGE_PERIODS:
            df[f"fg_change{period}d"] = df["fear_greed"].diff(period)

        return df.ffill().bfill()

    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("Fear & Greed Index 수집 실패: %s", e)
        return pd.DataFrame()


# ==================== Sector Rotation ====================

def get_sector_rotation(
    start_date: str = DATA_START_DATE,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Compute sector rotation features: relative strength of sector ETFs vs SPY.

    Returns DataFrame with features like tech_ret5d, tech_vs_spy5d, etc.
    """
    if end_date is None:
        end_date = datetime.now().date() + timedelta(days=1)

    try:
        spy_df = fdr.DataReader("SPY", start_date, end_date)
        if spy_df is None or spy_df.empty:
            return pd.DataFrame()
        spy_close = spy_df["Adj Close"] if "Adj Close" in spy_df.columns else spy_df["Close"]
        spy_close.index = pd.to_datetime(spy_close.index)

        features = pd.DataFrame(index=spy_close.index)

        for etf_code, sector_name in SECTOR_ETFS.items():
            try:
                etf_df = fdr.DataReader(etf_code, start_date, end_date)
                if etf_df is None or etf_df.empty:
                    continue
                etf_close = etf_df["Adj Close"] if "Adj Close" in etf_df.columns else etf_df["Close"]
                etf_close.index = pd.to_datetime(etf_close.index)
                etf_aligned = etf_close.reindex(spy_close.index).ffill()

                for period in SECTOR_LOOKBACK:
                    sector_ret = etf_aligned.pct_change(period)
                    spy_ret = spy_close.pct_change(period)
                    features[f"{sector_name}_ret{period}d"] = sector_ret
                    features[f"{sector_name}_vs_spy{period}d"] = sector_ret - spy_ret

            except (ValueError, KeyError, RuntimeError) as e:
                logger.warning("섹터 ETF %s 수집 실패: %s", etf_code, e)
                continue

        return features.ffill().bfill().fillna(0)

    except (ValueError, KeyError, RuntimeError) as e:
        logger.warning("섹터 로테이션 데이터 수집 실패: %s", e)
        return pd.DataFrame()


# ==================== Economic Calendar ====================

# FOMC meeting dates (public, scheduled 8 times per year)
_FOMC_DATES = [
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
]


def get_economic_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate economic event proximity features.

    Returns DataFrame with columns:
      - days_to_fomc: business days until next FOMC meeting
      - is_fomc_week: 1 if within ±ECON_EVENT_WINDOW_DAYS of FOMC, else 0
      - is_month_start: 1 if day <= 7 (NFP proxy, released first Friday)
      - is_mid_month: 1 if 10 <= day <= 15 (CPI proxy, released mid-month)
    """
    fomc_dates = pd.to_datetime(_FOMC_DATES)

    records = []
    for date in index:
        # Days to next FOMC
        future_fomc = fomc_dates[fomc_dates >= date]
        days_to_fomc = int((future_fomc[0] - date).days) if len(future_fomc) > 0 else 90

        # FOMC week flag
        is_fomc_week = int(any(
            abs((d - date).days) <= ECON_EVENT_WINDOW_DAYS for d in fomc_dates
        ))

        records.append({
            "Date": date,
            "days_to_fomc": min(days_to_fomc, 90),  # cap at 90 days
            "is_fomc_week": is_fomc_week,
            "is_month_start": int(date.day <= 7),
            "is_mid_month": int(10 <= date.day <= 15),
        })

    return pd.DataFrame(records).set_index("Date")


# ==================== Put/Call Ratio Proxy ====================

def get_put_call_proxy(vix_series: pd.Series) -> pd.DataFrame:
    """Compute a Put/Call ratio proxy from VIX data.

    Maps VIX range [10, 40] to approximate P/C ratio [0.5, 1.5].
    No external API needed - purely derived from existing VIX data.
    """
    pc_ratio = 0.5 + (vix_series - 10) / 30
    pc_ratio = pc_ratio.clip(0.3, 2.0)

    df = pd.DataFrame({"put_call_ratio": pc_ratio}, index=vix_series.index)
    df["pc_ma5"] = df["put_call_ratio"].rolling(5, min_periods=1).mean()
    df["pc_ma20"] = df["put_call_ratio"].rolling(20, min_periods=1).mean()
    df["pc_vs_ma20"] = df["put_call_ratio"] / df["pc_ma20"]

    return df.ffill().bfill()
