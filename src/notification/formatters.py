# -*- coding: utf-8 -*-
"""Format prediction results for Telegram (text + chart images)."""

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.config import INDEX_CONFIGS, SIGNAL_THRESHOLDS
from src.models.predictor import IndexPredictor
from src.strategy.allocation import AllocationResult


# ==================== Korean day name ====================
_DAY_NAMES_KR = ["월", "화", "수", "목", "금", "토", "일"]


def _day_kr(dt) -> str:
    return _DAY_NAMES_KR[dt.weekday()] if hasattr(dt, "weekday") else ""


# ==================== Text Formatters ====================

def format_daily_summary(
    predictions: dict[str, float],
    prices: dict[str, float],
    prev_predictions: dict[str, float] = None,
    indicators: dict = None,
    allocation: AllocationResult = None,
    rebalance_info: tuple[bool, str] = None,
) -> str:
    """
    Format the main daily summary message (HTML).
    predictions: {"NASDAQ": 0.723, ...}  (keys match INDEX_CONFIGS)
    prices: {"NASDAQ": 18456.23, ...}
    prev_predictions: yesterday's predictions for delta calculation
    indicators: {"vix": 16.42, "rsi": 58.3, "sma50_ratio": 0.682}
    allocation: AllocationResult from strategy
    rebalance_info: (bool, reason) from check_rebalance
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    day_str = _day_kr(now)

    lines = [
        f"<b>US Market Prediction - {date_str} ({day_str})</b>",
        "\u2501" * 20,
        "",
    ]

    for index_name, cfg in INDEX_CONFIGS.items():
        prob = predictions.get(index_name)
        if prob is None:
            continue

        price = prices.get(index_name, 0)
        signal_text, emoji = IndexPredictor.get_signal(prob)

        # Delta from previous day
        delta_str = ""
        if prev_predictions and index_name in prev_predictions:
            delta = (prob - prev_predictions[index_name]) * 100
            sign = "+" if delta >= 0 else ""
            delta_str = f" ({sign}{delta:.1f}%p)"

        display = cfg["display_name"]
        ticker = cfg["ticker"]
        lines.append(f"<b>{display} ({ticker})</b> {price:,.2f}")
        lines.append(f"  상승확률: <b>{prob*100:.1f}%</b>{delta_str} {emoji} {signal_text}")
        lines.append("")

    # Portfolio Allocation
    if allocation:
        lines.append("\U0001f4ca <b>Portfolio Allocation</b>")
        lines.append(f"  Tier: <b>{allocation.tier_label}</b>")
        lines.append(
            f"  TQQQ: {allocation.tqqq_weight*100:.0f}% | "
            f"SPY: {allocation.spy_weight*100:.0f}% | "
            f"Cash: {allocation.cash_weight*100:.0f}%"
        )
        if rebalance_info:
            rebal, reason = rebalance_info
            if rebal:
                lines.append(f"  \u26a1 Rebalance: <b>YES</b> ({reason})")
            else:
                lines.append(f"  Rebalance: NO ({reason})")
        lines.append("")

    # Indicators
    lines.append("\u2501" * 20)
    if indicators:
        parts = []
        if "vix" in indicators:
            parts.append(f"VIX: {indicators['vix']:.1f}")
        if "rsi" in indicators:
            parts.append(f"RSI: {indicators['rsi']:.1f}")
        if "sma50_ratio" in indicators:
            parts.append(f"SMA50 비율: {indicators['sma50_ratio']*100:.1f}%")
        if parts:
            lines.append("  ".join(parts))

    return "\n".join(lines)


def format_5day_table(
    history: dict[str, pd.DataFrame],
) -> str:
    """
    Format recent 5-day probability table (monospace text).
    history: {"NASDAQ": DataFrame with Probability column, ...}
    """
    index_names = list(history.keys())

    header_cols = "  ".join(f"{n:>7s}" for n in index_names)
    lines = [
        "<b>최근 5일 추이</b>",
        "",
        "<pre>",
        f"{'날짜':10s}  {header_cols}",
    ]

    # Get union of dates (last 5)
    all_dates = set()
    for df in history.values():
        if df is not None:
            all_dates.update(df.index[-5:])
    dates_sorted = sorted(all_dates)[-5:]

    for dt in dates_sorted:
        dt_str = dt.strftime("%m-%d") + f"({_day_kr(dt)})"
        values = []
        for index_name in index_names:
            df = history.get(index_name)
            if df is not None and dt in df.index:
                val = df.loc[dt, "Probability"] * 100
                values.append(f"{val:6.1f}%")
            else:
                values.append(f"{'--':>7s}")
        lines.append(f"{dt_str:10s}  {'  '.join(values)}")

    # Delta (5-day change)
    delta_parts = []
    for index_name in index_names:
        df = history.get(index_name)
        if df is not None and len(df) >= 5:
            recent = df.iloc[-5:]
            d = (recent.iloc[-1]["Probability"] - recent.iloc[0]["Probability"]) * 100
            sign = "+" if d >= 0 else ""
            delta_parts.append(f"{sign}{d:5.1f} ")
            continue
        delta_parts.append(f"{'--':>7s}")
    lines.append(f"{'변동(5d)':10s}  {'  '.join(delta_parts)}")

    lines.append("</pre>")
    return "\n".join(lines)


# ==================== Chart Image ====================

def create_probability_chart(
    history: dict[str, pd.DataFrame],
    prices: dict[str, pd.Series] = None,
    days: int = 60,
) -> bytes:
    """
    Create a chart image with probability trends and index prices.
    Returns PNG bytes.

    history: {"NASDAQ": DataFrame(Probability), ...}  (keys match INDEX_CONFIGS)
    prices: {"NASDAQ": Series(Adj Close), ...}
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), height_ratios=[3, 2], sharex=True,
    )
    fig.patch.set_facecolor("#f7f9ff")

    _palette = ["#FF4B4B", "#2ECC71", "#F1C40F", "#3498DB", "#9B59B6"]
    colors = {name: _palette[i % len(_palette)] for i, name in enumerate(INDEX_CONFIGS)}

    # ── Top: Probability trends ──
    ax1.set_facecolor("#ffffff")
    for index_name, df in history.items():
        if df is None or df.empty:
            continue
        recent = df.iloc[-days:]
        ax1.plot(
            recent.index, recent["Probability"] * 100,
            label=f"{index_name}",
            color=colors.get(index_name, "#333"),
            linewidth=1.8,
        )

    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.set_ylabel("상승 확률 (%)", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_title("US Market Prediction - Probability Trend", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.15)

    # Highlight zones
    ax1.axhspan(70, 100, alpha=0.05, color="green")
    ax1.axhspan(0, 40, alpha=0.05, color="red")

    # ── Bottom: Index prices (rebased to 100) ──
    if prices:
        ax2.set_facecolor("#ffffff")
        for index_name, series in prices.items():
            if series is None or series.empty:
                continue
            recent = series.iloc[-days:]
            if len(recent) > 0:
                rebased = (recent / float(recent.iloc[0])) * 100
                ax2.plot(
                    rebased.index, rebased,
                    label=f"{index_name}",
                    color=colors.get(index_name, "#333"),
                    linewidth=1.5,
                )

        ax2.set_ylabel("지수 (리베이스 100)", fontsize=11)
        ax2.legend(loc="upper left", fontsize=10)
        ax2.grid(True, alpha=0.15)

    ax2.set_xlabel("날짜", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
