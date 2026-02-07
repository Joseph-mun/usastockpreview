# -*- coding: utf-8 -*-
"""Portfolio backtest: simulate TQQQ/SPY allocation using model probabilities."""

import numpy as np
import pandas as pd

from src.strategy.allocation import get_allocation, check_rebalance


def run_portfolio_backtest(
    probabilities: pd.Series,
    nasdaq_prices: pd.Series,
    spy_prices: pd.Series,
    initial_capital: float = 10000.0,
) -> pd.DataFrame:
    """
    Simulate portfolio allocation over historical data.

    TQQQ return approximated as 3x daily NASDAQ return.
    """
    common_dates = (
        probabilities.index
        .intersection(nasdaq_prices.index)
        .intersection(spy_prices.index)
    )
    common_dates = common_dates.sort_values()

    nasdaq_ret = nasdaq_prices.pct_change()
    spy_ret = spy_prices.pct_change()
    tqqq_ret = nasdaq_ret * 3  # simplified 3x leverage

    records = []
    portfolio_value = initial_capital
    current_tier = "Defensive"
    prev_prob = 0.5

    for date in common_dates:
        prob = float(probabilities.loc[date])
        alloc = get_allocation(prob)

        should_rebalance, _ = check_rebalance(prob, prev_prob, current_tier)
        if should_rebalance:
            current_tier = alloc.tier_label
            alloc = get_allocation(prob)
        else:
            alloc = get_allocation(prev_prob)

        t_ret = float(tqqq_ret.get(date, 0)) if date in tqqq_ret.index and not np.isnan(tqqq_ret.get(date, 0)) else 0
        s_ret = float(spy_ret.get(date, 0)) if date in spy_ret.index and not np.isnan(spy_ret.get(date, 0)) else 0

        port_ret = (
            alloc.tqqq_weight * t_ret
            + alloc.spy_weight * s_ret
            + alloc.cash_weight * 0
        )
        portfolio_value *= (1 + port_ret)

        records.append({
            "date": date,
            "probability": prob,
            "tier": current_tier,
            "tqqq_weight": alloc.tqqq_weight,
            "spy_weight": alloc.spy_weight,
            "cash_weight": alloc.cash_weight,
            "tqqq_daily_ret": t_ret,
            "spy_daily_ret": s_ret,
            "portfolio_daily_ret": port_ret,
            "portfolio_value": portfolio_value,
            "rebalanced": should_rebalance,
        })
        prev_prob = prob

    return pd.DataFrame(records)


def compute_backtest_metrics(df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
    """Compute portfolio performance metrics."""
    if df.empty:
        return {}

    total_return = (df["portfolio_value"].iloc[-1] / initial_capital) - 1
    n_days = len(df)
    ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

    cummax = df["portfolio_value"].cummax()
    drawdown = (df["portfolio_value"] - cummax) / cummax
    max_drawdown = float(drawdown.min())

    daily_ret = df["portfolio_daily_ret"]
    sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252)) if daily_ret.std() > 0 else 0

    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "n_rebalances": int(df["rebalanced"].sum()),
        "tier_distribution": df["tier"].value_counts(normalize=True).to_dict(),
        "n_trading_days": n_days,
        "final_value": float(df["portfolio_value"].iloc[-1]),
    }


def compute_benchmark_returns(
    nasdaq_prices: pd.Series,
    spy_prices: pd.Series,
    dates: pd.DatetimeIndex,
    initial_capital: float = 10000.0,
) -> dict:
    """Compute buy-and-hold benchmark returns for comparison."""
    benchmarks = {}

    for name, prices in [("NASDAQ", nasdaq_prices), ("SPY", spy_prices)]:
        common = dates.intersection(prices.index)
        if len(common) < 2:
            continue
        start_price = float(prices.loc[common[0]])
        end_price = float(prices.loc[common[-1]])
        ret = (end_price / start_price) - 1
        benchmarks[name] = {
            "total_return": ret,
            "final_value": initial_capital * (1 + ret),
        }

    # TQQQ approximation
    if "NASDAQ" in benchmarks:
        nasdaq_common = dates.intersection(nasdaq_prices.index)
        daily_rets = nasdaq_prices.reindex(nasdaq_common).pct_change().fillna(0)
        tqqq_value = initial_capital
        for r in daily_rets:
            tqqq_value *= (1 + r * 3)
        benchmarks["TQQQ(3x)"] = {
            "total_return": (tqqq_value / initial_capital) - 1,
            "final_value": tqqq_value,
        }

    return benchmarks
