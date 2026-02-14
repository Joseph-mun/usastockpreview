# -*- coding: utf-8 -*-
"""Portfolio backtest: simulate TQQQ/SPY allocation using model probabilities.

Enhanced with:
  - P0-2: Transaction costs (slippage + commission) on rebalance days
  - P0-1/P1-2: VIX/ADX series forwarded to allocation logic
  - Risk management: stop-loss, MDD circuit breaker, volatility targeting, position aging
"""

import numpy as np
import pandas as pd

from src.config import (
    TRANSACTION_COST_ENABLED, SLIPPAGE_PCT, COMMISSION_PCT,
    TQQQ_REDUCTION_SPY_RATIO, TQQQ_REDUCTION_CASH_RATIO,
    STOP_LOSS_ENABLED, STOP_LOSS_THRESHOLD,
    MDD_CIRCUIT_BREAKER_ENABLED, MDD_CIRCUIT_BREAKER_THRESHOLD, MDD_CIRCUIT_BREAKER_RECOVERY,
    VOL_TARGETING_ENABLED, VOL_TARGET_ANNUAL, VOL_LOOKBACK_DAYS, VOL_SCALE_MIN, VOL_SCALE_MAX,
    POSITION_AGING_ENABLED, POSITION_AGING_DECAY_RATE, POSITION_AGING_MAX_DAYS,
)
from src.strategy.allocation import AllocationResult, get_allocation, check_rebalance


def _safe_float(series: pd.Series, date, default=None) -> float | None:
    """Safely extract a float value from a series at given date."""
    if series is None or date not in series.index:
        return default
    v = series.loc[date]
    if isinstance(v, float) and np.isnan(v):
        return default
    return float(v)


def _apply_position_aging(
    alloc: AllocationResult,
    days_since_change: int,
) -> AllocationResult:
    """Apply position aging decay to TQQQ weight."""
    if not POSITION_AGING_ENABLED or days_since_change <= 0:
        return alloc
    if alloc.tqqq_weight <= 0 or days_since_change > POSITION_AGING_MAX_DAYS:
        return alloc

    decay_factor = (1 - POSITION_AGING_DECAY_RATE) ** days_since_change
    aged_tqqq = alloc.tqqq_weight * decay_factor
    reduction = alloc.tqqq_weight - aged_tqqq

    return AllocationResult(
        tqqq_weight=round(aged_tqqq, 4),
        spy_weight=round(alloc.spy_weight + reduction * TQQQ_REDUCTION_SPY_RATIO, 4),
        cash_weight=round(alloc.cash_weight + reduction * TQQQ_REDUCTION_CASH_RATIO, 4),
        tier_label=alloc.tier_label,
        probability=alloc.probability,
        rebalance_needed=alloc.rebalance_needed,
        reason=alloc.reason,
        vix_level=alloc.vix_level,
        vix_filter_label=alloc.vix_filter_label,
        adx_level=alloc.adx_level,
        regime_label=alloc.regime_label,
    )


def _apply_vol_targeting(
    alloc: AllocationResult,
    daily_returns: list[float],
) -> tuple[AllocationResult, float]:
    """Scale risky asset weights by target_vol / realized_vol.

    Returns (adjusted_alloc, vol_scale_factor).
    """
    if not VOL_TARGETING_ENABLED or len(daily_returns) < VOL_LOOKBACK_DAYS:
        return alloc, 1.0

    recent = np.array(daily_returns[-VOL_LOOKBACK_DAYS:])
    realized_vol = float(recent.std() * np.sqrt(252))
    if realized_vol <= 0:
        return alloc, 1.0

    vol_scale = np.clip(VOL_TARGET_ANNUAL / realized_vol, VOL_SCALE_MIN, VOL_SCALE_MAX)

    scaled_tqqq = alloc.tqqq_weight * vol_scale
    scaled_spy = alloc.spy_weight * vol_scale
    scaled_cash = 1.0 - scaled_tqqq - scaled_spy
    scaled_cash = max(scaled_cash, 0.0)

    return AllocationResult(
        tqqq_weight=round(scaled_tqqq, 4),
        spy_weight=round(scaled_spy, 4),
        cash_weight=round(scaled_cash, 4),
        tier_label=alloc.tier_label,
        probability=alloc.probability,
        rebalance_needed=alloc.rebalance_needed,
        reason=alloc.reason,
        vix_level=alloc.vix_level,
        vix_filter_label=alloc.vix_filter_label,
        adx_level=alloc.adx_level,
        regime_label=alloc.regime_label,
    ), float(vol_scale)


def run_portfolio_backtest(
    probabilities: pd.Series,
    nasdaq_prices: pd.Series,
    spy_prices: pd.Series,
    initial_capital: float = 10000.0,
    vix_series: pd.Series = None,
    adx_series: pd.Series = None,
    include_costs: bool = TRANSACTION_COST_ENABLED,
) -> pd.DataFrame:
    """
    Simulate portfolio allocation over historical data.

    TQQQ return approximated as 3x daily NASDAQ return.
    Risk management: stop-loss, MDD circuit breaker, volatility targeting, position aging.
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
    portfolio_peak = initial_capital
    current_tier = "Defensive"
    prev_prob = 0.5
    total_costs = 0.0
    vix_filter_activations = 0
    prev_tqqq_w = 0.0
    prev_spy_w = 0.30

    # Risk management state
    circuit_breaker_active = False
    stop_loss_active = False
    days_since_tier_change = 0
    position_entry_value = initial_capital
    daily_returns_history: list[float] = []

    for date in common_dates:
        prob = float(probabilities.loc[date])
        vix = _safe_float(vix_series, date)
        adx = _safe_float(adx_series, date)

        # ── Risk Check A: MDD Circuit Breaker ──
        stop_loss_triggered = False
        vol_scale = 1.0

        if MDD_CIRCUIT_BREAKER_ENABLED and circuit_breaker_active:
            if portfolio_value >= portfolio_peak * MDD_CIRCUIT_BREAKER_RECOVERY:
                circuit_breaker_active = False
            else:
                alloc = AllocationResult(
                    tqqq_weight=0.0, spy_weight=0.0, cash_weight=1.0,
                    tier_label="Circuit Breaker", probability=prob,
                    rebalance_needed=False, reason="MDD Circuit Breaker Active",
                )
                t_ret = _safe_float(tqqq_ret, date, 0.0) or 0.0
                s_ret = _safe_float(spy_ret, date, 0.0) or 0.0
                port_ret = 0.0  # 100% cash
                portfolio_value *= (1 + port_ret)
                daily_returns_history.append(port_ret)

                records.append({
                    "date": date, "probability": prob, "tier": "Circuit Breaker",
                    "tqqq_weight": 0.0, "spy_weight": 0.0, "cash_weight": 1.0,
                    "tqqq_daily_ret": t_ret, "spy_daily_ret": s_ret,
                    "portfolio_daily_ret": port_ret, "portfolio_value": portfolio_value,
                    "rebalanced": False, "transaction_cost": 0.0,
                    "vix": vix, "adx": adx,
                    "vix_filter": None, "regime": None,
                    "stop_loss_triggered": False,
                    "circuit_breaker_active": True,
                    "position_aging_days": 0, "vol_scale": 1.0,
                })
                prev_prob = prob
                prev_tqqq_w = 0.0
                prev_spy_w = 0.0
                continue

        # ── Normal allocation ──
        alloc = get_allocation(prob, vix=vix, adx=adx)

        should_rebalance, rebal_reason = check_rebalance(
            prob, prev_prob, current_tier, vix=vix, adx=adx,
        )

        if should_rebalance:
            current_tier = alloc.tier_label
            alloc = get_allocation(prob, vix=vix, adx=adx)
            days_since_tier_change = 0
            stop_loss_active = False
            position_entry_value = portfolio_value
        else:
            alloc = get_allocation(prev_prob, vix=vix, adx=adx)
            days_since_tier_change += 1

        # ── Risk Check B: Stop Loss ──
        if STOP_LOSS_ENABLED and stop_loss_active and alloc.tqqq_weight > 0:
            alloc = AllocationResult(
                tqqq_weight=0.0,
                spy_weight=round(alloc.spy_weight + alloc.tqqq_weight * TQQQ_REDUCTION_SPY_RATIO, 4),
                cash_weight=round(alloc.cash_weight + alloc.tqqq_weight * TQQQ_REDUCTION_CASH_RATIO, 4),
                tier_label=alloc.tier_label,
                probability=alloc.probability,
                rebalance_needed=alloc.rebalance_needed,
                reason="Stop Loss Active",
                vix_level=alloc.vix_level,
                vix_filter_label=alloc.vix_filter_label,
                adx_level=alloc.adx_level,
                regime_label=alloc.regime_label,
            )

        # ── Risk Check C: Position Aging ──
        alloc = _apply_position_aging(alloc, days_since_tier_change)

        # ── Risk Check D: Volatility Targeting ──
        alloc, vol_scale = _apply_vol_targeting(alloc, daily_returns_history)

        # Transaction costs on rebalance
        day_cost = 0.0
        if should_rebalance and include_costs:
            turnover = (
                abs(alloc.tqqq_weight - prev_tqqq_w)
                + abs(alloc.spy_weight - prev_spy_w)
            )
            day_cost = turnover * (SLIPPAGE_PCT + COMMISSION_PCT)
            portfolio_value *= (1 - day_cost)
            total_costs += day_cost * portfolio_value

        if alloc.vix_filter_label and alloc.vix_filter_label != "Low Vol":
            vix_filter_activations += 1

        t_ret = _safe_float(tqqq_ret, date, 0.0) or 0.0
        s_ret = _safe_float(spy_ret, date, 0.0) or 0.0

        port_ret = (
            alloc.tqqq_weight * t_ret
            + alloc.spy_weight * s_ret
            + alloc.cash_weight * 0
        )
        portfolio_value *= (1 + port_ret)
        daily_returns_history.append(port_ret)

        # Update portfolio peak
        if portfolio_value > portfolio_peak:
            portfolio_peak = portfolio_value

        # ── Post-return risk checks ──

        # MDD circuit breaker trigger
        if MDD_CIRCUIT_BREAKER_ENABLED and not circuit_breaker_active:
            drawdown = (portfolio_value - portfolio_peak) / portfolio_peak
            if drawdown < -MDD_CIRCUIT_BREAKER_THRESHOLD:
                circuit_breaker_active = True

        # Stop loss trigger (based on TQQQ position loss since entry)
        if (STOP_LOSS_ENABLED and not stop_loss_active
                and alloc.tqqq_weight > 0 and position_entry_value > 0):
            tqqq_pnl = (portfolio_value - position_entry_value) / position_entry_value
            if tqqq_pnl < -STOP_LOSS_THRESHOLD:
                stop_loss_active = True
                stop_loss_triggered = True

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
            "transaction_cost": day_cost,
            "vix": vix,
            "adx": adx,
            "vix_filter": alloc.vix_filter_label,
            "regime": alloc.regime_label,
            "stop_loss_triggered": stop_loss_triggered,
            "circuit_breaker_active": circuit_breaker_active,
            "position_aging_days": days_since_tier_change,
            "vol_scale": vol_scale,
        })

        prev_prob = prob
        prev_tqqq_w = alloc.tqqq_weight
        prev_spy_w = alloc.spy_weight

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

    metrics = {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "n_rebalances": int(df["rebalanced"].sum()),
        "tier_distribution": df["tier"].value_counts(normalize=True).to_dict(),
        "n_trading_days": n_days,
        "final_value": float(df["portfolio_value"].iloc[-1]),
    }

    if "transaction_cost" in df.columns:
        total_costs = float(df["transaction_cost"].sum())
        metrics["total_transaction_costs"] = total_costs
        cost_free_return = total_return + total_costs
        metrics["cost_drag_pct"] = float(total_costs / cost_free_return) if cost_free_return != 0 else 0.0

    if "regime" in df.columns:
        regime_counts = df["regime"].value_counts(normalize=True, dropna=False).to_dict()
        metrics["regime_distribution"] = {k: v for k, v in regime_counts.items() if k is not None}

    if "vix_filter" in df.columns:
        vix_non_low = df["vix_filter"].apply(lambda x: x is not None and x != "Low Vol")
        metrics["vix_filter_activations"] = int(vix_non_low.sum())

    # Risk management metrics
    if "stop_loss_triggered" in df.columns:
        metrics["stop_loss_count"] = int(df["stop_loss_triggered"].sum())

    if "circuit_breaker_active" in df.columns:
        metrics["circuit_breaker_days"] = int(df["circuit_breaker_active"].sum())

    if "position_aging_days" in df.columns:
        aging = df[df["position_aging_days"] > 0]["position_aging_days"]
        metrics["avg_position_aging_days"] = float(aging.mean()) if not aging.empty else 0.0

    if "vol_scale" in df.columns:
        vs = df["vol_scale"]
        metrics["avg_vol_scale"] = float(vs.mean())
        metrics["min_vol_scale"] = float(vs.min())
        metrics["max_vol_scale"] = float(vs.max())

    return metrics


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
