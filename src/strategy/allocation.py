# -*- coding: utf-8 -*-
"""TQQQ/SPY/Cash allocation strategy based on calibrated probabilities.

Enhanced with:
  - P0-1: VIX-based volatility filter (reduce TQQQ in high-vol regimes)
  - P1-2: ADX-based market regime detection (penalize TQQQ in range-bound markets)
"""

from dataclasses import dataclass, field

from src.config import (
    ALLOCATION_TIERS,
    REBALANCE_HYSTERESIS,
    VIX_FILTER_ENABLED,
    VIX_FILTER_TIERS,
    REGIME_DETECTION_ENABLED,
    REGIME_ADX_RANGE_THRESHOLD,
    REGIME_ADX_TREND_THRESHOLD,
    REGIME_TQQQ_RANGE_PENALTY,
    TQQQ_REDUCTION_SPY_RATIO,
    TQQQ_REDUCTION_CASH_RATIO,
)


@dataclass
class AllocationResult:
    """Result of an allocation decision."""
    tqqq_weight: float
    spy_weight: float
    cash_weight: float
    tier_label: str
    probability: float
    rebalance_needed: bool
    reason: str
    vix_level: float | None = None
    vix_filter_label: str | None = None
    adx_level: float | None = None
    regime_label: str | None = None


def _lookup_tier(probability: float) -> tuple[float, float, float, str]:
    """Look up base allocation weights from probability tier."""
    for min_p, max_p, tqqq, spy, cash, label in ALLOCATION_TIERS:
        if min_p <= probability < max_p:
            return tqqq, spy, cash, label
    return 0.0, 0.30, 0.70, "Defensive"


def _apply_vix_filter(base_tqqq: float, vix: float | None) -> tuple[float, str | None]:
    """Reduce TQQQ weight based on VIX level.

    Returns:
        (adjusted_tqqq, vix_filter_label)
    """
    if not VIX_FILTER_ENABLED or vix is None:
        return base_tqqq, None

    for min_v, max_v, multiplier, label in VIX_FILTER_TIERS:
        if min_v <= vix < max_v:
            return base_tqqq * multiplier, label

    return 0.0, "Extreme Vol"


def _apply_regime_adjustment(tqqq_weight: float, adx: float | None) -> tuple[float, str | None]:
    """Adjust TQQQ weight based on ADX market regime.

    Trending (ADX > 25): no change (leverage benefits from trends)
    Range-bound (ADX < 20): additional reduction (volatility decay amplified)
    Transition (20-25): no change

    Returns:
        (adjusted_tqqq, regime_label)
    """
    if not REGIME_DETECTION_ENABLED or adx is None:
        return tqqq_weight, None

    if adx < REGIME_ADX_RANGE_THRESHOLD:
        return tqqq_weight * REGIME_TQQQ_RANGE_PENALTY, "Range"
    elif adx >= REGIME_ADX_TREND_THRESHOLD:
        return tqqq_weight, "Trend"
    else:
        return tqqq_weight, "Transition"


def get_allocation(
    probability: float,
    vix: float = None,
    adx: float = None,
) -> AllocationResult:
    """Determine TQQQ/SPY/Cash allocation with VIX filter and regime detection.

    Pipeline:
      1. Probability -> base tier lookup
      2. VIX filter -> reduce TQQQ in high-volatility
      3. Regime adjustment -> reduce TQQQ in range-bound markets
      4. Redistribute TQQQ reduction to SPY (60%) and Cash (40%)
    """
    base_tqqq, base_spy, base_cash, label = _lookup_tier(probability)

    adjusted_tqqq, vix_label = _apply_vix_filter(base_tqqq, vix)

    adjusted_tqqq, regime_label = _apply_regime_adjustment(adjusted_tqqq, adx)

    tqqq_reduction = base_tqqq - adjusted_tqqq
    final_tqqq = adjusted_tqqq
    final_spy = base_spy + tqqq_reduction * TQQQ_REDUCTION_SPY_RATIO
    final_cash = base_cash + tqqq_reduction * TQQQ_REDUCTION_CASH_RATIO

    reason_parts = [f"Prob {probability*100:.1f}% -> {label}"]
    if vix is not None and vix_label:
        reason_parts.append(f"VIX {vix:.1f} ({vix_label})")
    if adx is not None and regime_label:
        reason_parts.append(f"ADX {adx:.1f} ({regime_label})")
    if tqqq_reduction > 0.001:
        reason_parts.append(f"TQQQ adj {base_tqqq*100:.0f}%->{final_tqqq*100:.1f}%")

    return AllocationResult(
        tqqq_weight=round(final_tqqq, 4),
        spy_weight=round(final_spy, 4),
        cash_weight=round(final_cash, 4),
        tier_label=label,
        probability=probability,
        rebalance_needed=True,
        reason=" | ".join(reason_parts),
        vix_level=vix,
        vix_filter_label=vix_label,
        adx_level=adx,
        regime_label=regime_label,
    )


def check_rebalance(
    current_prob: float,
    prev_prob: float,
    current_tier: str,
    vix: float = None,
    adx: float = None,
) -> tuple[bool, str]:
    """Check if rebalancing is needed with hysteresis."""
    new_alloc = get_allocation(current_prob, vix=vix, adx=adx)

    if new_alloc.tier_label == current_tier:
        return False, f"Same tier ({current_tier})"

    prob_delta = abs(current_prob - prev_prob)
    if prob_delta < REBALANCE_HYSTERESIS:
        return False, f"Delta {prob_delta*100:.1f}%p < {REBALANCE_HYSTERESIS*100:.0f}%p hysteresis"

    return True, f"{current_tier} -> {new_alloc.tier_label} (delta: {prob_delta*100:+.1f}%p)"


def format_allocation_text(alloc: AllocationResult, rebalance_info: tuple[bool, str] | None = None) -> str:
    """Format allocation for Telegram display."""
    lines = [
        f"Tier: {alloc.tier_label}",
        f"TQQQ: {alloc.tqqq_weight*100:.0f}% | SPY: {alloc.spy_weight*100:.0f}% | Cash: {alloc.cash_weight*100:.0f}%",
    ]
    if alloc.vix_filter_label:
        lines.append(f"VIX Filter: {alloc.vix_filter_label} (VIX={alloc.vix_level:.1f})")
    if alloc.regime_label:
        lines.append(f"Regime: {alloc.regime_label} (ADX={alloc.adx_level:.1f})")
    if rebalance_info:
        rebal, reason = rebalance_info
        if rebal:
            lines.append(f"Rebalance: YES ({reason})")
        else:
            lines.append(f"Rebalance: NO ({reason})")
    return "\n".join(lines)
