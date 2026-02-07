# -*- coding: utf-8 -*-
"""TQQQ/SPY/Cash allocation strategy based on calibrated probabilities."""

from dataclasses import dataclass

from src.config import ALLOCATION_TIERS, REBALANCE_HYSTERESIS


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


def get_allocation(probability: float) -> AllocationResult:
    """Determine TQQQ/SPY/Cash allocation based on model probability."""
    for min_p, max_p, tqqq, spy, cash, label in ALLOCATION_TIERS:
        if min_p <= probability < max_p:
            return AllocationResult(
                tqqq_weight=tqqq,
                spy_weight=spy,
                cash_weight=cash,
                tier_label=label,
                probability=probability,
                rebalance_needed=True,
                reason=f"Probability {probability*100:.1f}% -> {label}",
            )
    return AllocationResult(0.0, 0.30, 0.70, "Defensive", probability, True, "Fallback")


def check_rebalance(
    current_prob: float,
    prev_prob: float,
    current_tier: str,
) -> tuple[bool, str]:
    """Check if rebalancing is needed with hysteresis."""
    new_alloc = get_allocation(current_prob)

    if new_alloc.tier_label == current_tier:
        return False, f"Same tier ({current_tier})"

    prob_delta = abs(current_prob - prev_prob)
    if prob_delta < REBALANCE_HYSTERESIS:
        return False, f"Delta {prob_delta*100:.1f}%p < {REBALANCE_HYSTERESIS*100:.0f}%p hysteresis"

    return True, f"{current_tier} -> {new_alloc.tier_label} (delta: {prob_delta*100:+.1f}%p)"


def format_allocation_text(alloc: AllocationResult, rebalance_info: tuple[bool, str] = None) -> str:
    """Format allocation for Telegram display."""
    lines = [
        f"Tier: {alloc.tier_label}",
        f"TQQQ: {alloc.tqqq_weight*100:.0f}% | SPY: {alloc.spy_weight*100:.0f}% | Cash: {alloc.cash_weight*100:.0f}%",
    ]
    if rebalance_info:
        rebal, reason = rebalance_info
        if rebal:
            lines.append(f"Rebalance: YES ({reason})")
        else:
            lines.append(f"Rebalance: NO ({reason})")
    return "\n".join(lines)
