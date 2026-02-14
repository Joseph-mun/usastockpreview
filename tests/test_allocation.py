# -*- coding: utf-8 -*-
"""Tests for allocation strategy."""

import pytest

from src.strategy.allocation import (
    AllocationResult,
    get_allocation,
    check_rebalance,
)


class TestGetAllocation:
    def test_high_probability_aggressive(self):
        result = get_allocation(0.65)
        assert result.tier_label == "Aggressive"
        assert result.tqqq_weight > 0

    def test_low_probability_defensive(self):
        result = get_allocation(0.30)
        assert result.tier_label == "Defensive"
        assert result.tqqq_weight == 0

    def test_mid_probability_moderate(self):
        result = get_allocation(0.52)
        assert result.tier_label == "Moderate"

    def test_weights_sum_to_one(self):
        for prob in [0.1, 0.3, 0.5, 0.55, 0.65, 0.9]:
            result = get_allocation(prob)
            total = result.tqqq_weight + result.spy_weight + result.cash_weight
            assert abs(total - 1.0) < 0.01, f"prob={prob}: weights sum to {total}"

    def test_vix_filter_reduces_tqqq(self):
        base = get_allocation(0.65, vix=10)
        high_vix = get_allocation(0.65, vix=30)
        assert high_vix.tqqq_weight <= base.tqqq_weight

    def test_extreme_vix_zeroes_tqqq(self):
        result = get_allocation(0.65, vix=40)
        assert result.tqqq_weight == 0

    def test_adx_range_penalizes_tqqq(self):
        trending = get_allocation(0.65, adx=30)
        ranging = get_allocation(0.65, adx=15)
        assert ranging.tqqq_weight <= trending.tqqq_weight

    def test_result_has_all_fields(self):
        result = get_allocation(0.55, vix=20, adx=25)
        assert isinstance(result, AllocationResult)
        assert result.probability == 0.55
        assert result.vix_level == 20
        assert result.adx_level == 25


class TestCheckRebalance:
    def test_same_tier_no_rebalance(self):
        rebal, reason = check_rebalance(0.52, 0.53, "Moderate")
        assert rebal is False

    def test_large_delta_triggers_rebalance(self):
        rebal, reason = check_rebalance(0.65, 0.40, "Defensive")
        assert rebal is True

    def test_small_delta_hysteresis_blocks(self):
        rebal, reason = check_rebalance(0.56, 0.54, "Moderate")
        assert rebal is False
        assert "hysteresis" in reason.lower() or "same tier" in reason.lower()
