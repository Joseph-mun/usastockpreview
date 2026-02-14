# -*- coding: utf-8 -*-
"""Tests for config module."""

from src.config import (
    ALLOCATION_TIERS,
    LGBM_PARAMS,
    PROB_CLIP_MIN,
    PROB_CLIP_MAX,
    SIGNAL_THRESHOLDS,
)


def test_allocation_tiers_cover_full_range():
    """Allocation tiers should cover probability range [0, 1]."""
    sorted_tiers = sorted(ALLOCATION_TIERS, key=lambda t: t[0])
    assert sorted_tiers[0][0] == 0.0, "Lowest tier should start at 0.0"
    assert sorted_tiers[-1][1] > 1.0, "Highest tier should extend past 1.0"

    # Check no gaps between tiers
    for i in range(len(sorted_tiers) - 1):
        assert sorted_tiers[i][1] == sorted_tiers[i + 1][0], (
            f"Gap between tier {i} end ({sorted_tiers[i][1]}) "
            f"and tier {i+1} start ({sorted_tiers[i+1][0]})"
        )


def test_allocation_tiers_weights_sum_to_one():
    """Each tier's weights (tqqq + spy + cash) should sum to 1.0."""
    for min_p, max_p, tqqq, spy, cash, label in ALLOCATION_TIERS:
        total = tqqq + spy + cash
        assert abs(total - 1.0) < 0.01, (
            f"Tier '{label}' weights sum to {total}, expected 1.0"
        )


def test_signal_thresholds_ordered():
    """Signal thresholds should be in descending order."""
    assert SIGNAL_THRESHOLDS["strong_buy"] > SIGNAL_THRESHOLDS["buy"]
    assert SIGNAL_THRESHOLDS["buy"] > SIGNAL_THRESHOLDS["neutral"]


def test_prob_clip_range():
    """Probability clip range should be valid."""
    assert 0.0 <= PROB_CLIP_MIN < PROB_CLIP_MAX <= 1.0


def test_lgbm_params_required_keys():
    """LightGBM params should contain required keys."""
    required = {"objective", "n_estimators", "max_depth", "learning_rate"}
    assert required.issubset(set(LGBM_PARAMS.keys()))
    assert LGBM_PARAMS["objective"] == "binary"
