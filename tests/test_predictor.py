# -*- coding: utf-8 -*-
"""Tests for model predictor."""

import pytest

from src.models.predictor import IndexPredictor
from src.config import SIGNAL_THRESHOLDS


class TestGetSignal:
    def test_strong_buy(self):
        text, emoji = IndexPredictor.get_signal(0.65)
        assert text == "Strong Buy"

    def test_buy(self):
        text, emoji = IndexPredictor.get_signal(0.57)
        assert text == "Buy"

    def test_neutral(self):
        text, emoji = IndexPredictor.get_signal(0.48)
        assert text == "Neutral"

    def test_sell(self):
        text, emoji = IndexPredictor.get_signal(0.30)
        assert text == "Sell"

    def test_boundary_strong_buy(self):
        text, _ = IndexPredictor.get_signal(SIGNAL_THRESHOLDS["strong_buy"])
        assert text == "Strong Buy"

    def test_boundary_buy(self):
        text, _ = IndexPredictor.get_signal(SIGNAL_THRESHOLDS["buy"])
        assert text == "Buy"

    def test_boundary_neutral(self):
        text, _ = IndexPredictor.get_signal(SIGNAL_THRESHOLDS["neutral"])
        assert text == "Neutral"

    def test_below_neutral_is_sell(self):
        text, _ = IndexPredictor.get_signal(SIGNAL_THRESHOLDS["neutral"] - 0.01)
        assert text == "Sell"
