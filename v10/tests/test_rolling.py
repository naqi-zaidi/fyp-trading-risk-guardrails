"""
v10/tests/test_rolling.py
Tests for core/rolling.py — rolling statistics, ATR, EMA, MACD, CUSUM.
"""

import math
import numpy as np
import pytest

from v10.core.rolling import (
    WelfordRolling, WilderATR, EMA, MACD, RollingDeque, CUSUMDetector,
)


class TestWelfordRolling:
    """Tests for the Welford online mean/std roller."""

    def test_not_ready_until_full(self):
        wr = WelfordRolling(20)
        for i in range(19):
            wr.update(float(i))
        assert not wr.ready
        wr.update(19.0)
        assert wr.ready

    def test_mean_matches_numpy(self, seed):
        data = np.random.randn(200)
        wr = WelfordRolling(50)
        for v in data:
            wr.update(v)
        expected_mean = np.mean(data[-50:])
        assert abs(wr.mean - expected_mean) < 1e-10

    def test_std_matches_numpy(self, seed):
        data = np.random.randn(200)
        wr = WelfordRolling(50)
        for v in data:
            wr.update(v)
        # Population std (ddof=0 matches our implementation)
        expected_std = np.std(data[-50:])
        assert abs(wr.std - expected_std) < 1e-6

    def test_z_score_zero_std(self):
        wr = WelfordRolling(5)
        for _ in range(5):
            wr.update(1.0)  # all identical → std = 0
        assert wr.z_score(1.0) == 0.0

    def test_z_score_normal(self, seed):
        wr = WelfordRolling(100)
        data = np.random.randn(100) * 2 + 5
        for v in data:
            wr.update(v)
        z = wr.z_score(5.0)  # should be near 0 (close to mean)
        assert abs(z) < 1.5

    def test_seed_bulk_load(self, seed):
        data = list(np.random.randn(50))
        wr1 = WelfordRolling(50)
        wr1.seed(data)
        wr2 = WelfordRolling(50)
        for v in data:
            wr2.update(v)
        assert abs(wr1.mean - wr2.mean) < 1e-10
        assert abs(wr1.std - wr2.std) < 1e-10


class TestWilderATR:
    """Tests for Wilder-smoothed ATR."""

    def test_not_ready_until_warmup(self):
        atr = WilderATR(14)
        for i in range(13):
            atr.update(1.1 + i * 0.001, 1.09 + i * 0.001, 1.095 + i * 0.001)
        assert not atr.ready
        atr.update(1.115, 1.105, 1.110)
        assert atr.ready

    def test_true_range_no_gap(self):
        tr = WilderATR.true_range(1.1050, 1.0990, None)
        assert abs(tr - 0.0060) < 1e-10

    def test_true_range_with_gap(self):
        tr = WilderATR.true_range(1.1050, 1.0990, 1.0980)
        # max(0.006, |1.105 - 1.098|, |1.099 - 1.098|) = max(0.006, 0.007, 0.001) = 0.007
        assert abs(tr - 0.0070) < 1e-10

    def test_atr_positive(self, seed):
        atr = WilderATR(14)
        for _ in range(50):
            h = 1.1 + np.random.rand() * 0.01
            l = 1.09 + np.random.rand() * 0.005
            atr.update(h, l, (h + l) / 2)
        assert atr.value > 0

    def test_atr_increases_with_volatility(self, seed):
        """ATR should increase when bar ranges widen."""
        atr = WilderATR(14)
        # Calm bars
        for _ in range(30):
            atr.update(1.1002, 1.0998, 1.1000)
        calm_atr = atr.value
        # Wide bars
        for _ in range(30):
            atr.update(1.1050, 1.0950, 1.1000)
        stressed_atr = atr.value
        assert stressed_atr > calm_atr * 2


class TestEMA:
    def test_single_value(self):
        ema = EMA(12)
        assert ema.update(100.0) == 100.0

    def test_convergence(self):
        """EMA of a constant series should converge to that constant."""
        ema = EMA(12)
        for _ in range(200):
            ema.update(50.0)
        assert abs(ema.value - 50.0) < 1e-10

    def test_alpha(self):
        ema = EMA(12)
        assert abs(ema.alpha - 2.0 / 13.0) < 1e-10


class TestMACD:
    def test_ready_after_warmup(self):
        macd = MACD(fast=12, slow=26, signal=9)
        for i in range(30):
            macd.update(1.1 + i * 0.0001)
        assert macd.ready

    def test_histogram_sign_trending(self):
        """In a clear uptrend, MACD histogram should be positive."""
        macd = MACD(fast=12, slow=26, signal=9)
        for i in range(100):
            macd.update(1.1 + i * 0.001)  # steady uptrend
        assert macd.histogram > 0

    def test_histogram_sign_downtrend(self):
        macd = MACD(fast=12, slow=26, signal=9)
        for i in range(100):
            macd.update(1.2 - i * 0.001)  # downtrend
        assert macd.histogram < 0


class TestRollingDeque:
    def test_mean(self):
        rd = RollingDeque(5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rd.append(v)
        assert rd.mean == 3.0

    def test_overflow(self):
        rd = RollingDeque(3)
        for v in [1, 2, 3, 4, 5]:
            rd.append(v)
        assert rd.values == [3, 4, 5]
        assert rd.mean == 4.0


class TestCUSUM:
    def test_no_trigger_on_noise(self, seed):
        cusum = CUSUMDetector(k_mult=1.5, h_mult=3.0)
        for _ in range(100):
            triggered = cusum.update(np.random.randn() * 0.001, 0.001)
        # Should not trigger constantly on pure noise
        # (may trigger occasionally, but not on every bar)

    def test_trigger_on_shift(self):
        """A sustained upward shift should trigger CUSUM."""
        cusum = CUSUMDetector(k_mult=1.5, h_mult=3.0)
        sigma = 0.001
        # Feed a large positive shift
        triggered = False
        for _ in range(20):
            result = cusum.update(0.005, sigma)  # 5σ shift
            if result:
                triggered = True
                break
        assert triggered, "CUSUM should trigger on a 5σ sustained shift"

    def test_reset(self):
        cusum = CUSUMDetector(k_mult=1.5, h_mult=3.0)
        cusum.update(0.01, 0.001)
        assert cusum.max_cusum > 0
        cusum.reset()
        assert cusum.s_plus == 0.0
        assert cusum.s_minus == 0.0
        assert not cusum.triggered
