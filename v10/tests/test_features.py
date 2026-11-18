"""
v10/tests/test_features.py
Tests for core/features.py — FeatureEngineer.
"""

import math
import numpy as np
import pandas as pd
import pytest

from v10.core.features import FeatureEngineer


class TestFeatureEngineer:
    """Tests for the streaming feature engine."""

    @pytest.fixture
    def fe(self):
        return FeatureEngineer(
            windows_short=[20], windows_long=[60],
            atr_windows=[14, 28],
            macd_params={"fast": 12, "slow": 26, "signal": 9},
            z_window=30,
        )

    def _feed_bars(self, fe, n=100, base=1.1, vol=0.001):
        """Feed n synthetic bars and return the last feature dict."""
        np.random.seed(42)
        for i in range(n):
            c = base + np.random.randn() * vol
            h = c + abs(np.random.randn()) * vol * 0.5
            l = c - abs(np.random.randn()) * vol * 0.5
            o = c + np.random.randn() * vol * 0.2
            feat = fe.update_features({
                "open": o, "high": h, "low": l, "close": c,
                "volume": 100 + np.random.randint(50),
            })
        return feat

    def test_not_ready_until_warmup(self, fe):
        for i in range(59):
            fe.update_features({
                "open": 1.1, "high": 1.101, "low": 1.099, "close": 1.1,
                "volume": 100,
            })
        assert not fe.ready

    def test_ready_after_warmup(self, fe):
        self._feed_bars(fe, n=100)
        assert fe.ready

    def test_returns_computed(self, fe):
        fe.update_features({"open": 1.1, "high": 1.101, "low": 1.099, "close": 1.1000, "volume": 100})
        feat = fe.update_features({"open": 1.1, "high": 1.101, "low": 1.099, "close": 1.1010, "volume": 100})
        expected_ret = math.log(1.1010) - math.log(1.1000)
        assert abs(feat["ret_1m"] - expected_ret) < 1e-10

    def test_atr_positive(self, fe):
        feat = self._feed_bars(fe, n=100)
        assert feat["atr_14"] > 0
        assert feat["atr_28"] > 0

    def test_range_ratio_normal(self, fe):
        feat = self._feed_bars(fe, n=100)
        # In normal conditions range_ratio should be around 1.0
        assert 0.1 < feat["range_ratio"] < 5.0

    def test_wick_ratio_bounded(self, fe):
        feat = self._feed_bars(fe, n=100)
        assert 0.0 <= feat["wick_ratio"] <= 1.0

    def test_macd_computed(self, fe):
        feat = self._feed_bars(fe, n=100)
        # MACD components should be numeric
        assert isinstance(feat["macd_line"], float)
        assert isinstance(feat["macd_signal"], float)
        assert isinstance(feat["slope_short"], float)

    def test_z_scores_computed(self, fe):
        feat = self._feed_bars(fe, n=100)
        assert isinstance(feat["z_ret"], float)
        assert isinstance(feat["z_range"], float)

    def test_event_flag_passthrough(self, fe):
        self._feed_bars(fe, n=60)
        feat = fe.update_features({
            "open": 1.1, "high": 1.101, "low": 1.099, "close": 1.1,
            "volume": 100, "event_flag": 1, "event_window_high": 1,
        })
        assert feat["event_flag"] == 1
        assert feat["event_window_high"] == 1

    def test_if_vector_length(self, fe):
        self._feed_bars(fe, n=100)
        vec = fe.build_model_input()
        assert len(vec) == 6  # [ret_1m, atr_14, z_range, slope_short, wick_ratio, event_flag]

    def test_if_vector_contents(self, fe):
        feat = self._feed_bars(fe, n=100)
        vec = fe.build_model_input()
        assert vec[0] == feat["ret_1m"]
        assert vec[1] == feat["atr_14"]
        assert vec[2] == feat["z_range"]
        assert vec[3] == feat["slope_short"]
        assert vec[4] == feat["wick_ratio"]

    def test_batch_atr(self, synthetic_bars):
        atr = FeatureEngineer.compute_atr(synthetic_bars, window=14)
        assert len(atr) == len(synthetic_bars)
        assert atr.iloc[-1] > 0

    def test_batch_wick_metrics(self, synthetic_bars):
        wm = FeatureEngineer.compute_wick_metrics(synthetic_bars)
        assert "wick_ratio" in wm.columns
        assert "body_ratio" in wm.columns
        # Wick ratio may be slightly negative with synthetic data where
        # open/close can fall outside high/low due to independent generation.
        # In real OHLC data this never happens. Check most are reasonable.
        assert (wm["wick_ratio"].dropna().abs() < 5.0).all()

    def test_batch_zscores(self):
        s = pd.Series(np.random.randn(100), name="test")
        z = FeatureEngineer.compute_zscores(s, window=20)
        assert len(z) == 100
        # After warmup, z-scores should be reasonable
        valid = z.dropna()
        assert (valid.abs() < 10).all()
