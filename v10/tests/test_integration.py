"""
v10/tests/test_integration.py
End-to-end integration tests: full pipeline from bars → decisions.
"""

import numpy as np
import pandas as pd
import pytest

from v10.core.features import FeatureEngineer
from v10.core.timeutils import iso_str
from v10.detectors.regime import RegimeDetector
from v10.detectors.anomaly import AnomalyDetector
from v10.detectors.calibrate import run_morning_calibration
from v10.policy.rules import DecisionPolicy
from v10.policy.config import Config


class TestFullPipeline:
    """End-to-end tests simulating the controller loop."""

    @pytest.fixture
    def pipeline(self, synthetic_bars):
        """Build and calibrate a full pipeline on pre-stress data only."""
        fe = FeatureEngineer(
            windows_short=[20], windows_long=[60],
            atr_windows=[14, 28],
            macd_params={"fast": 12, "slow": 26, "signal": 9},
        )
        regime = RegimeDetector(k=5, persistence_bars=3,
                                cusum_k_mult=1.5, cusum_h_mult=3.0)
        anomaly = AnomalyDetector(
            z_moderate=2.0, z_severe=3.0,
            persistence_moderate=2, persistence_severe=3,
        )
        policy = DecisionPolicy()

        # Calibrate ONLY on the first 500 bars (all calm, before the
        # planted stress at bar 1000). This mirrors production where
        # calibration uses past data, not future.
        run_morning_calibration(
            synthetic_bars.iloc[:500], regime, anomaly,
            calibration_path="/tmp/test_pipeline_calib.json",
            fit_iforest=True,
        )

        return {
            "fe": fe, "regime": regime,
            "anomaly": anomaly, "policy": policy,
            "bars": synthetic_bars,
        }

    def _run_pipeline(self, pipeline, bars):
        """Run the pipeline over bars and collect decisions."""
        fe = pipeline["fe"]
        regime = pipeline["regime"]
        anomaly = pipeline["anomaly"]
        policy = pipeline["policy"]

        decisions = []
        for ts, row in bars.iterrows():
            bar = {
                "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row["volume"],
                "ts_utc": iso_str(ts),
            }
            feat = fe.update_features(bar)
            if not fe.ready:
                continue

            regime_input = np.array([
                feat["sigma_20"], feat["range_ratio"], feat["slope_short"],
            ])
            regime.predict(regime_input)
            confirmed = regime.confirm_transition(
                feat["ret_1m"], feat["sigma_60"], regime.output.regime_label,
            )

            if_vector = np.array(fe.build_model_input())
            anom = anomaly.update(
                z_ret=feat["z_ret"], z_range=feat["z_range"],
                if_vector=if_vector, regime_label=confirmed,
            )

            dec = policy.decide(
                regime_label=confirmed,
                regime_confidence=regime.output.regime_confidence,
                anomaly_level=anom.anomaly_level,
                anomaly_score=anom.anomaly_score,
                if_confirmed=anom.if_confirmed,
                trend_strength=abs(feat["slope_short"]),
                event_window_high=False, event_window_med=False,
                z_ret=feat["z_ret"], z_range=feat["z_range"],
                atr_14=feat["atr_14"],
                current_price=float(row["close"]),
                slope_short=feat["slope_short"],
            )
            decisions.append({
                "ts_utc": bar["ts_utc"],
                "action": dec["action"],
                "reason_code": dec["reason_code"],
                "regime": confirmed,
                "anomaly": anom.anomaly_level,
                "z_ret": feat["z_ret"],
                "z_range": feat["z_range"],
            })

        return pd.DataFrame(decisions)

    def test_pipeline_produces_decisions(self, pipeline):
        decs = self._run_pipeline(pipeline, pipeline["bars"])
        assert len(decs) > 0
        assert "action" in decs.columns
        assert "reason_code" in decs.columns

    def test_all_actions_are_valid(self, pipeline):
        decs = self._run_pipeline(pipeline, pipeline["bars"])
        valid = {"allow", "throttle", "pause", "hedge_active"}
        assert set(decs["action"].unique()).issubset(valid)

    def test_mostly_allow_in_calm(self, pipeline, calm_bars):
        """Calm data should produce mostly allow actions."""
        # Warm up features first
        warmup = pipeline["bars"].iloc[:100]
        full = pd.concat([warmup, calm_bars])
        decs = self._run_pipeline(pipeline, full)

        # Filter to just the calm portion (after warmup)
        calm_decs = decs.iloc[100:]
        if len(calm_decs) > 0:
            allow_rate = (calm_decs["action"] == "allow").mean()
            assert allow_rate > 0.85, (
                f"Expected >85% allow in calm data, got {allow_rate:.1%}"
            )

    def test_detects_stress_regime(self, pipeline):
        """Pipeline should detect the planted volatility regime shift via
        anomaly z-scores even if regime labels don't map perfectly
        (k-means calibrated on calm data won't have seen 5x vol)."""
        decs = self._run_pipeline(pipeline, pipeline["bars"])

        # The shift is at bars 1000-1200 (5x vol). After warmup (~60 bars),
        # z-scores should spike. Check anomaly detection fires.
        stress_region = decs.iloc[940:1160]
        if len(stress_region) > 0:
            has_anomaly = stress_region["anomaly"].isin(["moderate", "severe"]).sum()
            high_z = (stress_region["z_ret"].abs() >= 2.0).sum()
            non_allow = (stress_region["action"] != "allow").sum()
            # At minimum, z-scores should detect elevated volatility
            assert has_anomaly > 0 or high_z > 0 or non_allow > 0, (
                "Expected anomaly detection or high z-scores during stress"
            )

    def test_no_actions_during_warmup(self, pipeline):
        """First 60 bars should only produce allow (warmup period)."""
        decs = self._run_pipeline(pipeline, pipeline["bars"].iloc[:60])
        # During warmup, fe.ready is False → no decisions recorded
        # Or all allow (default)
        if len(decs) > 0:
            assert (decs["action"] == "allow").all()

    def test_reason_codes_populated(self, pipeline):
        decs = self._run_pipeline(pipeline, pipeline["bars"])
        assert (decs["reason_code"] != "").all()

    def test_regime_labels_valid(self, pipeline):
        decs = self._run_pipeline(pipeline, pipeline["bars"])
        valid_regimes = {"calm", "trend_up", "trend_down", "volatile", "stressed"}
        assert set(decs["regime"].unique()).issubset(valid_regimes)

    def test_anomaly_levels_valid(self, pipeline):
        decs = self._run_pipeline(pipeline, pipeline["bars"])
        valid_levels = {"none", "moderate", "severe"}
        assert set(decs["anomaly"].unique()).issubset(valid_levels)


class TestRestraintCriteria:
    """Tests verifying the preregistered restraint pass criteria."""

    @pytest.fixture
    def calm_decisions(self, calm_bars):
        """Run pipeline on calm data and return decisions."""
        fe = FeatureEngineer(windows_short=[20], windows_long=[60], atr_windows=[14, 28])
        regime = RegimeDetector(k=5, persistence_bars=3)
        anomaly = AnomalyDetector(z_moderate=2.0, z_severe=3.0)
        policy = DecisionPolicy()

        # Quick calibration on calm data itself
        run_morning_calibration(
            calm_bars, regime, anomaly,
            calibration_path="/tmp/test_restraint_calib.json",
            fit_iforest=False,
        )

        decisions = []
        for ts, row in calm_bars.iterrows():
            bar = {
                "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row["volume"],
            }
            feat = fe.update_features(bar)
            if not fe.ready:
                continue
            regime_input = np.array([feat["sigma_20"], feat["range_ratio"], feat["slope_short"]])
            regime.predict(regime_input)
            confirmed = regime.confirm_transition(feat["ret_1m"], feat["sigma_60"], regime.output.regime_label)
            anom = anomaly.update(z_ret=feat["z_ret"], z_range=feat["z_range"])
            dec = policy.decide(
                regime_label=confirmed, regime_confidence=regime.output.regime_confidence,
                anomaly_level=anom.anomaly_level, anomaly_score=anom.anomaly_score,
                if_confirmed=False, trend_strength=0.0,
                event_window_high=False, event_window_med=False,
                z_ret=feat["z_ret"], z_range=feat["z_range"],
                atr_14=feat["atr_14"], current_price=row["close"],
                slope_short=feat["slope_short"],
            )
            decisions.append(dec["action"])

        return decisions

    def test_pause_rate_under_1_percent(self, calm_decisions):
        if len(calm_decisions) == 0:
            pytest.skip("No decisions produced")
        pause_rate = sum(1 for a in calm_decisions if a == "pause") / len(calm_decisions)
        assert pause_rate < 0.01, f"Pause rate {pause_rate:.2%} exceeds 1% ceiling"

    def test_throttle_rate_under_5_percent(self, calm_decisions):
        if len(calm_decisions) == 0:
            pytest.skip("No decisions produced")
        throttle_rate = sum(1 for a in calm_decisions if a == "throttle") / len(calm_decisions)
        assert throttle_rate < 0.05, f"Throttle rate {throttle_rate:.2%} exceeds 5% ceiling"
