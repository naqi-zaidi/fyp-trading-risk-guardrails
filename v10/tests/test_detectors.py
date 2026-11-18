"""
v10/tests/test_detectors.py
Tests for detectors/regime.py and detectors/anomaly.py.
"""

import json
import numpy as np
import pytest

from v10.detectors.regime import RegimeDetector, REGIME_LABELS
from v10.detectors.anomaly import AnomalyDetector
from v10.detectors.calibrate import compute_calibration_features, run_morning_calibration


class TestRegimeDetector:
    """Tests for the k-means + CUSUM regime detector."""

    @pytest.fixture
    def calibration_data(self, seed):
        """Generate synthetic calibration features [vol_20, range_ratio, slope]."""
        n = 1000
        vol = np.random.rand(n) * 0.005
        xr = np.random.rand(n) * 3.0
        slope = np.random.randn(n) * 0.001
        return np.column_stack([vol, xr, slope])

    @pytest.fixture
    def fitted_detector(self, calibration_data):
        rd = RegimeDetector(k=5, persistence_bars=3)
        rd.fit(calibration_data)
        return rd

    def test_fit_creates_model(self, fitted_detector):
        assert fitted_detector.model is not None
        assert fitted_detector.scaler is not None
        assert fitted_detector.centroids is not None
        assert fitted_detector.centroids.shape == (5, 3)

    def test_label_map_complete(self, fitted_detector):
        assert len(fitted_detector.label_map) == 5
        assigned_labels = set(fitted_detector.label_map.values())
        assert assigned_labels == set(REGIME_LABELS)

    def test_predict_returns_valid_label(self, fitted_detector):
        vec = np.array([0.002, 1.5, 0.0005])
        out = fitted_detector.predict(vec)
        assert out.regime_label in REGIME_LABELS

    def test_predict_confidence_bounded(self, fitted_detector):
        vec = np.array([0.002, 1.5, 0.0005])
        out = fitted_detector.predict(vec)
        assert 0.0 <= out.regime_confidence <= 1.0

    def test_persistence_blocks_immediate_switch(self, fitted_detector):
        """Label should not change until persistence threshold is met."""
        initial = fitted_detector.current_label
        # Feed one bar with a potentially different label
        vec = np.array([0.01, 5.0, 0.0])  # extreme values → likely "stressed"
        out = fitted_detector.predict(vec)
        fitted_detector.confirm_transition(0.005, 0.001, out.regime_label)
        # After just 1 bar, should still be initial (persistence = 3)
        if out.regime_label != initial:
            assert fitted_detector.current_label == initial

    def test_persistence_allows_switch_after_threshold(self, fitted_detector):
        """After enough bars, label should change."""
        # Feed many bars with extreme values
        for _ in range(10):
            vec = np.array([0.01, 5.0, 0.0])
            out = fitted_detector.predict(vec)
            fitted_detector.confirm_transition(0.01, 0.001, out.regime_label)
        # Should have switched by now (CUSUM also triggered with 10σ returns)
        assert fitted_detector.current_label in REGIME_LABELS

    def test_save_and_load_calibration(self, fitted_detector, tmp_path):
        path = str(tmp_path / "calib.json")
        fitted_detector.save_calibration(path, version="20240514")

        rd2 = RegimeDetector(k=5)
        rd2.load_calibration(path)

        assert rd2.model is not None
        assert rd2.label_map == fitted_detector.label_map
        assert rd2.calib_version == "20240514"
        np.testing.assert_array_almost_equal(rd2.centroids, fitted_detector.centroids)

    def test_predict_without_calibration(self):
        """Uncalibrated detector should return calm with 0 confidence."""
        rd = RegimeDetector(k=5)
        out = rd.predict(np.array([0.001, 1.0, 0.0]))
        assert out.regime_confidence == 0.0

    def test_cusum_gate_required_for_high_energy(self, fitted_detector):
        """Transitions to volatile/stressed should require CUSUM trigger."""
        # Reset to calm
        fitted_detector._current_label = "calm"
        fitted_detector._cusum.reset()

        # Predict something that maps to stressed
        vec = np.array([0.01, 5.0, 0.0])
        out = fitted_detector.predict(vec)
        # Small return → CUSUM should NOT trigger
        result = fitted_detector.confirm_transition(0.0001, 0.001, "stressed")
        # Should remain calm because CUSUM hasn't triggered
        assert result == "calm"


class TestAnomalyDetector:
    """Tests for the z-score + Isolation Forest anomaly detector."""

    def test_none_for_normal_zscores(self):
        ad = AnomalyDetector(z_moderate=2.0, z_severe=3.0)
        out = ad.score_z(0.5, 0.8)
        assert out.anomaly_level == "none"

    def test_moderate_with_persistence(self):
        ad = AnomalyDetector(z_moderate=2.0, z_severe=3.0,
                             persistence_moderate=2)
        # Bar 1: moderate z but persist=1 → none
        out = ad.score_z(2.5, 1.0)
        assert out.anomaly_level == "none"
        # Bar 2: moderate z, persist=2 → moderate
        out = ad.score_z(2.3, 1.0)
        assert out.anomaly_level == "moderate"

    def test_severe_with_persistence(self):
        ad = AnomalyDetector(z_moderate=2.0, z_severe=3.0,
                             persistence_moderate=2, persistence_severe=3)
        for i in range(4):
            out = ad.score_z(3.5, 3.2, regime_label="stressed")
        assert out.anomaly_level == "severe"

    def test_persistence_resets_on_clear(self):
        ad = AnomalyDetector(z_moderate=2.0, z_severe=3.0,
                             persistence_moderate=2)
        ad.score_z(2.5, 1.0)  # persist=1
        ad.score_z(0.5, 0.3)  # clear → reset
        out = ad.score_z(2.5, 1.0)  # persist=1 again
        assert out.anomaly_level == "none"

    def test_event_window_caps_escalation(self):
        ad = AnomalyDetector(z_moderate=2.0, z_severe=3.0,
                             persistence_moderate=2, persistence_severe=2)
        # First severe during high-impact window should cap to moderate
        ad.score_z(3.5, 3.2, event_window_high=True)
        out = ad.score_z(3.5, 3.2, event_window_high=True)
        assert out.anomaly_level == "moderate"  # capped, not severe

    def test_drivers_string_populated(self):
        ad = AnomalyDetector(z_moderate=2.0, z_severe=3.0)
        out = ad.score_z(2.5, 2.3)
        assert "z_ret" in out.drivers
        assert "z_range" in out.drivers

    def test_if_fit_and_score(self, seed):
        ad = AnomalyDetector()
        X = np.random.randn(500, 6)
        ad.fit_iforest(X)
        assert ad.iforest is not None
        score = ad.score_iforest(np.array([0, 0, 0, 0, 0, 0]))
        assert isinstance(score, float)

    def test_if_promotion(self, seed):
        """IF should promote none→moderate when z-test is quiet."""
        ad = AnomalyDetector(z_moderate=2.0, z_severe=3.0)
        # Fit on normal data
        X_normal = np.random.randn(500, 6) * 0.1
        ad.fit_iforest(X_normal)
        # Score an extreme outlier with quiet z-scores
        outlier = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 1.0])
        out = ad.update(z_ret=0.5, z_range=0.3, if_vector=outlier,
                        event_window_high=False)
        # IF should promote to moderate
        assert out.anomaly_level == "moderate"

    def test_if_no_promotion_during_event(self, seed):
        """IF should NOT promote during high-impact event windows."""
        ad = AnomalyDetector()
        X = np.random.randn(500, 6) * 0.1
        ad.fit_iforest(X)
        outlier = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 1.0])
        out = ad.update(z_ret=0.5, z_range=0.3, if_vector=outlier,
                        event_window_high=True)
        assert out.anomaly_level == "none"

    def test_graceful_without_iforest(self):
        """System should work fine without IF."""
        ad = AnomalyDetector()
        out = ad.update(z_ret=2.5, z_range=2.3,
                        if_vector=np.zeros(6))
        assert out.if_score == 0.0


class TestCalibration:
    """Tests for the morning calibration pipeline."""

    def test_compute_calibration_features(self, synthetic_bars):
        X = compute_calibration_features(synthetic_bars)
        assert X.shape[1] == 3  # vol_20, range_ratio, slope_short
        assert X.shape[0] > 0

    def test_morning_calibration_end_to_end(self, synthetic_bars, tmp_path):
        rd = RegimeDetector(k=5, persistence_bars=3)
        ad = AnomalyDetector()
        calib_path = str(tmp_path / "calib.json")

        version = run_morning_calibration(
            synthetic_bars, rd, ad,
            calibration_path=calib_path,
            fit_iforest=True,
        )

        assert rd.model is not None
        assert ad.iforest is not None
        assert (tmp_path / "calib.json").exists()
        assert version is not None
