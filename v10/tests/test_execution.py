"""
v10/tests/test_execution.py
Tests for execution/mt5_io.py, execution/audit.py, policy/config.py,
charting/mt5/overlay_writer.py.
"""

import csv
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from v10.execution.mt5_io import MT5Bridge
from v10.execution.audit import AuditLogger, Evaluator
from v10.policy.config import Config
from v10.charting.mt5.overlay_writer import OverlayWriter


class TestMT5Bridge:
    """Tests for the CSV command/ack bridge."""

    @pytest.fixture
    def bridge(self, tmp_path):
        return MT5Bridge(
            command_csv=str(tmp_path / "commands" / "control.csv"),
            ack_csv=str(tmp_path / "acks" / "ack.csv"),
            mirror_command=str(tmp_path / "mirror" / "control.csv"),
            mirror_ack=str(tmp_path / "mirror" / "ack.csv"),
        )

    def test_write_creates_file(self, bridge):
        bridge.write_command("throttle", params={"size_mult": 0.33})
        assert Path(bridge.command_csv).exists()

    def test_nonce_generated(self, bridge):
        nonce = bridge.write_command("pause")
        assert len(nonce) == 12
        assert nonce == bridge.last_nonce

    def test_unique_nonces(self, bridge):
        n1 = bridge.write_command("throttle")
        n2 = bridge.write_command("pause")
        assert n1 != n2

    def test_command_file_has_rows(self, bridge):
        bridge.write_command("throttle", reason_code="TEST_THROTTLE")
        bridge.write_command("pause", reason_code="TEST_PAUSE")
        with open(bridge.command_csv) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 3  # header + 2 commands

    def test_mirror_file_created(self, bridge):
        bridge.write_command("throttle")
        assert Path(bridge.mirror_command).exists()

    def test_read_ack_returns_none_when_empty(self, bridge):
        ack = bridge.read_ack("nonexistent")
        assert ack is None

    def test_read_ack_finds_matching_nonce(self, bridge, tmp_path):
        nonce = bridge.write_command("throttle")
        # Simulate EA writing an ack
        ack_path = Path(bridge.ack_csv)
        with open(ack_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["2024-05-14 13:30:00", "2024-05-14T13:30:00Z",
                             "throttle", "0", "ok", nonce])
        ack = bridge.read_ack(nonce)
        assert ack is not None
        assert ack["nonce"] == nonce
        assert ack["status_code"] == "0"


class TestAuditLogger:
    """Tests for the audit logging system."""

    @pytest.fixture
    def logger(self, tmp_path):
        return AuditLogger(log_path=str(tmp_path / "logs"))

    def test_log_row_creates_file(self, logger, tmp_path):
        logger.log_row(
            timestamp="2024-05-14T13:30:00Z",
            signals={"regime_label": "stressed", "anomaly_level": "severe"},
            action="pause",
            reason="test reason",
        )
        day_dir = tmp_path / "logs" / "20240514"
        assert day_dir.exists()
        assert (day_dir / "decisions.csv").exists()

    def test_log_row_appends(self, logger):
        for i in range(5):
            logger.log_row(
                timestamp=f"2024-05-14T13:{30+i}:00Z",
                signals={"regime_label": "calm"},
                action="allow",
                reason="clear",
            )
        # Read back
        with open(logger._decisions_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 6  # header + 5 rows

    def test_day_rotation(self, logger):
        logger.log_row("2024-05-14T13:30:00Z", {}, "allow", "r")
        logger.log_row("2024-05-15T09:00:00Z", {}, "allow", "r")
        # Should have two day directories
        assert logger._today == "20240515"


class TestEvaluator:
    """Tests for the evaluation metrics."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        return Evaluator(results_path=str(tmp_path / "results"))

    def test_max_drawdown_flat(self, evaluator):
        equity = pd.Series([100, 100, 100, 100])
        dd = evaluator._max_drawdown(equity)
        assert dd == 0.0

    def test_max_drawdown_simple(self, evaluator):
        equity = pd.Series([100, 90, 95, 80, 100])
        dd = evaluator._max_drawdown(equity)
        assert abs(dd - 0.20) < 1e-10  # 20% drawdown

    def test_max_drawdown_empty(self, evaluator):
        dd = evaluator._max_drawdown(pd.Series([], dtype=float))
        assert dd == 0.0

    def test_half_recovery(self, evaluator):
        # Peak at 100, trough at 80 (DD=20), half recovery at 90
        idx = pd.date_range("2024-01-01", periods=10, freq="min", tz="UTC")
        equity = pd.Series([100, 95, 90, 85, 80, 82, 85, 88, 90, 95], index=idx)
        t_half = evaluator._half_recovery(equity)
        assert t_half > 0  # should find recovery

    def test_compute_metrics_calm(self, evaluator):
        decisions = pd.DataFrame({
            "action": ["allow"] * 90 + ["throttle"] * 8 + ["pause"] * 2,
            "event_flag": [0] * 100,
            "regime_label": ["calm"] * 100,
        })
        calm_mask = pd.Series([True] * 100)
        metrics = evaluator.compute_metrics(decisions, calm_mask)
        assert metrics["pause_rate"] == 2.0
        assert metrics["throttle_rate"] == 8.0
        assert not metrics["thresholds_met"]  # throttle > 5%

    def test_compute_metrics_passing(self, evaluator):
        decisions = pd.DataFrame({
            "action": ["allow"] * 97 + ["throttle"] * 3,
        })
        calm_mask = pd.Series([True] * 100)
        metrics = evaluator.compute_metrics(decisions, calm_mask)
        assert metrics["pause_rate"] == 0.0
        assert metrics["throttle_rate"] == 3.0
        assert metrics["thresholds_met"]  # both under limits


class TestConfig:
    """Tests for YAML config loading."""

    def test_load_from_yaml(self):
        cfg = Config.from_yaml("v10/config.yaml")
        assert cfg.version == "1.0"
        assert cfg.seed == 42
        assert cfg.kmeans_k == 5
        assert cfg.z_moderate == 2.0
        assert cfg.z_severe == 3.0
        assert cfg.throttle_size_mult == 0.33
        assert cfg.pause_duration == 10

    def test_defaults_on_missing_file(self):
        cfg = Config.from_yaml("nonexistent.yaml")
        assert cfg.seed == 42
        assert cfg.kmeans_k == 5

    def test_config_hash_deterministic(self):
        cfg1 = Config.from_yaml("v10/config.yaml")
        cfg2 = Config.from_yaml("v10/config.yaml")
        assert cfg1.config_hash() == cfg2.config_hash()

    def test_all_parameters_loaded(self):
        cfg = Config.from_yaml("v10/config.yaml")
        assert cfg.returns_windows == [20, 60]
        assert cfg.atr_windows == [14, 28]
        assert cfg.macd_fast == 12
        assert cfg.macd_slow == 26
        assert cfg.macd_signal == 9
        assert cfg.cusum_k_mult == 1.5
        assert cfg.cusum_h_mult == 3.0
        assert cfg.persistence_label == 3
        assert cfg.persistence_moderate == 2
        assert cfg.persistence_severe == 3
        assert cfg.hedge_size_mult == 0.10
        assert cfg.hedge_stop_atr_mult == 0.20
        assert cfg.hedge_max_age == 10
        assert cfg.event_window_high_minutes == 15
        assert cfg.event_window_medium_minutes == 5
        assert cfg.baseline_ema_fast == 12
        assert cfg.baseline_stop_atr_mult == 0.8


class TestOverlayWriter:
    """Tests for charting/mt5/overlay_writer.py."""

    @pytest.fixture
    def writer(self, tmp_path):
        return OverlayWriter(output_dir=str(tmp_path / "overlays"))

    def test_write_sr(self, writer, tmp_path):
        writer.write_support_resistance([
            {"price": 1.10500, "type": "support", "strength": 3},
            {"price": 1.11200, "type": "resistance", "strength": 2},
        ])
        path = tmp_path / "overlays" / "support_resistance.csv"
        assert path.exists()
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2

    def test_write_poi(self, writer, tmp_path):
        writer.write_poi([
            {"ts_utc": "2024-05-14T12:30:00Z", "price": 1.1050,
             "label": "CPI", "notes": "surprise"},
        ])
        assert (tmp_path / "overlays" / "poi.csv").exists()

    def test_write_fvg(self, writer, tmp_path):
        writer.write_fvg([
            {"ts_utc": "2024-05-14T12:30:00Z", "high": 1.1060,
             "low": 1.1040, "direction": "bull"},
        ])
        assert (tmp_path / "overlays" / "fvg.csv").exists()

    def test_write_structure(self, writer, tmp_path):
        writer.write_market_structure([
            {"ts_utc": "2024-05-14T12:30:00Z", "price": 1.1050,
             "type": "swing_hi", "label": "HH"},
        ])
        assert (tmp_path / "overlays" / "market_structure.csv").exists()

    def test_write_regime_state_appends(self, writer, tmp_path):
        writer.write_regime_state("calm", "none", "allow", "clear")
        writer.write_regime_state("volatile", "moderate", "throttle", "z_range=2.3")
        path = tmp_path / "overlays" / "regime_state.csv"
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2 appends

    def test_clear_all(self, writer, tmp_path):
        writer.write_regime_state("calm", "none", "allow", "clear")
        writer.write_support_resistance([{"price": 1.1, "type": "support"}])
        writer.clear_all()
        assert not (tmp_path / "overlays" / "regime_state.csv").exists()
        assert not (tmp_path / "overlays" / "support_resistance.csv").exists()
