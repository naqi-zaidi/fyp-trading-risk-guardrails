"""
v10/controller.py
Main controller: live loop that runs the guardrail pipeline bar-by-bar.

This is the top-level entry point for real-time operation.
It orchestrates: ingest → features → detectors → policy → bridge → audit.

Usage:
    python -m v10.controller --config v10/config.yaml

For replay mode (non-live), use v10.scripts.replay_window instead.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v10.policy.config import Config
from v10.core.ingest import MarketDataLoader, EventCalendarLoader, ContextLoader
from v10.core.features import FeatureEngineer
from v10.core.timeutils import iso_str, utc_now
from v10.detectors.regime import RegimeDetector
from v10.detectors.anomaly import AnomalyDetector
from v10.detectors.calibrate import run_morning_calibration
from v10.policy.rules import DecisionPolicy
from v10.execution.mt5_io import MT5Bridge
from v10.execution.audit import AuditLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Graceful shutdown
_running = True


def _handle_signal(sig, frame):
    global _running
    logger.info("Shutdown signal received. Finishing current bar...")
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def build_pipeline(cfg: Config) -> dict:
    """Instantiate all pipeline components from config."""
    fe = FeatureEngineer(
        windows_short=[min(cfg.returns_windows)],
        windows_long=[max(cfg.returns_windows)],
        atr_windows=cfg.atr_windows,
        macd_params={"fast": cfg.macd_fast, "slow": cfg.macd_slow, "signal": cfg.macd_signal},
        z_window=cfg.z_window,
        small_body_lookback=cfg.small_body_lookback,
        small_body_threshold=cfg.small_body_threshold,
    )

    regime = RegimeDetector(
        k=cfg.kmeans_k,
        n_init=cfg.kmeans_n_init,
        seed=cfg.kmeans_seed,
        cusum_k_mult=cfg.cusum_k_mult,
        cusum_h_mult=cfg.cusum_h_mult,
        persistence_bars=cfg.persistence_label,
    )

    anomaly = AnomalyDetector(
        z_moderate=cfg.z_moderate,
        z_severe=cfg.z_severe,
        persistence_moderate=cfg.persistence_moderate,
        persistence_severe=cfg.persistence_severe,
        if_n_estimators=cfg.if_n_estimators,
        if_contamination=cfg.if_contamination,
        if_seed=cfg.if_seed,
    )

    policy = DecisionPolicy(
        throttle_size_mult=cfg.throttle_size_mult,
        throttle_duration=cfg.throttle_duration,
        pause_duration=cfg.pause_duration,
        hedge_size_mult=cfg.hedge_size_mult,
        hedge_stop_atr_mult=cfg.hedge_stop_atr_mult,
        hedge_max_age=cfg.hedge_max_age,
        hedge_cooldown=cfg.hedge_cooldown,
        persistence_label=cfg.persistence_label,
        persistence_moderate=cfg.persistence_moderate,
        persistence_severe=cfg.persistence_severe,
    )

    bridge = MT5Bridge(
        command_csv=cfg.paths.get("mt5_commands", "MQL5/Files/commands/control.csv"),
        ack_csv=cfg.paths.get("mt5_acks", "MQL5/Files/acks/ack.csv"),
        mirror_command=cfg.paths.get("execution_control", "data/execution/control.csv"),
        mirror_ack=cfg.paths.get("execution_ack", "data/execution/ack.csv"),
    )

    audit = AuditLogger(log_path=cfg.paths.get("logs", "logs"))

    return {
        "fe": fe,
        "regime": regime,
        "anomaly": anomaly,
        "policy": policy,
        "bridge": bridge,
        "audit": audit,
    }


def process_bar(
    bar: dict,
    pipeline: dict,
    cfg: Config,
    write_mt5: bool = True,
) -> dict:
    """
    Process a single bar through the full pipeline.
    Returns the decision dict.
    """
    fe = pipeline["fe"]
    regime = pipeline["regime"]
    anomaly = pipeline["anomaly"]
    policy = pipeline["policy"]
    bridge = pipeline["bridge"]
    audit = pipeline["audit"]

    ts = bar.get("ts_utc", iso_str(utc_now()))

    # 1. Features
    feat = fe.update_features(bar)
    if not fe.ready:
        return {"action": "allow", "reason_code": "WARMUP", "reason_text": "Feature warmup"}

    # 2. Regime
    regime_input = np.array([feat["sigma_20"], feat["range_ratio"], feat["slope_short"]])
    regime.predict(regime_input)
    confirmed_label = regime.confirm_transition(
        feat["ret_1m"], feat["sigma_60"], regime.output.regime_label
    )

    # 3. Anomaly
    if_vector = np.array(fe.build_model_input())
    anom = anomaly.update(
        z_ret=feat["z_ret"],
        z_range=feat["z_range"],
        if_vector=if_vector,
        regime_label=confirmed_label,
        event_window_high=bool(feat.get("event_window_high", 0)),
    )

    # 4. Decision
    decision = policy.decide(
        regime_label=confirmed_label,
        regime_confidence=regime.output.regime_confidence,
        anomaly_level=anom.anomaly_level,
        anomaly_score=anom.anomaly_score,
        if_confirmed=anom.if_confirmed,
        trend_strength=abs(feat["slope_short"]),
        event_window_high=bool(feat.get("event_window_high", 0)),
        event_window_med=bool(feat.get("event_window_med", 0)),
        z_ret=feat["z_ret"],
        z_range=feat["z_range"],
        atr_14=feat["atr_14"],
        current_price=float(bar.get("close", 0)),
        slope_short=feat["slope_short"],
    )

    # 5. Audit
    signals = {
        "regime_label": confirmed_label,
        "regime_confidence": regime.output.regime_confidence,
        "anomaly_level": anom.anomaly_level,
        "anomaly_score": anom.anomaly_score,
        "if_confirmed": anom.if_confirmed,
        "z_ret": feat["z_ret"],
        "z_range": feat["z_range"],
        "reason_code": decision["reason_code"],
        "params": decision.get("params", {}),
    }
    audit.log_row(
        timestamp=ts,
        signals=signals,
        action=decision["action"],
        reason=decision["reason_text"],
        config_version=cfg.version,
        calib_version=regime.calib_version,
    )

    # 6. MT5 bridge
    if write_mt5 and decision["action"] != "allow":
        nonce = bridge.write_command(
            action=decision["action"],
            symbol=cfg.symbol,
            params=decision.get("params"),
            config_version=cfg.version,
            reason_code=decision["reason_code"],
            reason_text=decision["reason_text"],
        )
        decision["nonce"] = nonce

    return decision


def main():
    parser = argparse.ArgumentParser(description="Guardrail controller")
    parser.add_argument("--config", default="v10/config.yaml")
    parser.add_argument("--no-mt5", action="store_true", help="Skip MT5 bridge writes")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    pipeline = build_pipeline(cfg)

    # Load calibration if available
    calib_path = cfg.paths.get("regime_calibration", "data/detectors/regime_calibration.json")
    if Path(calib_path).exists():
        pipeline["regime"].load_calibration(calib_path)
        logger.info("Loaded calibration from %s", calib_path)
    else:
        logger.warning("No calibration found at %s. Detectors will run uncalibrated.", calib_path)

    logger.info(
        "Controller started. Config version=%s, hash=%s",
        cfg.version, cfg.config_hash(),
    )
    logger.info("Waiting for bars... (Ctrl+C to stop)")

    # In a real deployment, this would poll MT5 for new bars.
    # For now, this serves as the structural entry point.
    # Integration with MT5's Python API would go here:
    #
    #   import MetaTrader5 as mt5
    #   while _running:
    #       rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
    #       bar = rates_to_dict(rates[-1])
    #       decision = process_bar(bar, pipeline, cfg, write_mt5=not args.no_mt5)
    #       time.sleep(60)  # wait for next bar

    print("Controller ready. Use process_bar() to feed bars programmatically.")
    print("For window replay, use: python -m v10.scripts.replay_window")


if __name__ == "__main__":
    main()
