"""
v10/scripts/replay_window.py
Replay a curated window end-to-end through the full guardrail pipeline.

Usage:
    python -m v10.scripts.replay_window --config config.yaml --window 2024-CPI-May

Steps:
  1. Load config and calibration.
  2. Load bars for the window range.
  3. Compute features bar-by-bar.
  4. Run detectors (regime + anomaly).
  5. Run decision policy.
  6. Log all decisions to audit trail.
  7. Optionally write commands to MT5 bridge.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v10.policy.config import Config
from v10.core.ingest import MarketDataLoader, EventCalendarLoader, ContextLoader
from v10.core.features import FeatureEngineer
from v10.core.timeutils import iso_str
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


def load_window_manifest(path: str) -> dict:
    """Load windows.json and return as dict."""
    with open(path) as f:
        return json.load(f)


def find_window(manifest: dict, code: str) -> dict:
    """Find a specific window by code."""
    for w in manifest.get("windows", []):
        if w["code"] == code:
            return w
    raise ValueError(f"Window '{code}' not found in manifest.")


def replay(
    config_path: str,
    window_code: str,
    write_mt5: bool = False,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Full replay of a single window.

    Returns DataFrame of per-bar decisions.
    """
    # ── Load config ──────────────────────────────────────────────────────
    cfg = Config.from_yaml(config_path)

    # ── Load window manifest ─────────────────────────────────────────────
    manifest = load_window_manifest(cfg.paths.get(
        "windows_manifest", "data/windows/windows.json"
    ))
    window = find_window(manifest, window_code)
    start_utc = window["start_utc"]
    end_utc = window["end_utc"]
    logger.info("Replaying window: %s [%s → %s]", window_code, start_utc, end_utc)

    # ── Load market data ─────────────────────────────────────────────────
    market = MarketDataLoader(
        source=cfg.paths.get("market_m1", "data/market/eurusd").rsplit("/m1", 1)[0],
        symbol=cfg.symbol,
        timeframe="m1",
    )
    # Convert UTC strings to YYYYMMDD for file selection
    start_date = start_utc[:10].replace("-", "")
    end_date = end_utc[:10].replace("-", "")
    bars = market.fetch_history(start=start_date, end=end_date)

    # Slice to exact window
    bars = bars[start_utc:end_utc]
    logger.info("Loaded %d bars for window.", len(bars))

    # ── Load events ──────────────────────────────────────────────────────
    try:
        events = EventCalendarLoader(
            csv_path=cfg.paths.get("events_calendar", "data/events/forexfactory_calendar.csv"),
            overrides_path=cfg.paths.get("events_overrides", "data/events/manual_overrides.csv"),
            impact_filter=2,
        )
        event_flags = events.make_event_flags(
            bars.index,
            high_minutes=cfg.event_window_high_minutes,
            med_minutes=cfg.event_window_medium_minutes,
        )
    except FileNotFoundError:
        logger.warning("Event calendar not found. Running without event flags.")
        event_flags = pd.DataFrame({
            "event_window_high": 0, "event_window_med": 0,
            "event_flag": 0, "impact": 0, "mins_to_event": np.nan,
        }, index=bars.index)

    # ── Load context ─────────────────────────────────────────────────────
    try:
        ctx = ContextLoader(dxy_source=cfg.paths.get("context_dxy", "data/context/dxy.csv"))
        dxy = ctx.load_dxy()
        dxy_aligned = ctx.align_to_index(dxy, bars.index)
    except FileNotFoundError:
        logger.warning("DXY context not found. Running without context.")
        dxy_aligned = pd.Series(np.nan, index=bars.index, name="dxy_close")

    # ── Merge features ───────────────────────────────────────────────────
    bars = bars.join(event_flags)
    bars["dxy_close"] = dxy_aligned

    # ── Initialize components ────────────────────────────────────────────
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

    audit = AuditLogger(log_path=cfg.paths.get("logs", "logs"))

    bridge = None
    if write_mt5:
        bridge = MT5Bridge(
            command_csv=cfg.paths.get("mt5_commands", "MQL5/Files/commands/control.csv"),
            ack_csv=cfg.paths.get("mt5_acks", "MQL5/Files/acks/ack.csv"),
        )

    # ── Try loading calibration ──────────────────────────────────────────
    calib_path = cfg.paths.get("regime_calibration", "data/detectors/regime_calibration.json")
    if Path(calib_path).exists():
        regime.load_calibration(calib_path)
        logger.info("Loaded existing calibration.")
    else:
        logger.info("No calibration found. Will fit from window data.")
        # Use first portion of bars for calibration
        run_morning_calibration(
            bars, regime, anomaly,
            calibration_path=calib_path,
            fit_iforest=True,
        )

    # ── Main loop ────────────────────────────────────────────────────────
    records = []
    for ts, row in bars.iterrows():
        bar = row.to_dict()
        bar["ts_utc"] = iso_str(ts)

        # 1. Features
        feat = fe.update_features(bar)
        if not fe.ready:
            continue

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
            current_price=float(row["close"]),
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
            timestamp=bar["ts_utc"],
            signals=signals,
            action=decision["action"],
            reason=decision["reason_text"],
            config_version=cfg.version,
            calib_version=regime.calib_version,
        )

        # 6. MT5 bridge (if enabled)
        if bridge and decision["action"] != "allow":
            bridge.write_command(
                action=decision["action"],
                symbol=cfg.symbol,
                params=decision.get("params"),
                config_version=cfg.version,
                reason_code=decision["reason_code"],
                reason_text=decision["reason_text"],
            )

        records.append({
            "ts_utc": bar["ts_utc"],
            **feat,
            "regime_label": confirmed_label,
            "regime_confidence": regime.output.regime_confidence,
            "anomaly_level": anom.anomaly_level,
            "anomaly_score": anom.anomaly_score,
            "action": decision["action"],
            "reason_code": decision["reason_code"],
        })

    df = pd.DataFrame(records)

    # Save results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{window_code}_decisions.csv", index=False)
    logger.info("Saved %d decisions to %s", len(df), out / f"{window_code}_decisions.csv")

    return df


def main():
    parser = argparse.ArgumentParser(description="Replay a guardrail window")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--window", required=True, help="Window code (e.g. 2024-CPI-May)")
    parser.add_argument("--mt5", action="store_true", help="Write commands to MT5 bridge")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    replay(args.config, args.window, write_mt5=args.mt5, output_dir=args.output)


if __name__ == "__main__":
    main()
