"""
v10/scripts/morning_fit.py
Standalone morning calibration script.

Run at 06:00 UTC each trading day to refit k-means and optionally IF.

Usage:
    python -m v10.scripts.morning_fit --config v10/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v10.policy.config import Config
from v10.core.ingest import MarketDataLoader
from v10.core.timeutils import calendar_days_back
from v10.detectors.regime import RegimeDetector
from v10.detectors.anomaly import AnomalyDetector
from v10.detectors.calibrate import run_morning_calibration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Morning calibration")
    parser.add_argument("--config", default="v10/config.yaml")
    parser.add_argument("--date", default=None, help="Override date (YYYY-MM-DD)")
    parser.add_argument("--no-iforest", action="store_true", help="Skip IF refit")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    today = date.fromisoformat(args.date) if args.date else date.today()

    # Load bars for calibration window
    market = MarketDataLoader(
        source=cfg.paths.get("market_m1", "data/market/eurusd").rsplit("/m1", 1)[0],
        symbol=cfg.symbol,
        timeframe="m1",
    )

    # Last N calendar days, excluding today
    end_date = (today - timedelta(days=1)).strftime("%Y%m%d")
    start_date = (today - timedelta(days=cfg.calibration_days)).strftime("%Y%m%d")

    try:
        bars = market.fetch_history(start=start_date, end=end_date)
    except FileNotFoundError as e:
        logger.error("Cannot load calibration data: %s", e)
        return

    logger.info("Loaded %d bars for calibration [%s → %s]", len(bars), start_date, end_date)

    # Initialize detectors
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
        if_n_estimators=cfg.if_n_estimators,
        if_contamination=cfg.if_contamination,
        if_seed=cfg.if_seed,
    )

    # Run calibration
    calib_path = cfg.paths.get("regime_calibration", "data/detectors/regime_calibration.json")
    version = run_morning_calibration(
        bars, regime, anomaly,
        calibration_path=calib_path,
        today=today,
        fit_iforest=not args.no_iforest,
    )

    logger.info("Calibration complete. Version: %s. Saved to: %s", version, calib_path)


if __name__ == "__main__":
    main()
