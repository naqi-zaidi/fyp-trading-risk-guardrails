"""
v10/detectors/calibrate.py
Morning calibration orchestrator.

At 06:00 UTC each trading day:
  1. Load last 30 calendar days of M1 bars.
  2. Compute features for calibration: vol_20, range_ratio, slope_short.
  3. Fit k-means (k=5, n_init=10, seed=42).
  4. Map centroids to semantic labels.
  5. Optionally refit Isolation Forest.
  6. Persist calibration artifacts.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from v10.core.features import FeatureEngineer
from v10.core.rolling import WelfordRolling, WilderATR, MACD
from v10.core.timeutils import trading_day_start, calendar_days_back, iso_str
from v10.detectors.regime import RegimeDetector
from v10.detectors.anomaly import AnomalyDetector

logger = logging.getLogger(__name__)


def compute_calibration_features(bars: pd.DataFrame) -> np.ndarray:
    """
    Compute the three features used for k-means calibration from raw bars.

    Parameters
    ----------
    bars : DataFrame with columns [open, high, low, close, volume]
           and a UTC DatetimeIndex.

    Returns
    -------
    ndarray of shape (n_bars, 3) with columns [vol_20, range_ratio, slope_short]
    """
    fe = FeatureEngineer(
        windows_short=[20],
        windows_long=[60],
        atr_windows=[14, 28],
        macd_params={"fast": 12, "slow": 26, "signal": 9},
    )

    records = []
    for _, row in bars.iterrows():
        bar = {
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row.get("volume", 0),
        }
        feat = fe.update_features(bar)
        if fe.ready:
            records.append([
                feat["sigma_20"],
                feat["range_ratio"],
                feat["slope_short"],
            ])

    return np.array(records) if records else np.empty((0, 3))


def compute_if_features(bars: pd.DataFrame) -> np.ndarray:
    """
    Compute the 6-dim Isolation Forest vector from raw bars.

    Returns
    -------
    ndarray of shape (n_bars, 6)
    Columns: [ret_1m, atr_14, z_range, slope_short, wick_ratio, event_flag]
    """
    fe = FeatureEngineer(
        windows_short=[20],
        windows_long=[60],
        atr_windows=[14, 28],
        macd_params={"fast": 12, "slow": 26, "signal": 9},
    )

    records = []
    for _, row in bars.iterrows():
        bar = {
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row.get("volume", 0),
            "event_flag": row.get("event_flag", 0),
        }
        feat = fe.update_features(bar)
        if fe.ready:
            records.append(fe.build_model_input())

    return np.array(records) if records else np.empty((0, 6))


def run_morning_calibration(
    bars: pd.DataFrame,
    regime_detector: RegimeDetector,
    anomaly_detector: AnomalyDetector,
    calibration_path: str = "data/detectors/regime_calibration.json",
    today: Optional[date] = None,
    fit_iforest: bool = True,
) -> str:
    """
    Execute the morning calibration sequence.

    Parameters
    ----------
    bars : DataFrame — historical M1 bars (at least 30 calendar days).
    regime_detector : RegimeDetector instance to calibrate.
    anomaly_detector : AnomalyDetector instance (IF refit if enabled).
    calibration_path : where to save the calibration JSON.
    today : override date for versioning (default: today).
    fit_iforest : whether to refit the Isolation Forest.

    Returns
    -------
    version : str — calibration version string (YYYYMMDD).
    """
    if today is None:
        today = date.today()

    version = today.strftime("%Y%m%d")
    logger.info("Starting morning calibration for %s", version)

    # 1. Compute calibration features
    X_regime = compute_calibration_features(bars)
    if X_regime.shape[0] < 50:
        logger.warning(
            "Only %d valid bars for calibration (need ≥50). "
            "Calibration may be unreliable.", X_regime.shape[0]
        )

    # 2. Fit k-means
    regime_detector.fit(X_regime)
    regime_detector.calib_version = version

    # 3. Save calibration
    regime_detector.save_calibration(calibration_path, version=version)

    # 4. Optionally fit Isolation Forest
    if fit_iforest:
        X_if = compute_if_features(bars)
        if X_if.shape[0] >= 100:
            anomaly_detector.fit_iforest(X_if)
            logger.info("Isolation Forest fitted on %d samples.", X_if.shape[0])
        else:
            logger.warning(
                "Not enough data for IF (%d samples). Skipping.", X_if.shape[0]
            )

    logger.info("Morning calibration complete (version=%s)", version)
    return version
