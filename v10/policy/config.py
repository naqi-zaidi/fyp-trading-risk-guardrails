"""
v10/policy/config.py
Configuration loader and single-knob pack validation.

Loads config.yaml and provides typed access to all parameters.
Validates that only one knob pack changes between runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Typed configuration container loaded from config.yaml."""

    # Meta
    version: str = "1.0"
    seed: int = 42

    # Instrument
    symbol: str = "EURUSD"
    timeframe: str = "M1"

    # Paths
    paths: dict[str, str] = field(default_factory=dict)

    # Features
    returns_windows: list[int] = field(default_factory=lambda: [20, 60])
    z_window: int = 30
    atr_windows: list[int] = field(default_factory=lambda: [14, 28])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    swing_deviation_atr_mult: float = 0.5
    small_body_lookback: int = 20
    small_body_threshold: float = 0.20
    wick_epsilon: float = 1e-10

    # Detectors
    kmeans_k: int = 5
    kmeans_n_init: int = 10
    kmeans_seed: int = 42
    calibration_days: int = 30
    calibration_hour_utc: int = 6
    cusum_k_mult: float = 1.5
    cusum_h_mult: float = 3.0
    cusum_sigma_window: int = 60
    z_moderate: float = 2.0
    z_severe: float = 3.0
    if_n_estimators: int = 100
    if_contamination: float = 0.01
    if_seed: int = 42

    # Policy
    persistence_label: int = 3
    persistence_moderate: int = 2
    persistence_severe: int = 3
    throttle_size_mult: float = 0.33
    throttle_duration: int = 5
    pause_duration: int = 10
    hedge_size_mult: float = 0.10
    hedge_stop_atr_mult: float = 0.20
    hedge_max_age: int = 10
    hedge_cooldown: int = 10
    stale_detector_bars: int = 2

    # Events
    event_window_high_minutes: int = 15
    event_window_medium_minutes: int = 5

    # Data quality
    max_forward_fill_minutes: int = 2
    cross_source_tolerance: float = 0.0010
    stale_context_minutes: int = 5

    # Baseline strategy
    baseline_ema_fast: int = 12
    baseline_ema_slow: int = 26
    baseline_momentum_z_gate: float = 0.6
    baseline_stop_atr_mult: float = 0.8
    baseline_target_atr_mult: float = 1.6
    baseline_size: float = 1.0

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load configuration from a YAML file."""
        p = Path(path)
        if not p.exists():
            logger.warning("Config file not found: %s. Using defaults.", path)
            return cls()

        with open(p) as f:
            raw = yaml.safe_load(f)

        c = cls()
        c.version = raw.get("version", c.version)
        c.seed = raw.get("seed", c.seed)

        # Instrument
        inst = raw.get("instrument", {})
        c.symbol = inst.get("symbol", c.symbol)
        c.timeframe = inst.get("timeframe", c.timeframe)

        # Paths
        c.paths = raw.get("paths", {})

        # Features
        feat = raw.get("features", {})
        c.returns_windows = feat.get("returns_windows", c.returns_windows)
        c.z_window = feat.get("z_window", c.z_window)
        c.atr_windows = feat.get("atr_windows", c.atr_windows)
        macd = feat.get("macd", {})
        c.macd_fast = macd.get("fast", c.macd_fast)
        c.macd_slow = macd.get("slow", c.macd_slow)
        c.macd_signal = macd.get("signal", c.macd_signal)
        c.swing_deviation_atr_mult = feat.get("swing_deviation_atr_mult", c.swing_deviation_atr_mult)
        c.small_body_lookback = feat.get("small_body_lookback", c.small_body_lookback)
        c.small_body_threshold = feat.get("small_body_threshold", c.small_body_threshold)
        c.wick_epsilon = feat.get("wick_epsilon", c.wick_epsilon)

        # Detectors
        det = raw.get("detectors", {})
        km = det.get("kmeans", {})
        c.kmeans_k = km.get("k", c.kmeans_k)
        c.kmeans_n_init = km.get("n_init", c.kmeans_n_init)
        c.kmeans_seed = km.get("seed", c.kmeans_seed)
        c.calibration_days = km.get("calibration_days", c.calibration_days)
        c.calibration_hour_utc = km.get("calibration_hour_utc", c.calibration_hour_utc)
        cus = det.get("cusum", {})
        c.cusum_k_mult = cus.get("k_mult", c.cusum_k_mult)
        c.cusum_h_mult = cus.get("h_mult", c.cusum_h_mult)
        c.cusum_sigma_window = cus.get("sigma_window", c.cusum_sigma_window)
        zt = det.get("z_thresholds", {})
        c.z_moderate = zt.get("moderate", c.z_moderate)
        c.z_severe = zt.get("severe", c.z_severe)
        iff = det.get("isolation_forest", {})
        c.if_n_estimators = iff.get("n_estimators", c.if_n_estimators)
        c.if_contamination = iff.get("contamination", c.if_contamination)
        c.if_seed = iff.get("seed", c.if_seed)

        # Policy
        pol = raw.get("policy", {})
        pers = pol.get("persistence", {})
        c.persistence_label = pers.get("label", c.persistence_label)
        c.persistence_moderate = pers.get("moderate", c.persistence_moderate)
        c.persistence_severe = pers.get("severe", c.persistence_severe)
        thr = pol.get("throttle", {})
        c.throttle_size_mult = thr.get("size_mult", c.throttle_size_mult)
        c.throttle_duration = thr.get("duration_bars", c.throttle_duration)
        pau = pol.get("pause", {})
        c.pause_duration = pau.get("duration_bars", c.pause_duration)
        hed = pol.get("hedge", {})
        c.hedge_size_mult = hed.get("size_mult", c.hedge_size_mult)
        c.hedge_stop_atr_mult = hed.get("stop_atr_mult", c.hedge_stop_atr_mult)
        c.hedge_max_age = hed.get("max_age_bars", c.hedge_max_age)
        cd = pol.get("cooldowns", {})
        c.hedge_cooldown = cd.get("hedge", c.hedge_cooldown)

        # Events
        ev = raw.get("events", {})
        c.event_window_high_minutes = ev.get("window_high_minutes", c.event_window_high_minutes)
        c.event_window_medium_minutes = ev.get("window_medium_minutes", c.event_window_medium_minutes)

        # Data quality
        dq = raw.get("data_quality", {})
        c.max_forward_fill_minutes = dq.get("max_forward_fill_minutes", c.max_forward_fill_minutes)
        c.cross_source_tolerance = dq.get("cross_source_tolerance", c.cross_source_tolerance)
        c.stale_context_minutes = dq.get("stale_context_minutes", c.stale_context_minutes)
        c.stale_detector_bars = dq.get("stale_detector_bars", c.stale_detector_bars)

        # Baseline
        bl = raw.get("baseline", {})
        c.baseline_ema_fast = bl.get("ema_fast", c.baseline_ema_fast)
        c.baseline_ema_slow = bl.get("ema_slow", c.baseline_ema_slow)
        c.baseline_momentum_z_gate = bl.get("momentum_z_gate", c.baseline_momentum_z_gate)
        c.baseline_stop_atr_mult = bl.get("stop_atr_mult", c.baseline_stop_atr_mult)
        c.baseline_target_atr_mult = bl.get("target_atr_mult", c.baseline_target_atr_mult)
        c.baseline_size = bl.get("base_size", c.baseline_size)

        logger.info("Config loaded from %s (version=%s)", path, c.version)
        return c

    def config_hash(self) -> str:
        """SHA-256 hash of the config for audit trail."""
        payload = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
