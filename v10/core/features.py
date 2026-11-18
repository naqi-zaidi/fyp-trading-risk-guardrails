"""
v10/core/features.py
Incremental, online feature engineering for the guardrail pipeline.

FeatureEngineer
    .update_features(bar)  — feed one OHLCV bar dict, updates all rolling stats
    .compute_atr(df)       — batch ATR for calibration
    .compute_range_expansion(df)
    .compute_wick_metrics(df)
    .compute_zscores(df)
    .build_model_input(seq_len) — return the compact IF vector

All features are O(1) amortised per bar (no re-scanning of history).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from v10.core.rolling import WelfordRolling, WilderATR, EMA, MACD, RollingDeque


class FeatureEngineer:
    """
    Streaming feature engine.  Feed bars one at a time via update_features().

    Attributes
    ----------
    windows_short : list[int] — rolling return windows, e.g. [20]
    windows_long  : list[int] — rolling return windows, e.g. [60]
    """

    def __init__(
        self,
        windows_short: list[int] | None = None,
        windows_long: list[int] | None = None,
        atr_windows: list[int] | None = None,
        macd_params: dict | None = None,
        z_window: int = 30,
        small_body_lookback: int = 20,
        small_body_threshold: float = 0.20,
        wick_epsilon: float = 1e-10,
    ):
        self.windows_short = windows_short or [20]
        self.windows_long = windows_long or [60]
        self.z_window = z_window
        self.small_body_lookback = small_body_lookback
        self.small_body_threshold = small_body_threshold
        self.wick_epsilon = wick_epsilon

        # Rolling return stats
        all_windows = sorted(set(self.windows_short + self.windows_long))
        self._ret_rollers: dict[int, WelfordRolling] = {
            w: WelfordRolling(w) for w in all_windows
        }

        # ATR
        atr_windows = atr_windows or [14, 28]
        self._atrs: dict[int, WilderATR] = {
            w: WilderATR(w) for w in atr_windows
        }

        # MACD
        mp = macd_params or {"fast": 12, "slow": 26, "signal": 9}
        self._macd = MACD(fast=mp["fast"], slow=mp["slow"], signal=mp["signal"])

        # Z-score rollers for range expansion
        self._z_range_roller = WelfordRolling(z_window)
        self._z_ret_roller = WelfordRolling(z_window)

        # Small body tracker
        self._body_ratios = RollingDeque(small_body_lookback)

        # State
        self._prev_close: Optional[float] = None
        self._bar_count: int = 0

        # Latest feature snapshot
        self.features: dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        """True when all rollers have enough data to produce valid features."""
        max_window = max(
            max(self.windows_short + self.windows_long),
            max(self._atrs.keys()),
        )
        return self._bar_count >= max_window

    def update_features(self, bar: dict) -> dict[str, Any]:
        """
        Feed one OHLCV bar and return the updated feature dict.

        Parameters
        ----------
        bar : dict with keys {open, high, low, close, volume}
              and optionally {event_flag, event_window_high, event_window_med,
              dxy_close, ts_utc}

        Returns
        -------
        dict of all current feature values.
        """
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        vol = float(bar.get("volume", 0))

        self._bar_count += 1

        # ── 1. Log return ────────────────────────────────────────────────
        if self._prev_close is not None and self._prev_close > 0:
            ret = math.log(c) - math.log(self._prev_close)
        else:
            ret = 0.0

        # ── 2. Rolling return stats ──────────────────────────────────────
        for w, roller in self._ret_rollers.items():
            roller.update(ret)

        sigma_20 = self._ret_rollers.get(20, list(self._ret_rollers.values())[0]).std
        sigma_60 = self._ret_rollers.get(60, list(self._ret_rollers.values())[-1]).std
        mu_20 = self._ret_rollers.get(20, list(self._ret_rollers.values())[0]).mean

        # ── 3. ATR ──────────────────────────────────────────────────────
        atr_values = {}
        tr = WilderATR.true_range(h, l, self._prev_close)
        for w, atr in self._atrs.items():
            atr.update(h, l, c)
            atr_values[w] = atr.value

        atr_14 = atr_values.get(14, 0.0)
        atr_28 = atr_values.get(28, 0.0)

        # ── 4. Range expansion ──────────────────────────────────────────
        range_ratio = tr / atr_28 if atr_28 > 0 else 0.0

        # ── 5. Wick metrics ─────────────────────────────────────────────
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        wick_ratio = (upper_wick + lower_wick) / (tr + self.wick_epsilon)

        body = abs(c - o)
        body_ratio = body / (tr + self.wick_epsilon)
        self._body_ratios.append(body_ratio)
        small_body_share = 0.0
        if self._body_ratios.ready:
            vals = self._body_ratios.values
            small_body_share = sum(
                1 for v in vals if v < self.small_body_threshold
            ) / len(vals)

        # ── 6. MACD / slope ─────────────────────────────────────────────
        self._macd.update(c)
        slope_short = self._macd.histogram  # macd_line − signal

        # ── 7. Z-scores (anomaly channels) ──────────────────────────────
        self._z_ret_roller.update(ret)
        self._z_range_roller.update(range_ratio)
        z_ret = self._z_ret_roller.z_score(ret)
        z_range = self._z_range_roller.z_score(range_ratio)

        # ── 8. Event proximity (passed through from bar) ────────────────
        event_flag = int(bar.get("event_flag", 0))
        event_window_high = int(bar.get("event_window_high", 0))
        event_window_med = int(bar.get("event_window_med", 0))
        mins_to_event = bar.get("mins_to_event", float("nan"))

        # ── 9. Context ──────────────────────────────────────────────────
        dxy_close = bar.get("dxy_close", float("nan"))

        # ── Assemble snapshot ────────────────────────────────────────────
        self.features = {
            "ts_utc": bar.get("ts_utc", ""),
            "ret_1m": ret,
            "sigma_20": sigma_20,
            "sigma_60": sigma_60,
            "mu_20": mu_20,
            "tr": tr,
            "atr_14": atr_14,
            "atr_28": atr_28,
            "range_ratio": range_ratio,
            "wick_ratio": wick_ratio,
            "small_body_share": small_body_share,
            "slope_short": slope_short,
            "macd_line": self._macd.line,
            "macd_signal": self._macd.signal_val,
            "macd_histogram": self._macd.histogram,
            "z_ret": z_ret,
            "z_range": z_range,
            "event_flag": event_flag,
            "event_window_high": event_window_high,
            "event_window_med": event_window_med,
            "mins_to_event": mins_to_event,
            "dxy_close": dxy_close,
            "volume": vol,
        }

        self._prev_close = c
        return self.features

    # ── Batch convenience methods (used during calibration / replay) ──────

    @staticmethod
    def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Compute ATR on a DataFrame with columns high, low, close."""
        atr = WilderATR(window)
        results = []
        for _, row in df.iterrows():
            results.append(atr.update(row["high"], row["low"], row["close"]))
        return pd.Series(results, index=df.index, name=f"atr_{window}")

    @staticmethod
    def compute_range_expansion(df: pd.DataFrame, atr_col: str = "atr_28") -> pd.Series:
        """TR / ATR ratio."""
        prev_close = df["close"].shift(1)
        hl = df["high"] - df["low"]
        hc = (df["high"] - prev_close).abs()
        lc = (df["low"] - prev_close).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return (tr / df[atr_col].replace(0, np.nan)).fillna(0.0).rename("range_ratio")

    @staticmethod
    def compute_wick_metrics(df: pd.DataFrame, epsilon: float = 1e-10) -> pd.DataFrame:
        """Return DataFrame with wick_ratio and body_ratio columns."""
        upper = df["high"] - df[["open", "close"]].max(axis=1)
        lower = df[["open", "close"]].min(axis=1) - df["low"]
        prev_close = df["close"].shift(1)
        hl = df["high"] - df["low"]
        hc = (df["high"] - prev_close).abs()
        lc = (df["low"] - prev_close).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        wr = (upper + lower) / (tr + epsilon)
        br = (df["close"] - df["open"]).abs() / (tr + epsilon)
        return pd.DataFrame({"wick_ratio": wr, "body_ratio": br}, index=df.index)

    @staticmethod
    def compute_zscores(series: pd.Series, window: int = 30) -> pd.Series:
        """Rolling z-score of a series."""
        mu = series.rolling(window).mean()
        sigma = series.rolling(window).std()
        return ((series - mu) / sigma.replace(0, np.nan)).fillna(0.0).rename(
            f"z_{series.name}"
        )

    def build_model_input(self, seq_len: int = 1) -> list[float]:
        """
        Return the compact Isolation Forest feature vector from the latest snapshot.
        [ret_1m, atr_14, z_range, slope_short, wick_ratio, event_flag]
        """
        f = self.features
        return [
            f.get("ret_1m", 0.0),
            f.get("atr_14", 0.0),
            f.get("z_range", 0.0),
            f.get("slope_short", 0.0),
            f.get("wick_ratio", 0.0),
            float(f.get("event_flag", 0)),
        ]
