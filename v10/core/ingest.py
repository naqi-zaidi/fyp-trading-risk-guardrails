"""
v10/core/ingest.py
Data loaders for market bars, event calendar, and context series.

Classes
-------
MarketDataLoader  — EURUSD M1/M5 bars from MT5 day-partitioned CSVs.
EventCalendarLoader — Forex Factory calendar + manual overrides.
ContextLoader     — DXY and backup EURUSD from Yahoo Finance CSVs.

All loaders enforce schema, dtype, and UTC timestamp checks at read time.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from v10.core.timeutils import to_utc, ensure_utc_index, check_monotonic, find_gaps

logger = logging.getLogger(__name__)

# ── Schema definitions ───────────────────────────────────────────────────────

MARKET_SCHEMA = {
    "ts_utc": "str",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

EVENT_SCHEMA = {
    "ts_utc": "str",
    "currency": "str",
    "event_name": "str",
    "actual": "str",
    "forecast": "str",
    "previous": "str",
    "impact": "int64",
}

CONTEXT_SCHEMA = {
    "ts_utc": "str",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

IMPACT_MAP = {
    "high": 3, "High": 3, "HIGH": 3, "3": 3, 3: 3,
    "medium": 2, "Medium": 2, "MEDIUM": 2, "2": 2, 2: 2,
    "low": 1, "Low": 1, "LOW": 1, "1": 1, 1: 1,
}


def _validate_schema(df: pd.DataFrame, schema: dict, name: str) -> None:
    """Raise ValueError if required columns are missing."""
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}. Got: {list(df.columns)}")


def _drop_duplicates_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate timestamps, keep last, and sort ascending."""
    if df.index.duplicated().any():
        n_dup = df.index.duplicated().sum()
        logger.warning("Dropped %d duplicate timestamps.", n_dup)
        df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


# ═════════════════════════════════════════════════════════════════════════════
# MarketDataLoader
# ═════════════════════════════════════════════════════════════════════════════

class MarketDataLoader:
    """
    Loads EURUSD minute bars from day-partitioned CSVs.

    Attributes
    ----------
    source : str   — base directory, e.g. "data/market/eurusd"
    symbol : str   — instrument name (default "EURUSD")
    timeframe : str — "m1" or "m5"
    timezone : str  — always "UTC"
    """

    def __init__(
        self,
        source: str,
        symbol: str = "EURUSD",
        timeframe: str = "m1",
        timezone: str = "UTC",
    ):
        self.source = source
        self.symbol = symbol
        self.timeframe = timeframe
        self.timezone = timezone

    def _dir(self) -> Path:
        return Path(self.source) / self.timeframe

    def fetch_latest(self) -> pd.DataFrame:
        """Load the most recent day file available."""
        d = self._dir()
        if not d.exists():
            raise FileNotFoundError(f"Market data directory not found: {d}")
        files = sorted(d.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files in {d}")
        return self._load_file(files[-1])

    def fetch_history(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load and concatenate day files in [start, end] range.
        *start*/*end* are YYYYMMDD strings or None for unbounded.
        """
        d = self._dir()
        if not d.exists():
            raise FileNotFoundError(f"Market data directory not found: {d}")

        files = sorted(d.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files in {d}")

        selected = []
        for f in files:
            stem = f.stem  # e.g. "20240514"
            if start and stem < start:
                continue
            if end and stem > end:
                continue
            selected.append(f)

        if not selected:
            raise FileNotFoundError(
                f"No files matched range [{start}, {end}] in {d}"
            )

        frames = [self._load_file(f) for f in selected]
        df = pd.concat(frames)
        return _drop_duplicates_and_sort(df)

    def _load_file(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        self.validate_schema(df)
        df["symbol"] = self.symbol
        df = self.to_utc(df)
        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        _validate_schema(df, MARKET_SCHEMA, "MarketData")
        # dtype coercion
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    @staticmethod
    def to_utc(df: pd.DataFrame) -> pd.DataFrame:
        return ensure_utc_index(df, col="ts_utc")


# ═════════════════════════════════════════════════════════════════════════════
# EventCalendarLoader
# ═════════════════════════════════════════════════════════════════════════════

class EventCalendarLoader:
    """
    Loads Forex Factory calendar CSV and manual overrides.

    Attributes
    ----------
    csv_path : str   — path to forexfactory_calendar.csv
    timezone : str   — always "UTC"
    impact_filter : int — minimum impact to include (default 2 = Medium+)
    """

    def __init__(
        self,
        csv_path: str,
        timezone: str = "UTC",
        impact_filter: int = 2,
        overrides_path: Optional[str] = None,
    ):
        self.csv_path = csv_path
        self.timezone = timezone
        self.impact_filter = impact_filter
        self.overrides_path = overrides_path

    def load_events(self) -> pd.DataFrame:
        """Load, clean, merge overrides, and filter by impact."""
        df = pd.read_csv(self.csv_path)

        # Normalise impact to integer
        if "impact" in df.columns:
            df["impact"] = df["impact"].map(IMPACT_MAP).fillna(0).astype(int)

        _validate_schema(df, EVENT_SCHEMA, "EventCalendar")

        # Merge overrides (last write wins)
        if self.overrides_path and os.path.exists(self.overrides_path):
            ov = pd.read_csv(self.overrides_path)
            if "impact" in ov.columns:
                ov["impact"] = ov["impact"].map(IMPACT_MAP).fillna(0).astype(int)
            df = pd.concat([df, ov], ignore_index=True)
            logger.info("Merged %d manual override events.", len(ov))

        df = self.to_utc(df)

        # Filter
        df = df[df["impact"] >= self.impact_filter]
        return _drop_duplicates_and_sort(df)

    def to_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert ts_utc column to UTC DatetimeIndex."""
        return ensure_utc_index(df, col="ts_utc")

    def make_event_flags(self, index: pd.DatetimeIndex,
                         high_minutes: int = 15,
                         med_minutes: int = 5) -> pd.DataFrame:
        """
        Build per-bar event flags aligned to *index*.

        Returns a DataFrame with columns:
            event_window_high, event_window_med, event_flag,
            impact, mins_to_event
        """
        events = self.load_events()
        n = len(index)
        flags = pd.DataFrame({
            "event_window_high": np.zeros(n, dtype=int),
            "event_window_med": np.zeros(n, dtype=int),
            "event_flag": np.zeros(n, dtype=int),
            "impact": np.zeros(n, dtype=int),
            "mins_to_event": np.full(n, np.nan),
        }, index=index)

        for ts in events.index:
            impact = events.loc[ts, "impact"]
            if isinstance(impact, pd.Series):
                impact = int(impact.iloc[0])
            else:
                impact = int(impact)

            if impact >= 3:
                delta = pd.Timedelta(minutes=high_minutes)
                mask = (index >= ts - delta) & (index <= ts + delta)
                flags.loc[mask, "event_window_high"] = 1
            if impact >= 2:
                delta = pd.Timedelta(minutes=med_minutes)
                mask = (index >= ts - delta) & (index <= ts + delta)
                flags.loc[mask, "event_window_med"] = 1

            # General event flag and distance
            if impact >= 3:
                w = high_minutes
            else:
                w = med_minutes
            delta = pd.Timedelta(minutes=w)
            mask = (index >= ts - delta) & (index <= ts + delta)
            flags.loc[mask, "event_flag"] = 1
            mins = (index[mask] - ts).total_seconds() / 60.0
            # Keep closest event if multiple
            closer = np.abs(mins) < np.abs(flags.loc[mask, "mins_to_event"].values)
            closer_idx = flags.loc[mask].index[
                closer | np.isnan(flags.loc[mask, "mins_to_event"].values)
            ]
            flags.loc[closer_idx, "mins_to_event"] = mins[
                closer | np.isnan(flags.loc[mask, "mins_to_event"].values)
            ]
            flags.loc[closer_idx, "impact"] = impact

        return flags


# ═════════════════════════════════════════════════════════════════════════════
# ContextLoader
# ═════════════════════════════════════════════════════════════════════════════

class ContextLoader:
    """
    Loads DXY or backup EURUSD context series from Yahoo Finance CSVs.

    Attributes
    ----------
    dxy_source : str   — path to dxy.csv
    cache_path : str   — optional backup path (eurusd_backup.csv)
    """

    def __init__(
        self,
        dxy_source: str,
        cache_path: Optional[str] = None,
    ):
        self.dxy_source = dxy_source
        self.cache_path = cache_path

    def load_dxy(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load DXY context series, optionally sliced by date strings."""
        df = pd.read_csv(self.dxy_source)
        _validate_schema(df, CONTEXT_SCHEMA, "ContextDXY")
        df = ensure_utc_index(df, col="ts_utc")
        df = _drop_duplicates_and_sort(df)

        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df

    def align_to_index(
        self,
        dxy: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Forward-fill DXY close to match bar timestamps.
        Returns a Series named 'dxy_close'.
        """
        close = dxy["close"].reindex(index, method="ffill")
        close.name = "dxy_close"
        return close
