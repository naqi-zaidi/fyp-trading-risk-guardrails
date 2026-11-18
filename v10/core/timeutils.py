"""
v10/core/timeutils.py
UTC helpers, ISO-8601 formatting, and timezone utilities.

All at-rest timestamps are UTC ISO-8601. Only the EA layer converts
to broker/platform time for order stamping.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, Union

import pandas as pd
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

UTC = dt.timezone.utc
ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"
ISO_FMT_MS = "%Y-%m-%dT%H:%M:%S.%fZ"


# ── Conversion helpers ───────────────────────────────────────────────────────

def to_utc(ts: Union[str, dt.datetime, pd.Timestamp, np.datetime64]) -> pd.Timestamp:
    """Convert any timestamp representation to a tz-aware UTC pd.Timestamp."""
    if isinstance(ts, str):
        parsed = pd.Timestamp(ts)
    elif isinstance(ts, np.datetime64):
        parsed = pd.Timestamp(ts)
    elif isinstance(ts, dt.datetime):
        parsed = pd.Timestamp(ts)
    else:
        parsed = ts

    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed


def ensure_utc_index(df: pd.DataFrame, col: str = "ts_utc") -> pd.DataFrame:
    """
    Set *col* as the DatetimeIndex (UTC).  If already set, just ensure UTC.
    Returns a copy; does not mutate the input.
    """
    df = df.copy()
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True)
        df = df.set_index(col)
    elif df.index.name == col or isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    else:
        raise ValueError(f"Column '{col}' not found and index is not DatetimeIndex.")
    df.index.name = "ts_utc"
    return df


def utc_now() -> pd.Timestamp:
    """Current time as tz-aware UTC pd.Timestamp."""
    return pd.Timestamp.now(tz="UTC")


def iso_str(ts: Union[pd.Timestamp, dt.datetime]) -> str:
    """Format a timestamp as UTC ISO-8601 string (second precision)."""
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.strftime(ISO_FMT)


# ── Date arithmetic ─────────────────────────────────────────────────────────

def trading_day_start(date: dt.date, hour_utc: int = 6) -> pd.Timestamp:
    """Return the calibration start time (default 06:00 UTC) for a given date."""
    return pd.Timestamp(
        year=date.year, month=date.month, day=date.day,
        hour=hour_utc, minute=0, second=0, tz="UTC",
    )


def calendar_days_back(
    anchor: pd.Timestamp, days: int
) -> pd.Timestamp:
    """Return anchor − *days* calendar days as UTC Timestamp."""
    return anchor - pd.Timedelta(days=days)


# ── Gap / monotonicity checks ───────────────────────────────────────────────

def check_monotonic(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Return the subset of *index* positions that violate strict monotonicity.
    An empty index means the series is clean.
    """
    diffs = index.to_series().diff()
    bad = diffs[diffs <= pd.Timedelta(0)]
    return bad.index


def find_gaps(
    index: pd.DatetimeIndex,
    max_fill_minutes: int = 2,
) -> pd.DataFrame:
    """
    Return a DataFrame of gaps in *index* with columns:
        gap_start, gap_end, gap_minutes, fillable (bool)
    """
    diffs = index.to_series().diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(minutes=1)]
    records = []
    for ts, delta in gaps.items():
        minutes = delta.total_seconds() / 60.0
        records.append({
            "gap_start": ts - delta,
            "gap_end": ts,
            "gap_minutes": minutes,
            "fillable": minutes <= max_fill_minutes,
        })
    return pd.DataFrame(records)


# ── Event window helpers ─────────────────────────────────────────────────────

def make_event_windows(
    event_times: pd.DatetimeIndex,
    impacts: pd.Series,
    high_minutes: int = 15,
    med_minutes: int = 5,
) -> pd.DataFrame:
    """
    Build a DataFrame with columns (start_utc, end_utc, impact) defining
    the active event windows for flagging bars.
    """
    records = []
    for ts, impact in zip(event_times, impacts):
        ts = to_utc(ts)
        if impact >= 3:
            delta = pd.Timedelta(minutes=high_minutes)
        elif impact >= 2:
            delta = pd.Timedelta(minutes=med_minutes)
        else:
            continue  # Low-impact: no window
        records.append({
            "start_utc": ts - delta,
            "end_utc": ts + delta,
            "event_utc": ts,
            "impact": int(impact),
        })
    return pd.DataFrame(records)
