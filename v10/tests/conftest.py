"""
v10/tests/conftest.py
Shared fixtures for the entire test suite.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure v10 is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))


@pytest.fixture
def seed():
    """Fix all randomness."""
    np.random.seed(42)
    return 42


@pytest.fixture
def synthetic_bars(seed):
    """
    Generate 2000 M1 bars of realistic EURUSD data.
    Includes a planted volatility regime shift at bar 1000.
    Highs/lows scale with return volatility for realistic range expansion.
    """
    n = 2000
    base_price = 1.10000
    vol = np.full(n, 0.0003)  # base volatility

    # Plant a regime shift: bars 1000-1200 have 5x volatility
    vol[1000:1200] = 0.0015

    returns = np.random.randn(n) * vol
    prices = base_price * np.exp(np.cumsum(returns))

    # Highs/lows scale with vol so ATR and range expansion track the shift
    highs = prices + np.abs(np.random.randn(n)) * vol * 2
    lows = prices - np.abs(np.random.randn(n)) * vol * 2
    opens = prices + np.random.randn(n) * vol * 0.5

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, prices))
    lows = np.minimum(lows, np.minimum(opens, prices))

    index = pd.date_range("2024-05-14T06:00:00", periods=n, freq="min", tz="UTC")

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": np.random.randint(50, 500, n).astype(float),
    }, index=index)
    df.index.name = "ts_utc"
    return df


@pytest.fixture
def calm_bars(seed):
    """500 bars of very calm, low-volatility data (no regime shift)."""
    n = 500
    base_price = 1.10000
    returns = np.random.randn(n) * 0.00005  # tiny moves
    prices = base_price * np.exp(np.cumsum(returns))
    highs = prices + np.abs(np.random.randn(n)) * 0.0001
    lows = prices - np.abs(np.random.randn(n)) * 0.0001

    index = pd.date_range("2024-06-10T08:00:00", periods=n, freq="min", tz="UTC")
    df = pd.DataFrame({
        "open": prices + np.random.randn(n) * 0.00005,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": np.random.randint(50, 200, n).astype(float),
    }, index=index)
    df.index.name = "ts_utc"
    return df


@pytest.fixture
def sample_events():
    """A small set of test events."""
    return pd.DataFrame({
        "ts_utc": ["2024-05-14T12:30:00Z", "2024-05-14T14:00:00Z"],
        "currency": ["USD", "USD"],
        "event_name": ["CPI m/m", "Fed Speaker"],
        "actual": ["0.3%", ""],
        "forecast": ["0.4%", ""],
        "previous": ["0.4%", ""],
        "impact": [3, 2],
    })


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    dirs = [
        "market/eurusd/m1", "market/eurusd/m5",
        "context", "events", "detectors",
        "execution", "windows", "logs",
    ]
    for d in dirs:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def populated_data_dir(tmp_data_dir, synthetic_bars, sample_events):
    """Data directory with bars, events, and windows written to disk."""
    # Write bars as day-partitioned CSVs
    m1_dir = tmp_data_dir / "market" / "eurusd" / "m1"
    for day, group in synthetic_bars.groupby(synthetic_bars.index.strftime("%Y%m%d")):
        out = group.copy()
        out.index = out.index.strftime("%Y-%m-%dT%H:%M:%SZ")
        out.index.name = "ts_utc"
        out["symbol"] = "EURUSD"
        out.to_csv(m1_dir / f"{day}.csv")

    # Write events
    sample_events.to_csv(tmp_data_dir / "events" / "forexfactory_calendar.csv", index=False)

    # Write window manifest
    manifest = {
        "version": "1.0",
        "windows": [
            {
                "code": "TEST-STRESS",
                "start_utc": "2024-05-14T06:00:00Z",
                "end_utc": "2024-05-14T23:00:00Z",
                "notes": "Synthetic stress test",
                "family": "macro_event",
            }
        ],
    }
    with open(tmp_data_dir / "windows" / "windows.json", "w") as f:
        json.dump(manifest, f)

    return tmp_data_dir


@pytest.fixture
def config_dict():
    """Return a config dict matching config.yaml defaults."""
    return {
        "version": "1.0",
        "seed": 42,
        "instrument": {"symbol": "EURUSD", "timeframe": "M1"},
        "features": {
            "returns_windows": [20, 60],
            "z_window": 30,
            "atr_windows": [14, 28],
            "macd": {"fast": 12, "slow": 26, "signal": 9},
        },
        "detectors": {
            "kmeans": {"k": 5, "n_init": 10, "seed": 42, "calibration_days": 30},
            "cusum": {"k_mult": 1.5, "h_mult": 3.0, "sigma_window": 60},
            "z_thresholds": {"moderate": 2.0, "severe": 3.0},
            "isolation_forest": {"n_estimators": 100, "contamination": 0.01, "seed": 42},
        },
        "policy": {
            "persistence": {"label": 3, "moderate": 2, "severe": 3},
            "throttle": {"size_mult": 0.33, "duration_bars": 5},
            "pause": {"duration_bars": 10},
            "hedge": {"size_mult": 0.10, "stop_atr_mult": 0.20, "max_age_bars": 10},
            "cooldowns": {"hedge": 10},
        },
        "events": {"window_high_minutes": 15, "window_medium_minutes": 5},
    }
