"""
v10/scripts/backfill_bars.py
Import MT5 History Center data into day-partitioned CSVs.

Connects to MT5 terminal via the MetaTrader5 Python package,
downloads historical bars, and writes them as day files.

Usage:
    python -m v10.scripts.backfill_bars --symbol EURUSD --timeframe M1 --days 60
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def backfill(
    symbol: str = "EURUSD",
    timeframe_str: str = "M1",
    days: int = 60,
    output_dir: str = "data/market/eurusd/m1",
) -> int:
    """
    Download bars from MT5 and write day-partitioned CSVs.
    Returns total number of bars written.
    """
    try:
        import MetaTrader5 as mt5
        import pandas as pd
    except ImportError:
        logger.error(
            "MetaTrader5 package not installed. "
            "Install with: pip install MetaTrader5"
        )
        logger.info(
            "If running on Linux, MT5 is Windows-only. "
            "You can export bars manually from MT5 History Center."
        )
        return 0

    # Map timeframe string to MT5 constant
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
    }
    timeframe = tf_map.get(timeframe_str.upper())
    if timeframe is None:
        logger.error("Unsupported timeframe: %s", timeframe_str)
        return 0

    # Initialize MT5
    if not mt5.initialize():
        logger.error("MT5 initialize failed: %s", mt5.last_error())
        return 0

    try:
        utc_now = datetime.now(timezone.utc)
        utc_from = utc_now - timedelta(days=days)

        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_now)
        if rates is None or len(rates) == 0:
            logger.error("No data returned for %s %s", symbol, timeframe_str)
            return 0

        df = pd.DataFrame(rates)
        # MT5 returns 'time' as Unix timestamp
        df["ts_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={
            "open": "open", "high": "high",
            "low": "low", "close": "close",
            "tick_volume": "volume",
        })
        df["symbol"] = symbol

        # Keep only needed columns
        df = df[["ts_utc", "open", "high", "low", "close", "volume", "symbol"]]

        # Write day-partitioned files
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        total = 0
        for day, group in df.groupby(df["ts_utc"].dt.strftime("%Y%m%d")):
            path = out / f"{day}.csv"
            group_out = group.copy()
            group_out["ts_utc"] = group_out["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            group_out.to_csv(path, index=False)
            total += len(group_out)

        logger.info(
            "Backfilled %d bars across %d days to %s",
            total, len(df.groupby(df["ts_utc"].dt.date)), out,
        )
        return total

    finally:
        mt5.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Backfill MT5 bars")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--output", default="data/market/eurusd/m1")
    args = parser.parse_args()

    count = backfill(args.symbol, args.timeframe, args.days, args.output)
    if count == 0:
        logger.warning(
            "No bars backfilled. If on Linux, export bars manually from MT5:\n"
            "  1. Open MT5 History Center (F2)\n"
            "  2. Select EURUSD → M1\n"
            "  3. Export as CSV with columns: Date,Time,Open,High,Low,Close,Volume\n"
            "  4. Save day files as data/market/eurusd/m1/YYYYMMDD.csv\n"
            "  5. Ensure ts_utc is ISO-8601 UTC format"
        )


if __name__ == "__main__":
    main()
