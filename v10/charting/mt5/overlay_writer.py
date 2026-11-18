"""
v10/charting/mt5/overlay_writer.py
Exports chart overlays to CSV files for the MT5 OverlayPlotterEA.

Writes to MQL5/Files/overlays/:
  - support_resistance.csv
  - poi.csv
  - fvg.csv
  - market_structure.csv
  - regime_state.csv

The EA polls these files and draws objects on the chart.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from v10.core.timeutils import iso_str, utc_now

logger = logging.getLogger(__name__)


class OverlayWriter:
    """
    Writes overlay CSVs for the MT5 OverlayPlotterEA.

    Parameters
    ----------
    output_dir : str — directory for overlay CSVs
                       (default: MQL5/Files/overlays)
    """

    def __init__(self, output_dir: str = "MQL5/Files/overlays"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Support / Resistance ─────────────────────────────────────────

    def write_support_resistance(
        self,
        levels: list[dict[str, Any]],
    ) -> None:
        """
        Write support/resistance levels.

        Parameters
        ----------
        levels : list of dicts with keys:
            ts_utc, price, type ("support" or "resistance"), strength (int)
        """
        path = self.output_dir / "support_resistance.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts_utc", "price", "type", "strength"])
            for lvl in levels:
                writer.writerow([
                    lvl.get("ts_utc", iso_str(utc_now())),
                    f"{lvl['price']:.5f}",
                    lvl.get("type", "support"),
                    lvl.get("strength", 1),
                ])
        logger.debug("Wrote %d S/R levels to %s", len(levels), path)

    # ── Points of Interest ───────────────────────────────────────────

    def write_poi(
        self,
        points: list[dict[str, Any]],
    ) -> None:
        """
        Write points of interest.

        Parameters
        ----------
        points : list of dicts with keys:
            ts_utc, price, label, notes
        """
        path = self.output_dir / "poi.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts_utc", "price", "label", "notes"])
            for pt in points:
                writer.writerow([
                    pt.get("ts_utc", ""),
                    f"{pt['price']:.5f}",
                    pt.get("label", ""),
                    pt.get("notes", ""),
                ])
        logger.debug("Wrote %d POIs to %s", len(points), path)

    # ── Fair Value Gaps ──────────────────────────────────────────────

    def write_fvg(
        self,
        gaps: list[dict[str, Any]],
    ) -> None:
        """
        Write fair value gaps.

        Parameters
        ----------
        gaps : list of dicts with keys:
            ts_utc, high, low, direction ("bull" or "bear")
        """
        path = self.output_dir / "fvg.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts_utc", "high", "low", "direction"])
            for gap in gaps:
                writer.writerow([
                    gap.get("ts_utc", ""),
                    f"{gap['high']:.5f}",
                    f"{gap['low']:.5f}",
                    gap.get("direction", "bull"),
                ])
        logger.debug("Wrote %d FVGs to %s", len(gaps), path)

    # ── Market Structure ─────────────────────────────────────────────

    def write_market_structure(
        self,
        structures: list[dict[str, Any]],
    ) -> None:
        """
        Write market structure points (swing highs/lows, BOS, CHoCH).

        Parameters
        ----------
        structures : list of dicts with keys:
            ts_utc, price, type ("swing_hi"/"swing_lo"/"bos"/"choch"), label
        """
        path = self.output_dir / "market_structure.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts_utc", "price", "type", "label"])
            for s in structures:
                writer.writerow([
                    s.get("ts_utc", ""),
                    f"{s['price']:.5f}",
                    s.get("type", "swing_hi"),
                    s.get("label", ""),
                ])
        logger.debug("Wrote %d structure points to %s", len(structures), path)

    # ── Regime / Action State ────────────────────────────────────────

    def write_regime_state(
        self,
        regime_label: str,
        anomaly_level: str,
        action: str,
        reason: str,
        ts_utc: str = "",
    ) -> None:
        """
        Write current regime and action state (append mode).
        The EA reads the last row to display the current state.
        """
        path = self.output_dir / "regime_state.csv"

        # Write header if file doesn't exist
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ts_utc", "regime_label", "anomaly_level",
                                 "action", "reason"])

        # Append current state
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts_utc or iso_str(utc_now()),
                regime_label,
                anomaly_level,
                action,
                reason,
            ])

    # ── Convenience: clear all overlays ──────────────────────────────

    def clear_all(self) -> None:
        """Remove all overlay CSV files (EA will clear chart objects)."""
        for name in ["support_resistance.csv", "poi.csv", "fvg.csv",
                      "market_structure.csv", "regime_state.csv"]:
            path = self.output_dir / name
            if path.exists():
                path.unlink()
        logger.info("Cleared all overlay files in %s", self.output_dir)
