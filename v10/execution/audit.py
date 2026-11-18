"""
v10/execution/audit.py
AuditLogger: per-minute decision logging with reasons and UTC timestamps.
Evaluator: compute metrics for event study, switch test, and restraint.

Logging structure (per day):
  - controller.log   : high-level info and warnings
  - decisions.csv    : one row per minute with features, labels, action
  - reasons.csv      : one row per action transition
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from v10.core.timeutils import iso_str, utc_now

logger = logging.getLogger(__name__)

DECISION_COLUMNS = [
    "ts_utc", "regime_label", "regime_confidence", "anomaly_level",
    "anomaly_score", "if_confirmed", "z_ret", "z_range",
    "action", "params_json", "reason_code", "reason_text",
    "features_json", "config_version", "calib_version",
]

REASON_COLUMNS = [
    "ts_utc", "transition", "triggers", "cooldown_status",
    "expected_effect",
]


class AuditLogger:
    """
    Append-only audit logger.

    Attributes
    ----------
    log_path : str — base directory for logs (e.g. "logs")
    """

    def __init__(self, log_path: str = "logs"):
        self.log_path = log_path
        self._today: Optional[str] = None
        self._decisions_path: Optional[str] = None
        self._reasons_path: Optional[str] = None
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

    def _rotate_day(self, day_str: str) -> None:
        """Set up new day files if the date has changed."""
        if day_str == self._today:
            return
        self._today = day_str
        day_dir = Path(self.log_path) / day_str
        day_dir.mkdir(parents=True, exist_ok=True)

        self._decisions_path = str(day_dir / "decisions.csv")
        self._reasons_path = str(day_dir / "reasons.csv")

        # Write headers if new
        if not Path(self._decisions_path).exists():
            self._write_header(self._decisions_path, DECISION_COLUMNS)
        if not Path(self._reasons_path).exists():
            self._write_header(self._reasons_path, REASON_COLUMNS)

    @staticmethod
    def _write_header(path: str, columns: list[str]) -> None:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(columns)

    def log_row(
        self,
        timestamp: str,
        signals: dict[str, Any],
        action: str,
        reason: str,
        config_version: str = "",
        calib_version: str = "",
    ) -> None:
        """
        Write one decision row to decisions.csv.

        Parameters
        ----------
        timestamp : ISO-8601 UTC string
        signals : dict with regime_label, anomaly_level, z_ret, z_range, etc.
        action : current action string
        reason : human-readable reason
        """
        day_str = timestamp[:10].replace("-", "")
        self._rotate_day(day_str)

        row = [
            timestamp,
            signals.get("regime_label", ""),
            signals.get("regime_confidence", 0.0),
            signals.get("anomaly_level", ""),
            signals.get("anomaly_score", 0.0),
            signals.get("if_confirmed", False),
            signals.get("z_ret", 0.0),
            signals.get("z_range", 0.0),
            action,
            json.dumps(signals.get("params", {})),
            signals.get("reason_code", ""),
            reason,
            json.dumps({k: v for k, v in signals.items()
                       if k not in ("params",)}),
            config_version,
            calib_version,
        ]

        with open(self._decisions_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def log_error(self, msg: str) -> None:
        """Log an error to the controller log."""
        logger.error(msg)


class Evaluator:
    """
    Compute evaluation metrics from logged decisions and equity curves.

    Attributes
    ----------
    results_path : str — directory for results output
    """

    def __init__(self, results_path: str = "results"):
        self.results_path = results_path
        Path(results_path).mkdir(parents=True, exist_ok=True)

    # ── Event study metrics ──────────────────────────────────────────────

    def run_event_study(
        self,
        equity_baseline: pd.Series,
        equity_guardrail: pd.Series,
        stress_mask: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute per-window stress metrics: MaxDD and HalfRecovery.

        Parameters
        ----------
        equity_baseline, equity_guardrail : equity curves indexed by ts_utc
        stress_mask : boolean Series marking stress bars

        Returns
        -------
        DataFrame with columns [MaxDD_base, MaxDD_guard, ΔDD, T½_base, T½_guard, ΔT½]
        """
        dd_base = self._max_drawdown(equity_baseline[stress_mask])
        dd_guard = self._max_drawdown(equity_guardrail[stress_mask])

        t_half_base = self._half_recovery(equity_baseline[stress_mask])
        t_half_guard = self._half_recovery(equity_guardrail[stress_mask])

        return pd.DataFrame([{
            "MaxDD_base": dd_base,
            "MaxDD_guard": dd_guard,
            "ΔDD": dd_guard - dd_base,
            "T½_base": t_half_base,
            "T½_guard": t_half_guard,
            "ΔT½": t_half_guard - t_half_base,
        }])

    def run_switch_test(
        self,
        detector_flags: pd.Series,
        large_range_times: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Compute lead times for synthetic switch experiments.

        Parameters
        ----------
        detector_flags : boolean Series (True when detector first flags)
        large_range_times : timestamps of first large-range bars

        Returns
        -------
        DataFrame with lead_time per switch instance
        """
        results = []
        for lr_time in large_range_times:
            flags_before = detector_flags[detector_flags.index <= lr_time]
            if flags_before.any():
                first_flag = flags_before[flags_before].index[-1]
                lead = (lr_time - first_flag).total_seconds() / 60.0
                results.append({"large_range_time": lr_time, "lead_time_bars": lead})
            else:
                results.append({"large_range_time": lr_time, "lead_time_bars": float("nan")})

        return pd.DataFrame(results)

    def compute_metrics(
        self,
        decisions: pd.DataFrame,
        calm_mask: pd.Series,
    ) -> dict[str, float]:
        """
        Compute restraint metrics for calm windows.

        Returns
        -------
        dict with Pause%, Throttle%, and thresholds_met
        """
        calm_decisions = decisions[calm_mask]
        total = len(calm_decisions)
        if total == 0:
            return {"pause_rate": 0.0, "throttle_rate": 0.0, "thresholds_met": True}

        pause_rate = (calm_decisions["action"] == "pause").sum() / total * 100
        throttle_rate = (calm_decisions["action"] == "throttle").sum() / total * 100

        return {
            "pause_rate": pause_rate,
            "throttle_rate": throttle_rate,
            "thresholds_met": pause_rate < 1.0 and throttle_rate < 5.0,
        }

    def export_plots(self, prefix: str = "results") -> None:
        """Placeholder for generating result figures."""
        logger.info("Export plots called (prefix=%s). Implement with matplotlib.", prefix)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        """Compute maximum drawdown as a fraction."""
        if len(equity) == 0:
            return 0.0
        peak = equity.cummax()
        dd = (peak - equity) / peak.replace(0, np.nan)
        return float(dd.max()) if not dd.empty else 0.0

    @staticmethod
    def _half_recovery(equity: pd.Series) -> float:
        """Bars to recover half of worst drawdown. NaN if not reached."""
        if len(equity) < 2:
            return float("nan")
        peak = equity.cummax()
        dd = peak - equity
        worst_idx = dd.idxmax()
        worst_dd = dd[worst_idx]
        if worst_dd <= 0:
            return 0.0

        half_target = worst_dd / 2.0
        after = dd[dd.index > worst_idx]
        recovered = after[after <= half_target]
        if recovered.empty:
            return float("nan")  # right-censored
        return float((recovered.index[0] - worst_idx).total_seconds() / 60.0)
