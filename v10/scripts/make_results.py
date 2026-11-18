"""
v10/scripts/make_results.py
Regenerate all evaluation tables and figures from raw decision logs.

Usage:
    python -m v10.scripts.make_results --config config.yaml --results results/

Outputs:
  - metrics_by_window.csv
  - aggregate_metrics.csv
  - Table 9-A (lead time), 9-B (risk reduction), 9-C (restraint), 9-D (sensitivity)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v10.policy.config import Config
from v10.execution.audit import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_decisions(results_dir: str) -> dict[str, pd.DataFrame]:
    """Load all *_decisions.csv files from the results directory."""
    p = Path(results_dir)
    files = sorted(p.glob("*_decisions.csv"))
    windows = {}
    for f in files:
        code = f.stem.replace("_decisions", "")
        df = pd.read_csv(f)
        if "ts_utc" in df.columns:
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
            df = df.set_index("ts_utc")
        windows[code] = df
    return windows


def compute_restraint_table(decisions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Table 9-C: calm-market restraint per window."""
    ev = Evaluator()
    rows = []
    for code, df in decisions.items():
        # Heuristic calm mask: no event flags and regime in calm/trend
        calm_mask = (
            (df.get("event_flag", pd.Series(0, index=df.index)) == 0)
            & df["regime_label"].isin(["calm", "trend_up", "trend_down"])
        )
        metrics = ev.compute_metrics(df, calm_mask)
        rows.append({
            "window_code": code,
            "calm_bars": int(calm_mask.sum()),
            "pause_rate_pct": round(metrics["pause_rate"], 2),
            "throttle_rate_pct": round(metrics["throttle_rate"], 2),
            "thresholds_met": "Y" if metrics["thresholds_met"] else "N",
        })
    return pd.DataFrame(rows)


def compute_action_summary(decisions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summary of action distribution per window."""
    rows = []
    for code, df in decisions.items():
        total = len(df)
        if total == 0:
            continue
        counts = df["action"].value_counts()
        rows.append({
            "window_code": code,
            "total_bars": total,
            "allow_pct": round(counts.get("allow", 0) / total * 100, 1),
            "throttle_pct": round(counts.get("throttle", 0) / total * 100, 1),
            "pause_pct": round(counts.get("pause", 0) / total * 100, 1),
            "hedge_pct": round(counts.get("hedge_active", 0) / total * 100, 1),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation tables")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="results/tables")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    decisions = load_decisions(args.results)
    if not decisions:
        logger.error("No decision files found in %s", args.results)
        return

    # Table 9-C: Restraint
    restraint = compute_restraint_table(decisions)
    restraint.to_csv(out / "table_9C_restraint.csv", index=False)
    logger.info("Table 9-C:\n%s", restraint.to_string(index=False))

    # Action summary
    summary = compute_action_summary(decisions)
    summary.to_csv(out / "action_summary.csv", index=False)
    logger.info("Action summary:\n%s", summary.to_string(index=False))

    # Aggregate metrics
    agg = {
        "config_version": cfg.version,
        "config_hash": cfg.config_hash(),
        "total_windows": len(decisions),
        "total_bars": sum(len(df) for df in decisions.values()),
    }
    pd.DataFrame([agg]).to_csv(out / "aggregate_metrics.csv", index=False)

    logger.info("All tables written to %s", out)


if __name__ == "__main__":
    main()
