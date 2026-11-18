"""
v10/detectors/anomaly.py
Anomaly detector: rolling z-scores (primary) + Isolation Forest (secondary).

AnomalyDetector
    .score_z(X)         — compute z-based anomaly level
    .fit_iforest(X)     — train Isolation Forest on calibration data
    .score_iforest(X)   — score a single vector with IF
    .is_severe(score)   — check if IF confirms severity

Anomaly levels: none, moderate, severe
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyOutput:
    """Per-bar output from the anomaly detector."""
    anomaly_level: str = "none"      # none | moderate | severe
    anomaly_score: float = 0.0       # continuous score for audit
    z_ret: float = 0.0
    z_range: float = 0.0
    if_score: float = 0.0
    if_confirmed: bool = False
    drivers: str = ""


class AnomalyDetector:
    """
    Two-stage anomaly detector.

    Primary (always-on): rolling z-scores on returns and range expansion.
    Secondary (advisory): Isolation Forest on compact 6-dim vector.

    Attributes
    ----------
    z_threshold : float — moderate threshold (default 2.0)
    iforest : IsolationForest | None
    """

    def __init__(
        self,
        z_moderate: float = 2.0,
        z_severe: float = 3.0,
        persistence_moderate: int = 2,
        persistence_severe: int = 3,
        if_n_estimators: int = 100,
        if_contamination: float = 0.01,
        if_seed: int = 42,
    ):
        self.z_moderate = z_moderate
        self.z_severe = z_severe
        self.persistence_moderate = persistence_moderate
        self.persistence_severe = persistence_severe

        # IF parameters (deferred import for optional dependency)
        self._if_n_estimators = if_n_estimators
        self._if_contamination = if_contamination
        self._if_seed = if_seed
        self.iforest = None  # sklearn.ensemble.IsolationForest or None

        # Persistence counters
        self._moderate_count: int = 0
        self._severe_count: int = 0

        # Confirmed label
        self._current_level: str = "none"

        # Latest output
        self.output = AnomalyOutput()

    # ── Z-score primary ──────────────────────────────────────────────────

    def score_z(
        self,
        z_ret: float,
        z_range: float,
        regime_label: str = "calm",
        event_window_high: bool = False,
    ) -> AnomalyOutput:
        """
        Evaluate anomaly level from z-scores with persistence.

        Rules from §8.3:
        - severe: |z_ret| ≥ 3.0 OR |z_range| ≥ 3.0
          (persistence ≥ 3 if regime=stressed, else ≥ 2)
        - moderate: |z_ret| ≥ 2.0 OR |z_range| ≥ 2.0 (persistence ≥ 2)
        - none: otherwise

        During high-impact windows, cap escalations to avoid constant Pause
        on predictable news bursts.
        """
        abs_z_ret = abs(z_ret)
        abs_z_range = abs(z_range)

        raw_severe = (abs_z_ret >= self.z_severe) or (abs_z_range >= self.z_severe)
        raw_moderate = (abs_z_ret >= self.z_moderate) or (abs_z_range >= self.z_moderate)

        # Persistence tracking
        if raw_severe:
            self._severe_count += 1
        else:
            self._severe_count = 0

        if raw_moderate:
            self._moderate_count += 1
        else:
            self._moderate_count = 0

        # Determine persistence threshold for severe
        sev_persist_needed = (
            self.persistence_severe if regime_label == "stressed"
            else self.persistence_moderate
        )

        # Apply persistence gates
        level = "none"
        if self._severe_count >= sev_persist_needed:
            level = "severe"
        elif self._moderate_count >= self.persistence_moderate:
            level = "moderate"

        # Event window cap: during High-impact, don't escalate beyond
        # what's already confirmed to avoid fighting the calendar
        if event_window_high and level == "severe" and self._current_level == "none":
            level = "moderate"  # cap initial escalation

        # Build driver string
        drivers_parts = []
        if abs_z_ret >= self.z_moderate:
            drivers_parts.append(f"z_ret={z_ret:.2f}")
        if abs_z_range >= self.z_moderate:
            drivers_parts.append(f"z_range={z_range:.2f}")
        drivers = ", ".join(drivers_parts)

        # Compute continuous score (max of abs z-scores)
        score = max(abs_z_ret, abs_z_range)

        self._current_level = level

        self.output = AnomalyOutput(
            anomaly_level=level,
            anomaly_score=score,
            z_ret=z_ret,
            z_range=z_range,
            drivers=drivers,
        )
        return self.output

    # ── Isolation Forest secondary ───────────────────────────────────────

    def fit_iforest(self, X: np.ndarray) -> None:
        """
        Fit Isolation Forest on calibration data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 6)
            Columns: [ret_1m, atr_14, z_range, slope_short, wick_ratio, event_flag]
        """
        try:
            from sklearn.ensemble import IsolationForest

            self.iforest = IsolationForest(
                n_estimators=self._if_n_estimators,
                contamination=self._if_contamination,
                random_state=self._if_seed,
            )
            self.iforest.fit(X)
            logger.info("Isolation Forest fitted on %d samples.", X.shape[0])
        except ImportError:
            logger.warning("sklearn not available; Isolation Forest disabled.")
            self.iforest = None

    def score_iforest(self, X: np.ndarray) -> float:
        """
        Score a single observation with the fitted IF.
        Returns the anomaly score (more negative = more anomalous).
        Returns 0.0 if IF is not available.
        """
        if self.iforest is None:
            return 0.0
        X_2d = X.reshape(1, -1) if X.ndim == 1 else X
        score = float(self.iforest.score_samples(X_2d)[0])
        self.output.if_score = score
        return score

    def is_severe(self, if_score: float) -> bool:
        """
        Check if IF score indicates anomaly (negative decision function).
        IF predict returns -1 for anomalies.
        """
        if self.iforest is None:
            return False
        return if_score < 0  # score_samples < 0 indicates outlier

    # ── Combined update ──────────────────────────────────────────────────

    def update(
        self,
        z_ret: float,
        z_range: float,
        if_vector: Optional[np.ndarray] = None,
        regime_label: str = "calm",
        event_window_high: bool = False,
    ) -> AnomalyOutput:
        """
        Full anomaly update: z-score primary + optional IF confirmation.

        Per §8.3: if IF flags anomaly but z-based test is quiet,
        raise to moderate only if outside High-impact event windows.
        """
        # Primary z-score evaluation
        output = self.score_z(z_ret, z_range, regime_label, event_window_high)

        # Secondary IF evaluation
        if if_vector is not None and self.iforest is not None:
            if_score = self.score_iforest(if_vector)
            if_flagged = self.is_severe(if_score)

            output.if_score = if_score
            output.if_confirmed = if_flagged

            # IF promotion rule: if z says "none" but IF flags anomaly,
            # promote to "moderate" unless in high-impact window
            if output.anomaly_level == "none" and if_flagged and not event_window_high:
                output.anomaly_level = "moderate"
                output.drivers += " [IF_promoted]"
                logger.debug("IF promoted anomaly to moderate (score=%.3f)", if_score)

        self.output = output
        return output

    # ── State management ─────────────────────────────────────────────────

    def reset_persistence(self) -> None:
        """Reset persistence counters (e.g., after confirmed transition)."""
        self._moderate_count = 0
        self._severe_count = 0
        self._current_level = "none"

    @property
    def current_level(self) -> str:
        return self._current_level
