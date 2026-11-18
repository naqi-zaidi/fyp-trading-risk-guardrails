"""
v10/detectors/regime.py
Regime detector: k-means clustering + CUSUM change-point gate + persistence.

RegimeDetector
    .fit(X)                 — morning calibration on recent feature matrix
    .predict(X)             — assign regime label to a single feature vector
    .confirm_transition(stream) — apply CUSUM gate and persistence

Regime labels: calm, trend_up, trend_down, volatile, stressed
Centroids are ranked by (vol_20, range_ratio) to deterministically map names.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from v10.core.rolling import CUSUMDetector

logger = logging.getLogger(__name__)

REGIME_LABELS = ["calm", "trend_up", "trend_down", "volatile", "stressed"]
HIGH_ENERGY = {"volatile", "stressed"}


@dataclass
class RegimeOutput:
    """Per-bar output from the regime detector."""
    regime_label: str = "calm"
    regime_confidence: float = 0.0
    vol_20: float = 0.0
    xr: float = 0.0
    slope: float = 0.0
    top_drivers: str = ""
    calib_version: str = ""


class RegimeDetector:
    """
    k-means regime detector with CUSUM confirmation and persistence gate.

    Parameters (from config.yaml)
    ----------
    k : int                — number of clusters (default 5)
    model : KMeans | None  — fitted model (set by .fit())
    cusum_threshold : dict — {k_mult, h_mult} for CUSUM
    persistence_bars : int — consecutive bars to confirm a label change
    """

    def __init__(
        self,
        k: int = 5,
        n_init: int = 10,
        seed: int = 42,
        cusum_k_mult: float = 1.5,
        cusum_h_mult: float = 3.0,
        persistence_bars: int = 3,
    ):
        self.k = k
        self.n_init = n_init
        self.seed = seed
        self.persistence_bars = persistence_bars

        # Internal state
        self.model: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_map: dict[int, str] = {}        # cluster_id → label name
        self.centroids: Optional[np.ndarray] = None
        self.calib_version: str = ""

        # CUSUM gate
        self._cusum = CUSUMDetector(k_mult=cusum_k_mult, h_mult=cusum_h_mult)

        # Persistence tracking
        self._current_label: str = "calm"
        self._provisional_label: str = "calm"
        self._provisional_count: int = 0

        # Latest output
        self.output = RegimeOutput()

    # ── Calibration (morning fit) ────────────────────────────────────────

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> None:
        """
        Fit k-means on calibration data.

        Parameters
        ----------
        X : ndarray of shape (n_bars, 3)
            Columns: [vol_20, range_ratio, slope_short]
        feature_names : optional list of feature names for logging
        """
        if X.shape[0] < self.k * 2:
            logger.warning("Calibration data too small (%d bars). Keeping previous model.", X.shape[0])
            return

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = KMeans(
            n_clusters=self.k,
            n_init=self.n_init,
            random_state=self.seed,
        )
        self.model.fit(X_scaled)

        self.centroids = self.scaler.inverse_transform(self.model.cluster_centers_)
        self.label_map = self._map_centroids_to_labels(self.centroids)

        logger.info(
            "Regime calibration complete: %d clusters on %d bars. Mapping: %s",
            self.k, X.shape[0], self.label_map,
        )

    def _map_centroids_to_labels(self, centroids: np.ndarray) -> dict[int, str]:
        """
        Deterministically map cluster IDs to semantic labels.

        Ranking logic (per report §8.2):
          - Sort by vol_20 (col 0) ascending.
          - Lowest vol → calm
          - Next two: check sign of slope (col 2) → trend_up / trend_down
          - High vol, modest expansion → volatile
          - High vol, high expansion → stressed
        """
        n = centroids.shape[0]
        # Sort cluster indices by vol_20 ascending
        vol_order = np.argsort(centroids[:, 0])

        label_map = {}

        if n >= 5:
            label_map[int(vol_order[0])] = "calm"

            # Middle two: trend_up vs trend_down by slope sign
            mid_indices = [int(vol_order[1]), int(vol_order[2])]
            slopes = [centroids[i, 2] for i in mid_indices]
            if slopes[0] >= slopes[1]:
                label_map[mid_indices[0]] = "trend_up"
                label_map[mid_indices[1]] = "trend_down"
            else:
                label_map[mid_indices[0]] = "trend_down"
                label_map[mid_indices[1]] = "trend_up"

            # Top two: volatile vs stressed by range_ratio (col 1)
            high_indices = [int(vol_order[3]), int(vol_order[4])]
            xrs = [centroids[i, 1] for i in high_indices]
            if xrs[0] <= xrs[1]:
                label_map[high_indices[0]] = "volatile"
                label_map[high_indices[1]] = "stressed"
            else:
                label_map[high_indices[0]] = "stressed"
                label_map[high_indices[1]] = "volatile"
        else:
            # Fallback for fewer clusters
            for i, idx in enumerate(vol_order):
                label_map[int(idx)] = REGIME_LABELS[min(i, len(REGIME_LABELS) - 1)]

        return label_map

    # ── Online prediction ────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> RegimeOutput:
        """
        Assign regime to a single feature vector [vol_20, range_ratio, slope_short].

        Returns RegimeOutput with provisional label (before persistence gate).
        """
        if self.model is None or self.scaler is None:
            self.output = RegimeOutput(
                regime_label=self._current_label,
                regime_confidence=0.0,
                calib_version=self.calib_version,
            )
            return self.output

        X_2d = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X_2d)

        cluster_id = int(self.model.predict(X_scaled)[0])

        # Confidence: inverse distance to assigned centroid (normalised)
        distances = np.linalg.norm(
            X_scaled - self.model.cluster_centers_, axis=1
        )
        min_dist = distances[cluster_id]
        confidence = 1.0 / (1.0 + min_dist)

        provisional = self.label_map.get(cluster_id, "calm")

        # Top drivers: rank features by contribution to centroid distance
        centroid = self.model.cluster_centers_[cluster_id]
        deltas = np.abs(X_scaled[0] - centroid)
        feat_names = ["vol_20", "range_ratio", "slope_short"]
        ranked = sorted(zip(feat_names, deltas), key=lambda x: -x[1])
        top_drivers = ", ".join(f"{n}={d:.3f}" for n, d in ranked[:2])

        self.output = RegimeOutput(
            regime_label=provisional,
            regime_confidence=float(confidence),
            vol_20=float(X[0]),
            xr=float(X[1]),
            slope=float(X[2]),
            top_drivers=top_drivers,
            calib_version=self.calib_version,
        )
        return self.output

    # ── CUSUM gate + persistence ─────────────────────────────────────────

    def confirm_transition(
        self,
        ret: float,
        sigma_60: float,
        provisional_label: str,
    ) -> str:
        """
        Apply CUSUM gate and persistence to decide if a label change is confirmed.

        Rules from §8.2:
        - Transitions INTO high-energy states (volatile, stressed) require
          CUSUM to be triggered.
        - All transitions require persistence_bars consecutive bars with the
          same provisional label.

        Returns the confirmed regime label (may remain the previous one).
        """
        cusum_fired = self._cusum.update(ret, sigma_60)

        entering_high_energy = (
            provisional_label in HIGH_ENERGY
            and self._current_label not in HIGH_ENERGY
        )

        # If entering high-energy, CUSUM must also agree
        if entering_high_energy and not cusum_fired:
            # CUSUM not triggered — reject transition, keep current
            self._provisional_label = self._current_label
            self._provisional_count = 0
            self.output.regime_label = self._current_label
            return self._current_label

        # Persistence gate
        if provisional_label == self._provisional_label:
            self._provisional_count += 1
        else:
            self._provisional_label = provisional_label
            self._provisional_count = 1

        if self._provisional_count >= self.persistence_bars:
            # Transition confirmed
            if self._current_label != provisional_label:
                logger.info(
                    "Regime transition confirmed: %s → %s (persist=%d, cusum=%.4f)",
                    self._current_label, provisional_label,
                    self._provisional_count, self._cusum.max_cusum,
                )
            self._current_label = provisional_label
            # Reset CUSUM on confirmed transition
            if provisional_label not in HIGH_ENERGY:
                self._cusum.reset()
        # else: not yet enough persistence

        self.output.regime_label = self._current_label
        return self._current_label

    @property
    def current_label(self) -> str:
        return self._current_label

    @property
    def cusum_value(self) -> float:
        return self._cusum.max_cusum

    # ── Serialisation ────────────────────────────────────────────────────

    def save_calibration(self, path: str, version: str = "") -> None:
        """Save centroids, scaler params, and label map to JSON."""
        if self.centroids is None or self.scaler is None:
            logger.warning("No calibration to save.")
            return

        payload = {
            "version": version,
            "k": self.k,
            "seed": self.seed,
            "centroids": self.centroids.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "label_map": {str(k): v for k, v in self.label_map.items()},
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2))
        self.calib_version = version
        logger.info("Saved regime calibration to %s (version=%s)", path, version)

    def load_calibration(self, path: str) -> None:
        """Load calibration from JSON (for restarts or replay)."""
        p = Path(path)
        if not p.exists():
            logger.warning("Calibration file not found: %s", path)
            return

        payload = json.loads(p.read_text())
        self.centroids = np.array(payload["centroids"])
        self.label_map = {int(k): v for k, v in payload["label_map"].items()}
        self.calib_version = payload.get("version", "")

        # Reconstruct scaler and model
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(payload["scaler_mean"])
        self.scaler.scale_ = np.array(payload["scaler_scale"])
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = len(self.scaler.mean_)

        # Reconstruct a KMeans with the loaded centroids
        self.model = KMeans(
            n_clusters=self.k, n_init=1, random_state=self.seed, max_iter=1,
        )
        # Trick: fit on centroids themselves so predict() works
        self.model.fit(self.scaler.transform(self.centroids))
        # Override centers with correct values
        self.model.cluster_centers_ = self.scaler.transform(self.centroids)

        logger.info("Loaded regime calibration from %s (version=%s)", path, self.calib_version)
