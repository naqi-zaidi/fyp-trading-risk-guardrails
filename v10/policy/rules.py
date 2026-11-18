"""
v10/policy/rules.py
Decision policy: maps regime + anomaly signals → actions.

DecisionPolicy
    .decide(regime, anomaly, event_flag) → action + reason
    .apply_persistence(new_state)
    .make_reason()

Precedence: Pause > Throttle > HedgeActive > Allow
See Appendix B for full specification.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from v10.policy.state import Action, PolicyState, HedgeState

logger = logging.getLogger(__name__)


class DecisionPolicy:
    """
    Rules-first decision policy.

    Attributes
    ----------
    cooldown_bars : int — general cooldown reference
    state : PolicyState
    """

    def __init__(
        self,
        throttle_size_mult: float = 0.33,
        throttle_duration: int = 5,
        pause_duration: int = 10,
        hedge_size_mult: float = 0.10,
        hedge_stop_atr_mult: float = 0.20,
        hedge_max_age: int = 10,
        hedge_cooldown: int = 10,
        persistence_label: int = 3,
        persistence_moderate: int = 2,
        persistence_severe: int = 3,
        stale_detector_bars: int = 2,
    ):
        self.throttle_size_mult = throttle_size_mult
        self.hedge_size_mult = hedge_size_mult
        self.hedge_stop_atr_mult = hedge_stop_atr_mult
        self.persistence_label = persistence_label
        self.persistence_moderate = persistence_moderate
        self.persistence_severe = persistence_severe
        self.stale_detector_bars = stale_detector_bars

        self.state = PolicyState(
            throttle_duration=throttle_duration,
            pause_duration=pause_duration,
            hedge_cooldown_duration=hedge_cooldown,
        )
        self.state.hedge.max_age = hedge_max_age
        self.state.hedge.size_mult = hedge_size_mult

        # Clear-conditions counter
        self._clear_bars: int = 0

    def decide(
        self,
        regime_label: str,
        regime_confidence: float,
        anomaly_level: str,
        anomaly_score: float,
        if_confirmed: bool,
        trend_strength: float,
        event_window_high: bool,
        event_window_med: bool,
        z_ret: float,
        z_range: float,
        atr_14: float,
        current_price: float,
        slope_short: float,
        detector_stale: bool = False,
        alignment_degraded: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate all rules and return the action decision.

        Returns
        -------
        dict with keys: action, reason_code, reason_text, params
        """
        # Tick timers
        self.state.tick()

        # Track clear conditions (both calm/trending AND no anomaly)
        is_clear = (
            regime_label in ("calm", "trend_up", "trend_down")
            and anomaly_level == "none"
        )
        if is_clear:
            self._clear_bars += 1
        else:
            self._clear_bars = 0

        # ── Check hedge closure first ────────────────────────────────────
        if self.state.is_hedge_active:
            if self.state.hedge.should_close(current_price):
                reason = (
                    f"hedge closed: age={self.state.hedge.age_bars}, "
                    f"price={current_price:.5f}"
                )
                self.state.transition_to(Action.ALLOW, "HEDGE_CLOSED", reason)
                return self._build_result()

        # ── Check time-based expirations ─────────────────────────────────
        if self.state.throttle_expired() and self._clear_bars >= 3:
            self.state.transition_to(
                Action.ALLOW, "THROTTLE_CLEARED",
                f"anomalies below moderate for {self._clear_bars} bars"
            )
            return self._build_result()

        if self.state.throttle_expired():
            self.state.transition_to(
                Action.ALLOW, "THROTTLE_EXPIRED",
                f"throttle duration ({self.state.throttle_duration}) reached"
            )
            return self._build_result()

        if self.state.pause_expired() and self._clear_bars >= 3:
            self.state.transition_to(
                Action.ALLOW, "PAUSE_CLEARED",
                f"pause expired and clear for {self._clear_bars} bars"
            )
            return self._build_result()

        if self.state.pause_expired():
            self.state.transition_to(
                Action.ALLOW, "PAUSE_EXPIRED",
                f"pause duration ({self.state.pause_duration}) reached"
            )
            return self._build_result()

        # ── Degradation guards ───────────────────────────────────────────
        if alignment_degraded:
            if self.state.action in (Action.PAUSE, Action.HEDGE_ACTIVE):
                self.state.transition_to(
                    Action.ALLOW, "ALIGNMENT_DEGRADED",
                    "data alignment degraded; bias to Allow"
                )
            return self._build_result()

        if detector_stale:
            # Cannot escalate to Pause when detectors are stale
            if self.state.is_allow:
                pass  # no new actions
            return self._build_result()

        # ── Rule 1: PAUSE ────────────────────────────────────────────────
        # regime=stressed AND anomaly=severe with persistence
        if (
            regime_label == "stressed"
            and anomaly_level == "severe"
            and not self.state.is_pause
            and self.state.can_pause()
            and not detector_stale
        ):
            reason = (
                f"regime {regime_label}, z_ret={z_ret:.2f}, "
                f"z_range={z_range:.2f}, persisted"
            )
            self.state.transition_to(
                Action.PAUSE, "SEVERE_ANOMALY_STRESSED_REGIME", reason
            )
            return self._build_result()

        # ── Rule 1b: Escalate from Throttle to Pause ────────────────────
        if (
            self.state.is_throttle
            and regime_label == "stressed"
            and anomaly_level == "severe"
            and self.state.bars_in_state >= 2
            and self.state.can_pause()
        ):
            reason = (
                f"escalation from throttle: regime {regime_label}, "
                f"severe persisted {self.state.bars_in_state} bars"
            )
            self.state.transition_to(
                Action.PAUSE, "THROTTLE_ESCALATED_TO_PAUSE", reason
            )
            return self._build_result()

        # ── Rule 2: THROTTLE ────────────────────────────────────────────
        # regime=volatile AND anomaly=moderate with persistence
        if (
            regime_label in ("volatile", "stressed", "trend_up", "trend_down")
            and anomaly_level in ("moderate", "severe")
            and self.state.is_allow
            and self.state.can_throttle()
        ):
            # During high-impact events, cap throttle to one cycle
            reason = (
                f"regime {regime_label}, z_range={z_range:.2f}, "
                f"anomaly={anomaly_level}"
            )
            self.state.transition_to(
                Action.THROTTLE, "VOLATILITY_MODERATE_EXPANSION", reason
            )
            return self._build_result()

        # ── Rule 3: HEDGE ────────────────────────────────────────────────
        # Trend strong + single severe spike against trend + cooldown clear
        if (
            self.state.is_allow
            and self.state.can_hedge()
            and not event_window_high  # no new hedges in high-impact windows
            and not alignment_degraded
            and abs(slope_short) > 0  # trend must exist
        ):
            trend_up = slope_short > 0
            severe_against = (
                (trend_up and z_ret <= -self.persistence_severe)
                or (not trend_up and z_ret >= self.persistence_severe)
            )
            if severe_against and abs(z_ret) >= 3.0:
                # Start hedge
                direction = "long" if not trend_up else "short"
                stop = (
                    current_price - self.hedge_stop_atr_mult * atr_14
                    if direction == "long"
                    else current_price + self.hedge_stop_atr_mult * atr_14
                )
                self.state.hedge = HedgeState(
                    active=True,
                    age_bars=0,
                    direction=direction,
                    entry_price=current_price,
                    stop_price=stop,
                    size_mult=self.hedge_size_mult,
                    max_age=self.state.hedge.max_age,
                )
                reason = (
                    f"trend {'up' if trend_up else 'down'} strong, "
                    f"severe spike against trend z_ret={z_ret:.2f}, "
                    f"hedge {direction} {self.hedge_size_mult}x "
                    f"stop={stop:.5f}"
                )
                self.state.transition_to(
                    Action.HEDGE_ACTIVE,
                    "ONE_BAR_SEVERE_AGAINST_TREND",
                    reason,
                )
                return self._build_result()

        # ── Rule 4: ALLOW (clear conditions) ─────────────────────────────
        if self._clear_bars >= 3 and not self.state.is_allow:
            self.state.transition_to(
                Action.ALLOW, "CLEAR_CONDITIONS",
                f"clear for {self._clear_bars} bars"
            )

        return self._build_result()

    def _build_result(self) -> dict[str, Any]:
        """Build the result dict for the current state."""
        params = {}
        if self.state.is_throttle:
            params = {"size_mult": self.throttle_size_mult}
        elif self.state.is_hedge_active:
            h = self.state.hedge
            params = {
                "direction": h.direction,
                "size_mult": h.size_mult,
                "stop_price": h.stop_price,
                "entry_price": h.entry_price,
                "age_bars": h.age_bars,
            }

        return {
            "action": self.state.action.value,
            "reason_code": self.state.last_reason_code,
            "reason_text": self.state.last_reason_text,
            "params": params,
            "state_snapshot": self.state.to_dict(),
        }

    def make_reason(self) -> str:
        """Return the human-readable reason string for the current action."""
        return (
            f"{self.state.last_reason_code}: {self.state.last_reason_text}"
        )
