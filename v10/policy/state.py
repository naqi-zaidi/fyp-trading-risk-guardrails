"""
v10/policy/state.py
Finite-state machine for the guardrail action policy.

States: ALLOW, THROTTLE, PAUSE, HEDGE_ACTIVE
Transitions governed by cooldowns, timers, and clear conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Action(str, Enum):
    ALLOW = "allow"
    THROTTLE = "throttle"
    PAUSE = "pause"
    HEDGE_ACTIVE = "hedge_active"


@dataclass
class HedgeState:
    """Tracks the active hedge position."""
    active: bool = False
    age_bars: int = 0
    direction: str = ""       # "long" or "short"
    entry_price: float = 0.0
    stop_price: float = 0.0
    size_mult: float = 0.10
    max_age: int = 10

    def tick(self) -> None:
        if self.active:
            self.age_bars += 1

    def should_close(self, current_price: float) -> bool:
        """Check if hedge should close (stop hit or max age)."""
        if not self.active:
            return False
        if self.age_bars >= self.max_age:
            return True
        if self.direction == "long" and current_price <= self.stop_price:
            return True
        if self.direction == "short" and current_price >= self.stop_price:
            return True
        return False

    def close(self) -> None:
        self.active = False
        self.age_bars = 0


@dataclass
class PolicyState:
    """
    Maintains the current action state, cooldown timers, and transition history.

    Attributes
    ----------
    action : Action — current action
    bars_in_state : int — how long we've been in the current action
    cooldown_throttle : int — bars remaining before throttle can re-engage
    cooldown_pause : int — bars remaining before pause can re-engage
    cooldown_hedge : int — bars remaining before a new hedge is allowed
    hedge : HedgeState — active hedge tracking
    """

    action: Action = Action.ALLOW
    bars_in_state: int = 0

    # Cooldown timers (count down to 0)
    cooldown_throttle: int = 0
    cooldown_pause: int = 0
    cooldown_hedge: int = 0

    # Duration limits
    throttle_duration: int = 5
    pause_duration: int = 10
    hedge_cooldown_duration: int = 10

    # Hedge sub-state
    hedge: HedgeState = field(default_factory=HedgeState)

    # Last transition metadata
    last_reason_code: str = "INIT"
    last_reason_text: str = "System initialised"

    def tick(self) -> None:
        """Advance one bar: increment timers, decrement cooldowns."""
        self.bars_in_state += 1
        self.hedge.tick()

        # Decrement cooldowns
        if self.cooldown_throttle > 0:
            self.cooldown_throttle -= 1
        if self.cooldown_pause > 0:
            self.cooldown_pause -= 1
        if self.cooldown_hedge > 0:
            self.cooldown_hedge -= 1

    def transition_to(
        self,
        new_action: Action,
        reason_code: str,
        reason_text: str,
    ) -> bool:
        """
        Attempt to transition to *new_action*.
        Returns True if transition occurred, False if blocked.
        """
        if new_action == self.action:
            return False  # no-op

        old = self.action

        # Set cooldowns when leaving a state
        if old == Action.THROTTLE:
            self.cooldown_throttle = self.throttle_duration
        elif old == Action.PAUSE:
            self.cooldown_pause = self.pause_duration
        elif old == Action.HEDGE_ACTIVE:
            self.hedge.close()
            self.cooldown_hedge = self.hedge_cooldown_duration

        self.action = new_action
        self.bars_in_state = 0
        self.last_reason_code = reason_code
        self.last_reason_text = reason_text

        logger.info(
            "Policy transition: %s → %s | %s | %s",
            old.value, new_action.value, reason_code, reason_text,
        )
        return True

    def can_throttle(self) -> bool:
        return self.cooldown_throttle == 0

    def can_pause(self) -> bool:
        return self.cooldown_pause == 0

    def can_hedge(self) -> bool:
        return self.cooldown_hedge == 0 and not self.hedge.active

    @property
    def is_allow(self) -> bool:
        return self.action == Action.ALLOW

    @property
    def is_throttle(self) -> bool:
        return self.action == Action.THROTTLE

    @property
    def is_pause(self) -> bool:
        return self.action == Action.PAUSE

    @property
    def is_hedge_active(self) -> bool:
        return self.action == Action.HEDGE_ACTIVE

    def throttle_expired(self) -> bool:
        return self.is_throttle and self.bars_in_state >= self.throttle_duration

    def pause_expired(self) -> bool:
        return self.is_pause and self.bars_in_state >= self.pause_duration

    def to_dict(self) -> dict:
        """Snapshot for audit logging."""
        return {
            "action": self.action.value,
            "bars_in_state": self.bars_in_state,
            "cooldown_throttle": self.cooldown_throttle,
            "cooldown_pause": self.cooldown_pause,
            "cooldown_hedge": self.cooldown_hedge,
            "hedge_active": self.hedge.active,
            "hedge_age": self.hedge.age_bars,
            "reason_code": self.last_reason_code,
            "reason_text": self.last_reason_text,
        }
