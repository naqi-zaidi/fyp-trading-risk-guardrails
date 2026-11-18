"""
v10/tests/test_policy.py
Tests for policy/state.py and policy/rules.py.
"""

import pytest
from v10.policy.state import Action, PolicyState, HedgeState
from v10.policy.rules import DecisionPolicy


class TestPolicyState:
    """Tests for the finite state machine."""

    def test_initial_state_is_allow(self):
        ps = PolicyState()
        assert ps.action == Action.ALLOW
        assert ps.is_allow

    def test_transition_to_throttle(self):
        ps = PolicyState()
        changed = ps.transition_to(Action.THROTTLE, "TEST", "test reason")
        assert changed
        assert ps.is_throttle
        assert ps.bars_in_state == 0

    def test_no_self_transition(self):
        ps = PolicyState()
        changed = ps.transition_to(Action.ALLOW, "TEST", "noop")
        assert not changed

    def test_tick_increments_bars(self):
        ps = PolicyState()
        ps.transition_to(Action.THROTTLE, "TEST", "test")
        ps.tick()
        ps.tick()
        assert ps.bars_in_state == 2

    def test_cooldown_set_on_leave(self):
        ps = PolicyState(throttle_duration=5)
        ps.transition_to(Action.THROTTLE, "T", "t")
        ps.transition_to(Action.ALLOW, "A", "a")
        assert ps.cooldown_throttle == 5

    def test_cooldown_decrements(self):
        ps = PolicyState(throttle_duration=5)
        ps.transition_to(Action.THROTTLE, "T", "t")
        ps.transition_to(Action.ALLOW, "A", "a")
        for _ in range(5):
            ps.tick()
        assert ps.cooldown_throttle == 0

    def test_throttle_expired(self):
        ps = PolicyState(throttle_duration=5)
        ps.transition_to(Action.THROTTLE, "T", "t")
        for _ in range(4):
            ps.tick()
        assert not ps.throttle_expired()
        ps.tick()
        assert ps.throttle_expired()

    def test_pause_expired(self):
        ps = PolicyState(pause_duration=10)
        ps.transition_to(Action.PAUSE, "P", "p")
        for _ in range(9):
            ps.tick()
        assert not ps.pause_expired()
        ps.tick()
        assert ps.pause_expired()

    def test_to_dict_snapshot(self):
        ps = PolicyState()
        d = ps.to_dict()
        assert "action" in d
        assert "bars_in_state" in d
        assert "reason_code" in d


class TestHedgeState:
    def test_not_active_by_default(self):
        h = HedgeState()
        assert not h.active

    def test_tick_increments_age(self):
        h = HedgeState(active=True, age_bars=0)
        h.tick()
        assert h.age_bars == 1

    def test_close_on_max_age(self):
        h = HedgeState(active=True, age_bars=9, max_age=10)
        h.tick()
        assert h.should_close(1.1)

    def test_close_on_stop_long(self):
        h = HedgeState(active=True, direction="long",
                       stop_price=1.0990, entry_price=1.1000)
        assert h.should_close(1.0985)
        assert not h.should_close(1.1010)

    def test_close_on_stop_short(self):
        h = HedgeState(active=True, direction="short",
                       stop_price=1.1010, entry_price=1.1000)
        assert h.should_close(1.1015)
        assert not h.should_close(1.0990)


class TestDecisionPolicy:
    """Tests for the decision rules engine."""

    @pytest.fixture
    def dp(self):
        return DecisionPolicy(
            throttle_size_mult=0.33,
            throttle_duration=5,
            pause_duration=10,
            hedge_size_mult=0.10,
            hedge_stop_atr_mult=0.20,
            hedge_max_age=10,
            hedge_cooldown=10,
            persistence_label=3,
            persistence_moderate=2,
            persistence_severe=3,
        )

    def _decide(self, dp, **overrides):
        defaults = dict(
            regime_label="calm", regime_confidence=0.9,
            anomaly_level="none", anomaly_score=0.3,
            if_confirmed=False, trend_strength=0.0,
            event_window_high=False, event_window_med=False,
            z_ret=0.5, z_range=0.4, atr_14=0.001,
            current_price=1.1000, slope_short=0.0,
            detector_stale=False, alignment_degraded=False,
        )
        defaults.update(overrides)
        return dp.decide(**defaults)

    def test_calm_returns_allow(self, dp):
        dec = self._decide(dp)
        assert dec["action"] == "allow"

    def test_volatile_moderate_triggers_throttle(self, dp):
        dec = self._decide(dp,
                           regime_label="volatile",
                           anomaly_level="moderate",
                           anomaly_score=2.3,
                           z_range=2.3)
        assert dec["action"] == "throttle"
        assert dec["reason_code"] == "VOLATILITY_MODERATE_EXPANSION"

    def test_stressed_severe_triggers_pause(self, dp):
        dec = self._decide(dp,
                           regime_label="stressed",
                           anomaly_level="severe",
                           anomaly_score=3.5,
                           z_ret=3.5, z_range=3.2)
        assert dec["action"] == "pause"
        assert dec["reason_code"] == "SEVERE_ANOMALY_STRESSED_REGIME"

    def test_throttle_expires_to_allow(self, dp):
        # Trigger throttle
        self._decide(dp, regime_label="volatile", anomaly_level="moderate")
        # Tick through 5 bars of calm
        for _ in range(6):
            dec = self._decide(dp, regime_label="calm", anomaly_level="none")
        assert dec["action"] == "allow"

    def test_pause_expires_to_allow(self, dp):
        # Trigger pause
        self._decide(dp, regime_label="stressed", anomaly_level="severe",
                     z_ret=3.5, z_range=3.2)
        # Tick through 11 bars of calm
        for _ in range(12):
            dec = self._decide(dp, regime_label="calm", anomaly_level="none")
        assert dec["action"] == "allow"

    def test_precedence_pause_over_throttle(self, dp):
        """Pause should win when both conditions are met."""
        dec = self._decide(dp,
                           regime_label="stressed",
                           anomaly_level="severe",
                           z_ret=3.5, z_range=3.2)
        assert dec["action"] == "pause"

    def test_alignment_degraded_biases_allow(self, dp):
        # Even with stressed regime, degraded alignment → allow
        dec = self._decide(dp,
                           regime_label="stressed",
                           anomaly_level="severe",
                           alignment_degraded=True)
        assert dec["action"] == "allow"

    def test_stale_detector_blocks_pause(self, dp):
        dec = self._decide(dp,
                           regime_label="stressed",
                           anomaly_level="severe",
                           detector_stale=True)
        # Should NOT pause with stale detectors
        assert dec["action"] != "pause"

    def test_make_reason(self, dp):
        self._decide(dp)
        reason = dp.make_reason()
        assert isinstance(reason, str)
        assert ":" in reason

    def test_throttle_params_include_size_mult(self, dp):
        dec = self._decide(dp,
                           regime_label="volatile",
                           anomaly_level="moderate")
        assert dec["params"]["size_mult"] == 0.33

    def test_clear_conditions_counter(self, dp):
        # Trigger throttle then clear
        self._decide(dp, regime_label="volatile", anomaly_level="moderate")
        # Need 3 clear bars + expiry
        for _ in range(8):
            dec = self._decide(dp, regime_label="calm", anomaly_level="none")
        assert dec["action"] == "allow"
        assert "CLEAR" in dec["reason_code"] or "EXPIRE" in dec["reason_code"] or dec["reason_code"] == "INIT"
