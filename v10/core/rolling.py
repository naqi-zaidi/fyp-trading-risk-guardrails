"""
v10/core/rolling.py
Numerically stable O(1) rolling statistics for the feature pipeline.

- WelfordRolling: running mean/variance over a fixed window (Welford's algorithm).
- WilderATR: Wilder-smoothed Average True Range.
- EMA: Exponential Moving Average with standard alpha = 2/(n+1).
- RollingDeque: fixed-size deque with sum tracking for re-seeding after restarts.
- CUSUM: one-sided cumulative sum change detector.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional


class WelfordRolling:
    """
    Rolling mean and std over a fixed window using Welford's online algorithm.
    O(1) per update.
    """

    def __init__(self, window: int):
        self.window = window
        self._buf: deque[float] = deque(maxlen=window)
        self._sum: float = 0.0
        self._sum_sq: float = 0.0  # sum of squared deviations via two-pass proxy

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.window

    @property
    def count(self) -> int:
        return len(self._buf)

    def update(self, value: float) -> None:
        if len(self._buf) == self.window:
            old = self._buf[0]
            self._sum -= old
            self._sum_sq -= old * old
        self._buf.append(value)
        self._sum += value
        self._sum_sq += value * value

    @property
    def mean(self) -> float:
        n = len(self._buf)
        if n == 0:
            return 0.0
        return self._sum / n

    @property
    def var(self) -> float:
        n = len(self._buf)
        if n < 2:
            return 0.0
        mu = self._sum / n
        return max(0.0, self._sum_sq / n - mu * mu)

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def z_score(self, value: float) -> float:
        """Compute z-score of *value* against current window stats."""
        s = self.std
        if s < 1e-15:
            return 0.0
        return (value - self.mean) / s

    def seed(self, values: list[float]) -> None:
        """Bulk-load historical values to warm up the buffer."""
        for v in values:
            self.update(v)


class WilderATR:
    """
    Wilder-smoothed ATR.  Recursion:
        ATR_t = ((window-1) * ATR_{t-1} + TR_t) / window
    """

    def __init__(self, window: int = 14):
        self.window = window
        self._atr: Optional[float] = None
        self._warmup: list[float] = []
        self._prev_close: Optional[float] = None

    @property
    def ready(self) -> bool:
        return self._atr is not None

    @property
    def value(self) -> float:
        return self._atr if self._atr is not None else 0.0

    @staticmethod
    def true_range(high: float, low: float, prev_close: Optional[float]) -> float:
        hl = high - low
        if prev_close is None:
            return hl
        hc = abs(high - prev_close)
        lc = abs(low - prev_close)
        return max(hl, hc, lc)

    def update(self, high: float, low: float, close: float) -> float:
        """Feed one bar; returns the current ATR value."""
        tr = self.true_range(high, low, self._prev_close)
        self._prev_close = close

        if self._atr is None:
            self._warmup.append(tr)
            if len(self._warmup) >= self.window:
                self._atr = sum(self._warmup) / self.window
                self._warmup.clear()
            return self._atr if self._atr is not None else 0.0

        self._atr = ((self.window - 1) * self._atr + tr) / self.window
        return self._atr

    def seed(self, bars: list[tuple[float, float, float]]) -> None:
        """Warm up from list of (high, low, close) tuples."""
        for h, l, c in bars:
            self.update(h, l, c)


class EMA:
    """Exponential Moving Average with α = 2/(span+1)."""

    def __init__(self, span: int):
        self.span = span
        self.alpha = 2.0 / (span + 1)
        self._value: Optional[float] = None

    @property
    def ready(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> float:
        return self._value if self._value is not None else 0.0

    def update(self, price: float) -> float:
        if self._value is None:
            self._value = price
        else:
            self._value = self.alpha * price + (1 - self.alpha) * self._value
        return self._value

    def seed(self, prices: list[float]) -> None:
        for p in prices:
            self.update(p)


class MACD:
    """
    MACD indicator: line = EMA_fast − EMA_slow, signal = EMA_signal(line).
    Histogram = line − signal.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.ema_fast = EMA(fast)
        self.ema_slow = EMA(slow)
        self.ema_signal = EMA(signal)
        self.line: float = 0.0
        self.signal_val: float = 0.0
        self.histogram: float = 0.0

    @property
    def ready(self) -> bool:
        return self.ema_fast.ready and self.ema_slow.ready

    def update(self, price: float) -> None:
        f = self.ema_fast.update(price)
        s = self.ema_slow.update(price)
        self.line = f - s
        self.signal_val = self.ema_signal.update(self.line)
        self.histogram = self.line - self.signal_val


class RollingDeque:
    """
    Fixed-size deque that tracks the running sum for O(1) mean.
    Useful for re-seeding after restarts.
    """

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._sum: float = 0.0

    def append(self, value: float) -> None:
        if len(self._buf) == self.maxlen:
            self._sum -= self._buf[0]
        self._buf.append(value)
        self._sum += value

    @property
    def mean(self) -> float:
        n = len(self._buf)
        return self._sum / n if n else 0.0

    @property
    def values(self) -> list[float]:
        return list(self._buf)

    @property
    def ready(self) -> bool:
        return len(self._buf) == self.maxlen

    def __len__(self) -> int:
        return len(self._buf)


class CUSUMDetector:
    """
    One-sided CUSUM change detector on a stream of values.

    Tracks both positive and negative shifts:
        S_plus_t  = max(0, S_plus_{t-1}  + x_t − k)
        S_minus_t = max(0, S_minus_{t-1} − x_t − k)

    Signals when either S exceeds threshold h.

    Parameters are scaled by an external σ estimate:
        k_actual = k_mult × σ
        h_actual = h_mult × σ
    """

    def __init__(self, k_mult: float = 1.5, h_mult: float = 3.0):
        self.k_mult = k_mult
        self.h_mult = h_mult
        self.s_plus: float = 0.0
        self.s_minus: float = 0.0
        self._triggered: bool = False

    def reset(self) -> None:
        self.s_plus = 0.0
        self.s_minus = 0.0
        self._triggered = False

    def update(self, value: float, sigma: float) -> bool:
        """
        Feed one observation.  Returns True if CUSUM boundary is breached.
        *sigma* is the current rolling std of the monitored series.
        """
        k = self.k_mult * sigma
        h = self.h_mult * sigma

        self.s_plus = max(0.0, self.s_plus + value - k)
        self.s_minus = max(0.0, self.s_minus - value - k)

        self._triggered = (self.s_plus > h) or (self.s_minus > h)
        return self._triggered

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def max_cusum(self) -> float:
        """Return the larger of S+ and S− (useful for audit logging)."""
        return max(self.s_plus, self.s_minus)
