"""Generic exponential moving average meter, used for logging smoothed loss/metric curves."""

from __future__ import annotations


class EMAMeter:
    """Simple scalar EMA tracker: value_t = beta * value_{t-1} + (1-beta) * x_t."""

    def __init__(self, beta: float = 0.98):
        self.beta = beta
        self._value = None
        self._raw_history = []

    def update(self, x: float) -> float:
        x = float(x)
        self._raw_history.append(x)
        if self._value is None:
            self._value = x
        else:
            self._value = self.beta * self._value + (1 - self.beta) * x
        return self._value

    @property
    def value(self) -> float:
        return self._value if self._value is not None else 0.0

    def reset(self):
        self._value = None
        self._raw_history = []

    def history(self):
        return list(self._raw_history)
