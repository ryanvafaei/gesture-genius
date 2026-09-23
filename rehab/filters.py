"""
One Euro filter (Casiez, Roussel & Vogel, CHI 2012).

A low-pass filter whose cutoff frequency rises with the speed of the signal:
heavy smoothing while the hand is still, almost no lag while it moves.
A moving average lags during motion, which matters for slow movers because
the hold timer would start late.

Works on floats and on numpy arrays (element-wise).
"""

import math

import numpy as np


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x, t):
        """Filter value x observed at time t (seconds)."""
        x = np.asarray(x, dtype=float)

        if self._x is None:
            self._x = x
            self._dx = np.zeros_like(x)
            self._t = t
            return self._out(x)

        dt = t - self._t
        if dt <= 0:
            return self._out(self._x)
        self._t = t

        a_d = _alpha(self.d_cutoff, dt)
        dx = (x - self._x) / dt
        self._dx = a_d * dx + (1 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = _alpha_array(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._out(self._x)

    @property
    def speed(self):
        """Filtered derivative of the signal (units per second)."""
        return self._out(self._dx) if self._dx is not None else 0.0

    @staticmethod
    def _out(v):
        return float(v) if np.ndim(v) == 0 else v.copy()


def _alpha_array(cutoff, dt):
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class FeatureFilter:
    """
    Keeps one One Euro filter per named feature.

    Features that disappear (e.g. hand left the image) reset their filter so
    that the next appearance does not glide in from a stale value.
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, reset_after_s=0.5):
        self.params = dict(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.reset_after_s = reset_after_s
        self._filters = {}
        self._last_seen = {}

    def reset(self):
        self._filters.clear()
        self._last_seen.clear()

    def __call__(self, name, value, t):
        f = self._filters.get(name)
        if f is None:
            f = self._filters[name] = OneEuroFilter(**self.params)
        elif t - self._last_seen.get(name, t) > self.reset_after_s:
            f.reset()
        self._last_seen[name] = t
        return f(value, t)

    def speed(self, name):
        f = self._filters.get(name)
        return f.speed if f is not None else 0.0
