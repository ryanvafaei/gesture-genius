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


class LowPass:
    """
    Causal 2nd-order Butterworth low-pass for body angles (default 6 Hz,
    as Gates et al. 2016 filtered their marker data). Starts at the first
    sample, so there is no start-up transient. NaN (a frame without a
    reading) returns the last output and leaves the filter unchanged.

    fs is only the starting guess: with dt (seconds since the last sample)
    the coefficients follow the real frame rate. Pose and hand tracking
    together often run at 10-15 frames a second, not the camera's 30; a
    filter designed for 30 then cuts at a third of 6 Hz and lags behind
    her arm.
    """

    def __init__(self, fs=30.0, fc=6.0):
        self.fc = fc
        self.fs = None
        self._design(fs)
        self.reset()

    def _design(self, fs):
        fs = float(min(max(fs, 2.0), 240.0))
        if self.fs is not None and abs(fs - self.fs) < 0.1 * self.fs:
            return
        self.fs = fs
        k = math.tan(math.pi * min(self.fc, 0.45 * fs) / fs)
        q = 1 / math.sqrt(2)
        norm = 1 / (1 + k / q + k * k)
        self.b = (k * k * norm, 2 * k * k * norm, k * k * norm)
        self.a = (2 * (k * k - 1) * norm, (1 - k / q + k * k) * norm)

    def reset(self):
        self.x = [None, None]
        self.y = [None, None]

    def __call__(self, v, dt=None):
        if v is None or v != v:
            return self.y[0] if self.y[0] is not None else float("nan")
        v = float(v)
        if self.x[0] is None:
            self.x = [v, v]
            self.y = [v, v]
            return v
        if dt is not None and dt > 0:
            self._design(1.0 / dt)
        b, a = self.b, self.a
        y = b[0] * v + b[1] * self.x[0] + b[2] * self.x[1] - a[0] * self.y[0] - a[1] * self.y[1]
        self.x = [v, self.x[0]]
        self.y = [y, self.y[0]]
        return y


class LowPassBank:
    """
    One LowPass per named value; a value not seen for reset_after_s starts
    fresh. The frame rate is measured from the timestamps (smoothed), so
    the cutoff stays at fc whatever the real frame rate is.
    """

    def __init__(self, fs=30.0, fc=6.0, reset_after_s=0.5):
        self.fs, self.fc = fs, fc
        self.reset_after_s = reset_after_s
        self._filters = {}
        self._last_seen = {}
        self._dt = 1.0 / fs
        self._last_t = None

    def reset(self):
        self._filters.clear()
        self._last_seen.clear()
        self._last_t = None

    def _frame_dt(self, t):
        """Smoothed time between frames (all values of a frame share one t)."""
        if self._last_t is not None and t > self._last_t:
            step = t - self._last_t
            if step < 0.5:
                self._dt += 0.2 * (step - self._dt)
        if self._last_t is None or t > self._last_t:
            self._last_t = t
        return self._dt

    def __call__(self, name, value, t):
        dt = self._frame_dt(t)
        if value is None or value != value:
            return float("nan")
        f = self._filters.get(name)
        if f is None:
            f = self._filters[name] = LowPass(1.0 / dt, self.fc)
        elif t - self._last_seen.get(name, t) > self.reset_after_s:
            f.reset()
        self._last_seen[name] = t
        return f(value, dt)
