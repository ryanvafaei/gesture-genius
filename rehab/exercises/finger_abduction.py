"""
Exercise 2: finger abduction (spread and close).

Metric: the sum of the three gaps between neighbouring fingers (index-middle,
middle-ring, ring-pinky), measured as angles in the palm plane and scaled to
her calibrated range. Only valid while the fingers are straight, so the
exercise pauses when the fingers curl.
"""

import numpy as np

from rehab.features import GAPS, GAP_NAMES
from rehab.exercises.base import (CalibrationStep, FINGER_WORDS, Phase, Say,
                                  TwoPhaseExercise, scale)

ANGLE_MIN_RANGE = 3.0     # degrees


def _spread(f):
    values = dict(f.spread)
    values["total"] = f.spread_total
    return values


class FingerAbduction(TwoPhaseExercise):
    name = "finger_abduction"
    title = "Spread your fingers"
    instructions = (
        "Now we spread the fingers.",
        "Keep your fingers straight and show me your palm.",
    )
    need_palm_facing = True
    best_phrase = "That's your widest spread yet today."
    progress_phrase = "Your fingers spread {pct}% wider than {when}."
    phases = (
        Phase("spread", "high", "Spread your fingers apart.", "SPREAD"),
        Phase("together", "low", "Now bring them together.", "TOGETHER"),
    )

    @classmethod
    def calibration_steps(cls):
        return [
            CalibrationStep("together", "Keep your fingers straight and close together. And hold.",
                            _spread, need_palm_facing=True, screen_text="Fingers together"),
            CalibrationStep("spread", "Now spread your fingers as wide as you comfortably can. And hold.",
                            _spread, need_palm_facing=True, screen_text="Spread wide and hold"),
        ]

    def __init__(self, *args, grip_calibration=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grip_cal = grip_calibration or {}
        self._gaps = {}

    def straightness(self, f):
        """Finger openness, scaled to her grip calibration when she has one."""
        if self.grip_cal:
            lo = self.grip_cal.get("closed", {}).get("mean", 0.0)
            hi = self.grip_cal.get("open", {}).get("mean", 1.0)
            return scale(f.openness_mean, lo, hi)
        return f.openness_mean

    def frame_problem(self, f):
        if self.straightness(f) < self.params.get("min_openness", 0.7):
            return Say("Keep your fingers straight.", "quality")
        return None

    def gap_values(self, f):
        lo = self.cal.get("together", {})
        hi = self.cal.get("spread", {})
        return {g: scale(f.spread[g], lo.get(g, 0.0), hi.get(g, 20.0), ANGLE_MIN_RANGE)
                for g in GAP_NAMES}

    def metric(self, f):
        self._gaps = self.gap_values(f)
        lo = self.cal.get("together", {}).get("total", 0.0)
        hi = self.cal.get("spread", {}).get("total", 60.0)
        return scale(f.spread_total, lo, hi, ANGLE_MIN_RANGE)

    def raw_metric(self, f):
        return f.spread_total

    def smallest_gap(self):
        if not self._gaps:
            return None
        gap = min(self._gaps, key=self._gaps.get)
        others = np.mean([v for k, v in self._gaps.items() if k != gap])
        if others - self._gaps[gap] >= self.params.get("lag_margin", 0.15):
            return gap
        return None

    def observe(self, f, now, value):
        if self.phases[self._phase].key == "spread":
            gap = self.smallest_gap()
            if gap:
                self._rep.setdefault("small_gap", {}).setdefault(gap, 0)
                self._rep["small_gap"][gap] += 1
        return []

    def stall_hint(self, f):
        gap = self.smallest_gap()
        if self.phases[self._phase].key == "spread" and gap:
            a, b = GAPS[GAP_NAMES.index(gap)]
            return (f"Spread your {FINGER_WORDS[a].replace(' finger', '')} "
                    f"and {FINGER_WORDS[b]} a bit more.")
        return super().stall_hint(f)

    def rep_extra(self):
        small = self._rep.get("small_gap", {})
        return {"smallest_gap": max(small, key=small.get) if small else ""}

    def _make_display(self, f, value):
        d = super()._make_display(f, value)
        d["gap_values"] = dict(self._gaps)
        return d
