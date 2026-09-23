"""
Exercise 1: grip and release (advanced).

Metric: average finger openness (index..pinky), each finger scaled to its own
calibrated range. A partial opening counts against her range, not a healthy
hand's. The built-in gesture classifier is only a secondary check: when it
disagrees with the measurement this is logged, never shown.
"""

import numpy as np

from rehab.features import FINGERS
from rehab.exercises.base import (CalibrationStep, FINGER_WORDS, Phase,
                                  TwoPhaseExercise, scale)

EXPECTED_GESTURE = {"open": "Open_Palm", "close": "Closed_Fist"}


def _openness(f):
    values = {finger: f.openness[finger] for finger in FINGERS}
    values["mean"] = f.openness_mean
    return values


class GripRelease(TwoPhaseExercise):
    name = "grip_release"
    title = "Open and close your hand"
    instructions = (
        "Let's open and close your hand.",
        "Rest your elbow and show me your palm.",
    )
    need_palm_facing = True
    best_phrase = "That's your widest yet today."
    progress_phrase = "Your hand opened {pct}% wider than {when}."
    phases = (
        Phase("open", "high", "Open your hand wide.", "OPEN"),
        Phase("close", "low", "Now close your hand.", "CLOSE"),
    )

    @classmethod
    def calibration_steps(cls):
        return [
            CalibrationStep("open", "Open your hand as wide as you comfortably can. And hold.",
                            _openness, need_palm_facing=True, screen_text="Open wide and hold"),
            CalibrationStep("closed", "Now close your hand as much as you can. And hold.",
                            _openness, need_palm_facing=False, screen_text="Close and hold"),
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._finger_values = {}
        self._lagging = None
        self._gesture_frames = 0
        self._gesture_disagree = 0

    def finger_values(self, f):
        """Openness of each finger as a fraction of her calibrated range."""
        lo = self.cal.get("closed", {})
        hi = self.cal.get("open", {})
        return {finger: scale(f.openness[finger], lo.get(finger, 0.0), hi.get(finger, 1.0))
                for finger in FINGERS}

    def metric(self, f):
        self._finger_values = self.finger_values(f)
        return float(np.mean(list(self._finger_values.values())))

    def raw_metric(self, f):
        return f.openness_mean

    def observe(self, f, now, value):
        phase = self.phases[self._phase]
        # lagging finger while opening
        self._lagging = None
        if phase.key == "open" and value > self.params["low"]:
            vals = self._finger_values
            finger = min(vals, key=vals.get)
            others = np.mean([v for k, v in vals.items() if k != finger])
            if others - vals[finger] >= self.params.get("lag_margin", 0.15):
                self._lagging = finger
                self._rep.setdefault("lagging", {}).setdefault(finger, 0)
                self._rep["lagging"][finger] += 1
        # gesture recognizer as a secondary check, only during holds
        if self._holding and f.gesture and f.gesture != "None":
            self._gesture_frames += 1
            other = "close" if phase.key == "open" else "open"
            if f.gesture == EXPECTED_GESTURE[other]:
                self._gesture_disagree += 1
        return []

    def stall_hint(self, f):
        if self.phases[self._phase].key == "open" and self._lagging:
            return f"Try to open your {FINGER_WORDS[self._lagging]} a little more."
        return super().stall_hint(f)

    def rep_extra(self):
        lag = self._rep.get("lagging", {})
        extra = {
            "lagging_finger": max(lag, key=lag.get) if lag else "",
            "gesture_disagreement": round(self._gesture_disagree / self._gesture_frames, 2)
            if self._gesture_frames else "",
        }
        self._gesture_frames = self._gesture_disagree = 0
        return extra

    def _make_display(self, f, value):
        d = super()._make_display(f, value)
        d["finger_values"] = dict(self._finger_values)
        if self._lagging:
            d["finger_colors"] = {self._lagging: "lagging"}
        return d
