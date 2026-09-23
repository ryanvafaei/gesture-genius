"""
Exercise 3: thumb flexion / extension.

Metric: how far the thumb is "out", combining two calibrated measures:
  * thumb MCP + IP flexion (less flexion = more out)
  * distance from the thumb tip (4) to the pinky MCP (17) / palm size
Using both is more robust than either one when the thumb is partly hidden.

Compensation: the other four fingers should stay still.
"""

import numpy as np

from rehab.features import FINGERS
from rehab.exercises.base import CalibrationStep, Phase, TwoPhaseExercise, scale

ANGLE_MIN_RANGE = 10.0


def _thumb(f):
    return {"flexion": f.thumb_flexion, "to_pinky": f.thumb_to_pinky_mcp}


class ThumbFlexion(TwoPhaseExercise):
    name = "thumb_flexion"
    title = "Bend and stretch your thumb"
    instructions = (
        "Now only the thumb moves.",
        "Keep your other fingers relaxed and open.",
    )
    need_palm_facing = True
    best_phrase = "That's your best thumb stretch yet today."
    progress_phrase = "Your thumb moved {pct}% further than {when}."
    phases = (
        Phase("in", "low", "Bend your thumb across your palm.", "IN"),
        Phase("out", "high", "Now stretch your thumb out.", "OUT"),
    )
    compensation_messages = dict(TwoPhaseExercise.compensation_messages,
                                 other_fingers="Try to move only your thumb.")

    @classmethod
    def calibration_steps(cls):
        return [
            CalibrationStep("in", "Bend your thumb across your palm as far as you can. And hold.",
                            _thumb, need_palm_facing=True, screen_text="Thumb in and hold"),
            CalibrationStep("out", "Now stretch your thumb away from your hand. And hold.",
                            _thumb, need_palm_facing=True, screen_text="Thumb out and hold"),
        ]

    def metric(self, f):
        c_in, c_out = self.cal.get("in", {}), self.cal.get("out", {})
        # flexion decreases going out; scale() handles the reversed range,
        # so in -> 0 and out -> 1 for both measures
        by_angle = scale(f.thumb_flexion, c_in.get("flexion", 120.0),
                         c_out.get("flexion", 20.0), ANGLE_MIN_RANGE)
        by_distance = scale(f.thumb_to_pinky_mcp, c_in.get("to_pinky", 0.4),
                            c_out.get("to_pinky", 1.2))
        return float(np.mean([by_angle, by_distance]))

    def raw_metric(self, f):
        return f.thumb_to_pinky_mcp

    def observe(self, f, now, value):
        rep = self._rep
        others = np.array([f.openness[k] for k in FINGERS])
        if "others_min" not in rep:
            rep["others_min"] = others.copy()
            rep["others_max"] = others.copy()
        rep["others_min"] = np.minimum(rep["others_min"], others)
        rep["others_max"] = np.maximum(rep["others_max"], others)
        change = float(np.max(rep["others_max"] - rep["others_min"]))
        rep["others_change"] = change
        if change > self.params.get("max_other_finger_change", 0.25):
            return self._compensation("other_fingers", now)
        return []

    def rep_extra(self):
        return {"other_finger_change": round(self._rep.get("others_change", 0.0), 3)}
