"""
Exercise 6: grip squeeze (rolled washcloth or soft ball).

A camera cannot measure force. This exercise tracks holding and timing:
the hand closes a little further around the object and holds, then relaxes.

Metric: finger closure (1 - openness), scaled between her gentle hold of the
object (0) and her squeeze (1). The object hides parts of the fingers, so the
thresholds in config are looser.
"""

from rehab.exercises.base import CalibrationStep, Phase, TwoPhaseExercise, scale


def _closure(f):
    return {"closure": 1.0 - f.openness_mean}


class GripSqueeze(TwoPhaseExercise):
    name = "grip_squeeze"
    title = "Squeeze the cloth"
    instructions = (
        "Take the rolled cloth in your hand.",
        "Squeeze when I say squeeze, and let go when I say relax.",
    )
    need_palm_facing = False
    best_phrase = "Lovely strong squeeze."
    progress_phrase = "You held your squeezes {pct}% longer than {when}."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phases = (
            Phase("squeeze", "high", "Squeeze.", "SQUEEZE", self.params["hold_s"]),
            Phase("relax", "low", "And relax.", "RELAX", self.params.get("relax_s", self.params["hold_s"])),
        )

    @classmethod
    def calibration_steps(cls):
        return [
            CalibrationStep("hold", "Hold the cloth gently, without squeezing. And hold.",
                            _closure, screen_text="Hold gently"),
            CalibrationStep("squeeze", "Now squeeze it as firmly as is comfortable. And hold.",
                            _closure, screen_text="Squeeze and hold"),
        ]

    def metric(self, f):
        lo = self.cal.get("hold", {}).get("closure", 0.5)
        hi = self.cal.get("squeeze", {}).get("closure", 0.7)
        return scale(1.0 - f.openness_mean, lo, hi)

    def raw_metric(self, f):
        return 1.0 - f.openness_mean

    def observe(self, f, now, value):
        if self.phases[self._phase].key == "relax":
            rep = self._rep
            rep["relax_min"] = min(rep.get("relax_min", value), value)
        return []

    def stall_hint(self, f):
        if self.phases[self._phase].key == "relax":
            return "Let your hand relax completely."
        return "Squeeze the cloth a little."

    def rep_extra(self):
        relax_min = self._rep.get("relax_min", float("nan"))
        return {
            # real time spent squeezing (can be longer than the target hold)
            "squeeze_hold_s": self._rep["zone_time"].get("high_s", 0.0),
            "relaxed_fully": bool(relax_min <= 0.15) if relax_min == relax_min else "",
        }

    def _finish_rep(self, now):
        out = super()._finish_rep(now)
        rec = self.reps[-1]
        # summaries compare hold time for this exercise, not "range"
        rec.raw_high = rec.extra.get("high_zone_s", rec.raw_high)
        return out
