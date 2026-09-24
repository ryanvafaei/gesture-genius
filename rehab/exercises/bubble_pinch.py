"""
Exercise 7: bubble pinch (thumb and index finger).

The pinch grip is what buttons, a pinch of salt and small seeds need. On
screen a soft bubble sits between her thumb and index fingertip; it shrinks
while she pinches and holds, and pops when the rep is done.

Metric: thumb tip -> index tip distance (divided by palm size), scaled
between her calibrated "open" (0) and "pinch" (1), so a closer pinch is a
higher value. The raw value is the pinch closure 1 - distance / 1.5, higher
is better, so bests and progress messages read the same way as elsewhere.
"""

from rehab.exercises.base import (CalibrationStep, Phase, TwoPhaseExercise,
                                  increases, scale)

# distance (palm sizes) that counts as "fully open" for the raw closure
OPEN_REFERENCE = 1.5


def _dist(f):
    return {"dist": f.thumb_tip_dist["index"]}


class BubblePinch(TwoPhaseExercise):
    name = "bubble_pinch"
    title = "Pinch the bubble"
    instructions = (
        "Now pinch your thumb and index finger together.",
        "Hold the pinch, and the bubble pops.",
    )
    need_palm_facing = True
    range_steps = ("open", "pinch")
    progress_phrase = "Your pinch closed {pct}% further than {when}."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phases = (
            Phase("pinch", "high", "Pinch your thumb and finger together.", "PINCH"),
            Phase("release", "low", "And let go.", "OPEN",
                  self.params.get("release_s", 1.0), count=False),
        )
        self._popped_t = None

    @classmethod
    def calibration_steps(cls):
        return [
            CalibrationStep("open", "Open your hand, with your thumb away from your index finger. "
                                    "And hold.", _dist, need_palm_facing=True,
                            screen_text="Thumb away and hold"),
            CalibrationStep("pinch", "Now pinch your thumb and index finger together. And hold.",
                            _dist, need_palm_facing=True, screen_text="Pinch and hold"),
        ]

    @classmethod
    def calibration_valid(cls, steps):
        # pinching brings the fingertips closer together
        return increases(steps, "pinch", "open", "dist", 0.1)

    def metric(self, f):
        lo = self.cal.get("open", {}).get("dist", 1.0)
        hi = self.cal.get("pinch", {}).get("dist", 0.2)
        return scale(f.thumb_tip_dist["index"], lo, hi)

    def raw_metric(self, f):
        return 1.0 - min(f.thumb_tip_dist["index"], OPEN_REFERENCE) / OPEN_REFERENCE

    def stall_hint(self, f):
        if self.phases[self._phase].key == "release":
            return "Open your fingers again."
        return "Bring your thumb and index finger together."

    def _finish_rep(self, now):
        out = super()._finish_rep(now)
        self._popped_t = now
        return out

    def _make_display(self, f, value):
        d = super()._make_display(f, value)
        pinching = self.phases[self._phase].key == "pinch"
        d["bubble"] = {
            # the bubble shrinks while the pinch is held
            "hold_progress": d["hold_progress"] if pinching else 0.0,
            "pinching": pinching,
            "popped_t": self._popped_t,
            "now": self._rep["times"][-1] if self._rep["times"] else None,
        }
        return d
