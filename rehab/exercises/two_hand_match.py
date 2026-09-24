"""
Exercise 8: two-hand match (open and close both hands together).

Her stronger hand leads and the weaker one follows, as in mirror therapy
where the healthy side guides the affected side. Two bars on screen show
how open each hand is; the closer they are, the better they match.

Metric (drives the reps): the affected hand's openness, scaled to its own
calibrated range, as in grip and release. The other hand is scaled to its
own range too, and every rep logs the symmetry: 1 - mean |affected - other|
over the rep (1 = the hands moved exactly together).
"""

import numpy as np

from rehab import config
from rehab.exercises.base import (CalibrationStep, Phase, TwoPhaseExercise,
                                  increases, scale)


def _both(f):
    other = f.other.openness_mean if f.other is not None and f.other.present else float("nan")
    return {"affected": f.openness_mean, "other": other}


class TwoHandMatch(TwoPhaseExercise):
    name = "two_hand_match"
    title = "Both hands together"
    instructions = (
        "Now open and close both hands together.",
        "Let your stronger hand lead the way.",
    )
    need_palm_facing = True
    need_both_hands = True
    range_steps = ("closed", "open")
    progress_phrase = "Your hands moved {pct}% more together than {when}."
    phases = (
        Phase("open", "high", "Open both hands.", "OPEN"),
        Phase("close", "low", "Now close both hands.", "CLOSE"),
    )

    def __init__(self, *args, hand=config.AFFECTED_HAND, **kwargs):
        super().__init__(*args, **kwargs)
        self.affected = hand
        self.other_word = "right" if hand == "Left" else "left"
        self._pair = {}

    @classmethod
    def calibration_steps(cls):
        return [
            CalibrationStep("open", "Open both hands as wide as you comfortably can. And hold.",
                            _both, need_palm_facing=True, screen_text="Open both and hold",
                            need_both_hands=True),
            CalibrationStep("closed", "Now close both hands as much as you can. And hold.",
                            _both, screen_text="Close both and hold", need_both_hands=True),
        ]

    @classmethod
    def calibration_valid(cls, steps):
        return (increases(steps, "closed", "open", "affected", config.CALIBRATION_MIN_RANGE / 2)
                and increases(steps, "closed", "open", "other", config.CALIBRATION_MIN_RANGE / 2))

    def _scaled(self, key, value):
        return scale(value, self.cal.get("closed", {}).get(key, 0.3),
                     self.cal.get("open", {}).get(key, 0.9))

    def metric(self, f):
        affected = self._scaled("affected", f.openness_mean)
        other = (self._scaled("other", f.other.openness_mean)
                 if f.other is not None and f.other.present else float("nan"))
        self._pair = {"affected": affected, "other": other}
        return affected

    def raw_metric(self, f):
        return f.openness_mean

    def observe(self, f, now, value):
        other = self._pair.get("other")
        if other is not None and other == other:
            self._rep.setdefault("gaps", []).append(abs(value - other))
        return []

    def stall_hint(self, f):
        if self.phases[self._phase].key == "open":
            return f"Open your {self.affected.lower()} hand as wide as your {self.other_word} one."
        return f"Close your {self.affected.lower()} hand like your {self.other_word} one."

    def rep_extra(self):
        gaps = self._rep.get("gaps") or []
        symmetry = float(np.clip(1.0 - np.mean(gaps), 0.0, 1.0)) if gaps else float("nan")
        return {"symmetry": round(symmetry, 3) if symmetry == symmetry else ""}

    def _finish_rep(self, now):
        out = super()._finish_rep(now)
        rec = self.reps[-1]
        # summaries compare how well the hands moved together, not range
        sym = rec.extra.get("symmetry")
        rec.raw_high = sym if isinstance(sym, float) else float("nan")
        return out

    def _make_display(self, f, value):
        d = super()._make_display(f, value)
        affected, other = self._pair.get("affected", value), self._pair.get("other", float("nan"))
        names = {"affected": self.affected.lower(), "other": self.other_word}
        d["pair_values"] = {names["affected"]: float(affected),
                            names["other"]: float(other) if other == other else 0.0}
        gap = abs(affected - other) if other == other else None
        d["symmetry"] = None if gap is None else float(np.clip(1.0 - gap, 0.0, 1.0))
        return d
