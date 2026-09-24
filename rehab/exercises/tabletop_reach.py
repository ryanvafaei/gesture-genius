"""
Reaching on the table ("light the lamps"): the Reaching Performance Scale
(Levin 2004) trunk component.

She reaches for a near lamp (about 1 cm from the table edge) and a far one
(about 30 cm), in turn. The camera sits at 45 degrees between facing her
and side-on, on the affected side. Measure: how far the hand moved, in arm
lengths, and how much of that the trunk did (shoulder middle moving along
the same direction). People after a stroke use up to 4.5 times the trunk
movement of unimpaired people (Levin 2004).

Score 0-3 per reach (benchmarks.rps_trunk_score):
  near   3: almost no trunk (share <= 0.10, proposed), 1: over half, 0: all
  far    3: about a quarter (0.25 +/- 0.10) with the elbow almost straight,
         1: about half, 0: over three quarters and the hand does not arrive
A reach scoring 2 or 3 is a success; 0-1 gives one cue:
"Let your arm do the reaching." There is no FMA item and no ladder.
"""

from dataclasses import replace

import numpy as np

from rehab import benchmarks, config
from rehab.exercises.arm import ArmExercise

TRUNK_CUE = "Let your arm do the reaching."


class TabletopReach(ArmExercise):
    name = "tabletop_reach"
    title = "Reach across the table"
    instructions = (
        "Let's reach across the table, to the near lamp and the far lamp.",
        "Sit with your back against the chair and your {side} hand at the edge of the table.",
        "Turn your chair halfway, with your {side} arm towards the screen.",
    )
    metric = "reach"
    metric_label = "Reach"
    unit = "arm"
    direction = "increase"
    view = "oblique"
    joints = ("hip", "shoulder", "elbow", "wrist")
    both_sides = ("shoulder", "hip")
    uses_ladder = False
    form_rules = ()
    compensation_rules = ()
    start_prompt = "Rest your hand at the edge of the table."
    return_prompt = "And back to the edge of the table."
    calibration_screen = "Reach"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tolerance = config.REACH_PRIMARY_TOLERANCE     # arm lengths, not degrees
        self._anchor = None
        self._peak_frame = None

    @classmethod
    def make_calibration(cls, therapist=None):
        return None             # scored against the scale, not a personal range

    @property
    def target_name(self):
        return "near" if self.reps_this_set % 2 == 0 else "far"

    @property
    def move_prompt(self):
        return f"Now reach for the {self.target_name} lamp."

    @property
    def again_prompt(self):
        return self.move_prompt

    def plausible_bounds(self):
        return None, None

    # --- measures -------------------------------------------------------------------

    def _points(self, f):
        side = self.side
        wrist = f.point(f"{side}_wrist")
        arm_len = (np.linalg.norm(f.point(f"{side}_elbow") - f.point(f"{side}_shoulder"))
                   + np.linalg.norm(wrist - f.point(f"{side}_elbow")))
        return wrist, f.shoulder_mid, float(arm_len)

    def primary(self, f, arm):
        """Hand displacement from where the rep started, in arm lengths."""
        if not f.present or f.points is None:
            return float("nan")
        wrist, shoulder_mid, arm_len = self._points(f)
        if self.state == "start" or self._anchor is None:
            self._anchor = (wrist.copy(), shoulder_mid.copy(), arm_len)
        start_wrist, _, start_len = self._anchor
        return float(np.linalg.norm(wrist - start_wrist) / max(start_len, 1e-6))

    def in_start(self, f, arm, value):
        return value <= self.tolerance

    def interrupt(self, now):
        super().interrupt(now)
        self._anchor = None

    def on_peak(self, f, arm):
        wrist, shoulder_mid, _ = self._points(f)
        self._peak_frame = (wrist.copy(), shoulder_mid.copy(), arm.get("elbow_flexion"))

    # --- scoring ----------------------------------------------------------------------

    def adjust_result(self, result, rep):
        target = "close" if self.target_name == "near" else "far"
        share, arrived, elbow = float("nan"), False, float("nan")
        if self._anchor is not None and self._peak_frame is not None:
            start_wrist, start_shoulder, arm_len = self._anchor
            wrist, shoulder, elbow = self._peak_frame
            reach = wrist - start_wrist
            dist = float(np.linalg.norm(reach))
            if dist > 1e-6:
                axis = reach / dist
                share = max(0.0, float(np.dot(shoulder - start_shoulder, axis)) / dist)
            reached = dist / max(arm_len, 1e-6)
            arrived = target == "close" or reached >= config.REACH_FAR_ARRIVED
        straight = benchmarks.finite(elbow) and elbow <= config.REACH_ELBOW_ALMOST_STRAIGHT
        score = benchmarks.rps_trunk_score(share, target, arrived, straight, self.bench)
        self._rps = {"rps_trunk_score": score, "trunk_share": round(share, 3) if share == share else None,
                     "reach_target": target, "hand_arrived": arrived,
                     "elbow_at_reach_deg": round(float(elbow), 1) if benchmarks.finite(elbow) else None}
        moved = result.moved if benchmarks.finite(result.moved) else 0.0
        success = moved >= self.tolerance and score >= 2 and arrived
        compensation = score <= 1 and moved >= self.tolerance
        return replace(result, coaching_success=success, with_compensation=compensation,
                       compensations=["trunk_reach"] if compensation else [],
                       cue="trunk_reach" if compensation else ("more_range" if not arrived else None),
                       feedback_key=("praise_specific" if success else "one_cue"),
                       fma_style_score=0)

    def cue_text(self, rule):
        if rule == "trunk_reach":
            return TRUNK_CUE
        return super().cue_text(rule)

    def rep_extra(self, result, rep):
        extra = dict(getattr(self, "_rps", {}))
        self._peak_frame = None
        return extra

    def display_range(self):
        return 0.0, 1.2
