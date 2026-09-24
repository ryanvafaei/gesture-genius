"""
Exercise 5: finger tapping (lift one finger at a time, hand flat on the table).

Camera placement is the main risk: a laptop webcam sees the table at a very
low angle. Check it with tools/tracking_check.py first. If tracking is poor,
use a tilted phone stand or an external camera looking down at the hand,
or ask her to lift the finger higher than in the video.

Metric: per finger, how far the fingertip has left its "flat" position
(distance from the palm plane, in palm sizes) and how much the MCP angle
has changed, each relative to that finger's lift threshold, averaged (one
noisy measure alone does not make a lift). Both are compared as absolute
changes, so they do not depend on which way the palm normal points.

Each finger has its own threshold: the index lift measured during
calibration, scaled by finger_scale (the ring finger is tied to its
neighbours and lifts much less than the index and middle fingers). The flat
position is measured in calibration and then slowly follows the hand while
no finger is lifted, so a hand that settles or slides on the table does
not look like a lift. When the prompted finger and a neighbour that moves
along with it both rise, the prompted finger is taken (target_bias).

Isolation score: how little the other fingers moved while the target finger
was lifted (1 = only the target moved), allowing the neighbours of the
lifted finger a little coupled movement. This is the main quality measure.

Modes (config "mode"):
  in_order    index -> pinky and back
  called_out  the coach names a random finger; reaction time is logged
  pattern     repeat a short pattern from memory
"""

import numpy as np

from rehab.features import FINGERS
from rehab.exercises.base import (GUIDED_ORDER, CalibrationStep, FINGER_WORDS,
                                  Say, SequenceExercise)


def _flat(f):
    values = {}
    for finger in FINGERS:
        values[f"h_{finger}"] = f.tip_height[finger]
        values[f"mcp_{finger}"] = float(f.joint_flexion[finger][0])
    return values


def _lift_index(f):
    return {"h_index": f.tip_height["index"], "mcp_index": float(f.joint_flexion["index"][0])}


MODES = {"in_order": "guided", "called_out": "called_out", "pattern": "memory"}


class FingerTapping(SequenceExercise):
    name = "finger_tapping"
    title = "Lift one finger at a time"
    instructions = (
        "Rest your hand flat on the table.",
        "Lift one finger at a time, and keep the others down.",
    )
    need_palm_facing = False
    action_word = "Lift"
    wrong_phrase = "That was your {got}. Let's try the {want}."
    progress_phrase = "You lifted your fingers {pct}% quicker than {when}."

    @classmethod
    def calibration_steps(cls):
        return [
            CalibrationStep("flat", "Rest your hand flat on the table and relax it. And hold.",
                            _flat, screen_text="Hand flat and still"),
            CalibrationStep("lift", "Now lift your index finger as high as is comfortable. And hold.",
                            _lift_index, screen_text="Lift index finger and hold"),
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flat = self.cal.get("flat", {})
        lift = self.cal.get("lift", {})
        measured = abs(lift.get("h_index", 0.0) - flat.get("h_index", 0.0))
        self.lift_threshold = max(self.params.get("min_lift", 0.08),
                                  self.params.get("lift_factor", 0.5) * measured)
        self.mcp_threshold = self.params.get("mcp_lift_deg", 15.0)
        scale = self.params.get("finger_scale", {})
        floor = self.params.get("min_lift_floor", 0.04)
        # per finger: (height threshold, MCP angle threshold)
        self.thresholds = {k: (max(floor, scale.get(k, 1.0) * self.lift_threshold),
                               max(5.0, scale.get(k, 1.0) * self.mcp_threshold))
                           for k in FINGERS}
        self._base = dict(flat)         # the flat position, following the hand
        self._base_t = None
        self._lifted = None
        self._candidate = None
        self._candidate_t = None
        self._peaks = None
        self._lift_round = None
        self._scores = {}

    def initial_mode(self):
        mode = MODES.get(self.params.get("mode", "in_order"), "guided")
        if mode == "guided":
            return mode, len(GUIDED_ORDER)
        if mode == "called_out":
            return mode, int(self.params.get("called_out_count", 8))
        return mode, int(self.params.get("pattern_length", 3))

    def lift_scores(self, f):
        """Per finger: movement away from flat, 1.0 = at that finger's lift threshold."""
        scores = {}
        for finger in FINGERS:
            h_thr, a_thr = self.thresholds[finger]
            by_height = abs(f.tip_height[finger] - self._base.get(f"h_{finger}", 0.0)) / h_thr
            by_angle = abs(float(f.joint_flexion[finger][0])
                           - self._base.get(f"mcp_{finger}", 0.0)) / a_thr
            scores[finger] = (by_height + by_angle) / 2.0
        return scores

    def _follow_hand(self, f, s, now):
        """While nothing is lifted, the resting fingers' flat position drifts with the hand."""
        dt = 0.0 if self._base_t is None else max(0.0, now - self._base_t)
        self._base_t = now
        if self._lifted or not dt:
            return
        k = min(1.0, dt / self.params.get("baseline_tau_s", 3.0))
        below = self.params.get("baseline_below", 0.6)
        for finger in FINGERS:
            if s[finger] < below:
                for key, value in ((f"h_{finger}", f.tip_height[finger]),
                                   (f"mcp_{finger}", float(f.joint_flexion[finger][0]))):
                    base = self._base.get(key, value)
                    self._base[key] = base + k * (value - base)

    def _prompted(self):
        r = self._round
        if r is None or r["step"] >= len(r["seq"]):
            return None
        return r["seq"][r["step"]]

    def _isolation(self, lifted):
        """1 - mean movement of the other fingers; neighbours may move a little with it."""
        i = FINGERS.index(lifted)
        allowance = self.params.get("coupling_allowance", 0.3)
        others = []
        for j, k in enumerate(FINGERS):
            if k == lifted:
                continue
            peak = self._peaks[k] - (allowance if abs(i - j) == 1 else 0.0)
            others.append(min(1.0, max(0.0, peak)))
        return float(1.0 - np.mean(others))

    def detect(self, f, now):
        s = self._scores = self.lift_scores(f)
        self._follow_hand(f, s, now)
        events = []
        if self._lifted:
            for finger in FINGERS:
                self._peaks[finger] = max(self._peaks[finger], s[finger])
            if s[self._lifted] < self.params.get("release_ratio", 0.6):
                events.append(("end", self._lifted, {"isolation": self._isolation(self._lifted)}))
                self._lifted = None
            return events

        best = max(s, key=s.get)
        target = self._prompted()
        keep = self.params.get("candidate_keep", 0.75)
        pick = None
        if target and s[target] >= 1.0 and s[target] >= self.params.get("target_bias", 0.75) * s[best]:
            pick = target               # a neighbour rising with it does not take the lift
        elif s[best] >= 1.0:
            pick = best
        if (self._candidate and pick != self._candidate and pick != target
                and s[self._candidate] >= keep):
            pick = self._candidate      # still up (a noisy frame below 1): keep its timer running
        if pick is None:
            self._candidate = None
        elif pick != self._candidate:
            self._candidate, self._candidate_t = pick, now
        elif now - self._candidate_t >= self.params.get("min_lift_s", 0.25):
            self._lifted = pick
            self._candidate = None
            self._peaks = {k: 0.0 for k in FINGERS}
            self._peaks[pick] = s[pick]
            self._lift_round = self._round
            events.append(("start", pick, {}))
        return events

    def on_event_end(self, finger, info, now):
        isolation = info.get("isolation", float("nan"))
        if self._lift_round is not None and self._lift_round is self._round:
            self._round["extra"].append(isolation)
        elif self.reps:
            # the round just finished on this lift; attach to the last record
            rec = self.reps[-1]
            rec.extra.setdefault("isolations", []).append(isolation)
            rec.extra["isolation"] = round(float(np.mean(rec.extra["isolations"])), 3)
        if isolation >= self.params.get("good_isolation", 0.75):
            if self._hints.ready("isolation_praise", now):
                return [Say(f"Nice, only your {FINGER_WORDS[finger]} moved.", "praise", optional=True)]
        elif isolation < 0.4 and self._hints.ready("isolation_hint", now):
            return [Say("Try to keep the other fingers resting on the table.", "hint")]
        return []

    def round_extra(self, r):
        iso = [v for v in r["extra"] if v == v]
        return {"isolation": round(float(np.mean(iso)), 3) if iso else ""}

    def interrupt(self, now):
        super().interrupt(now)
        self._lifted = None
        self._candidate = None

    def debug_settings(self):
        return {"thresholds": {k: {"height": h, "mcp_deg": a} for k, (h, a) in self.thresholds.items()},
                "lift_threshold": self.lift_threshold, "flat": self.cal.get("flat", {})}

    def debug_state(self):
        return {"scores": dict(self._scores), "baseline": dict(self._base),
                "candidate": self._candidate, "lifted": self._lifted,
                "prompted": self._prompted(),
                "peaks": dict(self._peaks) if self._lifted else None}

    def _make_display(self, f):
        d = super()._make_display(f)
        d["lift_scores"] = dict(self._scores)
        if self._lifted:
            d["finger_colors"] = dict(d["finger_colors"], **{self._lifted: "active"})
        return d
