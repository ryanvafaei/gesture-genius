"""
Exercise 5: finger tapping (lift one finger at a time, hand flat on the table).

The camera sees the back of the hand here, unlike in every other exercise,
and often from a low angle. That changes three things:

  * MediaPipe's Left/Right label is unreliable for the back of a hand. The
    knuckle triangle (features.dorsal_side) vouches for the hand instead
    (palm_down), and a frame whose label flickers while the geometry says
    it is the right hand is skipped silently: nothing is decided on it.
    A new label that persists for label_switch_s is accepted (and the
    resting position measured again).
  * The 3D hand shape jumps as a whole at this angle, so single joint
    angles are useless. The lift is measured in the picture instead
    (features.tip_rise: fingertip above its own knuckle, out of the back
    of the hand, in hand sizes), and each finger is compared with the
    other three, which cancels a whole-hand jump.
  * A calibration's "flat" does not match today's resting hand. The
    resting position (baseline) is taken from the live hand: the median of
    baseline_seed_s of frames at the start of each set, after every pause
    or lost hand, after a jump of the hand (a new or ghost hand) and after
    a label change. It then follows the resting fingers.

Score per finger: (its lift - median lift of the other three) / its
threshold, signed: a finger moving down is never a lift. 1.0 = at the
threshold. The threshold is the index lift measured during calibration
(times lift_factor), scaled per finger by finger_scale.

A lift starts when a finger scores >= 1 for min_lift_s (the prompted finger
wins when a neighbour rises with it: target_bias). It ends when its score
drops below release_ratio, when another finger clearly takes over
(switch_ratio), or after max_lift_s: a stuck lift must never block the
next finger.

Isolation score: how little the other fingers moved while the target finger
was lifted (1 = only the target moved), allowing the neighbours of the
lifted finger a little coupled movement. The other fingers' movement is
their lift above their own resting position (per threshold), not the score
against each other: three fingers rising together would cancel out there.
This is the main quality measure.

Modes (config "mode"):
  in_order    index -> pinky and back
  called_out  the coach names a random finger; reaction time is logged
  pattern     repeat a short pattern from memory

tools/replay.py re-runs a verbose log through this detector (for tuning).
"""

import numpy as np

from rehab import config
from rehab.features import FINGERS
from rehab.exercises.base import (GUIDED_ORDER, CalibrationStep, FINGER_WORDS,
                                  Say, SequenceExercise)

# a calibrated index lift smaller than this (hand sizes) is not a lift
MIN_CALIBRATED_LIFT = 0.02


def _flat(f):
    return {f"r_{finger}": f.tip_rise[finger] for finger in FINGERS}


def _lift_index(f):
    return {"r_index": f.tip_rise["index"]}


MODES = {"in_order": "guided", "called_out": "called_out", "pattern": "memory"}


class FingerTapping(SequenceExercise):
    name = "finger_tapping"
    title = "Lift one finger at a time"
    instructions = (
        "Rest your hand flat on the table.",
        "Lift one finger at a time, and keep the others down.",
    )
    need_palm_facing = False
    palm_down = True
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

    @classmethod
    def calibration_valid(cls, steps):
        """Needs the image-space rises (older calibrations are measured again) and a real lift."""
        flat, lift = (steps or {}).get("flat", {}), (steps or {}).get("lift", {})
        if not all(f"r_{finger}" in flat for finger in FINGERS) or "r_index" not in lift:
            return False
        return lift["r_index"] - flat["r_index"] >= MIN_CALIBRATED_LIFT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flat = self.cal.get("flat", {})
        lift = self.cal.get("lift", {})
        measured = max(0.0, lift.get("r_index", 0.0) - flat.get("r_index", 0.0))
        self.lift_threshold = max(self.params.get("min_lift", 0.12),
                                  self.params.get("lift_factor", 0.5) * measured)
        scale = self.params.get("finger_scale", {})
        floor = self.params.get("min_lift_floor", 0.08)
        self.thresholds = {k: max(floor, scale.get(k, 1.0) * self.lift_threshold) for k in FINGERS}
        self._base = None               # finger -> resting rise; None while it is measured
        self._seed = []
        self._seed_t = None
        self._base_t = None
        self._label = None              # the accepted Left/Right label
        self._trust_low = False         # accepted although its score is low
        self._odd_since = None          # first frame of a run with another label
        self._last_wrist = None
        self._high_since = {}           # finger -> since when it scores high without a lift
        self._lifted = None
        self._lifted_t = None
        self._candidate = None
        self._candidate_t = None
        self._peaks = None
        self._lift_round = None
        self._scores = {}
        self._skipped = False

    def initial_mode(self):
        mode = MODES.get(self.params.get("mode", "in_order"), "guided")
        if mode == "guided":
            return mode, len(GUIDED_ORDER)
        if mode == "called_out":
            return mode, int(self.params.get("called_out_count", 8))
        return mode, int(self.params.get("pattern_length", 3))

    # --- the resting position -------------------------------------------------------

    def _reseed(self, why):
        """Measure the resting position again; nothing is detected meanwhile."""
        self._base = None
        self._seed = []
        self._seed_t = None
        self._lifted = None
        self._candidate = None
        self._high_since = {}
        self.trace("baseline", exercise=self.name, reason=why)

    def _seeding(self, f, now):
        """Collect the resting position; True while it is still being measured."""
        if self._base is not None:
            return False
        if self._seed_t is None:
            self._seed_t = now
        self._seed.append(dict(f.tip_rise))
        if now - self._seed_t >= self.params.get("baseline_seed_s", 0.4) and len(self._seed) >= 3:
            self._base = {k: float(np.median([s[k] for s in self._seed])) for k in FINGERS}
            self._base_t = now
            self._seed = []
        return True

    def _lifts(self, f):
        return {k: f.tip_rise[k] - self._base[k] for k in FINGERS}

    def _reset_finger(self, finger, lifts):
        """Its resting position becomes where it is now (relative to the others)."""
        others = np.median([lifts[k] for k in FINGERS if k != finger])
        self._base[finger] += lifts[finger] - others
        self._high_since.pop(finger, None)

    def lift_scores(self, f):
        """Per finger: lift compared with the other three, 1.0 = at that finger's threshold."""
        if self._base is None:
            return {k: 0.0 for k in FINGERS}
        lifts = self._lifts(f)
        return {k: (lifts[k] - float(np.median([lifts[j] for j in FINGERS if j != k])))
                / self.thresholds[k] for k in FINGERS}

    def _follow_hand(self, f, s, now):
        """
        The resting fingers' baseline drifts with the hand (faster when a
        finger sits below it: that is never a lift). A finger that sits high
        without counting as a lift for baseline_stale_s is rested where it is.
        """
        dt = max(0.0, now - self._base_t) if self._base_t is not None else 0.0
        self._base_t = now
        lifts = self._lifts(f)
        below = self.params.get("baseline_below", 0.6)
        tau = self.params.get("baseline_tau_s", 3.0)
        down_tau = self.params.get("baseline_down_tau_s", 0.5)
        for finger in FINGERS:
            if finger == self._lifted:
                continue
            if s[finger] >= below:
                since = self._high_since.setdefault(finger, now)
                if now - since > self.params.get("baseline_stale_s", 3.0):
                    self._reset_finger(finger, lifts)
                continue
            self._high_since.pop(finger, None)
            if dt:
                k = min(1.0, dt / (down_tau if lifts[finger] < 0 else tau))
                self._base[finger] += k * lifts[finger]

    # --- what the camera shows ---------------------------------------------------------

    def _clean(self, f, now):
        """
        False for a frame to skip: its label differs from the accepted one or
        is unsure, while the geometry says it is her hand (else it would be a
        quality problem). A different label that lasts label_switch_s is
        accepted, and the resting position measured again.
        """
        min_score = config.MIN_HANDEDNESS_SCORE
        if self._label is None and f.correct_hand:
            self._label = f.handedness
        if f.handedness == self._label and (f.handedness_score >= min_score or self._trust_low):
            if f.handedness_score >= min_score:
                self._trust_low = False
            self._odd_since = None
            return True
        if self._odd_since is None:
            self._odd_since = now
        if now - self._odd_since < self.params.get("label_switch_s", 1.0):
            return False
        self._label = f.handedness
        self._trust_low = f.handedness_score < min_score
        self._odd_since = None
        self._reseed("label")
        return True

    def _jumped(self, f):
        """The hand moved more than baseline_jump hand sizes since the last frame."""
        wrist, last = f.wrist_image, self._last_wrist
        self._last_wrist = None if wrist is None else np.array(wrist, dtype=float)
        if wrist is None or last is None or not f.image_size:
            return False
        size = f.hand_size_image * f.image_size[1]
        return float(np.linalg.norm(self._last_wrist - last)) > self.params.get("baseline_jump", 0.8) * size

    # --- detection ----------------------------------------------------------------------

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

    def _end(self, reason):
        finger = self._lifted
        self._lifted = None
        return ("end", finger, {"isolation": self._isolation(finger), "reason": reason})

    def detect(self, f, now):
        self._skipped = False
        if self._jumped(f):
            self._reseed("jump")
        if not self._clean(f, now):
            self._skipped = True        # hold everything: timers, baseline, the lift
            return []
        if self._seeding(f, now):
            self._scores = {k: 0.0 for k in FINGERS}
            return []
        s = self._scores = self.lift_scores(f)
        self._follow_hand(f, s, now)
        events = []
        if self._lifted:
            # isolation: the others' own lift, not compared with each other
            # (three fingers rising together would cancel out)
            lifts = self._lifts(f)
            for finger in FINGERS:
                v = s[finger] if finger == self._lifted else lifts[finger] / self.thresholds[finger]
                self._peaks[finger] = max(self._peaks[finger], v)
            lifted = self._lifted
            rival = max((k for k in FINGERS if k != lifted), key=s.get)
            if s[lifted] < self.params.get("release_ratio", 0.6):
                events.append(self._end("down"))
            elif s[rival] >= 1.0 and s[rival] > self.params.get("switch_ratio", 1.5) * s[lifted]:
                events.append(self._end("switch"))
            elif now - self._lifted_t >= self.params.get("max_lift_s", 5.0):
                events.append(self._end("too_long"))
                self._reset_finger(lifted, self._lifts(f))
                s = self._scores = self.lift_scores(f)
            if self._lifted:
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
            self._lifted, self._lifted_t = pick, now
            self._candidate = None
            self._high_since.pop(pick, None)
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

    def start_set(self, set_no, now):
        super().start_set(set_no, now)
        self._reseed("set")

    def interrupt(self, now):
        super().interrupt(now)
        self._reseed("interrupt")

    def debug_settings(self):
        return {"thresholds": dict(self.thresholds), "lift_threshold": self.lift_threshold,
                "calibration": {k: self.cal.get(k, {}) for k in ("flat", "lift")}}

    def debug_state(self):
        return {"scores": dict(self._scores),
                "baseline": dict(self._base) if self._base is not None else None,
                "candidate": self._candidate, "lifted": self._lifted,
                "prompted": self._prompted(), "label": self._label, "skipped": self._skipped,
                "peaks": dict(self._peaks) if self._lifted else None}

    def _make_display(self, f):
        d = super()._make_display(f)
        d["lift_scores"] = dict(self._scores)
        if self._lifted:
            d["finger_colors"] = dict(d["finger_colors"], **{self._lifted: "active"})
        return d
