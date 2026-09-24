"""
Exercise 1: grip and release (advanced).

Metric: average finger openness (index..pinky), each finger scaled to its own
calibrated range. A partial opening counts against her range, not a healthy
hand's. The built-in gesture classifier is only a secondary check: when it
disagrees with the measurement this is logged, never shown.

Benchmarks (the plan's "Bloom", benchmarks/BENCHMARK_PLAN.md 7.2): from the
joint angles held in each position every rep also gets
  FMA 25-style mass extension   2 when all five fingers are straight within
                                the finger tolerance (10 degrees, proposed)
  FMA 24-style mass flexion     2 when all fingertips touch the palm (0.35
                                palm sizes, proposed) with the thumb outside
  Bain 2015 functional open     MCP <= 19, PIP <= 23, DIP <= 10 degrees
  Bain 2015 functional grasp    MCP >= 71, PIP >= 87, DIP >= 64 degrees
The Bain values and a first FMA 25 score of 2 are milestones, announced
once. MediaPipe finger angles are not validated, so a gain is only claimed
for a mean change of FINGER_CHANGE_DEG per joint, or a steady rise over
TREND_SESSIONS sessions (claim_rule).
"""

import numpy as np

from rehab import benchmarks, config
from rehab.features import FINGERS, FULL_CURL_DEG
from rehab.storage import to_float
from rehab.exercises.base import (CalibrationStep, FINGER_WORDS, Phase,
                                  TwoPhaseExercise, increases, scale)

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
    range_steps = ("closed", "open")
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

    @classmethod
    def calibration_valid(cls, steps):
        # "open" must be more open than "closed", or every prompt is reversed
        return increases(steps, "closed", "open", "mean", config.CALIBRATION_MIN_RANGE / 2)

    @staticmethod
    def claim_rule(now, before, history):
        """
        May "opened N% wider" be said? Only for a mean change of at least the
        provisional finger threshold per joint (openness is 1 - flexion sum /
        FULL_CURL_DEG over three joints), or for a rise in every one of the
        last TREND_SESSIONS sessions.
        """
        per_joint = (now - before) * FULL_CURL_DEG / 3
        threshold = benchmarks.load().tol["finger_joints"]["provisional_change_threshold_deg"]
        if per_joint >= threshold:
            return True
        values = [to_float(r.get("mean_raw_high")) for r in history[-config.TREND_SESSIONS:]]
        values = [v for v in values if v == v] + [now]
        return (len(values) == config.TREND_SESSIONS + 1
                and all(b > a for a, b in zip(values, values[1:])))

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
        # each finger during the open hold, for "your ring finger opened more"
        if self._holding and phase.key == "open":
            for finger, v in self._finger_values.items():
                self._rep.setdefault("finger_hold", {}).setdefault(finger, []).append(v)
        # joint angles held in each position, for the benchmark scores
        if self._holding and f.joint_flexion:
            sample = {k: list(v) for k, v in f.joint_flexion.items()}
            key = "open_samples" if phase.key == "open" else "close_samples"
            self._rep.setdefault(key, []).append((sample, dict(f.tip_to_palm), f.aperture))
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
        held = self._rep.get("finger_hold", {})
        extra = {
            "lagging_finger": max(lag, key=lag.get) if lag else "",
            "finger_hold": {k: round(float(np.median(v)), 3) for k, v in held.items() if v},
            "gesture_disagreement": round(self._gesture_disagree / self._gesture_frames, 2)
            if self._gesture_frames else "",
        }
        self._gesture_frames = self._gesture_disagree = 0
        extra.update(self._benchmark_extra())
        return extra

    def _benchmark_extra(self):
        """FMA 24/25-style scores and Bain functional ranges from the median held angles."""
        def median(samples):
            fingers = samples[0][0].keys()
            joints = {k: list(np.median([s[0][k] for s in samples], axis=0)) for k in fingers}
            tips = {k: float(np.median([s[1][k] for s in samples])) for k in samples[0][1]}
            return joints, tips, float(np.median([s[2] for s in samples]))

        bench = benchmarks.load()
        tol = bench.tol["finger_joints"]["tolerance_deg"]
        touch = bench.exercise(self.name)["close_phase"]["fma24_fingertip_to_palm_max"]
        out, reached = {}, []
        opened = self._rep.get("open_samples")
        closed = self._rep.get("close_samples")
        if opened:
            joints, _, aperture = median(opened)
            score = benchmarks.hand_open_score(benchmarks.bain_joints(joints), tol, bench)
            out.update(fma25_style=score["fma25_style"], aperture_open=round(aperture, 3),
                       bain_functional_open=score["bain_functional_open"],
                       fingers_short=",".join(score["fingers_short"]))
            if score["bain_functional_open"]:
                reached.append("open_hand")
            if score["fma25_style"] == 2:
                reached.append("fma25")
        if closed:
            joints, tips, aperture = median(closed)
            thumb_outside = tips.get("thumb", 0.0) > touch
            score = benchmarks.hand_close_score(benchmarks.bain_joints(joints), tips,
                                                thumb_outside, touch, bench)
            out.update(fma24_style=score["fma24_style"], aperture_closed=round(aperture, 3),
                       bain_functional_grasp=score["bain_functional_grasp"])
            if score["bain_functional_grasp"]:
                reached.append("grasp")
        if reached:
            out["milestones_reached"] = reached
        return out

    def _make_display(self, f, value):
        d = super()._make_display(f, value)
        d["finger_values"] = dict(self._finger_values)
        if self._lagging:
            d["finger_colors"] = {self._lagging: "lagging"}
        return d
