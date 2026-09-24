"""
Finger to nose, five times with each arm, timed (FMA items 31-33, adapted).

The unaffected arm goes first (FMA), then the affected arm. One round =
five touches: from the lap to the nose and back to the lap. The time runs
from the moment the hand leaves the lap until it is back after the fifth
touch (plan section 11). She sits facing the camera.

Scores, logged for the weekly profile:
  FMA 33  speed: the affected arm less than 2 s slower -> 2; 2-5.9 s -> 1;
          6 s or more, or fewer than five touches -> 0
  FMA 32  dysmetria: the worst touch within 0.35 eye distances of the nose
          -> 2, within 0.8 -> 1, else 0 (thresholds proposed)
  FMA 31  tremor, a rough stand-in: at most TREMOR_MAX_PEAKS speed peaks per
          approach -> 2, else 1 (proposed; a webcam cannot score tremor well)
Adapted: Eleanor keeps her eyes open (mild cognitive impairment, fall risk),
so these are not comparable with a clinical FMA 31-33 and are logged as
"adapted".

Counted as reps: one round per arm (reps = 2 in the therapist profile).
"""

import numpy as np

from rehab import benchmarks, config
from rehab.benchmarks import finite
from rehab.exercises.arm import ArmExercise
from rehab.exercises.base import RepRecord, Say, count_speed_peaks, number_word


class FingerToNose(ArmExercise):
    name = "finger_to_nose_timed"
    title = "Finger to your nose"
    instructions = (
        "Let's touch your nose with your finger, five times.",
        "Sit facing the screen, with your hands in your lap.",
        "First your {other} hand, then your {side} hand. Keep your eyes open.",
    )
    metric = "nose_error"
    metric_label = "Finger to nose"
    direction = "decrease"
    view = "frontal"
    joints = ("shoulder", "elbow", "wrist", "hip")
    both_sides = ("shoulder", "hip")
    head = ("nose", "left_eye", "right_eye")
    fma_item = "33"
    uses_ladder = False
    form_rules = ()
    compensation_rules = ()
    return_prompt = "And back to your lap."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.affected = self.side
        self.round_times = {}
        self._round = None

    @classmethod
    def make_calibration(cls, therapist=None):
        return None             # the exercise measures both arms itself

    @property
    def reps_per_set(self):
        return 2                # the unaffected arm, then the affected arm

    @property
    def round_side(self):
        """The arm of the current round: the unaffected one first (FMA)."""
        return self.other_side_of(self.affected) if self.reps_this_set == 0 else self.affected

    @staticmethod
    def other_side_of(side):
        return "right" if side == "left" else "left"

    def required(self, side=None):
        return super().required(side or self.round_side)

    def quality_problem(self, f):
        if not f.present:
            return "no_body"
        if not f.visible(self.required()):
            return "arm_hidden"
        return None

    @property
    def start_prompt(self):
        return f"Rest your {self.round_side} hand in your lap."

    @property
    def move_prompt(self):
        return (f"Now touch your nose with your {self.round_side} index finger, five times, "
                "going back to your lap each time.")

    # --- rounds ------------------------------------------------------------------

    def start_set(self, set_no, now):
        super().start_set(set_no, now)
        self._round = None

    def interrupt(self, now):
        super().interrupt(now)
        self._round = None          # a round with lost tracking starts again

    def first_messages(self):
        return [Say(self.start_prompt, valid=self._while_state())]

    def _in_lap(self, f):
        side = self.round_side
        wrist_y = f.point(f"{side}_wrist")[1]
        hip_y = f.hip_mid[1]
        return wrist_y >= hip_y - config.REST_ZONE_TORSO * f.torso_len

    def update(self, f, now):
        out = []
        if not f.present or f.points is None:
            return out
        side = self.round_side
        err = f.arm.get(side, {}).get("nose_error", float("nan"))
        in_lap = self._in_lap(f)
        r = self._round
        if r is None:
            r = self._round = {"state": "lap", "lap_frames": 0, "t0": None, "touches": [],
                               "min_err": float("nan"), "away": True, "errs": [], "times": [],
                               "peaks": [], "hints": 0, "since": now}
        state = r["state"]
        if state == "lap":
            r["lap_frames"] = r["lap_frames"] + 1 if in_lap else 0
            if r["lap_frames"] >= self.min_frames:
                r["state"] = "ready"
                self.state = "ready"
                self._progress(now)
                out.append(Say(self.move_prompt, valid=self._while_state()))
            elif self._stalled(now) and self._hints.ready("stall", now):
                self._progress(now)
                out.append(Say(self.start_prompt, "hint", valid=self._while_state()))
        elif state == "ready":
            if not in_lap:
                r["state"] = "running"
                self.state = "moving"
                r["t0"] = now
                self._start_excursion(r)
                self._progress(now)
        if r["state"] == "running":
            out += self._update_running(r, err, in_lap, now)
        self.display = self._display(err)
        return out

    def _start_excursion(self, r):
        r["min_err"] = float("nan")
        r["errs"], r["times"] = [], []

    def _update_running(self, r, err, in_lap, now):
        out = []
        cfg = config
        if finite(err):
            r["errs"].append(err)
            r["times"].append(now)
            if not finite(r["min_err"]) or err < r["min_err"]:
                r["min_err"] = err
                self._progress(now)
        if in_lap and r["errs"]:
            # back in the lap: one excursion is over; it counts when it reached the nose area
            if finite(r["min_err"]) and r["min_err"] <= cfg.NOSE_AWAY:
                r["touches"].append(r["min_err"])
                i = int(np.argmin(r["errs"]))
                r["peaks"].append(count_speed_peaks(r["times"][:i + 1], r["errs"][:i + 1]))
                n = len(r["touches"])
                if n >= cfg.NOSE_TOUCHES:
                    return out + self._finish_round(r, now)
                out.append(Say(number_word(n), "count"))
            self._start_excursion(r)
        if now - r["t0"] >= cfg.NOSE_ROUND_TIMEOUT_S:
            return out + self._finish_round(r, now)
        if self._stalled(now) and self._hints.ready("stall", now):
            self._progress(now)
            r["hints"] += 1
            out.append(Say("Touch your nose, then back to your lap.", "hint"))
        return out

    def _finish_round(self, r, now):
        side = self.round_side
        touches = r["touches"]
        duration = now - r["t0"]
        self.round_times[side] = (duration, len(touches))
        fma32 = benchmarks.fma32_dysmetria_score(touches, self.bench)
        mean_peaks = float(np.mean(r["peaks"])) if r["peaks"] else float("nan")
        fma31 = 2 if finite(mean_peaks) and mean_peaks <= config.TREMOR_MAX_PEAKS else 1
        extra = {
            "benchmark": True, "mode": self.mode, "side": side, "level": 0,
            "metric": "time_s", "direction": "decrease", "fma_item": "31-33",
            "adapted": "eyes open",
            "duration_s": round(duration, 2), "touches": len(touches),
            "touch_errors": [round(e, 3) for e in touches],
            "fma31_style": fma31, "fma32_style": fma32,
            "start_value": None, "peak_value": round(duration, 2),
            "coaching_success": len(touches) >= config.NOSE_TOUCHES,
            "with_compensation": False, "form_rules_broken": [], "compensation_flags": [],
            "feedback_key": "praise_specific" if len(touches) >= config.NOSE_TOUCHES else "one_cue",
            "cue_given": "",
        }
        fma33 = None
        if side == self.affected:
            other = self.round_times.get(self.other_side_of(side))
            fma33 = benchmarks.fma33_time_score(duration, other[0] if other else None, len(touches))
            extra["unaffected_time_s"] = round(other[0], 2) if other else None
        extra["fma_style_score"] = fma33
        rec = RepRecord(
            exercise=self.name, set_no=self.set_no, rep_no=self.reps_this_set + 1,
            t_start=r["t0"], t_end=now,
            range_high=len(touches) / config.NOSE_TOUCHES, raw_high=float(duration),
            hold_raw=float(duration), movement_time=duration / max(1, len(touches)),
            success=len(touches) >= config.NOSE_TOUCHES, hints=r["hints"], extra=extra,
        )
        self._add_rep(rec)
        self._round = None
        self.state = "start"
        out = [Say(f"That's all five with your {side} hand." if rec.success
                   else f"Thank you. That's enough with your {side} hand.", "praise", tag="rep_done")]
        if not self.set_done:
            out.append(Say(f"Now your {self.round_side} hand.", valid=self._while_state()))
        return out

    def _display(self, err):
        r = self._round or {}
        return {
            "kind": "count",
            "label": f"{self.round_side.capitalize()} hand",
            "count": len(r.get("touches", [])),
            "of": config.NOSE_TOUCHES,
            "prompt": self.move_prompt if r.get("state") in ("ready", "running") else self.start_prompt,
            "phase_label": "TOUCH" if r.get("state") == "running" else "LAP",
            "value": err if finite(err) else None,
            "finger_colors": {},
        }
