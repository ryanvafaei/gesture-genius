"""
Arm exercises: one angle, in degrees, judged against published benchmarks
(benchmarks/BENCHMARK_PLAN.md).

ArmExercise runs the rep state machine of the plan (section 10.1):

  start      waiting for the start posture (e.g. arm at the side), which
             must hold ARM_START_HOLD_FRAMES frames in a row
  ready      in the start posture; the prompt to move is said
  moving     the main angle has left the start by more than the tolerance;
             the peak is tracked
  hold       today's target reached (minus the tolerance): hold timer
  returning  on the way back; the rep ends when the angle is within the
             tolerance of the start again
             -> benchmarks.score_rep() -> RepRecord -> the Coach

Tracking lost is a pause, handled by the Coach with a message that blames
the system ("I can't see your arm"), never her.

Every frame the form rules (elbow straight, arm in its plane, shoulder not
hiking) and compensations (trunk moving more than 10 degrees, Gates 2016)
are checked. They count only when they last ARM_START_HOLD_FRAMES frames in
a row. After a rep she hears at most one cue, and at most one technique
cue per set (plan 10.5).

Modes
  training    sets of reps against today's target from the ladder
              (rehab/benchmarks.py TargetLadder, level in profile.json)
  assessment  no target and no hold: used by the calibration, which is also
              the weekly assessment (rehab/calibration.py SideCalibration)

The subclasses (arm_raise.py, elbow_bend.py, wrist_extension.py,
tabletop_reach.py, finger_to_nose.py) choose the angle, the start posture,
the rules and the words.
"""

import numpy as np

from rehab import benchmarks, body, config
from rehab.benchmarks import finite, further
from rehab.exercises.base import Exercise, Say, RepRecord, count_speed_peaks, number_word

OTHER_SIDE = {"left": "right", "right": "left"}

# on-screen labels of the states
STATE_LABEL = {"start": "START", "ready": "READY", "moving": "MOVE", "hold": "HOLD",
               "returning": "BACK"}

MORE_RANGE_CUE = "A little further next time, if you can."
START_CUE = "Try to start with your elbow as straight as it goes."
START_BUFFER = 15           # frames of the start posture kept for the start reference
ENCOURAGE = "That's all right. Let's go through it once more."


class ArmExercise(Exercise):
    benchmark = True
    # --- what is measured (override) ------------------------------------------
    metric = "shoulder_elevation"      # key in BodyFeatures.arm[side]
    metric_label = "Arm height"
    unit = "deg"                       # "arm": arm lengths (reaching)
    direction = "increase"
    view = "sagittal"                  # camera view for the setup check
    joints = ("hip", "shoulder", "elbow", "wrist")     # landmarks needed, per side
    both_sides = ()                    # landmarks needed on both sides (e.g. "shoulder")
    head = ()                          # e.g. ("nose", "left_eye", "right_eye")
    needs_hand = False                 # the hand of that side must be tracked too
    fma_item = None                    # e.g. "13"; None: no FMA-style score
    fma_threshold = None
    fma_max_score = 2
    at_scale_end = False
    milestone_metric = None            # Gates 2016 milestones for this angle
    secondary_metrics = ()             # other angles whose peak is logged (e.g. shoulder in hand to mouth)
    form_rules = ()                    # ids handled by _broken()
    compensation_rules = ("trunk",)
    uses_ladder = True
    claims_from_assessment_only = False   # elbow tasks (plan T5)
    # --- words (override) ------------------------------------------------------
    start_prompt = "Let your arm hang down by your side."
    move_prompt = "Now lift your arm."
    again_prompt = "And again."        # the same movement, from the second rep of a set on
    return_prompt = "And slowly down."
    calibration_screen = "Lift and lower"
    # --- engine ------------------------------------------------------------------
    calibration_max_age_days = config.ASSESSMENT_EVERY_DAYS
    range_steps = ()                   # no percentage targets: the ladder is in degrees

    def __init__(self, params=None, calibration=None, thresholds=None, side=None, level=0,
                 mode="training", therapist=None, bench=None, **kwargs):
        self.bench = bench or benchmarks.load()
        self.therapist = therapist or benchmarks.load_therapist_profile()
        settings = benchmarks.exercise_settings(self.therapist, self.name)
        merged = {k: settings[k] for k in ("sets", "reps", "hold_s", "rest_s")}
        merged.update(params or {})
        super().__init__(params=merged, calibration=calibration)
        self.side = side or self.therapist.get("affected_side", "left")
        self.mode = mode
        self.level = int(level or 0)
        self.sex = self.therapist.get("sex_for_norms", "female")
        self.tolerance = self.bench.tolerance(self.name)
        self.elbow_tol = self.bench.elbow_tolerance(self.name)
        self.min_frames = int(self.bench.signal("state_debounce_frames"))
        self.quality_grace_s = float(self.bench.signal("invalid_frames_pause_s"))
        self.max_cues = self.bench.max_cues_per_set
        self.max_score = min(self.fma_max_score, benchmarks.contracture_cap(self.therapist, self.name))
        self.bounds = self.plausible_bounds()
        self.ladder = self.build_ladder() if mode == "training" and self.uses_ladder else None
        self.target_deg = self.ladder.target(self.level) if self.ladder else None
        self._epoch = 0
        self._cues = 0
        self._demo_replayed = False
        self.last_result = None
        self.state = "start"
        self._reset_rep(None)

    # --- benchmarks -----------------------------------------------------------------

    @property
    def other_side(self):
        return OTHER_SIDE[self.side]

    @property
    def hold_s(self):
        return 0.0 if self.mode == "assessment" else float(self.params.get("hold_s", 0))

    def required(self, side=None):
        """Pose landmarks that must be visible for this exercise and side."""
        side = side or self.side
        names = [f"{side}_{j}" for j in self.joints]
        names += [f"{s}_{j}" for j in self.both_sides for s in ("left", "right")]
        return list(dict.fromkeys(names + list(self.head)))

    @classmethod
    def setup_view(cls):
        return cls.view

    @classmethod
    def instruction_lines(cls, side="left"):
        """The spoken instructions, with {side} / {other} filled in."""
        other = OTHER_SIDE.get(side, "right")
        return [t.format(side=side, other=other) for t in cls.instructions]

    @classmethod
    def not_testable(cls, therapist):
        """True when the therapist's entries rule the exercise out (e.g. elbow contracture)."""
        return False

    def plausible_bounds(self):
        """(upper, lower) for the main angle; readings outside are tracking errors."""
        upper = self.bench.plausible_max(self.name, self.sex)
        lower = None
        if self.metric == "elbow_flexion":
            lower = self.bench.plausible_min_elbow(self.sex) - self.tolerance
        return upper, lower

    def normative_ceiling(self):
        return None

    def therapist_limit(self):
        """The therapist's passive range limit for this movement (degrees), or None."""
        return benchmarks.passive_limit(self.therapist, self.side, self.metric)

    def milestones(self):
        """FMA threshold and daily-task values, each {deg, task, source}."""
        out = []
        if self.fma_threshold is not None and self.fma_item:
            out.append({"deg": float(self.fma_threshold), "task": f"FMA {self.fma_item}",
                        "source": "FMA2026", "kind": "fma"})
        if self.milestone_metric:
            out += [dict(m, kind="task") for m in self.bench.milestones(self.milestone_metric)]
        return out

    def build_ladder(self):
        cal = self.cal or {}
        affected = cal.get(self.side)
        if not affected or not finite(affected.get("best")):
            return None
        other = cal.get(self.other_side) or {}
        baseline = cal.get("baseline", affected["best"])
        return benchmarks.build_ladder(baseline, other.get("best"), self.tolerance,
                                       self.milestones(), self.normative_ceiling(),
                                       self.therapist_limit(), self.direction)

    def change_threshold(self):
        """Smallest change between sessions worth calling an improvement (MDC)."""
        return self.bench.mdc(self.name, self.therapist.get("mdc_setting", "laboratory"))

    @classmethod
    def calibration_valid(cls, steps):
        """The affected side was measured (a baseline without movement is a baseline too)."""
        side = benchmarks.load_therapist_profile().get("affected_side", "left")
        entry = (steps or {}).get(side) or {}
        return finite(entry.get("best"))

    @classmethod
    def calibration_steps(cls):
        return []           # SideCalibration runs reps instead of held positions

    @classmethod
    def make_calibration(cls, therapist=None):
        from rehab.calibration import SideCalibration
        return SideCalibration(cls, therapist=therapist)

    # --- per frame measures (override) -------------------------------------------

    def primary(self, f, arm):
        return arm.get(self.metric, float("nan"))

    def in_start(self, f, arm, value):
        """The start posture that defines a rep (e.g. arm at the side)."""
        return value <= config.ARM_START_ELEVATION_MAX

    def start_ok(self, refs):
        """The FMA item's start position (e.g. elbow straight); False -> the rep scores 0."""
        return True

    def capture_refs(self, frames):
        """Reference values at the start of a rep from the frames in the start posture."""
        def med(values):
            values = [v for v in values if finite(v)]
            return float(np.median(values)) if values else float("nan")

        arms = [a for _, a, _ in frames]
        fs = [f for f, _, _ in frames]
        return {
            "primary": med([v for _, _, v in frames]),
            "elbow": med([a.get("elbow_flexion") for a in arms]),
            "upper_arm_len": med([a.get("upper_arm_len") for a in arms]),
            "ear_gap": med([a.get("ear_gap") for a in arms]),
            "trunk": med([f.trunk_angle for f in fs]),
        }

    def _broken(self, rule, f, arm, refs):
        """True while a form rule or compensation is broken in this frame."""
        if rule == "elbow_straight":
            return arm["elbow_flexion"] - refs["elbow"] > self.elbow_tol
        if rule == "elbow_steady":
            return abs(arm["elbow_flexion"] - refs["elbow"]) > self.elbow_tol
        if rule == "in_plane":
            ratio = self.bench.compensation("arm_leaves_movement_plane")["threshold_ratio"]
            return arm["upper_arm_len"] / max(refs["upper_arm_len"], 1e-6) < ratio
        if rule == "no_shoulder_hike":
            limit = self.bench.compensation("shoulder_girdle_elevation")["threshold_ratio"]
            return (refs["ear_gap"] - arm["ear_gap"]) / max(f.shoulder_width, 1e-6) > limit
        if rule == "trunk":
            return abs(f.trunk_angle - refs["trunk"]) > self.bench.trunk_threshold
        return False

    def cue_text(self, rule):
        """The one cue for a rule (words from benchmarks.json where it has them)."""
        for r in self.bench.exercise(self.name).get("form_rules", []):
            if r.get("id") == rule and r.get("cue"):
                return r["cue"]
        if rule == "trunk":
            return self.bench.compensation("trunk_flexion_or_lean")["cue"]
        if rule == "no_shoulder_hike":
            return self.bench.compensation("shoulder_girdle_elevation")["cue"]
        if rule == "elbow_straight":
            return self.bench.compensation("elbow_bends_during_straight_arm_items")["cue"]
        if rule == "start":
            return START_CUE
        return MORE_RANGE_CUE

    def on_peak(self, f, arm):
        """Called on the frame of a new peak (e.g. where the hand and trunk were)."""

    def adjust_result(self, result, rep):
        """Exercise-specific scoring after score_rep (e.g. hand to head: the ear)."""
        return result

    def rep_extra(self, result, rep):
        return {}

    # --- quality ------------------------------------------------------------------

    def quality_problem(self, f):
        """None, or "no_body" / "arm_hidden" / "hand_hidden" (said as the system's fault)."""
        if not f.present:
            return "no_body"
        if not f.visible(self.required()):
            return "arm_hidden"
        if self.needs_hand:
            hand = f.hands.get(self.side)
            if hand is None or not hand.present:
                return "hand_hidden"
        return None

    def skip_frame(self, now):
        """A frame the Coach did not pass on (tracking problem): counted for the log."""
        if self.state in ("moving", "hold", "returning"):
            self._rep["invalid"] += 1

    # --- state ----------------------------------------------------------------------

    def _reset_rep(self, now):
        self._rep = {
            "t_start": now, "values": [], "times": [], "start_buffer": [],
            "refs": None, "start_ok": True, "peak": float("nan"), "peak_t": None,
            "form": {}, "comp": {}, "hold_values": [], "hold_s": 0.0,
            "moving_samples": [], "t_move": None, "hints": 0, "invalid": 0, "rejects": 0,
            "visibility": [], "trunk_max": 0.0, "hike_max": 0.0, "in_plane_min": float("nan"),
            "wrist_to_ear_min": float("nan"), "in_start_frames": 0, "back_frames": 0,
            "stalls": 0,
        }
        self._hold_start = None

    def start_set(self, set_no, now):
        super().start_set(set_no, now)
        self._cues = 0
        self._demo_replayed = False
        self.state = "start"
        self._reset_rep(now)

    def interrupt(self, now):
        """Tracking lost or paused: the rep in progress is dropped, she starts again from rest."""
        super().interrupt(now)
        self._epoch += 1
        self.state = "start"
        self._reset_rep(now)

    def _state(self):
        return (self.set_no, len(self.reps), self.state, self._epoch)

    def _while_state(self):
        state = self._state()
        return lambda: self._state() == state

    def first_messages(self):
        return [Say(self.start_prompt, valid=self._while_state())]

    def resume_messages(self):
        return self.first_messages()

    # --- update -----------------------------------------------------------------------

    def update(self, f, now):
        out = []
        rep = self._rep
        if rep["t_start"] is None:
            rep["t_start"] = now
        arm = f.arm.get(self.side, {}) if f.present else {}
        value = self.primary(f, arm)
        if not finite(value):
            self.display = self._make_display(value)
            return out
        if not benchmarks.is_plausible(value, *self.bounds):
            rep["rejects"] += 1             # a tracking error, never a success
            self.display = self._make_display(float("nan"))
            return out
        if f.visibility is not None:
            rep["visibility"].append(float(np.mean([f.visibility[body.POSE[n]]
                                                    for n in self.required()])))
        state = self.state
        if state == "start":
            out += self._update_start(f, arm, value, now)
        elif state == "ready":
            out += self._update_ready(f, arm, value, now)
        else:
            out += self._update_moving(f, arm, value, now)
        self.display = self._make_display(value)
        return out

    def _update_start(self, f, arm, value, now):
        rep = self._rep
        if self.in_start(f, arm, value):
            rep["in_start_frames"] += 1
            rep["start_buffer"] = (rep["start_buffer"] + [(f, arm, value)])[-START_BUFFER:]
            if rep["in_start_frames"] >= self.min_frames:
                self.state = "ready"
                self._progress(now)
                prompt = self.again_prompt if self.reps_this_set else self.move_prompt
                return [Say(prompt, valid=self._while_state())]
            return []
        rep["in_start_frames"] = 0
        rep["start_buffer"] = []
        if self._stalled(now) and self._hints.ready("stall", now):
            self._progress(now)
            rep["hints"] += 1
            return [Say(self.start_prompt, "hint", valid=self._while_state())]
        return []

    def _update_ready(self, f, arm, value, now):
        rep = self._rep
        if self.in_start(f, arm, value) and not rep["values"]:
            rep["start_buffer"] = (rep["start_buffer"] + [(f, arm, value)])[-START_BUFFER:]
        start = self._start_refs()
        if start and self._moved_from(start, value):
            rep["refs"] = start
            rep["start_ok"] = self.start_ok(start)
            rep["values"] = [start["primary"]]
            rep["times"] = [now]
            rep["t_move"] = now
            rep["t_start"] = now
            self.state = "moving"
            self._progress(now)
            return self._update_moving(f, arm, value, now)
        if self._stalled(now) and self._hints.ready("stall", now):
            self._progress(now)
            rep["stalls"] += 1
            rep["hints"] += 1
            if rep["stalls"] >= 3:
                # no movement after several prompts: log an attempt without movement, move on
                rep["refs"] = start or {"primary": value}
                rep["values"] = [value]
                rep["times"] = [now]
                return self._finish_rep(now)
            return [Say(self.move_prompt, "hint", valid=self._while_state())]
        return []

    def _start_refs(self):
        """
        References from the most rest-like frames of the start posture, so
        the first frames of the movement do not raise the start angle.
        """
        buf = self._rep["start_buffer"]
        if not buf:
            return None
        rest = sorted(buf, key=lambda item: self.sgn * item[2])[:self.min_frames]
        return self.capture_refs(rest)

    def _moved_from(self, refs, value):
        return self.sgn * (value - refs["primary"]) > self.tolerance

    @property
    def sgn(self):
        return 1.0 if self.direction == "increase" else -1.0

    def _update_moving(self, f, arm, value, now):
        out = []
        rep = self._rep
        refs = rep["refs"]
        rep["values"].append(value)
        rep["times"].append(now)
        i = len(rep["values"]) - 1
        self._track_rules(f, arm, refs, i)
        if not finite(rep["peak"]) or further(value, rep["peak"], self.direction):
            rep["peak"], rep["peak_t"] = value, now
            self._progress(now)
            self.on_peak(f, arm)
        if self.state == "moving":
            rep["moving_samples"].append((now, value))
            target = self.target_deg
            if target is not None and self.sgn * (value - target) >= -self.tolerance:
                if self.hold_s > 0:
                    self.state = "hold"
                    self._hold_start = now
                    rep["hold_values"] = []
                    out.append(Say("And hold.", "count", valid=self._while_state()))
                else:
                    self.state = "returning"
                    out.append(Say(self.return_prompt, valid=self._while_state()))
            elif self.sgn * (rep["peak"] - value) > self.tolerance:
                # on the way back without reaching the target: the attempt is over
                self.state = "returning"
            elif self._stalled(now) and self._hints.ready("stall", now):
                self._progress(now)
                rep["hints"] += 1
                out.append(Say(MORE_RANGE_CUE if target is None else "A little further, if you can.",
                               "hint", valid=self._while_state()))
        elif self.state == "hold":
            target = self.target_deg
            if self.sgn * (value - target) < -self.tolerance:
                self.state = "moving"           # dropped out of the target: no penalty
            else:
                rep["hold_values"].append(value)
                rep["hold_s"] = now - self._hold_start
                if rep["hold_s"] >= self.hold_s:
                    self.state = "returning"
                    out.append(Say(self.return_prompt, valid=self._while_state()))
        if self.state == "returning":
            back = self.sgn * (value - refs["primary"]) <= self.tolerance
            rep["back_frames"] = rep["back_frames"] + 1 if back else 0
            if rep["back_frames"] >= self.min_frames:
                out += self._finish_rep(now)
            elif self._stalled(now) and self._hints.ready("stall", now):
                self._progress(now)
                out.append(Say(self.return_prompt, "hint", valid=self._while_state()))
        return out

    def _track_rules(self, f, arm, refs, i):
        rep = self._rep
        for rule in self.form_rules:
            if self._broken(rule, f, arm, refs):
                rep["form"].setdefault(rule, []).append(i)
        for rule in self.compensation_rules:
            if self._broken(rule, f, arm, refs):
                rep["comp"].setdefault(rule, []).append(i)
        if finite(f.trunk_angle) and finite(refs.get("trunk")):
            rep["trunk_max"] = max(rep["trunk_max"], abs(f.trunk_angle - refs["trunk"]))
        if finite(arm.get("ear_gap")) and finite(refs.get("ear_gap")) and f.shoulder_width > 0:
            rep["hike_max"] = max(rep["hike_max"], (refs["ear_gap"] - arm["ear_gap"]) / f.shoulder_width)
        if finite(arm.get("upper_arm_len")) and finite(refs.get("upper_arm_len")):
            ratio = arm["upper_arm_len"] / max(refs["upper_arm_len"], 1e-6)
            rep["in_plane_min"] = ratio if not finite(rep["in_plane_min"]) else min(rep["in_plane_min"], ratio)
        for name in self.secondary_metrics:
            v = arm.get(name)
            if finite(v):
                peaks = rep.setdefault("secondary", {})
                peaks[name] = max(peaks.get(name, v), v)
        if finite(arm.get("wrist_to_ear")):
            w = arm["wrist_to_ear"]
            rep["wrist_to_ear_min"] = w if not finite(rep["wrist_to_ear_min"]) else min(rep["wrist_to_ear_min"], w)

    # --- a finished rep ------------------------------------------------------------------

    def score(self):
        rep = self._rep
        result = benchmarks.score_rep(
            rep["values"], self.direction, self.tolerance, self.target_deg,
            self.fma_threshold if self.fma_item else None, rep["form"], rep["comp"],
            rep["start_ok"], self.min_frames, self.max_score, self.at_scale_end)
        return self.adjust_result(result, rep)

    def fraction(self, deg):
        """Degrees as a share of her range (start of the ladder .. ceiling), for summaries."""
        lo = (self.cal.get(self.side) or {}).get("start")
        lo = lo if finite(lo) else self._rep["refs"]["primary"] if self._rep["refs"] else 0.0
        hi = self.ladder.ceiling if self.ladder else None
        if not finite(hi) or abs(hi - lo) < 1e-6 or not finite(deg):
            return float("nan")
        return float((deg - lo) / (hi - lo))

    def _finish_rep(self, now):
        rep = self._rep
        result = self.score()
        values = np.asarray(rep["values"], float)
        refs = rep["refs"] or {"primary": float("nan")}
        peak = result.peak
        hold = float(np.median(rep["hold_values"])) if rep["hold_values"] else peak
        samples = rep["moving_samples"]
        n_frames = len(rep["values"]) + rep["invalid"]
        extra = {
            "benchmark": True, "mode": self.mode, "side": self.side, "level": self.level,
            "set_no": self.set_no,
            "metric": self.metric, "direction": self.direction,
            "personal_target_deg": self.target_deg,
            "fma_item": self.fma_item or "",
            "fma_threshold_deg": self.fma_threshold if self.fma_item else None,
            "tolerance_deg": self.tolerance,
            "start_value": refs.get("primary"), "peak_value": peak,
            "peak_minus_start": result.moved,
            "hold_s": round(rep["hold_s"], 2),
            "fma_style_score": result.fma_style_score if self.fma_item else None,
            "coaching_success": result.coaching_success,
            "with_compensation": result.with_compensation,
            "form_rules_broken": list(result.broken),
            "compensation_flags": list(result.compensations),
            "reasons": list(result.reasons),
            "trunk_max_change_deg": round(rep["trunk_max"], 2),
            "shoulder_girdle_elevation_max": round(rep["hike_max"], 3),
            "in_plane_ratio_min": rep["in_plane_min"],
            "feedback_key": result.feedback_key,
            "mean_visibility": round(float(np.mean(rep["visibility"])), 3) if rep["visibility"] else None,
            "invalid_frame_pct": round(100.0 * rep["invalid"] / n_frames, 1) if n_frames else 0.0,
            "plausibility_rejects": rep["rejects"],
        }
        for name, v in (rep.get("secondary") or {}).items():
            extra[f"{name}_max"] = round(v, 2)
        extra.update(self.rep_extra(result, rep))
        cue = self._cue(result)
        extra["cue_given"] = cue.text if cue else ""
        rec = RepRecord(
            exercise=self.name, set_no=self.set_no, rep_no=self.reps_this_set + 1,
            t_start=rep["t_start"] if rep["t_start"] is not None else now, t_end=now,
            range_high=self.fraction(peak), range_low=float(np.nan),
            raw_high=float(peak), hold_value=self.fraction(hold), hold_raw=float(hold),
            target=self.fraction(self.target_deg) if self.target_deg is not None else float("nan"),
            success=bool(result.coaching_success),
            movement_time=(rep["peak_t"] - rep["t_move"]) if rep["peak_t"] and rep["t_move"] else float("nan"),
            smoothness_peaks=count_speed_peaks([s[0] for s in samples], [s[1] for s in samples])
            if samples else 0,
            hold_stability=(float(np.std(rep["hold_values"])) / max(self.tolerance, 1e-6)
                            if len(rep["hold_values"]) > 2 else float("nan")),
            compensation=list(result.compensations),
            hints=rep["hints"],
            extra=extra,
        )
        self.last_result = result
        self._add_rep(rec)
        out = [Say(f"That's {number_word(rec.rep_no)}.", "praise", tag="rep_done")]
        if cue:
            out.append(cue)
        # back in the start posture: the prompt for the next rep comes when it is held
        self.state = "start"
        self._reset_rep(now)
        return out

    def _cue(self, result):
        """At most one cue after a rep, at most max_cues technique cues per set (plan 10.5)."""
        if result.fma_style_score == 0 and result.feedback_key == "encourage_and_demo":
            if self._demo_replayed:
                return None
            self._demo_replayed = True
            return Say(" ".join([ENCOURAGE] + list(self.instructions[-1:])), "hint")
        if result.cue is None or self._cues >= self.max_cues:
            return None
        self._cues += 1
        return Say(self.cue_text(result.cue), "hint")

    # --- display ------------------------------------------------------------------------

    def display_range(self):
        """(low, high) degrees for the bar."""
        lo = 0.0
        hi = 180.0 if self.metric in ("shoulder_elevation",) else 160.0
        if self.metric == "wrist_extension":
            lo, hi = -30.0, 80.0
        return lo, hi

    def _make_display(self, value):
        best = [r.raw_high for r in self.reps if finite(r.raw_high)]
        best = (max(best) if self.direction == "increase" else min(best)) if best else None
        held = 0.0
        if self.state == "hold" and self._hold_start is not None and self.hold_s > 0:
            held = min(1.0, self._rep["hold_s"] / self.hold_s)
        lo, hi = self.display_range()
        milestones = []
        if self.ladder:
            milestones = [{"deg": m["deg"], "label": m.get("task", "")} for m in self.ladder.milestones]
        prompt = {"start": self.start_prompt, "ready": self.move_prompt,
                  "hold": "Hold", "returning": self.return_prompt}.get(self.state, self.move_prompt)
        return {
            "kind": "angle",
            "unit": self.unit,
            "value": float(value) if finite(value) else None,
            "target": self.target_deg,
            "tolerance": self.tolerance,
            "direction": self.direction,
            "lo": lo, "hi": hi,
            "ceiling": self.ladder.ceiling if self.ladder else None,
            "milestones": milestones,
            "label": self.metric_label,
            "side": self.side,
            "level": self.level,
            "phase_label": STATE_LABEL.get(self.state, ""),
            "prompt": prompt,
            "holding": self.state == "hold",
            "hold_progress": held,
            "best": best,
            "finger_colors": {},
        }
