"""
Personal calibration: capture her own range for one exercise.

For each CalibrationStep of an exercise (e.g. "open", then "closed") the
routine says the prompt, gives her CALIBRATION_SETTLE_S to get into position
and then records the step's values for CALIBRATION_HOLD_S. The settle time
only starts once the coach has finished speaking: speech is slow, and
measuring while she is still listening to the previous prompt stores the
wrong position (e.g. "open" and "closed" swapped, which reverses every
prompt of the exercise). The median over the hold is stored, which ignores
single bad frames. Frames with a quality problem (hand lost, wrong hand, too
far) are skipped and the hold timer pauses, so she never has to start over.
When the result is clearly wrong (exercise_cls.calibration_valid) the steps
are repeated, up to CALIBRATION_MAX_ATTEMPTS times.

Result: {step name: {key: value}} -> profile.json via storage.

Arm exercises measure reps instead of held positions: SideCalibration (the
unaffected arm first, then the affected arm) is also their weekly
assessment. See its docstring.
"""

from datetime import date
from types import SimpleNamespace

import numpy as np

from rehab import benchmarks, body, config
from rehab.exercises.base import Say, number_word

# Positioning before each side of an arm calibration and before arm
# exercises (plan 3.1). The fault is always the camera's, never hers.
SETUP_TEXT = {
    "no_body": "I can't see you yet. Please sit where the camera can see you.",
    "arm_hidden": "I can't see your whole {side} arm. Please move into the box.",
    "move_back": "Please move back a little, so I can see all of your arm.",
    "turn_side": "Please turn your chair so your {side} arm is nearest the screen.",
    "face_camera": "Please turn your chair to face the screen.",
    "turn_45": "Please turn your chair halfway, with your {side} arm towards the screen.",
}
SETUP_SCREEN = {
    "no_body": "Sit where the camera can see you",
    "arm_hidden": "Move into the box",
    "move_back": "Move back a little",
    "turn_side": "Turn side-on",
    "face_camera": "Face the screen",
    "turn_45": "Turn halfway",
}


class CalibrationRoutine:

    def __init__(self, exercise_cls,
                 settle_s=config.CALIBRATION_SETTLE_S,
                 hold_s=config.CALIBRATION_HOLD_S,
                 min_samples=config.CALIBRATION_MIN_SAMPLES,
                 max_attempts=config.CALIBRATION_MAX_ATTEMPTS):
        self.exercise_cls = exercise_cls
        self.steps = exercise_cls.calibration_steps()
        self.settle_s = settle_s
        self.hold_s = hold_s
        self.min_samples = min_samples
        self.max_attempts = max_attempts
        self.attempt = 1
        self.result = {}
        self._i = -1
        self._samples = []
        self._stage = None
        self._stage_t = None
        self._held = 0.0
        self._last_t = None

    @property
    def done(self):
        return self._i >= len(self.steps)

    @property
    def step(self):
        return self.steps[self._i] if 0 <= self._i < len(self.steps) else None

    @property
    def progress(self):
        """0..1 for the current step's hold (for the on-screen ring)."""
        if self._stage != "hold":
            return 0.0
        return min(1.0, self._held / self.hold_s)

    def start(self, now):
        self._i = -1
        self.attempt = 1
        self.result = {}
        return [Say("Let's measure your hand, so the exercise fits you.")] + self._next(now)

    def _next(self, now):
        self._i += 1
        if self.done:
            if (not self.exercise_cls.calibration_valid(self.result)
                    and self.attempt < self.max_attempts):
                self.attempt += 1
                self._i = -1
                self.result = {}
                return [Say("That didn't quite work. Let's try once more.")] + self._next(now)
            return [Say("Thank you. That's all I need.")]
        self._begin_step(now)
        return [Say(self.step.prompt)]

    def _begin_step(self, now):
        self._samples = []
        self._stage = "settle"
        self._stage_t = now
        self._held = 0.0
        self._last_t = now

    def interrupt(self, now):
        self._last_t = now

    def quality_problem(self, f):
        """Tracking problem for the current step (see Think.QUALITY_TEXT), or None."""
        step = self.step
        return f.quality_problem(step.need_palm_facing if step else False)

    def resume(self, now):
        """After a pause: she may have moved, so measure this step again."""
        if self.done or self._i < 0:
            return []
        self._begin_step(now)
        return [Say(self.step.prompt)]

    def update(self, features, now, quality_ok=True, speaking=False):
        """Feed one frame. Returns a list of Say."""
        if self.done or self._i < 0:
            return []
        dt = max(0.0, now - (self._last_t if self._last_t is not None else now))
        self._last_t = now
        if self._stage == "settle":
            if speaking:
                self._stage_t = now      # she is still listening to the prompt
            elif now - self._stage_t >= self.settle_s:
                self._stage = "hold"
            return []
        # hold: only time with a good view counts
        if not quality_ok:
            return []
        self._held += dt
        self._samples.append(self.step.extract(features))
        if self._held >= self.hold_s and len(self._samples) >= self.min_samples:
            self.result[self.step.name] = _median(self._samples)
            # neutral: "relax" would be wrong before e.g. "Now squeeze it"
            return [Say("Good.", "count")] + self._next(now)
        return []

    def as_profile_entry(self):
        return {"date": date.today().isoformat(), "steps": self.result}


def _median(samples):
    keys = samples[0].keys()
    return {k: float(np.median([s[k] for s in samples])) for k in keys}


def is_stale(entry, today=None, max_age_days=config.RECALIBRATE_AFTER_DAYS):
    """True when there is no calibration or it is older than max_age_days."""
    if not entry or not entry.get("steps"):
        return True
    try:
        made = date.fromisoformat(entry["date"])
    except (KeyError, ValueError):
        return True
    today = today or date.today()
    return (today - made).days > max_age_days


class SetupCheck:
    """
    The camera must see what an arm exercise needs (plan 3.1): the landmarks
    of that arm, not at the edge of the picture, from the right direction
    (side-on, facing, or halfway). Passed after SETUP_CHECK_HOLD_S in a row.
    Problems are said calmly, after QUALITY_GRACE_S, at most every
    QUALITY_MESSAGE_REPEAT_S.
    """

    def __init__(self, required, view, side):
        self.required, self.view, self.side = required, view, side
        self.problem = None
        self.passed = False
        self.reported = False           # a problem was said (then "I can see you well" is worth saying)
        self._since = None
        self._ok_since = None
        self._said = {}

    @property
    def progress(self):
        return 0.0 if self._ok_since is None else 1.0

    def update(self, f, now, speaking=False):
        """Returns a list of Say; sets passed once the view has been good long enough."""
        problem = body.setup_problem(f, self.required, self.view, self.side)
        if problem != self.problem:
            self.problem, self._since = problem, now
        if problem is None:
            if self._ok_since is None:
                self._ok_since = now
            if now - self._ok_since >= config.SETUP_CHECK_HOLD_S and not speaking:
                self.passed = True
            return []
        self._ok_since = None
        if now - self._since < config.QUALITY_GRACE_S:
            return []
        if now - self._said.get(problem, -1e9) < config.QUALITY_MESSAGE_REPEAT_S:
            return []
        self._said[problem] = now
        self.reported = True
        return [Say(SETUP_TEXT[problem].format(side=self.side), "quality",
                    valid=lambda p=problem: self.problem == p)]

    @property
    def screen_text(self):
        return SETUP_SCREEN.get(self.problem, "Hold still") if self.problem else "Hold still"


class SideCalibration:
    """
    Calibration of an arm exercise, which is also its weekly assessment
    (plan 9.1; FMA 2026, Lazem 2026, Levin 2004). The exercise's spoken
    instructions are the demonstration. Then, for the unaffected arm first
    and the affected arm second:

      position  the camera must see that arm (she may need to turn her chair
                for a side view): SetupCheck
      practice  BENCH_PRACTICE_TRIALS reps, not recorded (first arm only)
      record    BENCH_CALIBRATION_REPS reps with BENCH_REST_BETWEEN_REPS_S
                rest in between

    Each rep is run by the exercise itself in assessment mode (no target,
    no hold); an attempt without movement is recorded too. Stored per side:
    the best of the recorded reps (FMA: the best performance counts), their
    mean and the start angle; plus the symmetry index and the best FMA-style
    score per side. The session keeps the affected side's first best as her
    baseline.

    Same interface as CalibrationRoutine, so the session runs either.
    """

    def __init__(self, exercise_cls, therapist=None, practice=None, reps=None, rest_s=None):
        self.exercise_cls = exercise_cls
        self.therapist = therapist or benchmarks.load_therapist_profile()
        self.affected = self.therapist.get("affected_side", "left")
        self.unaffected = "right" if self.affected == "left" else "left"
        practice = config.BENCH_PRACTICE_TRIALS if practice is None else practice
        reps = config.BENCH_CALIBRATION_REPS if reps is None else reps
        self.rest_s = config.BENCH_REST_BETWEEN_REPS_S if rest_s is None else rest_s
        self.plan = [(self.unaffected, practice, reps), (self.affected, 0, reps)]
        self.steps = []
        self.attempt = 1
        self.result = {}
        self._i = -1
        self._stage = None
        self._stage_t = 0.0
        self._ex = None
        self._setup = None
        self._recorded = {}
        self._t = 0.0

    # --- state --------------------------------------------------------------------

    @property
    def done(self):
        return self._i >= len(self.plan)

    @property
    def side(self):
        return self.plan[self._i][0] if 0 <= self._i < len(self.plan) else self.affected

    @property
    def display(self):
        return self._ex.display if self._ex is not None and self._stage == "rep" else {}

    def _counts(self):
        side, practice, reps = self.plan[self._i]
        done = len(self._ex.reps) if self._ex else 0
        return done, practice, reps

    @property
    def step(self):
        if self.done or self._i < 0:
            return None
        side = self.side.capitalize()
        if self._stage == "position":
            text = f"{side} arm: " + self._setup.screen_text
        elif self._stage == "rest":
            text = "Rest"
        else:
            done, practice, reps = self._counts()
            if done < practice:
                text = f"{side} arm: practice"
            else:
                text = f"{side} arm: {self.exercise_cls.calibration_screen} ({done - practice + 1} of {reps})"
        return SimpleNamespace(name=self.side, need_palm_facing=False, screen_text=text)

    @property
    def progress(self):
        if self._stage == "rest" and self.rest_s > 0:
            return min(1.0, (self._t - self._stage_t) / self.rest_s)
        if self._stage == "position":
            return self._setup.progress
        if self._stage == "rep" and self._ex is not None:
            done, practice, reps = self._counts()
            return min(1.0, done / max(1, practice + reps))
        return 0.0

    # --- flow -----------------------------------------------------------------------

    def start(self, now):
        self._i = -1
        self._t = now
        self.result = {}
        self._recorded = {self.unaffected: [], self.affected: []}
        return [Say("Let's measure how your arms move, so the exercise fits you.")] + self._next_side(now)

    def _next_side(self, now):
        self._i += 1
        if self.done:
            self._finish()
            return [Say("Thank you. That's all I need.")]
        side = self.side
        self._ex = self.exercise_cls(side=side, mode="assessment", therapist=self.therapist)
        self._ex.start_set(1, now)
        self._setup = SetupCheck(self._ex.required(), self._ex.setup_view(), side)
        self._enter("position", now)
        if self._i == 0:
            return [Say(f"First your {side} arm.")]
        return [Say(f"Now the same with your {side} arm.")]

    def _enter(self, stage, now):
        self._stage = stage
        self._stage_t = now

    def quality_problem(self, f):
        if self._stage != "rep" or self._ex is None:
            return None             # positioning speaks for itself, rest needs nothing
        return self._ex.quality_problem(f)

    def interrupt(self, now):
        if self._ex is not None:
            self._ex.interrupt(now)

    def resume(self, now):
        """After a pause she may have moved: check the camera again, then carry on."""
        if self.done or self._i < 0:
            return []
        self._ex.interrupt(now)
        self._setup = SetupCheck(self._ex.required(), self._ex.setup_view(), self.side)
        self._enter("position", now)
        return []

    def update(self, features, now, quality_ok=True, speaking=False):
        self._t = now
        if self.done or self._i < 0:
            return []
        if self._stage == "position":
            out = self._setup.update(features, now, speaking)
            if self._setup.passed:
                self._enter("rep", now)
                self._ex.interrupt(now)
                if self._setup.reported:
                    out.append(Say("Good, I can see you well."))
                out += self._ex.first_messages()
            return out
        if self._stage == "rest":
            if now - self._stage_t >= self.rest_s and not speaking:
                self._enter("rep", now)
                self._ex.interrupt(now)
                return self._ex.resume_messages()
            return []
        if not quality_ok:
            self._ex.skip_frame(now)
            return []
        self._ex.speaking = speaking
        says = self._ex.update(features, now)
        new = list(self._ex.new_reps)
        self._ex.new_reps.clear()
        if not new:
            return says
        # replace the exercise's "That's three." with this routine's own count
        says = [m for m in says if m.tag != "rep_done"]
        done, practice, reps = self._counts()
        rec = new[-1]
        if done <= practice:
            out = [Say("Good. That one was for practice.", "praise")]
        else:
            self._recorded[self.side].append(rec)
            out = [Say(f"That's {number_word(done - practice)}.", "praise")]
        out += says
        if done >= practice + reps:
            return out + self._next_side(now)
        if self.rest_s > 0:
            self._enter("rest", now)
            n = int(round(self.rest_s))
            out.append(Say(f"Rest for {number_word(n)} second{'s' if n != 1 else ''}.",
                           valid=lambda: self._stage == "rest"))
            return out
        self._ex.interrupt(now)
        return out + self._ex.resume_messages()

    def _finish(self):
        cls = self.exercise_cls
        result = {}
        scores = {}
        for side, recs in self._recorded.items():
            peaks = [r.raw_high for r in recs]
            starts = [r.extra.get("start_value") for r in recs]
            summary = benchmarks.side_summary(peaks, cls.direction, starts)
            if summary:
                result[side] = summary
            fma = [r.extra.get("fma_style_score") for r in recs
                   if isinstance(r.extra.get("fma_style_score"), int)]
            if fma:
                scores[side] = max(fma)
        result["symmetry_index"] = benchmarks.symmetry_index(
            result.get(self.affected), result.get(self.unaffected), cls.direction)
        result["fma_style_best"] = scores
        self.result = result

    def as_profile_entry(self):
        return {"date": date.today().isoformat(), "steps": self.result}
