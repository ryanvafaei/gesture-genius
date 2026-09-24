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
"""

from datetime import date

import numpy as np

from rehab import config
from rehab.exercises.base import Say


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
    def index(self):
        """The current step, counted from 0."""
        return max(0, self._i)

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
