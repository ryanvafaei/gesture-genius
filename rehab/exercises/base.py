"""
Generic exercise engine.

An exercise turns a stream of HandFeatures into
  * messages for the speaker (`Say`),
  * finished repetitions with quality measures (`RepRecord`),
  * a `display` dict that Act draws.

Two families cover all six exercises:

TwoPhaseExercise  one continuous metric, two positions to reach and hold
                  (grip and release, finger abduction, thumb flexion,
                  grip squeeze).
SequenceExercise  discrete finger events that must follow a sequence
                  (thumb opposition, finger tapping), with guided and
                  memory levels.

Exercises only import `config` and `features`; they never speak or draw
themselves. The Coach (Think.py) passes their messages on.
"""

from dataclasses import dataclass, field

import numpy as np

from rehab import config
from rehab.features import FINGERS, angle_between

FINGER_WORDS = {
    "thumb": "thumb",
    "index": "index finger",
    "middle": "middle finger",
    "ring": "ring finger",
    "pinky": "little finger",
}

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen"]


def number_word(n):
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)


# ---------------------------------------------------------------------------
# Small data types
# ---------------------------------------------------------------------------

@dataclass
class Say:
    """Something the coach should say. Counts may be dropped if speech is busy."""
    text: str
    kind: str = "instruction"     # instruction | count | praise | hint | quality

    @property
    def ephemeral(self):
        return self.kind == "count"


@dataclass
class CalibrationStep:
    """
    One position held during calibration.

    extract  HandFeatures -> {key: value}; the median of each key over the
             hold is stored (robust against single bad frames).
    """
    name: str
    prompt: str
    extract: callable
    need_palm_facing: bool = False
    screen_text: str = None


@dataclass
class RepRecord:
    exercise: str
    set_no: int
    rep_no: int
    t_start: float
    t_end: float
    range_high: float = float("nan")      # best reached, fraction of calibrated range
    range_low: float = float("nan")       # lowest reached, fraction of calibrated range
    raw_high: float = float("nan")        # same peak, uncalibrated (comparable over recalibrations)
    movement_time: float = float("nan")   # s, from leaving one position to reaching the other
    smoothness_peaks: int = 0             # speed peaks during that movement (1 = smooth)
    hold_stability: float = float("nan")  # std of the metric during the holds
    compensation: list = field(default_factory=list)
    hints: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def duration(self):
        return self.t_end - self.t_start


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scale(value, lo, hi, min_range=config.CALIBRATION_MIN_RANGE):
    """Map value to 0..1 between her calibrated lo and hi (not clipped)."""
    span = hi - lo
    if abs(span) < min_range:
        span = min_range if span >= 0 else -min_range
    return (value - lo) / span


def count_speed_peaks(times, values, rel_threshold=0.2):
    """
    Number of peaks in |speed| of a movement.

    One peak = one smooth movement. Several peaks = stop-and-go (jerky).
    Peaks smaller than rel_threshold * the largest peak are ignored.
    """
    if len(values) < 5:
        return 1 if len(values) else 0
    t = np.asarray(times, float)
    v = np.asarray(values, float)
    dt = np.diff(t)
    dt[dt <= 0] = 1e-3
    speed = np.abs(np.diff(v) / dt)
    # light smoothing so single-frame noise is not a peak
    if len(speed) >= 3:
        speed = np.convolve(speed, np.ones(3) / 3, mode="same")
    top = speed.max()
    if top <= 1e-9:
        return 0
    thr = rel_threshold * top
    peaks = 0
    above = False
    low_since_peak = True
    for s in speed:
        if s >= thr and not above and low_since_peak:
            peaks += 1
            above = True
            low_since_peak = False
        elif s < thr * 0.5:
            above = False
            low_since_peak = True
        elif s < thr:
            above = False
    return max(peaks, 1)


# Stability is measured after this much of a hold, so it reflects wobble
# rather than the last part of the movement into position.
HOLD_SETTLE_S = 0.5


class Hysteresis:
    """
    Three zones: 'high', 'mid', 'low'.

    Entering 'high' needs value >= high; leaving it needs value < high - gap.
    Entering 'low' needs value <= low; leaving it needs value > low + gap.
    So a value hovering around a threshold does not flicker in and out.
    """

    def __init__(self, high, low, gap):
        assert low < high
        self.high, self.low, self.gap = high, low, gap
        self.zone = "mid"

    def reset(self):
        self.zone = "mid"

    def update(self, v):
        if self.zone == "high":
            if v < self.high - self.gap:
                self.zone = "low" if v <= self.low else "mid"
        elif self.zone == "low":
            if v > self.low + self.gap:
                self.zone = "high" if v >= self.high else "mid"
        else:
            if v >= self.high:
                self.zone = "high"
            elif v <= self.low:
                self.zone = "low"
        return self.zone


class RateLimiter:
    def __init__(self, interval):
        self.interval = interval
        self.last = {}

    def ready(self, key, now):
        if now - self.last.get(key, -1e9) >= self.interval:
            self.last[key] = now
            return True
        return False


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Exercise:
    name = "exercise"
    title = "Exercise"
    # One short sentence each, said one at a time before the first set.
    instructions = ()
    need_palm_facing = False
    # How the progress message describes an improvement of raw_high.
    progress_phrase = "You moved {pct}% further than {when}."
    # "lower is better" for the summary number (e.g. time per touch)
    progress_lower_is_better = False

    def __init__(self, params=None, calibration=None, thresholds=None):
        self.params = dict(config.EXERCISES.get(self.name, {}))
        if params:
            self.params.update(params)
        if thresholds:
            self.params.update(thresholds)
        self.cal = calibration or {}
        self.set_no = 1
        self.reps = []            # all RepRecords of this exercise today
        self.new_reps = []        # filled by update(), emptied by the coach
        self.display = {}
        self.fatigue = False
        self._hints = RateLimiter(config.HINT_REPEAT_S)
        self._last_progress_t = None

    # --- to override ----------------------------------------------------

    @classmethod
    def calibration_steps(cls):
        return []

    @property
    def reps_per_set(self):
        return int(self.params.get("reps", 10))

    @property
    def sets(self):
        return int(self.params.get("sets", 3))

    def first_messages(self):
        """Said when a set starts (the instructions are said once, by the session)."""
        return []

    def frame_problem(self, f):
        """Exercise-specific reason to pause (e.g. fingers not straight). Returns a Say or None."""
        return None

    def update(self, f, now):
        raise NotImplementedError

    def interrupt(self, now):
        """Hand lost or paused: cancel any hold in progress."""
        self._last_progress_t = now

    # --- shared -----------------------------------------------------------

    def start_set(self, set_no, now):
        self.set_no = set_no
        self.fatigue = False
        self._last_progress_t = now

    @property
    def reps_this_set(self):
        return sum(1 for r in self.reps if r.set_no == self.set_no)

    @property
    def set_done(self):
        return self.reps_this_set >= self.reps_per_set

    def _progress(self, now):
        self._last_progress_t = now

    def _stalled(self, now):
        return (self._last_progress_t is not None
                and now - self._last_progress_t >= config.HINT_DELAY_S)

    def _add_rep(self, rec):
        self.reps.append(rec)
        self.new_reps.append(rec)
        self._check_fatigue()

    def _check_fatigue(self):
        n = config.FATIGUE_WINDOW
        this_set = [r.range_high for r in self.reps
                    if r.set_no == self.set_no and np.isfinite(r.range_high)]
        if len(this_set) >= 2 * n:
            first = np.mean(this_set[:n])
            last = np.mean(this_set[-n:])
            if first > 0 and last < config.FATIGUE_DROP * first:
                self.fatigue = True


# ---------------------------------------------------------------------------
# Two positions, one metric
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    key: str              # "open"
    zone: str             # "high" or "low"
    prompt: str           # said when this position is next: "Open your hand."
    label: str            # on screen: "OPEN"
    hold_s: float = None  # None -> params["hold_s"]


class TwoPhaseExercise(Exercise):
    """
    Reach phase 0, hold; reach phase 1, hold -> one repetition.

    Subclasses define `phases`, `metric()` (calibrated 0..1) and `raw_metric()`,
    and may add checks in `observe()` and details in `rep_extra()`.
    """

    phases = ()
    best_phrase = "That's your best yet today."
    compensation_messages = {
        "palm_rotation": "Try to keep your wrist still.",
        "wrist_moved": "Try to keep your wrist still.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = self.params
        self.hyst = Hysteresis(p["high"], p["low"], p["gap"])
        self._phase = 0
        self._holding = False
        self._reset_rep(None)

    # --- to override ----------------------------------------------------

    def metric(self, f):
        raise NotImplementedError

    def raw_metric(self, f):
        raise NotImplementedError

    def observe(self, f, now, value):
        """Called every valid frame; may return a list of Say."""
        return []

    def rep_extra(self):
        return {}

    def stall_hint(self, f):
        return self.phases[self._phase].prompt

    # --- state machine ----------------------------------------------------

    def hold_time(self, phase):
        return phase.hold_s if phase.hold_s is not None else self.params["hold_s"]

    def _reset_rep(self, now):
        self._rep = {
            "t_start": now,
            "values": [], "times": [], "raw": [],
            "hold_values": {},        # phase index -> values of the last hold
            "phase_done_t": None, "movement_samples": None,
            "movement_time": float("nan"), "smoothness": 0,
            "normal0": None, "wrist0": None,
            "max_rotation": 0.0, "max_shift": 0.0,
            "hints": 0, "comp": [], "comp_said": set(),
            "zone_time": {},
        }

    def start_set(self, set_no, now):
        super().start_set(set_no, now)
        self._phase = 0
        self._holding = False
        self.hyst.reset()
        self._reset_rep(now)

    def first_messages(self):
        return [Say(self.phases[0].prompt)]

    def interrupt(self, now):
        super().interrupt(now)
        self._holding = False
        self.hyst.reset()

    def update(self, f, now):
        out = []
        value = self.metric(f)
        zone = self.hyst.update(value)
        rep = self._rep
        if rep["t_start"] is None:
            rep["t_start"] = now
        rep["values"].append(value)
        rep["times"].append(now)
        rep["raw"].append(self.raw_metric(f))
        if rep["movement_samples"] is not None:
            rep["movement_samples"].append((now, value))

        out += self._track_compensation(f, now)
        out += self.observe(f, now, value) or []

        phase = self.phases[self._phase]
        in_zone = zone == phase.zone

        # time spent continuously in each zone (e.g. real squeeze duration)
        zt = rep["zone_time"]
        if zone != zt.get("zone"):
            if zt.get("zone") in ("high", "low"):
                zt[zt["zone"] + "_s"] = max(zt.get(zt["zone"] + "_s", 0.0), now - zt["since"])
            zt["zone"], zt["since"] = zone, now

        if not self._holding:
            if in_zone:
                self._holding = True
                self._hold_start = now
                self._counted = 0
                rep["hold_values"][self._phase] = []
                self._progress(now)
                if rep["movement_samples"] is not None:
                    samples = rep["movement_samples"]
                    rep["movement_time"] = now - rep["phase_done_t"]
                    rep["smoothness"] = count_speed_peaks([s[0] for s in samples],
                                                          [s[1] for s in samples])
                    rep["movement_samples"] = None
                if self.params.get("count_aloud"):
                    out.append(Say("And hold.", "count"))
            elif self._stalled(now) and self._hints.ready("stall", now):
                out.append(Say(self.stall_hint(f), "hint"))
                rep["hints"] += 1
                self._progress(now)
        else:
            if not in_zone:
                self._holding = False
                self._progress(now)
                if self._hints.ready("hold", now):
                    out.append(Say("Hold it a little longer.", "hint"))
                    rep["hints"] += 1
            else:
                held = now - self._hold_start
                if held >= HOLD_SETTLE_S:       # ignore the tail of the movement into the hold
                    rep["hold_values"][self._phase].append(value)
                if self.params.get("count_aloud"):
                    k = int(held)
                    if k > self._counted and held < self.hold_time(phase):
                        self._counted = k
                        out.append(Say(number_word(k + 1), "count"))
                if held >= self.hold_time(phase):
                    out += self._phase_complete(f, now)

        self.display = self._make_display(f, value)
        return out

    def _phase_complete(self, f, now):
        out = []
        self._holding = False
        self._progress(now)
        rep = self._rep
        if self._phase == 0:
            self._phase = 1
            rep["phase_done_t"] = now
            rep["movement_samples"] = [(now, rep["values"][-1])]
            out.append(Say(self.phases[1].prompt))
        else:
            out += self._finish_rep(now)
            self._phase = 0
            out.append(Say(self.phases[0].prompt))
        return out

    def _finish_rep(self, now):
        rep = self._rep
        values = np.asarray(rep["values"], float)
        raw = np.asarray(rep["raw"], float)
        zt = rep["zone_time"]
        if zt.get("zone") in ("high", "low"):
            zt[zt["zone"] + "_s"] = max(zt.get(zt["zone"] + "_s", 0.0), now - zt["since"])
        prev_best = max((r.range_high for r in self.reps if np.isfinite(r.range_high)),
                        default=None)
        rec = RepRecord(
            exercise=self.name,
            set_no=self.set_no,
            rep_no=self.reps_this_set + 1,
            t_start=rep["t_start"], t_end=now,
            range_high=float(values.max()),
            range_low=float(values.min()),
            raw_high=float(raw[np.argmax(values)]),
            movement_time=float(rep["movement_time"]),
            smoothness_peaks=int(rep["smoothness"]),
            hold_stability=float(np.mean([np.std(v) for v in rep["hold_values"].values() if v]))
            if any(rep["hold_values"].values()) else float("nan"),
            compensation=list(rep["comp"]),
            hints=rep["hints"],
            extra={"high_zone_s": round(zt.get("high_s", 0.0), 2),
                   "low_zone_s": round(zt.get("low_s", 0.0), 2),
                   **self.rep_extra()},
        )
        self._add_rep(rec)
        out = [Say(f"{number_word(rec.rep_no)}.", "praise")]
        if prev_best is not None and rec.range_high > prev_best + 0.02:
            out.append(Say(self.best_phrase, "praise"))
        self._reset_rep(now)
        return out

    # --- compensation -----------------------------------------------------

    def _track_compensation(self, f, now):
        rep = self._rep
        out = []
        if f.palm_normal_image is None or f.wrist_image is None:
            return out
        if rep["normal0"] is None:
            rep["normal0"] = f.palm_normal_image.copy()
            rep["wrist0"] = f.wrist_image.copy()
            return out
        rotation = angle_between(rep["normal0"], f.palm_normal_image)
        size = max(f.palm_size_px, 1.0)
        shift = float(np.linalg.norm(f.wrist_image - rep["wrist0"]) / size)
        rep["max_rotation"] = max(rep["max_rotation"], rotation)
        rep["max_shift"] = max(rep["max_shift"], shift)
        if rotation > self.params.get("max_palm_rotation_deg", 1e9):
            out += self._compensation("palm_rotation", now)
        if shift > self.params.get("max_wrist_shift", 1e9):
            out += self._compensation("wrist_moved", now)
        return out

    def _compensation(self, key, now):
        rep = self._rep
        if key not in rep["comp"]:
            rep["comp"].append(key)
        text = self.compensation_messages.get(key)
        if text and text not in rep["comp_said"] and self._hints.ready("comp", now):
            rep["comp_said"].add(text)
            rep["hints"] += 1
            return [Say(text, "hint")]
        return []

    # --- display ------------------------------------------------------------

    def _make_display(self, f, value):
        phase = self.phases[self._phase]
        held = 0.0
        if self._holding:
            held = min(1.0, (self._rep["times"][-1] - self._hold_start) / max(self.hold_time(phase), 1e-6))
        best = max((r.range_high for r in self.reps), default=None)
        return {
            "kind": "bar",
            "value": float(value),
            "high": self.params["high"],
            "low": self.params["low"],
            "target_zone": phase.zone,
            "phase_label": phase.label,
            "prompt": phase.prompt,
            "hold_progress": held,
            "holding": self._holding,
            "best": best,
            "finger_colors": {},
        }


# ---------------------------------------------------------------------------
# Finger sequences (opposition, tapping)
# ---------------------------------------------------------------------------

GUIDED_ORDER = ["index", "middle", "ring", "pinky", "ring", "middle", "index"]


class SequenceExercise(Exercise):
    """
    The user produces finger events (a touch, a lift) that should follow a
    sequence. One completed sequence = one repetition ("round").

    Modes
      guided      the target is highlighted and spoken each step
      called_out  a random finger is called; reaction time is measured
      memory      a sequence is shown and spoken, then hidden; she repeats it
    """

    action_word = "Touch"           # "Touch your ring finger."
    wrong_phrase = "That was your {got}. Let's try the {want}."
    progress_phrase = "You were {pct}% quicker than {when}."
    progress_lower_is_better = True

    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = rng or np.random.default_rng()
        self.mode, self.length = self.initial_mode()
        self._clean_rounds = 0
        self._round = None

    # --- to override ------------------------------------------------------

    def initial_mode(self):
        return "guided", len(GUIDED_ORDER)

    def detect(self, f, now):
        """Return a list of (kind, finger, info) events; kind is 'start' or 'end'."""
        raise NotImplementedError

    def on_event_end(self, finger, info, now):
        """Called when a movement ends; may return a list of Say."""
        return []

    def level_up(self):
        """Called after enough error-free rounds. Returns a Say or None."""
        return None

    # --- rounds -------------------------------------------------------------

    def make_sequence(self):
        if self.mode == "guided":
            return list(GUIDED_ORDER[:self.length])
        seq = []
        for _ in range(self.length):
            choices = [c for c in FINGERS if not seq or c != seq[-1]]
            seq.append(str(self.rng.choice(choices)))
        return seq

    def start_set(self, set_no, now):
        super().start_set(set_no, now)
        self._round = None

    def _new_round(self, now):
        seq = self.make_sequence()
        self._round = {
            "seq": seq, "step": 0, "t_start": now,
            "step_start": now, "step_times": [],
            "correct": 0, "wrong": 0, "hints": 0,
            "hidden": self.mode != "memory",
            "show_until": now + self.params.get("memory_show_s", 5.0),
            "extra": [],
            "revealed": False,
        }
        out = []
        if self.mode == "memory":
            words = ", ".join(FINGER_WORDS[s].replace(" finger", "") for s in seq)
            out.append(Say(f"Remember: {words}."))
            self._round["hidden"] = False
        else:
            out.append(self._target_say())
        return out

    def _target(self):
        r = self._round
        return r["seq"][r["step"]]

    def _target_say(self):
        return Say(f"{self.action_word} your {FINGER_WORDS[self._target()]}.")

    def interrupt(self, now):
        super().interrupt(now)
        if self._round:
            self._round["step_start"] = now

    def update(self, f, now):
        out = []
        if self._round is None:
            out += self._new_round(now)
            self._progress(now)

        r = self._round
        if self.mode == "memory" and not r["hidden"] and not r["revealed"] and now >= r["show_until"]:
            r["hidden"] = True
            r["step_start"] = now
            self._progress(now)
            out.append(Say("Now you."))

        events = self.detect(f, now)
        for kind, finger, info in events:
            if kind == "end":
                out += self.on_event_end(finger, info, now) or []
                continue
            if self.mode == "memory" and not r["hidden"]:
                continue        # still showing the sequence
            out += self._on_start(finger, now)
            if self._round is None:
                break

        if self._round is not None and self._stalled(now) and self._hints.ready("stall", now):
            r = self._round
            r["hints"] += 1
            self._progress(now)
            if self.mode == "memory" and r["hidden"] and not r["revealed"] and r["hints"] == 1:
                out.append(Say("Which finger comes next?", "hint"))
            else:
                if self.mode == "memory":
                    r["revealed"] = True        # show the target instead of failing
                out.append(Say(self._target_say().text, "hint"))

        self.display = self._make_display(f)
        return out

    def _on_start(self, finger, now):
        r = self._round
        target = self._target()
        self._progress(now)
        if finger == target:
            r["correct"] += 1
            r["step_times"].append(now - r["step_start"])
            r["step"] += 1
            r["step_start"] = now
            if r["step"] >= len(r["seq"]):
                return self._finish_round(now)
            if self.mode in ("guided", "called_out") or r["revealed"]:
                return [self._target_say()]
            return [Say("Good.", "count")]
        r["wrong"] += 1
        if self.mode == "memory" and r["hidden"] and not r["revealed"]:
            # don't give the answer away, just invite another try
            text = f"That was your {FINGER_WORDS[finger]}. Let's try again."
        else:
            text = self.wrong_phrase.format(got=FINGER_WORDS[finger], want=FINGER_WORDS[target])
        return [Say(text, "hint")]

    def _finish_round(self, now):
        r = self._round
        times = r["step_times"]
        rec = RepRecord(
            exercise=self.name, set_no=self.set_no, rep_no=self.reps_this_set + 1,
            t_start=r["t_start"], t_end=now,
            # accuracy stands in for "range" so fatigue and summaries work alike
            range_high=r["correct"] / max(1, r["correct"] + r["wrong"]),
            raw_high=float(np.mean(times)) if times else float("nan"),
            movement_time=float(np.mean(times)) if times else float("nan"),
            hints=r["hints"],
            extra={"mode": self.mode, "length": len(r["seq"]),
                   "sequence": "-".join(r["seq"]),
                   "correct": r["correct"], "wrong": r["wrong"],
                   **self.round_extra(r)},
        )
        self._add_rep(rec)
        out = [Say("Well done, that's the whole sequence.", "praise")]
        if r["wrong"] == 0 and not r["revealed"]:
            self._clean_rounds += 1
        else:
            self._clean_rounds = 0
        if self._clean_rounds >= self.params.get("rounds_to_level_up", 2):
            msg = self.level_up()
            if msg:
                self._clean_rounds = 0
                out.append(msg)
        self._round = None
        return out

    def round_extra(self, r):
        return {}

    def _make_display(self, f):
        r = self._round
        if r is None:
            return {"kind": "sequence", "sequence": [], "step": 0, "hidden": False,
                    "target": None, "mode": self.mode, "finger_colors": {}}
        showing = self.mode != "memory" or not r["hidden"] or r["revealed"]
        target = self._target() if r["step"] < len(r["seq"]) else None
        colors = {}
        if target and (self.mode != "memory" or r["revealed"]):
            colors[target] = "target"
        return {
            "kind": "sequence",
            "sequence": r["seq"],
            "step": r["step"],
            "hidden": not showing,
            "target": target if showing else None,
            "mode": self.mode,
            "prompt": (f"{self.action_word} your {FINGER_WORDS[target]}" if target and
                       (self.mode != "memory" or r["revealed"]) else
                       ("Remember the fingers" if self.mode == "memory" and not r["hidden"]
                        else "Your turn")),
            "finger_colors": colors,
        }
