"""
Exercise 4: thumb opposition (touch each fingertip).

A touch is registered when one thumb-to-fingertip distance (divided by palm
size) drops below her calibrated touch threshold, is clearly the smallest of
the four, and stays there for a moment. The thumb has to move away again
before the next touch counts.

Levels (cognitive goal):
  1  guided        index -> pinky -> index, target highlighted and spoken
  2  short memory  3 fingers shown for a few seconds, then hidden
  3+ longer        4-5 fingers, only after error-free rounds

Benchmark (the plan's "pinch", FMA item 28 position): every index touch is
also scored as a pad-to-pad pincer grasp: thumb-index gap at most 0.12 palm
sizes (proposed) while the other fingertips stay away from the thumb. At
most 1, because the tug on the pencil cannot be felt by a webcam. Logged per
round as fma28_style and pinch_gap_min.
"""

from rehab import benchmarks
from rehab.features import FINGERS
from rehab.exercises.base import (GUIDED_ORDER, CalibrationStep, Say,
                                  SequenceExercise, increases)


def _distances(f):
    return dict(f.thumb_tip_dist)


def _touch_index(f):
    return {"index": f.thumb_tip_dist["index"]}


class ThumbOpposition(SequenceExercise):
    name = "thumb_opposition"
    title = "Touch your fingertips"
    instructions = (
        "Now touch your thumb to each fingertip.",
        "Take your time.",
    )
    need_palm_facing = True
    uses_level = True
    action_word = "Touch"
    progress_phrase = "You touched your fingers {pct}% quicker than {when}."

    @classmethod
    def calibration_steps(cls):
        return [
            CalibrationStep("open", "Open your hand and hold your thumb away from your fingers. And hold.",
                            _distances, need_palm_facing=True, screen_text="Thumb away and hold"),
            CalibrationStep("touch", "Now touch your thumb to your index fingertip. And hold.",
                            _touch_index, need_palm_facing=True, screen_text="Touch index and hold"),
        ]

    @classmethod
    def calibration_valid(cls, steps):
        # touching brings the thumb closer to the index fingertip
        return increases(steps, "touch", "open", "index", 0.05)

    def __init__(self, *args, level=None, **kwargs):
        self.level = level          # None -> params["start_level"], see initial_mode()
        super().__init__(*args, **kwargs)
        self._touching = None
        self._candidate = None
        self._candidate_t = None
        self._pinch = None              # (smallest gap, other gaps) during an index touch
        self._pinch_scores = []         # FMA 28-style score of each index touch this round
        self._pinch_gaps = []
        self.thresholds = self._thresholds()

    # --- levels -------------------------------------------------------------

    def _mode_for_level(self, level):
        lengths = self.params.get("level_lengths", {1: 7, 2: 3, 3: 4, 4: 5})
        lengths = {int(k): v for k, v in lengths.items()}
        level = min(level, max(lengths))
        if level == 1:
            return "guided", min(lengths[1], len(GUIDED_ORDER))
        return "memory", lengths[level]

    def initial_mode(self):
        self.level = int(self.level or self.params.get("start_level", 1))
        return self._mode_for_level(self.level)

    def level_up(self):
        lengths = self.params.get("level_lengths", {1: 7, 2: 3, 3: 4, 4: 5})
        if self.level >= max(int(k) for k in lengths):
            return None
        self.level += 1
        self.mode, self.length = self._mode_for_level(self.level)
        if self.level == 2:
            return Say("You're doing so well. Now let's try one from memory.", "praise")
        return Say("Great memory. Let's make it a little longer.", "praise")

    # --- touch detection ------------------------------------------------------

    def _thresholds(self):
        open_d = self.cal.get("open", {})
        touch_d = self.cal.get("touch", {}).get("index", 0.25)
        k_touch = self.params.get("touch_factor", 0.35)
        k_release = self.params.get("release_factor", 0.55)
        th = {}
        for finger in FINGERS:
            far = max(open_d.get(finger, 1.0), touch_d + 0.1)
            th[finger] = (touch_d + k_touch * (far - touch_d),
                          touch_d + k_release * (far - touch_d))
        return th

    def detect(self, f, now):
        d = f.thumb_tip_dist
        events = []
        if self._touching:
            if self._touching == "index":
                gap = d["index"]
                if self._pinch is None or gap < self._pinch[0]:
                    self._pinch = (gap, [d[k] for k in FINGERS if k != "index"])
            if d[self._touching] > self.thresholds[self._touching][1]:
                if self._touching == "index" and self._pinch is not None:
                    self._pinch_gaps.append(self._pinch[0])
                    self._pinch_scores.append(benchmarks.pinch_score(*self._pinch))
                    self._pinch = None
                events.append(("end", self._touching, {}))
                self._touching = None
            return events

        ranked = sorted(FINGERS, key=lambda k: d[k])
        best, second = ranked[0], ranked[1]
        touch_thr = self.thresholds[best][0]
        dominant = d[best] < self.params.get("dominance_ratio", 0.75) * d[second]
        if d[best] < touch_thr and dominant:
            if self._candidate != best:
                self._candidate, self._candidate_t = best, now
            elif now - self._candidate_t >= self.params.get("min_touch_s", 0.3):
                self._touching = best
                self._candidate = None
                events.append(("start", best, {}))
        else:
            self._candidate = None
        return events

    def interrupt(self, now):
        super().interrupt(now)
        self._touching = None
        self._candidate = None
        self._pinch = None

    def round_extra(self, r):
        extra = {"level": self.level}
        if self._pinch_scores:
            extra["fma28_style"] = max(self._pinch_scores)
            extra["pinch_gap_min"] = round(min(self._pinch_gaps), 3)
        self._pinch_scores, self._pinch_gaps = [], []
        return extra

    def _make_display(self, f):
        d = super()._make_display(f)
        d["level"] = self.level
        if self._touching:
            d["finger_colors"] = dict(d["finger_colors"], **{self._touching: "active"})
        return d
