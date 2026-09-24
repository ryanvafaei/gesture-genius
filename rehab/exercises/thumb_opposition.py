"""
Exercise 4: thumb opposition (touch each fingertip).

Calibration measures the thumb away from the fingers and a touch on each
of the four fingertips: the little and ring fingers are further from the
thumb and are tracked less well, so each finger gets its own touch distance.

Per finger, "closeness" is 0 at her calibrated touch and 1 at her open
hand, from the 3D (world) distance and from the distance in the picture
(with the palm to the camera the picture has no depth noise; the world
distance does not change with a tilted hand); the two are averaged. A touch
is registered when one finger's closeness drops below touch_factor, it is
clearly closer than the next finger (dominance_margin), and it stays there
for a moment. When the prompted finger and a neighbour are both touching
(the ring and little fingertips are close together) the prompted finger is
taken (prefer_target). The thumb has to move away again before the next
touch counts.

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
    d = dict(f.thumb_tip_dist)
    d.update({f"img_{k}": v for k, v in f.thumb_tip_dist_image.items()})
    return d


def _touch(finger):
    def extract(f):
        return {finger: f.thumb_tip_dist[finger], f"img_{finger}": f.thumb_tip_dist_image[finger]}
    return extract


# the index touch keeps its old step name ("touch")
TOUCH_STEPS = {"index": "touch", "middle": "touch_middle", "ring": "touch_ring",
               "pinky": "touch_pinky"}


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
                            _touch("index"), need_palm_facing=True,
                            screen_text="Touch index and hold"),
            CalibrationStep("touch_middle", "Now touch your middle fingertip. And hold.",
                            _touch("middle"), need_palm_facing=True,
                            screen_text="Touch middle finger and hold"),
            CalibrationStep("touch_ring", "Now touch your ring fingertip. And hold.",
                            _touch("ring"), need_palm_facing=True,
                            screen_text="Touch ring finger and hold"),
            CalibrationStep("touch_pinky", "And now your little fingertip. And hold.",
                            _touch("pinky"), need_palm_facing=True,
                            screen_text="Touch little finger and hold"),
        ]

    @classmethod
    def calibration_valid(cls, steps):
        # touching brings the thumb closer to each fingertip; an older
        # calibration with only the index touch is measured again
        return all(increases(steps, step, "open", finger, 0.05)
                   for finger, step in TOUCH_STEPS.items())

    def __init__(self, *args, level=None, **kwargs):
        self.level = level          # None -> params["start_level"], see initial_mode()
        super().__init__(*args, **kwargs)
        self._touching = None
        self._candidate = None
        self._candidate_t = None
        self._pinch = None              # (smallest gap, other gaps) during an index touch
        self._pinch_scores = []         # FMA 28-style score of each index touch this round
        self._pinch_gaps = []
        self._closeness = {}
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
        """Per finger and view ("world", "image"): (touch distance, open distance)."""
        open_d = self.cal.get("open", {})
        th = {}
        for finger in FINGERS:
            step = self.cal.get(TOUCH_STEPS[finger])
            # an older calibration measured only the index touch
            src = finger if step else "index"
            step = step or self.cal.get("touch", {})
            ranges = {}
            for view, prefix in (("world", ""), ("image", "img_")):
                near = step.get(prefix + src)
                far = open_d.get(prefix + finger)
                if view == "world":
                    near = 0.25 if near is None else near
                    far = 1.0 if far is None else far
                elif near is None or far is None:
                    continue
                ranges[view] = (near, max(far, near + 0.1))
            th[finger] = ranges
        return th

    def closeness(self, f):
        """Per finger: 0 at her calibrated touch, 1 with the thumb away (world and picture averaged)."""
        c = {}
        for finger in FINGERS:
            values = []
            for view, d in (("world", f.thumb_tip_dist), ("image", f.thumb_tip_dist_image)):
                rng = self.thresholds[finger].get(view)
                if rng is not None and finger in d:
                    near, far = rng
                    values.append((d[finger] - near) / (far - near))
            c[finger] = float(sum(values) / len(values)) if values else 1.0
        return c

    def _prompted(self):
        """The finger she is asked for now, or None."""
        r = self._round
        if r is None or r["step"] >= len(r["seq"]):
            return None
        return r["seq"][r["step"]]

    def detect(self, f, now):
        d = f.thumb_tip_dist
        c = self._closeness = self.closeness(f)
        touch_at = self.params.get("touch_factor", 0.35)
        events = []
        if self._touching:
            if self._touching == "index":
                gap = d["index"]
                if self._pinch is None or gap < self._pinch[0]:
                    self._pinch = (gap, [d[k] for k in FINGERS if k != "index"])
            if c[self._touching] > self.params.get("release_factor", 0.55):
                if self._touching == "index" and self._pinch is not None:
                    self._pinch_gaps.append(self._pinch[0])
                    self._pinch_scores.append(benchmarks.pinch_score(*self._pinch))
                    self._pinch = None
                events.append(("end", self._touching, {}))
                self._touching = None
            return events

        ranked = sorted(FINGERS, key=c.get)
        best, second = ranked[0], ranked[1]
        margin = self.params.get("dominance_margin", 0.15)
        target = self._prompted() if self.params.get("prefer_target", True) else None
        pick = None
        if target and c[target] < touch_at and c[target] - c[best] < margin:
            pick = target               # e.g. ring and little fingertips both at the thumb
        elif c[best] < touch_at and c[second] - c[best] >= margin:
            pick = best
        elif self._candidate and c[self._candidate] < touch_at:
            pick = self._candidate      # a moment of doubt does not restart the hold
        if pick is None:
            self._candidate = None
        elif pick != self._candidate:
            self._candidate, self._candidate_t = pick, now
        elif now - self._candidate_t >= self.params.get("min_touch_s", 0.3):
            self._touching = pick
            self._candidate = None
            events.append(("start", pick, {}))
        return events

    def interrupt(self, now):
        super().interrupt(now)
        self._touching = None
        self._candidate = None
        self._pinch = None

    def debug_settings(self):
        return {"thresholds": self.thresholds, "level": self.level}

    def debug_state(self):
        return {"closeness": dict(self._closeness), "candidate": self._candidate,
                "touching": self._touching, "prompted": self._prompted(), "level": self.level,
                "mode": self.mode}

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
