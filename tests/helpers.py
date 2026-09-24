"""Drive exercises with synthetic hands at 30 fps."""

import collections

import numpy as np

from rehab import features, storage
from rehab.Act import SpeechBase, drop_older, enqueue, interrupts, next_message
from rehab.calibration import CalibrationRoutine
from synthetic_hand import IMAGE_SIZE, hand

FPS = 30.0


def returning_profile():
    """A profile after the first-time setup (coach name and activities chosen)."""
    p = storage.new_profile()
    p["coach_name"] = "Iris"
    p["chosen_activities"] = ["tea", "gardening", "reading"]
    return p


def answer(session, t, yes=True):
    """Answer a question on screen (name, activities, check-in, plant) like the space / n keys."""
    from rehab.Think import QUESTION_STAGES
    if session.stage in QUESTION_STAGES:
        session.on_key("y" if yes else "n", t)
        return True
    return False


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def tick(self):
        self.t += 1.0 / FPS
        return self.t


def feat(t=0.0, **kw):
    return features.extract(hand(**kw), t, IMAGE_SIZE)


def run(exercise, seconds, clock, **pose):
    """Feed the same pose for some seconds; returns all messages."""
    said = []
    for _ in range(int(seconds * FPS)):
        t = clock.tick()
        said += exercise.update(feat(t, **pose), t)
    return said


def ramp(exercise, seconds, clock, pose_fn):
    """pose_fn(u) -> pose kwargs for u in 0..1."""
    said = []
    n = int(seconds * FPS)
    for i in range(n):
        t = clock.tick()
        said += exercise.update(feat(t, **pose_fn(i / max(1, n - 1))), t)
    return said


def calibrate(exercise_cls, poses, clock=None):
    """Run the real calibration routine with one pose per step."""
    clock = clock or Clock()
    routine = CalibrationRoutine(exercise_cls)
    routine.start(clock.t)
    for pose in poses:
        name = routine.step.name
        while not routine.done and routine.step.name == name:
            t = clock.tick()
            routine.update(feat(t, **pose), t)
    assert routine.done
    return routine.result


def texts(messages):
    return [m.text for m in messages]


def lerp(a, b, u):
    return tuple(np.asarray(a) + (np.asarray(b) - np.asarray(a)) * u)


class TimedSpeaker(SpeechBase):
    """
    Speaker with the real queue rules, but speech takes simulated time
    (about 140 words per minute), so tests hear what she would hear and when.
    Call advance() once per frame.
    """

    S_PER_WORD = 60.0 / 140
    START_S = 0.3

    def __init__(self, clock, max_queue=6, on_speak=None, feedback=None):
        self.clock = clock
        self.max_queue = max_queue
        self.on_speak = on_speak or (lambda msg: None)
        self.feedback = feedback
        self.items = collections.deque()
        self.current = None
        self.until = 0.0
        self.heard = []           # (time, text) in the order she hears them
        self.cut_off = []         # texts interrupted by an instruction
        self.chimes = 0
        self.last_text = ""

    @property
    def busy(self):
        return self.current is not None or bool(self.items)

    def _say(self, msg):
        if msg.ephemeral and self.busy:
            return
        if interrupts(self.current, msg, self.clock.t):
            self.cut_off.append(self.current.text)
            self.current = None
        enqueue(self.items, msg, self.max_queue, self.clock.t)

    def chime(self):
        self.chimes += 1

    def clear(self):
        self.items.clear()

    def drop_before(self, mark):
        drop_older(self.items, mark)
        if self.current is not None and self.current.seq is not None and self.current.seq < mark:
            self.cut_off.append(self.current.text)
            self.current = None

    def advance(self):
        t = self.clock.t
        if self.current is not None and t >= self.until:
            self.current = None
        while self.current is None and self.items:
            msg = next_message(self.items, t)
            if msg is None:
                break
            self.current = msg
            self.until = t + self.START_S + self.S_PER_WORD * len(msg.text.split())
            self.last_text = msg.text
            self.heard.append((t, msg.text))
            self.on_speak(msg)

    def close(self):
        pass
