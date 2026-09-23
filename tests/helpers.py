"""Drive exercises with synthetic hands at 30 fps."""

import collections

import numpy as np

from rehab import features
from rehab.Act import enqueue
from rehab.calibration import CalibrationRoutine
from rehab.exercises.base import Say
from synthetic_hand import IMAGE_SIZE, hand

FPS = 30.0


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


class TimedSpeaker:
    """
    Speaker with the real queue rules, but speech takes simulated time
    (about 140 words per minute), so tests hear what she would hear and when.
    Call advance() once per frame.
    """

    S_PER_WORD = 60.0 / 140
    START_S = 0.3

    def __init__(self, clock, max_queue=3, on_speak=None):
        self.clock = clock
        self.max_queue = max_queue
        self.on_speak = on_speak or (lambda msg: None)
        self.items = collections.deque()
        self.current = None
        self.until = 0.0
        self.heard = []           # (time, text) in the order she hears them
        self.last_text = ""

    @property
    def busy(self):
        return self.current is not None or bool(self.items)

    def say(self, msg):
        if isinstance(msg, str):
            msg = Say(msg)
        if msg.ephemeral and self.busy:
            return
        enqueue(self.items, msg, self.max_queue)

    def say_all(self, messages):
        for m in messages or []:
            self.say(m)

    def clear(self):
        self.items.clear()

    def advance(self):
        t = self.clock.t
        if self.current is not None and t >= self.until:
            self.current = None
        while self.current is None and self.items:
            msg = self.items.popleft()
            if not msg.still_valid():
                continue
            self.current = msg
            self.until = t + self.START_S + self.S_PER_WORD * len(msg.text.split())
            self.last_text = msg.text
            self.heard.append((t, msg.text))
            self.on_speak(msg)

    def close(self):
        pass
