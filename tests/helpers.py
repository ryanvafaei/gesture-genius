"""Drive exercises with synthetic hands at 30 fps."""

import numpy as np

from rehab import features
from rehab.calibration import CalibrationRoutine
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
