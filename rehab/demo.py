"""
The demo hand: shows each movement instead of only saying it.

Every instruction reaches her three ways: a short sentence on screen, the
spoken line, and this picture. The exercise card before an exercise loops
through the whole movement; during the exercise a small still picture shows
the position to reach now.

The hand is the same kinematic model the tests use (rehab/handmodel.py),
drawn from a three-quarter view so that bending fingers can be seen. It is
a left hand as she sees it in the mirror; for a right hand it is flipped.

Nothing here decides anything: poses in, pixel points out (ui draws them).
"""

import numpy as np

from rehab.exercises.base import GUIDED_ORDER
from rehab.features import FINGERS
from rehab.handmodel import hand

OPEN = {"flex": (0.0, 0.0, 0.0), "spread": 8.0, "thumb_out": 1.0}
CLOSED = {"flex": (80.0, 95.0, 60.0), "spread": 2.0, "thumb_out": 0.0}
TOGETHER = {"flex": (0.0, 0.0, 0.0), "spread": 1.0, "thumb_out": 1.0}
SPREAD = {"flex": (0.0, 0.0, 0.0), "spread": 16.0, "thumb_out": 1.0}
THUMB_IN = {"flex": (0.0, 0.0, 0.0), "spread": 6.0, "thumb_out": 0.0}
HOLD_CLOTH = {"flex": (45.0, 55.0, 35.0), "spread": 3.0, "thumb_out": 0.4}
SQUEEZE = {"flex": (65.0, 80.0, 50.0), "spread": 2.0, "thumb_out": 0.2}
POINT = {"flex": (80.0, 95.0, 60.0), "finger_flex": {"index": (0.0, 0.0, 0.0)},
         "spread": 4.0, "thumb_out": 0.0}
PINCH_OPEN = {"flex": (20.0, 25.0, 15.0), "spread": 6.0, "thumb_out": 1.0}


def _touch(finger):
    return {"flex": (15.0, 20.0, 10.0), "spread": 6.0, "thumb_out": 0.6, "thumb_touch": finger}


def _lift(finger):
    return {"flex": (0.0, 0.0, 0.0), "spread": 6.0, "thumb_out": 1.0,
            "finger_flex": {finger: (-50.0, 0.0, 0.0)}}


# the position for each phase key (the exercise display's "demo_key")
POSES = {
    "grip_release": {"open": OPEN, "close": CLOSED},
    "finger_abduction": {"spread": SPREAD, "together": TOGETHER},
    "thumb_flexion": {"in": THUMB_IN, "out": OPEN},
    "thumb_opposition": {f"touch_{f}": _touch(f) for f in FINGERS},
    "finger_tapping": {f"lift_{f}": _lift(f) for f in FINGERS},
    "grip_squeeze": {"squeeze": SQUEEZE, "relax": HOLD_CLOTH},
    "bubble_pinch": {"pinch": _touch("index"), "release": PINCH_OPEN},
    "two_hand_match": {"open": OPEN, "close": CLOSED},
    "memory_pairs": {"point": POINT},
}
# the loop on the exercise card: (pose, seconds to get there and stay)
LOOPS = {
    "grip_release": [(OPEN, 1.6), (CLOSED, 1.6)],
    "finger_abduction": [(SPREAD, 1.6), (TOGETHER, 1.6)],
    "thumb_flexion": [(THUMB_IN, 1.6), (OPEN, 1.6)],
    "thumb_opposition": [(_touch(f), 1.0) for f in GUIDED_ORDER[:4]] + [(OPEN, 1.0)],
    "finger_tapping": [(_lift(f), 1.0) for f in FINGERS] + [(OPEN, 1.0)],
    "grip_squeeze": [(SQUEEZE, 1.8), (HOLD_CLOTH, 1.8)],
    "bubble_pinch": [(PINCH_OPEN, 1.4), (_touch("index"), 1.8)],
    "two_hand_match": [(OPEN, 1.6), (CLOSED, 1.6)],
    "memory_pairs": [(POINT, 2.0)],
}
TWO_HANDS = {"two_hand_match"}

# three-quarter view: turned about the vertical axis, then tilted a little
_YAW, _PITCH = np.radians(35.0), np.radians(-20.0)
_ROT = (np.array([[1, 0, 0], [0, np.cos(_PITCH), -np.sin(_PITCH)],
                  [0, np.sin(_PITCH), np.cos(_PITCH)]])
        @ np.array([[np.cos(_YAW), 0, np.sin(_YAW)], [0, 1, 0],
                    [-np.sin(_YAW), 0, np.cos(_YAW)]]))


def _view(pose):
    """3D landmarks turned to the three-quarter view, the wrist at the origin."""
    world = hand(**pose).world
    return (world - world[0]) @ _ROT.T


# the scale and centre are fixed (from the open hand), so the hand never
# jumps or changes size between positions; the wrist stays in one place
_OPEN_VIEW = _view(OPEN)
_CENTRE = (_OPEN_VIEW[:, :2].min(axis=0) + _OPEN_VIEW[:, :2].max(axis=0)) / 2
_EXTENT = float((_OPEN_VIEW[:, :2].max(axis=0) - _OPEN_VIEW[:, :2].min(axis=0)).max()) * 1.15


def _blend(a, b, u):
    """Pose between a and b (u = 0..1); the thumb target switches halfway."""
    out = {}
    for key in ("flex", "spread", "thumb_out"):
        va, vb = np.asarray(a.get(key, OPEN[key]), float), np.asarray(b.get(key, OPEN[key]), float)
        out[key] = tuple(va + (vb - va) * u) if va.ndim else float(va + (vb - va) * u)
    fingers = set(a.get("finger_flex", {})) | set(b.get("finger_flex", {}))
    if fingers:
        base_a, base_b = a.get("flex", OPEN["flex"]), b.get("flex", OPEN["flex"])
        out["finger_flex"] = {
            f: tuple(np.asarray(a.get("finger_flex", {}).get(f, base_a), float) * (1 - u)
                     + np.asarray(b.get("finger_flex", {}).get(f, base_b), float) * u)
            for f in fingers}
    touch = a.get("thumb_touch") if u < 0.5 else b.get("thumb_touch")
    if touch:
        out["thumb_touch"] = touch
    return out


def _smooth(u):
    return u * u * (3 - 2 * u)


def pose_at(exercise, t):
    """The loop's pose at t seconds: move for the first 45% of a step, then stay."""
    loop = LOOPS.get(exercise) or [(OPEN, 1.0)]
    total = sum(s for _, s in loop)
    t = float(t) % total
    for i, (pose, seconds) in enumerate(loop):
        if t < seconds:
            before = loop[i - 1][0]
            return _blend(before, pose, _smooth(min(1.0, t / (0.45 * seconds))))
        t -= seconds
    return loop[-1][0]


def _project(pose, box, mirror=False, offset=0.0):
    """21 pixel points of one hand in box (x0, y0, x1, y1)."""
    p = _view(pose)[:, :2] - _CENTRE
    x, y = p[:, 0], p[:, 1]
    if mirror:
        x = -x
    x0, y0, x1, y1 = box
    k = min(x1 - x0, y1 - y0) / _EXTENT
    cx, cy = (x0 + x1) / 2 + offset * (x1 - x0), (y0 + y1) / 2
    return np.column_stack([cx + x * k, cy + y * k])


def _hands(exercise, pose, box, right=False):
    if exercise in TWO_HANDS:
        # her pair of hands side by side, the smaller scale to fit both
        x0, y0, x1, y1 = box
        half = (x1 - x0) / 2
        left_box, right_box = (x0, y0, x0 + half, y1), (x0 + half, y0, x1, y1)
        return [_project(pose, left_box, mirror=False), _project(pose, right_box, mirror=True)]
    return [_project(pose, box, mirror=right)]


def demo_points(exercise, t, box, hand_side="Left"):
    """The looping demo at t seconds: a list of (21, 2) pixel arrays (one per hand)."""
    return _hands(exercise, pose_at(exercise, t), box, right=hand_side == "Right")


def phase_points(exercise, key, box, hand_side="Left"):
    """The still picture of one position (the display's demo_key), or [] if unknown."""
    pose = POSES.get(exercise, {}).get(key)
    if pose is None:
        return []
    return _hands(exercise, pose, box, right=hand_side == "Right")
