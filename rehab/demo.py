"""
The demo hand: shows each movement instead of only saying it.

Every instruction reaches her three ways: a short sentence on screen, the
spoken line, and this picture. The exercise card before an exercise loops
through the whole movement; during the exercise a small still picture shows
the position to reach now.

The hand is the same kinematic model the tests use (rehab/handmodel.py),
drawn from a three-quarter view so that bending fingers can be seen. It is
a left hand as she sees it in the mirror; for a right hand it is flipped.

The arm exercises show a seated figure instead (side-on or facing, like the
camera view of the exercise): trunk, head, thigh and the arm, from the
shoulder, elbow and wrist angles of each position (arm_points).

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

# --- the arm figure (arm exercises) --------------------------------------------------------------
# A pose: shoulder elevation, elbow flexion and wrist extension in degrees,
# the trunk lean, and the view ("side": she faces to the right, "front").
ARM_REST = {"shoulder": 0.0, "elbow": 0.0, "wrist": 0.0}
ARM_LAP = {"shoulder": 15.0, "elbow": 75.0, "wrist": 0.0}
ARM_POSES = {
    "shoulder_flexion_raise": {"rest": ARM_REST, "up": {"shoulder": 110.0, "elbow": 0.0}},
    "shoulder_abduction_raise": {"rest": dict(ARM_REST, view="front"),
                                 "up": {"shoulder": 100.0, "elbow": 0.0, "view": "front"}},
    "hand_to_mouth": {"rest": ARM_LAP, "up": {"shoulder": 35.0, "elbow": 135.0}},
    "hand_to_head": {"rest": ARM_LAP, "up": {"shoulder": 70.0, "elbow": 130.0}},
    "elbow_extension": {"rest": ARM_LAP, "up": {"shoulder": 30.0, "elbow": 0.0}},
    "wrist_extension": {"rest": {"shoulder": 10.0, "elbow": 80.0, "wrist": -20.0, "table": True},
                        "up": {"shoulder": 10.0, "elbow": 80.0, "wrist": 40.0, "table": True}},
    "tabletop_reach": {"rest": {"shoulder": 20.0, "elbow": 90.0, "table": True},
                       "up": {"shoulder": 60.0, "elbow": 15.0, "table": True}},
    "finger_to_nose_timed": {"rest": dict(ARM_LAP, view="front"),
                             "touch": {"shoulder": 40.0, "elbow": 150.0, "view": "front"}},
}
ARM_LOOPS = {name: [(p["rest"], 1.8), (p.get("up", p.get("touch")), 1.8)]
             for name, p in ARM_POSES.items()}
POSES.update(ARM_POSES)
LOOPS.update(ARM_LOOPS)
# the figure's points: 0 head, 1 shoulder, 2 hip, 3 knee, 4 elbow, 5 wrist,
# 6 fingertips, 7-9 the other shoulder, elbow and wrist (facing view only)
ARM_POINTS = 10
ARM_FIGURE_LINES = ((1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (1, 7), (7, 8), (8, 9))
_TORSO, _UPPER, _FORE, _HAND = 1.0, 0.55, 0.5, 0.18
_ARM_EXTENT = 2.5           # fixed scale: the figure never jumps between positions


def _rot(v, deg):
    r = np.radians(deg)
    return np.array([v[0] * np.cos(r) - v[1] * np.sin(r), v[0] * np.sin(r) + v[1] * np.cos(r)])


def _figure(pose):
    """ARM_POINTS points in figure units, the shoulder at the origin (y down)."""
    front = pose.get("view") == "front"
    down = np.array([0.0, 1.0])
    shoulder = np.zeros(2)
    hip = np.array([0.0, _TORSO])
    knee = hip + ([0.0, 0.0] if front else [0.6, 0.0])
    # side view: forward is +x; facing view: her arm goes out to her side (-x in a mirror)
    turn = pose.get("shoulder", 0.0) if front else -pose.get("shoulder", 0.0)
    upper = _rot(down, turn)
    elbow = shoulder + _UPPER * upper
    fore = _rot(upper, pose.get("elbow", 0.0) if front else -pose.get("elbow", 0.0))
    wrist = elbow + _FORE * fore
    hand = _rot(fore, -pose.get("wrist", 0.0))
    tip = wrist + _HAND * hand
    if front:
        other = np.array([0.45, 0.0])
        hip = np.array([0.22, _TORSO])
        knee = hip
        others = [other, other + [0.0, _UPPER], other + [0.0, _UPPER + _FORE]]
    else:
        others = [shoulder, shoulder, shoulder]
    head = np.array([0.22 if front else 0.0, -0.3])
    return np.array([head, shoulder, hip, knee, elbow, wrist, tip] + others, float)


def _blend_arm(a, b, u):
    out = dict(b)
    for key in ("shoulder", "elbow", "wrist"):
        out[key] = a.get(key, 0.0) + (b.get(key, 0.0) - a.get(key, 0.0)) * u
    return out


def arm_points(pose, box, mirror=False):
    """(ARM_POINTS, 2) pixel points of the arm figure in box (x0, y0, x1, y1)."""
    p = _figure(pose)
    if mirror:
        p[:, 0] = -p[:, 0]
    x0, y0, x1, y1 = box
    k = min(x1 - x0, y1 - y0) / _ARM_EXTENT
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.1 * k
    return np.column_stack([cx + p[:, 0] * k, cy + p[:, 1] * k])

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
            blend = _blend_arm if exercise in ARM_POSES else _blend
            return blend(before, pose, _smooth(min(1.0, t / (0.45 * seconds))))
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
    if exercise in ARM_POSES:
        # her right arm is shown on the other side, like in a mirror
        return [arm_points(pose, box, mirror=right)]
    if exercise in TWO_HANDS:
        # her pair of hands side by side, the smaller scale to fit both
        x0, y0, x1, y1 = box
        half = (x1 - x0) / 2
        left_box, right_box = (x0, y0, x0 + half, y1), (x0 + half, y0, x1, y1)
        return [_project(pose, left_box, mirror=False), _project(pose, right_box, mirror=True)]
    return [_project(pose, box, mirror=right)]


def demo_points(exercise, t, box, hand_side="Left"):
    """
    The looping demo at t seconds: a list of (21, 2) pixel arrays (one per
    hand), or one (ARM_POINTS, 2) array for an arm exercise's figure.
    """
    return _hands(exercise, pose_at(exercise, t), box, right=hand_side == "Right")


def phase_points(exercise, key, box, hand_side="Left"):
    """The still picture of one position (the display's demo_key), or [] if unknown."""
    pose = POSES.get(exercise, {}).get(key)
    if pose is None:
        return []
    return _hands(exercise, pose, box, right=hand_side == "Right")
