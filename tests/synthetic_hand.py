"""
A simple kinematic hand for tests: joint angles in, 21 landmarks out.

Left hand, palm facing the camera, in the mirrored image frame
(x right, y down, z away from the camera). Fingers flex towards the
camera (the palm side); negative flexion lifts them backwards.
"""

import numpy as np

from rehab.features import HandObservation, TIPS

MCP = {
    "index": np.array([0.025, -0.085, 0.0]),
    "middle": np.array([0.005, -0.090, 0.0]),
    "ring": np.array([-0.015, -0.085, 0.0]),
    "pinky": np.array([-0.033, -0.075, 0.0]),
}
SEGMENTS = {"index": (0.040, 0.025, 0.020), "middle": (0.045, 0.028, 0.021),
            "ring": (0.042, 0.026, 0.020), "pinky": (0.032, 0.020, 0.018)}
SPREAD_SIGN = {"index": 1.0, "middle": 0.0, "ring": -1.0, "pinky": -2.0}
PALM_SIDE = np.array([0.0, 0.0, -1.0])    # towards the camera


def _rot(v, axis, deg):
    axis = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    return (v * np.cos(t) + np.cross(axis, v) * np.sin(t)
            + axis * np.dot(axis, v) * (1 - np.cos(t)))


def _chain(start, direction, lengths, flex, bend_axis):
    pts = [start]
    d = direction / np.linalg.norm(direction)
    p = start
    for length, angle in zip(lengths, flex):
        d = _rot(d, bend_axis, angle)
        p = p + length * d
        pts.append(p)
    return pts


def hand(flex=None, spread=8.0, thumb_out=1.0, thumb_touch=None, finger_flex=None,
         handedness="Left", noise=0.0, rng=None):
    """
    flex         (mcp, pip, dip) degrees for all four fingers
    finger_flex  {finger: (mcp, pip, dip)} overrides per finger
    spread       degrees between neighbouring fingers
    thumb_out    1 = thumb stretched out, 0 = bent across the palm
    thumb_touch  finger name: put the thumb tip on that fingertip
    """
    flex = flex if flex is not None else (0.0, 0.0, 0.0)
    finger_flex = finger_flex or {}
    w = np.zeros((21, 3))
    for i, name in enumerate(("index", "middle", "ring", "pinky")):
        base = 5 + 4 * i
        angle = np.radians(SPREAD_SIGN[name] * spread)
        direction = np.array([np.sin(angle), -np.cos(angle), 0.0])
        # flexing bends the finger towards the palm side
        bend_axis = np.cross(direction, PALM_SIDE)
        pts = _chain(MCP[name], direction, SEGMENTS[name],
                     finger_flex.get(name, flex), bend_axis)
        w[base:base + 4] = pts

    # thumb
    cmc = np.array([0.030, -0.025, -0.008])
    out_dir = np.array([0.85, -0.5, -0.15])
    in_dir = np.array([-0.55, -0.55, -0.6])
    d = thumb_out * out_dir + (1 - thumb_out) * in_dir
    bend = (1 - thumb_out) * 55.0
    axis = np.cross(d, PALM_SIDE)
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([0.0, 0.0, 1.0])
    pts = _chain(cmc, d, (0.035, 0.03, 0.025), (0.0, bend, bend), axis)
    w[1:5] = pts
    if thumb_touch:
        tip = w[TIPS[thumb_touch]] + np.array([0.0, 0.0, -0.004])
        w[4] = tip
        w[3] = w[2] + 0.55 * (tip - w[2]) + np.array([0.0, 0.0, -0.01])

    if noise:
        rng = rng or np.random.default_rng(0)
        w = w + rng.normal(0, noise, w.shape)

    world = w - w.mean(axis=0)
    image = np.column_stack([0.5 + w[:, 0] * 3.0, 0.6 + w[:, 1] * 3.0 * 16 / 9, w[:, 2] * 3.0])
    return HandObservation(image=image, world=world, handedness=handedness,
                           handedness_score=0.95, gesture=None, gesture_score=0.0)


IMAGE_SIZE = (1280, 720)
