"""
A simple 2D body for tests: joint angles in, 33 pose landmarks out.

Seated, in the mirrored display frame (x right, y down). Side-on
("sagittal"): the tested arm nearest the camera moves forward (+x) and the
shoulders overlap. Facing the camera ("frontal"): the shoulders are apart
and the arm moves out to its side (her left is on the left of a mirror).
The angles come out exactly as given once converted to pixels.
"""

import math

import numpy as np

from rehab import body, features
from rehab.body import POSE, PoseObservation

IMAGE_SIZE = (1280, 720)
UPPER_ARM = 150.0
FOREARM = 130.0
TORSO = 250.0


def rot(v, deg):
    r = math.radians(deg)
    return np.array([v[0] * math.cos(r) - v[1] * math.sin(r), v[0] * math.sin(r) + v[1] * math.cos(r)])


def pose(shoulder=0.0, elbow=0.0, side="left", view="sagittal", trunk=0.0, hike=0.0,
         abduct_out_of_plane=0.0, wrist_at=None, hidden=(), other_shoulder=0.0,
         other_elbow=0.0, index_at=None, size=IMAGE_SIZE):
    """
    shoulder   shoulder elevation of `side` (degrees)
    elbow      elbow flexion of `side` (degrees)
    trunk      trunk lean from vertical (degrees, forward = +x)
    hike       shoulder raised towards the ear (pixels)
    abduct_out_of_plane  upper arm leaving the image plane (degrees): shorter in the picture
    wrist_at   (x, y) pixels: put the wrist there instead (e.g. at the ear)
    index_at   (x, y) pixels for the index fingertip of `side`
    hidden     landmark names with low visibility
    """
    w, h = size
    pts = np.zeros((33, 2))
    vis = np.full(33, 0.99)
    hip_mid = np.array([w * 0.5, h * 0.85])
    up = rot(np.array([0.0, -1.0]), trunk)
    sh_mid = hip_mid + TORSO * up
    # shoulder width / torso length: facing 0.8, halfway 0.4 (as MediaPipe measures a
    # real 45 degree turn), side-on 0.08
    half_sh = 100.0 if view == "frontal" else (50.0 if view == "oblique" else 10.0)
    half_hip = half_sh          # hip under shoulder: the angles come out exact
    sign = {"left": -1.0, "right": 1.0}          # her left is on the left of the (mirrored) picture
    for s in ("left", "right"):
        pts[POSE[f"{s}_shoulder"]] = sh_mid + [sign[s] * half_sh, 0.0]
        pts[POSE[f"{s}_hip"]] = hip_mid + [sign[s] * half_hip, 0.0]
        pts[POSE[f"{s}_ear"]] = sh_mid + [sign[s] * half_sh * 0.4, -90.0]
    nose = sh_mid + [0.0, -110.0]
    pts[POSE["nose"]] = nose
    pts[POSE["left_eye"]] = nose + [-15.0, -12.0]
    pts[POSE["right_eye"]] = nose + [15.0, -12.0]

    def arm(s, sh_deg, el_deg):
        shoulder_pt = pts[POSE[f"{s}_shoulder"]]
        down = np.array([0.0, 1.0])
        if view == "frontal":
            turn = sh_deg if s == "left" else -sh_deg      # out to her side
        else:
            turn = -sh_deg                                  # forward (+x)
        upper = rot(down, turn) * UPPER_ARM
        if s == side and abduct_out_of_plane:
            upper = upper * math.cos(math.radians(abduct_out_of_plane))
        elbow_pt = shoulder_pt + upper
        fturn = el_deg if (view == "frontal" and s == "left") else -el_deg
        fore = rot(upper / np.linalg.norm(upper), fturn) * FOREARM
        return elbow_pt, elbow_pt + fore

    for s in ("left", "right"):
        if s == side:
            e, wr = arm(s, shoulder, elbow)
        else:
            e, wr = arm(s, other_shoulder, other_elbow)
        pts[POSE[f"{s}_elbow"]] = e
        pts[POSE[f"{s}_wrist"]] = wr
        pts[POSE[f"{s}_index"]] = wr + (wr - e) / max(np.linalg.norm(wr - e), 1e-6) * 30.0
    for j in ("shoulder", "elbow", "wrist", "index"):       # a hiked shoulder lifts the whole arm
        pts[POSE[f"{side}_{j}"]][1] -= hike
    if wrist_at is not None:
        pts[POSE[f"{side}_wrist"]] = np.asarray(wrist_at, float)
    if index_at is not None:
        pts[POSE[f"{side}_index"]] = np.asarray(index_at, float)
    if view == "sagittal":
        other = "right" if side == "left" else "left"
        for j in ("shoulder", "elbow", "wrist", "hip", "ear", "index"):
            vis[POSE[f"{other}_{j}"]] = 0.6         # the far arm is partly hidden
    for name in hidden:
        vis[POSE[name]] = 0.1
    image = np.column_stack([pts[:, 0] / w, pts[:, 1] / h, np.zeros(33)])
    return PoseObservation(image=image, visibility=vis)


def hand_for_wrist(pose_obs, side="left", wrist_ext=0.0, size=IMAGE_SIZE):
    """HandFeatures of `side` whose hand points wrist_ext degrees up from the forearm."""
    w, h = size
    pts = body.to_px(pose_obs.image, w, h)
    elbow, wrist = pts[POSE[f"{side}_elbow"]], pts[POSE[f"{side}_wrist"]]
    fore = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    # rotate towards "up" in the picture
    a, b = rot(fore, wrist_ext), rot(fore, -wrist_ext)
    hand_dir = a if a[1] < b[1] else b
    if wrist_ext < 0:
        hand_dir = a if a[1] > b[1] else b
    img = np.zeros((21, 2))
    img[0] = wrist
    for i in range(1, 21):
        img[i] = wrist + hand_dir * (40.0 + 3.0 * i)
    f = features.HandFeatures(t=0.0, present=True, handedness=side.capitalize(),
                              handedness_score=0.95, correct_hand=True)
    f.image_points = img
    return f


def feat(t, size=IMAGE_SIZE, hand_ext=None, side="left", **kw):
    """BodyFeatures for one frame (unfiltered)."""
    p = pose(side=side, size=size, **kw)
    hands = {}
    if hand_ext is not None:
        hands[side] = hand_for_wrist(p, side, hand_ext, size)
    return body.extract(p, t, size, hands)
