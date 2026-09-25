"""
Pose landmarks -> BodyFeatures: the arm, trunk and head measures the arm
exercises use (benchmarks/BENCHMARK_PLAN.md, section 3).

  shoulder_elevation   angle(hip, shoulder, elbow); 0 = arm at the side
                       (Lazem 2026; approximates Gates 2016 humeral elevation).
                       Seated at a table the hips are usually hidden or below
                       the picture; MediaPipe then guesses them, and a guess
                       tilts the angle by up to 20 degrees. Without visible
                       hips the trunk is taken as upright (the image vertical).
  elbow_flexion        180 - angle(shoulder, elbow, wrist); 0 = straight
  wrist_extension      forearm (pose elbow -> wrist) against the hand (hand
                       wrist -> middle MCP); positive = hand lifted up, for a
                       forearm resting roughly level (Jayavel 2025)
  trunk_angle          hip middle -> shoulder middle against the vertical;
                       the exercises use its change from the start (NaN
                       while the hips are not seen)
  ear_gap              ear above shoulder (pixels): the shoulder hiking up
                       makes it smaller (FMA item 15)
  upper_arm_len        shoulder -> elbow in the picture: shorter than at the
                       start = the arm left the movement plane
  wrist_to_ear         / torso length (hand to head, FMA item 7)
  wrist_drop           wrist below the shoulder / torso length (hand in the
                       lap or hanging down: about 0.6 or more)
  nose_error           index fingertip to nose / distance between the eyes
                       (finger to nose, FMA item 32)

A measure is NaN while one of its landmarks is not seen well enough
(visibility below visibility_min): MediaPipe always returns all 33 points,
and a guessed point (an arm out of the picture, the far arm side-on) must
not count as a movement.

All 2D measures are in pixels (x * width, y * height), never in normalised
coordinates: with a 16:9 image a 45 degree arm would read more than 5
degrees wrong. Pose "left" / "right" is the person's own side, because the
landmarker sees the camera's unmirrored image (see Sense); the points are
mirrored afterwards, like the hand landmarks, to match the display.
The pose model sometimes swaps left and right for a frame, mostly side-on;
canonical_sides() puts them back from the geometry (which way she faces,
which arm is nearest the camera), so the trained arm stays the trained arm.
Hands are given to an arm by where they are (the hand's wrist next to that
arm's wrist, assign_hands), not by MediaPipe's handedness label, which is
unreliable for a hand seen from the side.

Landmark numbers (MediaPipe Pose): nose 0, eyes 2/5, ears 7/8, shoulders
11/12, elbows 13/14, wrists 15/16, index fingers 19/20, hips 23/24.
"""

from dataclasses import dataclass, field

import numpy as np

from rehab import config

POSE = {
    "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_index": 19, "right_index": 20,
    "left_hip": 23, "right_hip": 24,
}
SIDES = ("left", "right")

# left/right pairs of the 33 pose landmarks (eyes, ears, mouth, arms, hands, legs)
LR_PAIRS = ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10)) + tuple((i, i + 1) for i in range(11, 33, 2))
_SWAP = np.arange(33)
for _a, _b in LR_PAIRS:
    _SWAP[_a], _SWAP[_b] = _b, _a

# lines drawn for the skeleton: the trunk, then each arm
TRUNK_LINES = (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
               ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"))
ARM_LINES = (("shoulder", "elbow"), ("elbow", "wrist"))

# measures smoothed with the 6 Hz low-pass filter
ARM_MEASURES = ("shoulder_elevation", "elbow_flexion", "wrist_extension", "upper_arm_len",
                "ear_gap", "wrist_to_ear", "wrist_drop", "nose_error")


@dataclass
class PoseObservation:
    image: np.ndarray            # (33, 3) normalised image coords (z: depth, same scale as x)
    visibility: np.ndarray       # (33,) 0..1


@dataclass
class BodyFeatures:
    t: float
    present: bool = False
    image_size: tuple = (0, 0)
    points: np.ndarray = None           # (33, 2) pixels, display (mirrored) coordinates
    visibility: np.ndarray = None       # (33,)
    depth: np.ndarray = None            # (33,) pixels, smaller = nearer the camera
    hips_visible: bool = False          # both hips seen: the trunk axis is hip -> shoulder
    sides_swapped: bool = False         # the model's left/right were swapped back this frame
    arm: dict = field(default_factory=dict)      # side -> {measure: value}
    trunk_angle: float = float("nan")   # degrees from vertical, signed
    shoulder_mid: np.ndarray = None
    hip_mid: np.ndarray = None
    shoulder_width: float = 0.0         # pixels
    torso_len: float = 0.0              # pixels, hip middle -> shoulder middle
    view_ratio: float = float("nan")    # shoulder width / torso length
    view: str = None                    # "frontal", "sagittal", "oblique"
    hands: dict = field(default_factory=dict)    # side -> HandFeatures of that hand
    gesture: str = None
    gesture_score: float = 0.0

    def point(self, name):
        return self.points[POSE[name]]

    def visible(self, names, min_visibility=None):
        """True when every named landmark is seen well enough."""
        if not self.present or self.visibility is None:
            return False
        v = _min_visibility() if min_visibility is None else min_visibility
        return all(self.visibility[POSE[n]] >= v for n in names)

    def near_edge(self, names, margin=None):
        """Names of landmarks closer to the image edge than margin (part of the arm may be cut off)."""
        if not self.present:
            return []
        m = config.SETUP_EDGE_MARGIN if margin is None else margin
        w, h = self.image_size
        out = []
        for n in names:
            x, y = self.points[POSE[n]]
            if x < m * w or x > (1 - m) * w or y < m * h or y > (1 - m) * h:
                out.append(n)
        return out

    def seen(self, name):
        """One landmark seen well enough (visibility gate)."""
        return (self.present and self.visibility is not None
                and self.visibility[POSE[name]] >= _min_visibility())

    def quality_problem(self, need_palm_facing=False, **_):
        """Like HandFeatures.quality_problem: only "no_body" here; exercises check their own landmarks."""
        return None if self.present else "no_body"


def _min_visibility():
    """Visibility gate (plan 3.4, proposed 0.5), kept in benchmarks.json."""
    from rehab import benchmarks
    return float(benchmarks.load().signal("visibility_min"))


# ---------------------------------------------------------------------------
# Geometry (pixels)
# ---------------------------------------------------------------------------

def to_px(landmarks_norm, width, height):
    """(N, 2|3) normalised landmarks -> (N, 2) pixels. Required before any 2D angle."""
    lm = np.asarray(landmarks_norm, dtype=float)
    return np.stack([lm[:, 0] * width, lm[:, 1] * height], axis=1)


def angle_at(a, b, c):
    """Unsigned angle abc in degrees (2D or 3D); NaN when two points coincide."""
    ba = np.asarray(a, float) - np.asarray(b, float)
    bc = np.asarray(c, float) - np.asarray(b, float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def signed_angle_2d(v1, v2):
    """Signed angle from v1 to v2 in degrees, (-180, 180]."""
    return float(np.degrees(np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])))


def shoulder_elevation(pts, side, upright=False):
    """
    Angle between the trunk (shoulder -> hip) and the upper arm; with
    upright=True the trunk is taken as the image vertical (hips not seen).
    """
    sh = pts[POSE[f"{side}_shoulder"]]
    hip = sh + np.array([0.0, 100.0]) if upright else pts[POSE[f"{side}_hip"]]
    return angle_at(hip, sh, pts[POSE[f"{side}_elbow"]])


def elbow_flexion(pts, side):
    return 180.0 - angle_at(pts[POSE[f"{side}_shoulder"]], pts[POSE[f"{side}_elbow"]],
                            pts[POSE[f"{side}_wrist"]])


def trunk_angle(pts):
    """Signed angle of hip middle -> shoulder middle against the image vertical (y points down)."""
    hip = (pts[POSE["left_hip"]] + pts[POSE["right_hip"]]) / 2
    sh = (pts[POSE["left_shoulder"]] + pts[POSE["right_shoulder"]]) / 2
    return signed_angle_2d(np.array([0.0, -1.0]), sh - hip)


def ear_gap(pts, side):
    """Pixels the ear is above the shoulder."""
    return float(pts[POSE[f"{side}_shoulder"]][1] - pts[POSE[f"{side}_ear"]][1])


def upper_arm_len(pts, side):
    return float(np.linalg.norm(pts[POSE[f"{side}_elbow"]] - pts[POSE[f"{side}_shoulder"]]))


def wrist_extension(elbow, wrist, hand_wrist, hand_middle_mcp):
    """
    Angle of the hand against the forearm in the picture, positive when
    the hand points more upwards than the forearm (dorsal extension with the
    forearm resting level, palm down). NaN when either segment has no length.
    """
    forearm = np.asarray(wrist, float) - np.asarray(elbow, float)
    hand = np.asarray(hand_middle_mcp, float) - np.asarray(hand_wrist, float)
    if np.linalg.norm(forearm) < 1e-6 or np.linalg.norm(hand) < 1e-6:
        return float("nan")
    u = forearm / np.linalg.norm(forearm)
    up = np.array([u[1], -u[0]])            # perpendicular to the forearm ...
    if up[1] > 0:
        up = -up                            # ... pointing up in the picture
    return float(np.degrees(np.arctan2(np.dot(hand, up), np.dot(hand, u))))


def estimate_view(ratio):
    """Camera view from shoulder width / torso length (thresholds proposed)."""
    if not np.isfinite(ratio):
        return None
    if ratio <= config.VIEW_SAGITTAL_MAX:
        return "sagittal"
    if ratio >= config.VIEW_FRONTAL_MIN:
        return "frontal"
    return "oblique"


def view_accepted(ratio, view):
    """
    True when a camera view is good enough for an exercise that needs
    `view`. Wider than estimate_view's bands, so a small turn of the chair
    or a narrow pair of shoulders does not stop the exercise.
    """
    if view is None:
        return True
    if not np.isfinite(ratio):
        return False
    lo, hi = config.VIEW_ACCEPT[view]
    return lo <= ratio <= hi


# ---------------------------------------------------------------------------
# Left and right, near and far
# ---------------------------------------------------------------------------

def _facing(pts, vis, v_min):
    """
    +1 when she faces the right of the (mirrored) picture, -1 the left,
    0 when it cannot be told (facing the camera, or the head not seen).
    Side-on the nose is clearly in front of the ears.
    """
    if vis[POSE["nose"]] < v_min:
        return 0
    ears = [pts[POSE[e]] for e in ("left_ear", "right_ear") if vis[POSE[e]] >= v_min * 0.5]
    if not ears:
        return 0
    ear_mid = np.mean(ears, axis=0)
    head = max(float(np.linalg.norm(pts[POSE["nose"]] - ear_mid)), 1e-6)
    dx = float(pts[POSE["nose"]][0] - ear_mid[0])
    shoulders = abs(float(pts[POSE["left_shoulder"]][0] - pts[POSE["right_shoulder"]][0]))
    if abs(dx) < 0.6 * head or abs(dx) < 0.5 * shoulders:
        return 0
    return 1 if dx > 0 else -1


def _nearer(pts, vis, depth, v_min):
    """
    The side ("left"/"right") whose arm is nearer the camera, or None:
    by depth when the model gives it, else by how well each arm is seen
    (the far arm is hidden behind the body).
    """
    def arm(side):
        return [POSE[f"{side}_{j}"] for j in ("shoulder", "elbow", "wrist")]

    if depth is not None:
        dz = float(np.mean(depth[arm("left")]) - np.mean(depth[arm("right")]))
        span = float(np.linalg.norm(pts[POSE["left_shoulder"]] - pts[POSE["left_elbow"]]))
        if abs(dz) > 0.25 * max(span, 1.0):
            return "left" if dz < 0 else "right"
    dv = float(np.mean(vis[arm("left")]) - np.mean(vis[arm("right")]))
    if abs(dv) > 0.15:
        return "left" if dv > 0 else "right"
    return None


def near_side(f):
    """
    The side nearest the camera in a side-on view: from which way she
    faces (facing the right of the mirrored picture = her left side is
    nearest), else from depth and visibility. None when it cannot be told.
    """
    if not f.present:
        return None
    v_min = _min_visibility()
    facing = _facing(f.points, f.visibility, v_min)
    if facing:
        return "left" if facing > 0 else "right"
    return _nearer(f.points, f.visibility, f.depth, v_min)


def canonical_sides(pts, vis, depth, torso_len):
    """
    (pts, vis, depth, swapped) with left and right put back when the model
    swapped them in this frame. Facing the camera (shoulders apart), her
    left shoulder is on the left of the mirrored picture. Side-on, the arm
    nearest the camera is on the side she turns towards the camera: facing
    the right of the mirrored picture means her left side is nearest.
    Only changed on clear evidence; otherwise the model's labels stay.
    """
    v_min = _min_visibility()
    ls, rs = pts[POSE["left_shoulder"]], pts[POSE["right_shoulder"]]
    apart = abs(float(ls[0] - rs[0]))
    swap = False
    if torso_len > 1e-6 and apart >= config.SIDES_APART_TORSO * torso_len:
        swap = ls[0] > rs[0]
    else:
        facing = _facing(pts, vis, v_min)
        nearer = _nearer(pts, vis, depth, v_min)
        if facing and nearer:
            swap = nearer != ("left" if facing > 0 else "right")
    if not swap:
        return pts, vis, depth, False
    return pts[_SWAP], vis[_SWAP], (depth[_SWAP] if depth is not None else None), True


def assign_hands(observations, f):
    """
    {side: HandObservation}: each hand seen goes to the arm whose pose wrist
    it is next to (within a forearm's length), whatever its handedness label
    says; the label is set to match, so the palm direction is computed the
    right way round. Without a body the labels are used.
    """
    from dataclasses import replace
    obs = [o for o in observations or [] if o is not None]
    if not obs:
        return {}
    if not f.present:
        out = {}
        for o in sorted(obs, key=lambda o: o.handedness_score, reverse=True):
            if o.handedness in ("Left", "Right"):
                out.setdefault(o.handedness.lower(), o)
        return out
    w, h = f.image_size
    pairs = []
    for i, o in enumerate(obs):
        wrist = np.array([o.image[0][0] * w, o.image[0][1] * h])
        for side in SIDES:
            pw, pe = f.point(f"{side}_wrist"), f.point(f"{side}_elbow")
            reach = max(float(np.linalg.norm(pw - pe)), 0.1 * h)
            d = float(np.linalg.norm(wrist - pw))
            if d <= reach:
                pairs.append((d / reach, i, side))
    out, used = {}, set()
    for _, i, side in sorted(pairs):
        if i in used or side in out:
            continue
        used.add(i)
        o = obs[i]
        label = side.capitalize()
        out[side] = o if o.handedness == label else replace(o, handedness=label)
    return out


class HipGate:
    """
    Whether the hips are seen, with hysteresis: the trunk axis (hips, or
    upright when they are not seen) must not flip every frame while their
    visibility hovers around the gate, or the shoulder angle would jump.
    """

    def __init__(self, on=None, off=None):
        v = _min_visibility()
        self.on = v + 0.1 if on is None else on
        self.off = v - 0.1 if off is None else off
        self.state = False

    def __call__(self, level):
        self.state = level >= (self.off if self.state else self.on)
        return self.state


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract(pose, t, image_size, hands=None, gesture=(None, 0.0), hip_gate=None):
    """
    BodyFeatures for one frame.

    pose        PoseObservation (display coordinates) or None
    image_size  (width, height) in pixels
    hands       {"left": HandFeatures, "right": HandFeatures} of the hands
                seen this frame (either may be missing; see assign_hands)
    hip_gate    HipGate kept by the caller (hysteresis), or None
    """
    f = BodyFeatures(t=t, image_size=tuple(image_size), hands=dict(hands or {}))
    f.gesture, f.gesture_score = gesture
    if pose is None:
        return f
    width, height = image_size
    pts = to_px(pose.image, width, height)
    vis = np.asarray(pose.visibility, dtype=float)
    image = np.asarray(pose.image, dtype=float)
    depth = image[:, 2] * width if image.shape[1] > 2 else None
    v_min = _min_visibility()
    torso_guess = float(np.linalg.norm((pts[POSE["left_shoulder"]] + pts[POSE["right_shoulder"]]) / 2
                                       - (pts[POSE["left_hip"]] + pts[POSE["right_hip"]]) / 2))
    pts, vis, depth, f.sides_swapped = canonical_sides(pts, vis, depth, torso_guess)
    f.present = True
    f.points = pts
    f.visibility = vis
    f.depth = depth
    f.shoulder_mid = (pts[POSE["left_shoulder"]] + pts[POSE["right_shoulder"]]) / 2
    f.hip_mid = (pts[POSE["left_hip"]] + pts[POSE["right_hip"]]) / 2
    f.shoulder_width = float(np.linalg.norm(pts[POSE["left_shoulder"]] - pts[POSE["right_shoulder"]]))
    # the hips' position is guessed when they are hidden (under the table,
    # below the picture): still a usable length, but not a usable direction
    f.torso_len = float(np.linalg.norm(f.shoulder_mid - f.hip_mid))
    hip_level = min(vis[POSE["left_hip"]], vis[POSE["right_hip"]])
    if f.hip_mid[1] > height:
        hip_level = 0.0
    f.hips_visible = hip_gate(hip_level) if hip_gate is not None else hip_level >= v_min
    f.trunk_angle = trunk_angle(pts) if f.hips_visible else float("nan")
    f.view_ratio = f.shoulder_width / f.torso_len if f.torso_len > 1e-6 else float("nan")
    f.view = estimate_view(f.view_ratio)
    torso = max(f.torso_len, 1e-6)
    nan = float("nan")

    def seen(*names):
        return all(vis[POSE[n]] >= v_min for n in names)

    for side in SIDES:
        sh, el, wr, ear = (f"{side}_{j}" for j in ("shoulder", "elbow", "wrist", "ear"))
        arm = {
            "shoulder_elevation": (shoulder_elevation(pts, side, upright=not f.hips_visible)
                                   if seen(sh, el) else nan),
            "elbow_flexion": elbow_flexion(pts, side) if seen(sh, el, wr) else nan,
            "upper_arm_len": upper_arm_len(pts, side) if seen(sh, el) else nan,
            "ear_gap": ear_gap(pts, side) if seen(sh, ear) else nan,
            "wrist_to_ear": (float(np.linalg.norm(pts[POSE[wr]] - pts[POSE[ear]]) / torso)
                             if seen(wr, ear) else nan),
            "wrist_drop": (float((pts[POSE[wr]][1] - pts[POSE[sh]][1]) / torso)
                           if seen(sh, wr) else nan),
        }
        f.arm[side] = arm
    attach_hands(f, f.hands)
    return f


def attach_hands(f, hands):
    """
    Give f the hands seen this frame ({side: HandFeatures}) and the
    measures that need a hand: wrist extension (pose forearm against the
    hand) and the fingertip for finger to nose (the hand's index tip when
    the hand is tracked, else the pose's index point).
    """
    f.hands = dict(hands or {})
    if not f.present:
        return f
    pts, v_min, nan = f.points, _min_visibility(), float("nan")
    eyes = float(np.linalg.norm(pts[POSE["left_eye"]] - pts[POSE["right_eye"]]))

    def seen(*names):
        return all(f.visibility[POSE[n]] >= v_min for n in names)

    for side in SIDES:
        arm = f.arm.setdefault(side, {})
        el, wr = f"{side}_elbow", f"{side}_wrist"
        arm["wrist_extension"] = nan
        tip = pts[POSE[f"{side}_index"]] if seen(f"{side}_index") or seen(wr) else None
        hand = f.hands.get(side)
        if hand is not None and getattr(hand, "present", False) and hand.image_points is not None:
            hp = hand.image_points
            if seen(el, wr):
                arm["wrist_extension"] = wrist_extension(pts[POSE[el]], pts[POSE[wr]], hp[0], hp[9])
            tip = hp[8]
        arm["nose_error"] = (float(np.linalg.norm(tip - pts[POSE["nose"]]) / eyes)
                             if tip is not None and eyes > 1e-6 and seen("nose") else nan)
    return f


def smooth(features, bank):
    """6 Hz low-pass on every arm measure and the trunk angle (LowPassBank). Returns the same object."""
    if not features.present:
        return features
    t = features.t
    features.trunk_angle = bank("trunk_angle", features.trunk_angle, t)
    for side, arm in features.arm.items():
        for name in ARM_MEASURES:
            if name in arm:
                arm[name] = bank(f"{side}.{name}", arm[name], t)
    return features


# ---------------------------------------------------------------------------
# Setup check (plan 3.1): framing and camera view before an arm exercise
# ---------------------------------------------------------------------------

def setup_problem(f, required, view, side):
    """
    None when the camera sees what the exercise needs, else a short key:
      no_body       nobody in the picture
      arm_hidden    a needed landmark is not seen well enough
      move_back     a needed landmark is at the edge of the picture
      turn_side     side-on view needed, with `side` nearest the camera
      face_camera   she should face the camera
      turn_45       halfway between facing the camera and side-on
    Lazem 2026 found most home recordings unusable without such guidance.
    """
    if not f.present:
        return "no_body"
    if not f.visible(required):
        # a point the model guesses outside the picture: she is too close
        hidden = [n for n in required if not f.seen(n)]
        return "move_back" if f.near_edge(hidden) else "arm_hidden"
    if f.near_edge(required):
        return "move_back"
    if view == "sagittal":
        if not view_accepted(f.view_ratio, "sagittal") or near_side(f) not in (side, None):
            return "turn_side"
    elif view == "frontal" and not view_accepted(f.view_ratio, "frontal"):
        return "face_camera"
    elif view == "oblique" and not view_accepted(f.view_ratio, "oblique"):
        return "turn_45"
    return None
