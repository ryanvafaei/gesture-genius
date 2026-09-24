"""
Landmarks -> HandFeatures.

Turns the 21 hand landmarks of one frame into the measurements the exercises
use: joint angles, finger openness, spread between fingers, thumb-to-fingertip
distances, fingertip heights and a few quality flags.

Angles and distances come from the *world* landmarks (metres, centred on the
hand). Unlike the image landmarks they do not change with the distance to the
camera or with perspective. The image landmarks are used only for where the
hand is in the picture (drawing, "move closer", wrist movement) and for the
palm direction, because their axes are fixed: x right, y down, z away from
the camera.

Landmark numbers: 0 wrist; 1-4 thumb (CMC, MCP, IP, tip); 5-8 index;
9-12 middle; 13-16 ring; 17-20 pinky (each MCP, PIP, DIP, tip).
"""

from dataclasses import dataclass, field

import numpy as np

from rehab import config

WRIST = 0
THUMB = (1, 2, 3, 4)
FINGER_LANDMARKS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGERS = tuple(FINGER_LANDMARKS)            # the four long fingers
ALL_FINGERS = ("thumb",) + FINGERS
TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
GAPS = (("index", "middle"), ("middle", "ring"), ("ring", "pinky"))
GAP_NAMES = tuple(f"{a}_{b}" for a, b in GAPS)

# Sum of MCP + PIP + DIP flexion of a fully curled finger (degrees).
# Used to turn curl into a 0..1 "anatomical" openness before calibration.
FULL_CURL_DEG = 250.0

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


# ---------------------------------------------------------------------------
# Input from Sense: one detected hand
# ---------------------------------------------------------------------------

@dataclass
class HandObservation:
    image: np.ndarray            # (21, 3) normalised image coords (x, y in 0..1, z relative)
    world: np.ndarray            # (21, 3) metres, origin at the hand centre
    handedness: str              # "Left" / "Right"
    handedness_score: float
    gesture: str = None          # built-in classifier label, secondary check only
    gesture_score: float = 0.0


# ---------------------------------------------------------------------------
# Output: everything an exercise may look at
# ---------------------------------------------------------------------------

@dataclass
class HandFeatures:
    t: float
    present: bool = False

    # quality
    handedness: str = None
    handedness_score: float = 0.0
    correct_hand: bool = False
    too_small: bool = False
    palm_facing: float = 0.0          # cosine: 1 palm to camera, -1 back to camera

    # geometry
    palm_size: float = 0.0            # metres, wrist -> middle MCP
    palm_width: float = 0.0           # metres, index MCP -> pinky MCP
    palm_normal: np.ndarray = None    # unit vector out of the palm (world frame)
    palm_normal_image: np.ndarray = None

    joint_flexion: dict = field(default_factory=dict)   # finger -> [MCP, PIP, DIP] deg (thumb: CMC, MCP, IP)
    curl: dict = field(default_factory=dict)            # finger -> sum of flexion (deg)
    openness: dict = field(default_factory=dict)        # finger -> 0..1, 1 = straight (uncalibrated)
    spread: dict = field(default_factory=dict)          # "index_middle" ... -> deg
    thumb_tip_dist: dict = field(default_factory=dict)  # finger -> thumb tip distance / palm size
    # the same in the picture (pixels / palm size in pixels): no depth noise
    thumb_tip_dist_image: dict = field(default_factory=dict)
    tip_reach: dict = field(default_factory=dict)       # finger -> |tip - wrist| / |PIP - wrist|
    tip_height: dict = field(default_factory=dict)      # finger -> tip distance from palm plane / palm size
    thumb_flexion: float = 0.0        # thumb MCP + IP flexion (deg)
    thumb_to_pinky_mcp: float = 0.0   # / palm size
    thumb_to_index_mcp: float = 0.0   # / palm size
    # benchmark measures (benchmarks/BENCHMARK_PLAN.md 3.3)
    aperture: float = 0.0             # mean fingertip -> wrist distance / palm size
    tip_to_palm: dict = field(default_factory=dict)     # finger -> tip to palm centre / palm size

    # image space
    image_points: np.ndarray = None   # (21, 2) pixel coordinates
    palm_size_image: float = 0.0      # wrist -> middle MCP / image height
    wrist_image: np.ndarray = None    # pixels

    gesture: str = None
    gesture_score: float = 0.0

    image_size: tuple = None          # (width, height) of the camera image in pixels
    # the other hand's features (two-hand match, finger counting), or None
    other: "HandFeatures" = None

    @property
    def openness_mean(self):
        return float(np.mean([self.openness[f] for f in FINGERS])) if self.openness else 0.0

    @property
    def spread_total(self):
        return float(sum(self.spread.values())) if self.spread else 0.0

    @property
    def pinch_gap(self):
        """Thumb tip to index tip / palm size (FMA item 28 pad-to-pad)."""
        return self.thumb_tip_dist.get("index", float("nan"))

    @property
    def palm_size_px(self):
        if self.image_points is None:
            return 0.0
        return float(np.linalg.norm(self.image_points[9] - self.image_points[WRIST]))

    def quality_problem(self, need_palm_facing=False, need_both=False, any_hand=False):
        """
        First quality problem as a short key, or None when all is fine.

        need_both  the other hand has to be in view too (two-hand exercises)
        any_hand   either hand will do (e.g. pointing in the memory game)
        """
        if not self.present:
            return "no_hand"
        if not self.correct_hand and not any_hand:
            return "wrong_hand"
        if self.too_small:
            return "too_far"
        if need_palm_facing and self.palm_facing < config.MIN_PALM_FACING:
            return "palm_away"
        if need_both and (self.other is None or not self.other.present
                          or self.other.handedness == self.handedness):
            return "no_other_hand"
        return None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0.0


def angle_between(v1, v2):
    """Angle between two vectors in degrees."""
    u1, u2 = _unit(np.asarray(v1, float)), _unit(np.asarray(v2, float))
    return float(np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))))


def joint_angle(a, b, c):
    """Angle at b formed by a-b-c, in degrees (180 = straight)."""
    return angle_between(a - b, c - b)


def flexion(a, b, c):
    """Flexion at b: 0 when a-b-c is straight, grows as the joint bends."""
    return 180.0 - joint_angle(a, b, c)


def palm_normal(points, handedness):
    """
    Unit vector pointing out of the palm.

    The cross product of wrist->index MCP and wrist->pinky MCP points out of
    the back of a right hand and out of the palm of a left hand (x right,
    y down, z away from the camera, mirrored image). Flip it for right hands.
    """
    n = np.cross(points[5] - points[WRIST], points[17] - points[WRIST])
    if handedness == "Right":
        n = -n
    return _unit(n)


def project_on_plane(v, n):
    return v - np.dot(v, n) * n


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def choose_hand(observations, affected_hand=config.AFFECTED_HAND):
    """The observation of the affected hand, or the most confident other hand."""
    if not observations:
        return None
    for obs in observations:
        if obs.handedness == affected_hand:
            return obs
    return max(observations, key=lambda o: o.handedness_score)


def choose_other(observations, chosen):
    """The observation of the other hand (not `chosen`), or None."""
    others = [o for o in observations or [] if o is not chosen]
    return max(others, key=lambda o: o.handedness_score) if others else None


# Finger counting (answers 1-5 without a keyboard). A long finger counts as
# raised when its tip is clearly farther from the wrist than its middle
# joint (tip_reach: about 1.35 straight, 1.0 half bent, 0.6 in a fist) and it
# is not bent far at the knuckles. This does not depend on the small
# flexion angles of a straight finger, which are noisy when the fingers
# point up at the camera (the old openness rule counted mostly the index).
# The thumb also has to stand away from the palm and from the index finger
# (a straight thumb lying along the index finger does not count).
RAISED_REACH = 1.15
RAISED_MAX_BEND_DEG = 90.0        # MCP + PIP flexion
RAISED_THUMB_OPENNESS = 0.6
RAISED_THUMB_DIST = 0.9           # thumb tip -> little finger MCP, in palm sizes
RAISED_THUMB_INDEX_DIST = 0.6     # thumb tip -> index finger MCP, in palm sizes


def finger_raised(f, name):
    """True when this long finger is held up (finger counting)."""
    reach = f.tip_reach.get(name)
    if reach is None:
        return f.openness.get(name, 0.0) > 0.75
    bend = f.joint_flexion.get(name)
    bend = float(bend[0] + bend[1]) if bend is not None else 0.0
    return reach > RAISED_REACH and bend < RAISED_MAX_BEND_DEG


def thumb_raised(f):
    return (f.openness.get("thumb", 0.0) > RAISED_THUMB_OPENNESS
            and f.thumb_to_pinky_mcp > RAISED_THUMB_DIST
            and f.thumb_to_index_mcp > RAISED_THUMB_INDEX_DIST)


def count_extended(f):
    """Number of raised fingers (0-5) of one hand, or 0 when it is not seen."""
    if f is None or not f.present or not f.openness:
        return 0
    n = sum(1 for name in FINGERS if finger_raised(f, name))
    return n + (1 if thumb_raised(f) else 0)


def extract(obs, t, image_size, affected_hand=config.AFFECTED_HAND):
    """
    Compute HandFeatures for one hand.

    obs         HandObservation or None
    t           timestamp in seconds
    image_size  (width, height) in pixels
    """
    f = HandFeatures(t=t, image_size=tuple(image_size))
    if obs is None:
        return f

    w = np.asarray(obs.world, dtype=float)
    width, height = image_size
    img = np.asarray(obs.image, dtype=float)
    # z of image landmarks uses roughly the same scale as x
    img3 = img * np.array([width, height, width])

    f.present = True
    f.handedness = obs.handedness
    f.handedness_score = float(obs.handedness_score)
    f.correct_hand = (obs.handedness == affected_hand
                      and obs.handedness_score >= config.MIN_HANDEDNESS_SCORE)
    f.gesture = obs.gesture
    f.gesture_score = float(obs.gesture_score or 0.0)

    # --- image space -----------------------------------------------------
    f.image_points = img3[:, :2]
    f.wrist_image = img3[WRIST, :2].copy()
    f.palm_size_image = float(np.linalg.norm(img3[9, :2] - img3[WRIST, :2]) / height)
    f.too_small = f.palm_size_image < config.MIN_PALM_SIZE_IMAGE
    f.palm_normal_image = palm_normal(img3, obs.handedness)
    f.palm_facing = float(-f.palm_normal_image[2])      # camera looks along +z

    # --- world space -----------------------------------------------------
    f.palm_size = float(np.linalg.norm(w[9] - w[WRIST]))
    f.palm_width = float(np.linalg.norm(w[17] - w[5]))
    scale = f.palm_size if f.palm_size > 1e-6 else 1.0
    n = palm_normal(w, obs.handedness)
    f.palm_normal = n

    # joint flexion and openness
    t1, t2, t3, t4 = THUMB
    f.joint_flexion["thumb"] = np.array([
        flexion(w[WRIST], w[t1], w[t2]),
        flexion(w[t1], w[t2], w[t3]),
        flexion(w[t2], w[t3], w[t4]),
    ])
    for name, (mcp, pip, dip, tip) in FINGER_LANDMARKS.items():
        f.joint_flexion[name] = np.array([
            flexion(w[WRIST], w[mcp], w[pip]),
            flexion(w[mcp], w[pip], w[dip]),
            flexion(w[pip], w[dip], w[tip]),
        ])
    for name, angles in f.joint_flexion.items():
        f.curl[name] = float(np.sum(angles))
    for name in FINGERS:
        f.openness[name] = float(np.clip(1.0 - f.curl[name] / FULL_CURL_DEG, 0.0, 1.0))
    # thumb: MCP + IP only, about 150 deg when fully bent
    f.thumb_flexion = float(f.joint_flexion["thumb"][1] + f.joint_flexion["thumb"][2])
    f.openness["thumb"] = float(np.clip(1.0 - f.thumb_flexion / 150.0, 0.0, 1.0))

    # spread between neighbouring fingers, in the palm plane
    directions = {}
    for name, (mcp, _, _, tip) in FINGER_LANDMARKS.items():
        directions[name] = project_on_plane(w[tip] - w[mcp], n)
    for (a, b), gap in zip(GAPS, GAP_NAMES):
        f.spread[gap] = angle_between(directions[a], directions[b])

    # thumb distances
    for name in FINGERS:
        f.thumb_tip_dist[name] = float(np.linalg.norm(w[TIPS["thumb"]] - w[TIPS[name]]) / scale)
    f.thumb_to_pinky_mcp = float(np.linalg.norm(w[TIPS["thumb"]] - w[17]) / scale)
    f.thumb_to_index_mcp = float(np.linalg.norm(w[TIPS["thumb"]] - w[5]) / scale)
    palm_px = float(np.linalg.norm(img3[9, :2] - img3[WRIST, :2]))
    palm_px = palm_px if palm_px > 1e-6 else 1.0
    for name in FINGERS:
        f.thumb_tip_dist_image[name] = float(
            np.linalg.norm(img3[TIPS["thumb"], :2] - img3[TIPS[name], :2]) / palm_px)
    for name, (mcp, pip, dip, tip) in FINGER_LANDMARKS.items():
        pip_d = float(np.linalg.norm(w[pip] - w[WRIST]))
        f.tip_reach[name] = float(np.linalg.norm(w[tip] - w[WRIST]) / pip_d) if pip_d > 1e-9 else 0.0

    # hand aperture and fingertips to the palm centre (FMA items 24 and 25)
    f.aperture = float(np.mean([np.linalg.norm(w[t] - w[WRIST]) for t in TIPS.values()]) / scale)
    centre = w[[WRIST, 5, 9, 13, 17]].mean(axis=0)
    for name in ALL_FINGERS:
        f.tip_to_palm[name] = float(np.linalg.norm(w[TIPS[name]] - centre) / scale)

    # fingertip height above the palm plane, positive towards the back of the hand
    for name in ALL_FINGERS:
        f.tip_height[name] = float(np.dot(w[TIPS[name]] - w[WRIST], -n) / scale)

    return f


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

_SCALARS = ("palm_facing", "palm_size", "palm_width", "thumb_flexion",
            "thumb_to_pinky_mcp", "thumb_to_index_mcp", "palm_size_image", "aperture")
_DICTS = ("curl", "openness", "spread", "thumb_tip_dist", "thumb_tip_dist_image", "tip_height",
          "joint_flexion", "tip_to_palm", "tip_reach")
_ARRAYS = ("image_points", "wrist_image", "palm_normal", "palm_normal_image")


def smooth(features, feature_filter):
    """
    Run every numeric feature through its own One Euro filter.

    Filtering the features (not only the landmarks) lets each one adapt to its
    own speed: the moving finger stays responsive while the still ones are
    smoothed. Quality flags are not filtered. Returns the same object.
    """
    if not features.present:
        return features
    t = features.t
    for name in _SCALARS:
        setattr(features, name, feature_filter(name, getattr(features, name), t))
    for name in _DICTS:
        d = getattr(features, name)
        for key in d:
            d[key] = feature_filter(f"{name}.{key}", d[key], t)
    for name in _ARRAYS:
        v = getattr(features, name)
        if v is not None:
            setattr(features, name, feature_filter(name, v, t))
    if features.palm_normal is not None:
        features.palm_normal = _unit(features.palm_normal)
    features.too_small = features.palm_size_image < config.MIN_PALM_SIZE_IMAGE
    return features
