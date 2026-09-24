"""
Pose landmarks -> BodyFeatures: the arm, trunk and head measures the arm
exercises use (benchmarks/BENCHMARK_PLAN.md, section 3).

  shoulder_elevation   angle(hip, shoulder, elbow); 0 = arm at the side
                       (Lazem 2026; approximates Gates 2016 humeral elevation)
  elbow_flexion        180 - angle(shoulder, elbow, wrist); 0 = straight
  wrist_extension      forearm (pose elbow -> wrist) against the hand (hand
                       wrist -> middle MCP); positive = hand lifted up, for a
                       forearm resting roughly level (Jayavel 2025)
  trunk_angle          hip middle -> shoulder middle against the vertical;
                       the exercises use its change from the start
  ear_gap              ear above shoulder (pixels): the shoulder hiking up
                       makes it smaller (FMA item 15)
  upper_arm_len        shoulder -> elbow in the picture: shorter than at the
                       start = the arm left the movement plane
  wrist_to_ear         / torso length (hand to head, FMA item 7)
  wrist_drop           wrist below the shoulder / torso length (hand in the
                       lap or hanging down: about 0.6 or more)
  nose_error           index fingertip to nose / distance between the eyes
                       (finger to nose, FMA item 32)

All 2D measures are in pixels (x * width, y * height), never in normalised
coordinates: with a 16:9 image a 45 degree arm would read more than 5
degrees wrong. Pose "left" / "right" is the person's own side, because the
landmarker sees the camera's unmirrored image (see Sense); the points are
mirrored afterwards, like the hand landmarks, to match the display.

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

# lines drawn for the skeleton: the trunk, then each arm
TRUNK_LINES = (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
               ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"))
ARM_LINES = (("shoulder", "elbow"), ("elbow", "wrist"))

# measures smoothed with the 6 Hz low-pass filter
ARM_MEASURES = ("shoulder_elevation", "elbow_flexion", "wrist_extension", "upper_arm_len",
                "ear_gap", "wrist_to_ear", "wrist_drop", "nose_error")


@dataclass
class PoseObservation:
    image: np.ndarray            # (33, 3) normalised image coords
    visibility: np.ndarray       # (33,) 0..1


@dataclass
class BodyFeatures:
    t: float
    present: bool = False
    image_size: tuple = (0, 0)
    points: np.ndarray = None           # (33, 2) pixels, display (mirrored) coordinates
    visibility: np.ndarray = None       # (33,)
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

    def quality_problem(self, need_palm_facing=False):
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


def shoulder_elevation(pts, side):
    return angle_at(pts[POSE[f"{side}_hip"]], pts[POSE[f"{side}_shoulder"]], pts[POSE[f"{side}_elbow"]])


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


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract(pose, t, image_size, hands=None, gesture=(None, 0.0)):
    """
    BodyFeatures for one frame.

    pose        PoseObservation (display coordinates) or None
    image_size  (width, height) in pixels
    hands       {"left": HandFeatures, "right": HandFeatures} of the hands
                seen this frame (either may be missing)
    """
    f = BodyFeatures(t=t, image_size=tuple(image_size), hands=dict(hands or {}))
    f.gesture, f.gesture_score = gesture
    if pose is None:
        return f
    width, height = image_size
    pts = to_px(pose.image, width, height)
    f.present = True
    f.points = pts
    f.visibility = np.asarray(pose.visibility, dtype=float)
    f.shoulder_mid = (pts[POSE["left_shoulder"]] + pts[POSE["right_shoulder"]]) / 2
    f.hip_mid = (pts[POSE["left_hip"]] + pts[POSE["right_hip"]]) / 2
    f.shoulder_width = float(np.linalg.norm(pts[POSE["left_shoulder"]] - pts[POSE["right_shoulder"]]))
    f.torso_len = float(np.linalg.norm(f.shoulder_mid - f.hip_mid))
    f.trunk_angle = trunk_angle(pts)
    f.view_ratio = f.shoulder_width / f.torso_len if f.torso_len > 1e-6 else float("nan")
    f.view = estimate_view(f.view_ratio)
    torso = max(f.torso_len, 1e-6)
    eyes = float(np.linalg.norm(pts[POSE["left_eye"]] - pts[POSE["right_eye"]]))
    for side in SIDES:
        arm = {
            "shoulder_elevation": shoulder_elevation(pts, side),
            "elbow_flexion": elbow_flexion(pts, side),
            "upper_arm_len": upper_arm_len(pts, side),
            "ear_gap": ear_gap(pts, side),
            "wrist_to_ear": float(np.linalg.norm(pts[POSE[f"{side}_wrist"]]
                                                 - pts[POSE[f"{side}_ear"]]) / torso),
            "wrist_drop": float((pts[POSE[f"{side}_wrist"]][1]
                                 - pts[POSE[f"{side}_shoulder"]][1]) / torso),
            "wrist_extension": float("nan"),
        }
        hand = f.hands.get(side)
        tip = pts[POSE[f"{side}_index"]]
        if hand is not None and getattr(hand, "present", False) and hand.image_points is not None:
            hp = hand.image_points
            arm["wrist_extension"] = wrist_extension(pts[POSE[f"{side}_elbow"]],
                                                     pts[POSE[f"{side}_wrist"]], hp[0], hp[9])
            tip = hp[8]
        arm["nose_error"] = (float(np.linalg.norm(tip - pts[POSE["nose"]]) / eyes)
                             if eyes > 1e-6 else float("nan"))
        f.arm[side] = arm
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
        return "arm_hidden"
    if f.near_edge(required):
        return "move_back"
    if view == "sagittal":
        other = "right" if side == "left" else "left"
        near = [f.visibility[POSE[f"{side}_{j}"]] for j in ("shoulder", "elbow", "wrist")]
        far = [f.visibility[POSE[f"{other}_{j}"]] for j in ("shoulder", "elbow", "wrist")]
        if f.view != "sagittal" or np.mean(near) < np.mean(far):
            return "turn_side"
    elif view == "frontal" and f.view != "frontal":
        return "face_camera"
    elif view == "oblique" and f.view != "oblique":
        return "turn_45"
    return None
