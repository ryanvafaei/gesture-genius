"""
Settings for the hand rehabilitation coach.

Everything a therapist might want to change lives here: which hand is
trained, how long holds last, how many sets and reps, the thresholds for
each exercise and whether the program is allowed to raise targets by itself.

All thresholds are fractions of Eleanor's own calibrated range
(0.0 = the least she could do during calibration, 1.0 = the most),
never of a healthy hand.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "gesture_recognizer.task"
DATA_DIR = ROOT_DIR / "data"
PROFILE_PATH = DATA_DIR / "profile.json"
REP_LOG_PATH = DATA_DIR / "reps.csv"
HISTORY_PATH = DATA_DIR / "history.csv"

# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

USER_NAME = "Eleanor"
# Hand to train. Eleanor has left-sided hemiparesis.
AFFECTED_HAND = "Left"

# ---------------------------------------------------------------------------
# Camera and tracking
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0
# The image is shown mirrored (like a mirror, easier to follow).
# MediaPipe Tasks labels the real hand ("Left"/"Right") when it sees the
# camera's own, unmirrored view. Tested on MediaPipe's sample photos:
# a raised left hand is labelled "Left" unmirrored and "Right" mirrored.
# So the recognizer gets the unmirrored frame and the landmarks are mirrored
# afterwards. If `python -m tools.tracking_check` shows "Right" while you hold
# up your left hand, set this to False.
RECOGNIZE_UNMIRRORED = True
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MIRROR = True
NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Quality checks
MIN_HANDEDNESS_SCORE = 0.6
# Hand smaller than this (wrist -> middle MCP, as a fraction of the image
# height) is considered too far away.
MIN_PALM_SIZE_IMAGE = 0.07
# Palm-facing score is the cosine between the palm normal and the direction
# towards the camera: 1 = palm faces the camera, -1 = back of the hand does.
MIN_PALM_FACING = 0.45
# Seconds a problem has to last before the coach mentions it.
QUALITY_GRACE_S = 1.5

# One Euro filter (applied to every feature). Lower min_cutoff = smoother when
# still; higher beta = less lag when moving.
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.05
ONE_EURO_D_CUTOFF = 1.0

# ---------------------------------------------------------------------------
# Speech and pacing (slower movement, mild cognitive impairment)
# ---------------------------------------------------------------------------

SPEECH_ENABLED = True
SPEECH_RATE = 140          # words per minute (`say -r 140`)
HINT_DELAY_S = 8.0         # no progress for this long -> one short hint
HINT_REPEAT_S = 10.0       # at most one hint per this many seconds
QUALITY_MESSAGE_REPEAT_S = 8.0
REST_BETWEEN_SETS_S = 30
REST_BETWEEN_EXERCISES_S = 45
FATIGUE_REST_S = 60
RECALIBRATION_OFFER_S = 6  # press space within this time to recalibrate

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

CALIBRATION_SETTLE_S = 2.0   # time to get into position after the prompt
CALIBRATION_HOLD_S = 3.0     # samples are collected during this hold
CALIBRATION_MIN_SAMPLES = 10
# Smallest calibrated range we accept (in raw metric units, per metric).
# Protects against dividing by an almost-zero range.
CALIBRATION_MIN_RANGE = 0.05
# A clearly wrong calibration (e.g. "open" less open than "closed") is
# repeated up to this many times in total.
CALIBRATION_MAX_ATTEMPTS = 3
RECALIBRATE_AFTER_DAYS = 14

# ---------------------------------------------------------------------------
# Progression (therapist can switch this off)
# ---------------------------------------------------------------------------

AUTO_PROGRESSION = True
PROGRESSION_STEP = 0.03           # raise the "high" threshold by this much
PROGRESSION_MAX_HIGH = 0.90
# A session is "good" when at least this share of reps reached high + margin.
PROGRESSION_GOOD_SHARE = 0.8
PROGRESSION_MARGIN = 0.10

# Fatigue: range of the last reps drops below this share of the first reps.
FATIGUE_WINDOW = 3
FATIGUE_DROP = 0.80

# ---------------------------------------------------------------------------
# Session order
# ---------------------------------------------------------------------------

SESSION_ORDER = [
    "grip_release",
    "finger_abduction",
    "thumb_flexion",
    "thumb_opposition",
    "finger_tapping",
    "grip_squeeze",
]
# Strengthening (stage 5) only every other day.
EVERY_OTHER_DAY = {"grip_squeeze"}

# ---------------------------------------------------------------------------
# Per-exercise parameters
#
# high / low   thresholds (fraction of calibrated range) to enter each zone
# gap          hysteresis: leave the zone only after moving this much back
# hold_s       how long each position has to be held
# sets / reps  defaults from the video (stage 4: 3-5 sets of 10-15 reps),
#              starting at the low end. Confirm with a therapist.
# ---------------------------------------------------------------------------

EXERCISES = {
    "grip_release": {
        "sets": 3, "reps": 10,
        "high": 0.70, "low": 0.30, "gap": 0.10,
        "hold_s": 2.0,
        "count_aloud": True,
        # a finger whose openness is this far below the others is "lagging"
        "lag_margin": 0.15,
        # compensation limits
        "max_palm_rotation_deg": 30.0,
        "max_wrist_shift": 0.5,        # in palm sizes (image)
    },
    "finger_abduction": {
        "sets": 3, "reps": 10,
        "high": 0.70, "low": 0.30, "gap": 0.10,
        "hold_s": 2.0,
        "count_aloud": True,
        # fingers have to be at least this straight (grip openness) for the
        # spread to be meaningful
        "min_openness": 0.70,
        "lag_margin": 0.15,
        "max_palm_rotation_deg": 30.0,
        "max_wrist_shift": 0.5,
    },
    "thumb_flexion": {
        "sets": 3, "reps": 10,
        "high": 0.70, "low": 0.30, "gap": 0.10,
        "hold_s": 2.0,
        "count_aloud": True,
        # other fingers' openness may change this much before we mention it
        "max_other_finger_change": 0.25,
        "max_palm_rotation_deg": 35.0,
        "max_wrist_shift": 0.5,
    },
    "thumb_opposition": {
        "sets": 2, "reps": 3,               # reps = rounds of the sequence
        # touch threshold = touch + factor * (open - touch), per finger
        "touch_factor": 0.35,
        "release_factor": 0.55,
        # the touched finger's distance must be this much smaller than the
        # next closest fingertip
        "dominance_ratio": 0.75,
        "min_touch_s": 0.3,
        "start_level": 1,
        "rounds_to_level_up": 2,            # error-free rounds in a row
        "memory_show_s": 5.0,
        "level_lengths": {1: 7, 2: 3, 3: 4, 4: 5},
    },
    "finger_tapping": {
        "sets": 2, "reps": 2,               # reps = rounds of the sequence
        "mode": "in_order",                 # in_order | called_out | pattern
        # lift: tip height above the flat baseline, as a share of the index
        # lift measured during calibration
        "lift_factor": 0.5,
        "min_lift": 0.08,                   # in palm sizes
        "mcp_lift_deg": 15.0,
        "release_ratio": 0.6,
        "min_lift_s": 0.25,
        "good_isolation": 0.75,
        "called_out_count": 8,
        "pattern_length": 3,
        "memory_show_s": 5.0,
    },
    "grip_squeeze": {
        "sets": 2, "reps": 8,
        "high": 0.65, "low": 0.35, "gap": 0.15,    # looser: object hides fingers
        "hold_s": 4.0,                               # squeeze 3-5 s
        "relax_s": 4.0,                              # rest 3-5 s
        "count_aloud": True,
    },
}
