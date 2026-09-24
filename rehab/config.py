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
SESSIONS_PATH = DATA_DIR / "sessions.csv"
GARDEN_PATH = DATA_DIR / "garden.json"
# "Delete my profile" (profile screen) moves her files here instead of erasing
# them, so a therapist can still restore them. False erases them for good.
KEEP_DELETED_PROFILE = True
DELETED_PROFILES_DIR = DATA_DIR / "deleted"
CONTENT_DIR = ROOT_DIR / "content"
ASSETS_DIR = ROOT_DIR / "assets"
FONT_REGULAR = ASSETS_DIR / "fonts" / "AtkinsonHyperlegible-Regular.ttf"
FONT_BOLD = ASSETS_DIR / "fonts" / "AtkinsonHyperlegible-Bold.ttf"

# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

USER_NAME = "Eleanor"
# Hand to train. Eleanor has left-sided hemiparesis.
AFFECTED_HAND = "Left"

# ---------------------------------------------------------------------------
# Camera and tracking
# ---------------------------------------------------------------------------

CAMERA_INDEX = 1
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

# Screen: the window opens full screen (f toggles, --windowed starts in a
# window) and the layout takes the screen's shape, so nothing falls off the
# edge. DESIGN_HEIGHT is the height everything is drawn for before the
# window scales it; the camera image is scaled to it.
FULLSCREEN = True
DESIGN_HEIGHT = 720

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
SPEECH_RATE = 145          # words per minute (`say -r 145`, pyttsx3 rate); per user in profile.json
VOICE = None               # macOS voice name (e.g. "Samantha"); None = system default
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

AUTO_PROGRESSION = True           # False: targets stay where the therapist set them

# Adaptive targets (see rehab/progress.py). The target is the "high"
# threshold: a fraction of her calibrated range. It is evaluated once per
# set, never during a set.
TARGET_STEP = 0.03
TARGET_RAISE_AT = 0.80            # success rate in a set >= this -> one step up
TARGET_LOWER_AT = 0.50            # success rate in a set < this -> one step down
TARGET_FLOOR = 0.50
TARGET_CEILING = 1.00
# She holds beyond her calibrated maximum (median of the set's holds above
# 1.0 + this): the calibrated range grows with her.
CALIBRATION_GROW_MARGIN = 0.05
LONG_GAP_DAYS = 7                 # warm-up: 1 step below last time, 2 after a long gap

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

# ---------------------------------------------------------------------------
# Motivation: personal bests, praise, difficult days, activities, garden.
#
# All of these are starting values for testing, not clinical values.
# ---------------------------------------------------------------------------

# Personal bests. A rep's value is the median during its hold (not the peak
# frame), so landmark jitter cannot make a fake best. Improvements are
# relative to the old best: 0.03 = 3% better.
PB_MIN_IMPROVEMENT = 0.03
PB_MIN_STEADINESS_IMPROVEMENT = 0.10    # hold wobble is noisier
PB_MAX_PER_SET = 1
PB_WEEK_DAYS = 7
PB_KEEP_DAYS = 14                       # daily bests kept for "this week"

# Praise
PRAISE_EVERY_N_REPS = 3        # a spoken phrase at most about every n reps ...
PHRASE_NO_REPEAT = 3           # ... never one of the last 3 of its category
PRAISE_MAX_WAIT_S = 3.0        # praise waiting longer than this is dropped, not played late
IMPROVEMENT_WINDOW = 5         # recent reps for "opened more than usual"
IMPROVEMENT_MARGIN = 0.08      # fraction of her range above that recent average
STEADY_HOLD_MAX_STD = 0.03     # hold wobble (fraction of range) for "very steady"

# Difficult day mode (any one signal switches it on for the rest of the session)
DIFFICULT_DROP = 0.15                  # first set this far below her baseline
DIFFICULT_BASELINE_SESSIONS = 5        # baseline = median of her last 5 normal sessions
DIFFICULT_MIN_BASELINE_SESSIONS = 3    # fewer normal sessions: no warm-up check
DIFFICULT_LOW_SUCCESS_SETS = 2         # success < TARGET_LOWER_AT this many sets in a row
DIFFICULT_TARGET_FACTOR = 0.8
DIFFICULT_REST_FACTOR = 1.5
DIFFICULT_FEWER_SETS = 1
DIFFICULT_SKIP = {"grip_squeeze"}      # strength work is skipped
# Repeated difficult days: a note in sessions.csv for her therapist or family
DIFFICULT_REPEAT_WINDOW = 5
DIFFICULT_REPEAT_COUNT = 3

# Session flow
CHECK_IN_TIMEOUT_S = 25        # no answer: carry on as a normal day
QUESTION_TIMEOUT_S = 20        # setup questions (name, activities, plant)
INTRO_CARD_S = 5.0             # activity card before an exercise
CARD_PAUSE_S = 1.0             # after a card's speech, before moving on
GARDEN_SCREEN_S = 8.0
GOODBYE_S = 4.0

# Yes / no without a keyboard: thumbs up / thumbs down, held briefly.
# Either hand counts. Space or "y" = yes and "n" = no stay as a backup.
GESTURE_HOLD_S = 0.7
GESTURE_MIN_SCORE = 0.6
YES_GESTURE = "Thumb_Up"
NO_GESTURE = "Thumb_Down"

# Daily activities
ACTIVITY_CHOICES = 3
MILESTONES = (50, 100, 250, 500)

# Garden: grows from showing up, never wilts
GARDEN_STAGES = 5              # seed, sprout, leaves, bud, flower
GARDEN_PLOTS = 8               # a full bed starts "a new season"
GARDEN_GROW_S = 1.0            # growth animation
# Two are offered when a new seed is planted (thumbs up: the first one)
PLANT_TYPES = ("rose", "tulip", "sunflower", "lavender")
