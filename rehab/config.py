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
# Arm exercises track the body with MediaPipe Pose (loaded only when needed).
POSE_MODEL_PATH = ROOT_DIR / "models" / "pose_landmarker_lite.task"
# Movement benchmarks (see benchmarks/BENCHMARK_PLAN.md): every number there
# has a source or is marked proposed. The therapist profile sets her sex and
# age for the norms, limits, and which arm exercises she may do alone.
BENCHMARKS_DIR = ROOT_DIR / "benchmarks"
BENCHMARKS_PATH = BENCHMARKS_DIR / "benchmarks.json"
THERAPIST_PROFILE_PATH = BENCHMARKS_DIR / "therapist_profile.json"
# benchmark logs (plan section 12): one row per rep, one per session
BENCH_REP_LOG_PATH = DATA_DIR / "benchmark_reps.csv"
BENCH_SESSION_LOG_PATH = DATA_DIR / "benchmark_sessions.csv"
# guest sessions (python main.py --guest) keep their data apart from hers
GUESTS_DIR = DATA_DIR / "guests"


def use_data_dir(path):
    """
    Keep all personal data in another folder (e.g. a marketplace guest's),
    so her own files in data/ are never touched. storage reads these paths
    when it is called, so this works at any time before a session starts.
    """
    global DATA_DIR, PROFILE_PATH, REP_LOG_PATH, HISTORY_PATH, SESSIONS_PATH
    global GARDEN_PATH, DELETED_PROFILES_DIR, BENCH_REP_LOG_PATH, BENCH_SESSION_LOG_PATH
    DATA_DIR = Path(path)
    PROFILE_PATH = DATA_DIR / "profile.json"
    REP_LOG_PATH = DATA_DIR / "reps.csv"
    HISTORY_PATH = DATA_DIR / "history.csv"
    SESSIONS_PATH = DATA_DIR / "sessions.csv"
    GARDEN_PATH = DATA_DIR / "garden.json"
    DELETED_PROFILES_DIR = DATA_DIR / "deleted"
    BENCH_REP_LOG_PATH = DATA_DIR / "benchmark_reps.csv"
    BENCH_SESSION_LOG_PATH = DATA_DIR / "benchmark_sessions.csv"

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

# The menu, in this order (keys 1-9).
SESSION_ORDER = [
    "grip_release",
    "finger_abduction",
    "thumb_flexion",
    "thumb_opposition",
    "finger_tapping",
    "grip_squeeze",
    "bubble_pinch",
    "two_hand_match",
    "memory_pairs",
]
# "All of today's exercises": the six hand exercises and the memory game as a
# restful end. Bubble pinch and two-hand match are in the menu only.
DAILY_PLAN = [
    "grip_release",
    "finger_abduction",
    "thumb_flexion",
    "thumb_opposition",
    "finger_tapping",
    "grip_squeeze",
    "memory_pairs",
]
# python main.py --short (e.g. for guests at the marketplace): one short set.
# "reps" is used for the range exercises, "rounds" for sequences and boards.
SHORT_SESSION = {"sets": 1, "reps": 5, "rounds": 1}
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
        # "finger piano": every correct touch plays the next note of a tune
        "play_notes": True,
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
    "bubble_pinch": {
        "sets": 2, "reps": 8,
        "high": 0.70, "low": 0.30, "gap": 0.10,
        "hold_s": 2.0,                               # pinch and hold
        "release_s": 1.0,
        "count_aloud": True,
        "max_palm_rotation_deg": 35.0,
        "max_wrist_shift": 0.5,
    },
    "two_hand_match": {
        "sets": 2, "reps": 8,
        "high": 0.70, "low": 0.30, "gap": 0.10,
        "hold_s": 1.0,
        "count_aloud": False,
        "max_palm_rotation_deg": 35.0,
        "max_wrist_shift": 0.6,
    },
    "memory_pairs": {
        "sets": 1, "reps": 2,                        # reps = boards
        "start_level": 1,
        "level_pairs": {1: 2, 2: 3, 3: 4},           # pairs on the board per level
        "boards_to_level_up": 2,                     # good boards in a row
        "dwell_s": 1.5,                              # hold the finger still to choose
        "dwell_radius": 0.04,                        # how still (fraction of the image width)
        "show_mismatch_s": 2.0,
    },
}

# ---------------------------------------------------------------------------
# Arm exercises and movement benchmarks (benchmarks/BENCHMARK_PLAN.md)
#
# Angles are in degrees and judged with the measurement tolerance of each
# task (MediaPipe limits of agreement in stroke, Lazem 2026). Sets, reps,
# rest and hold per exercise come from benchmarks/therapist_profile.json.
# Values marked "proposed" are project defaults, not from the sources.
# ---------------------------------------------------------------------------

# Order in the "Arm exercises" menu.
ARM_EXERCISES = [
    "shoulder_flexion_raise",
    "shoulder_abduction_raise",
    "hand_to_mouth",
    "hand_to_head",
    "elbow_extension",
    "wrist_extension",
    "tabletop_reach",
    "finger_to_nose_timed",
]
ARM_DEFAULT_SETS = 2                # when the therapist profile gives none (proposed)

# Sense: pose landmarks (proposed values from the plan, section 14)
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
LOW_PASS_HZ = 6.0                   # Gates 2016 filtered markers at 6 Hz
# Framing check before an arm exercise: the needed landmarks must be seen
# this long in a row, and not closer than EDGE_MARGIN to the image edge.
SETUP_CHECK_HOLD_S = 1.5
SETUP_EDGE_MARGIN = 0.03            # fraction of the image
# Camera view from the shoulder width / torso length ratio (proposed):
# side-on (sagittal) below the first, facing the camera (frontal) above
# the second, 45 degrees in between.
VIEW_SAGITTAL_MAX = 0.35
VIEW_FRONTAL_MIN = 0.60

# Calibration and weekly assessment (FMA: demonstrate, practise, the
# unaffected side first; Lazem: about 10 s rest between reps)
BENCH_PRACTICE_TRIALS = 2
BENCH_CALIBRATION_REPS = 3
BENCH_REST_BETWEEN_REPS_S = 10
ASSESSMENT_EVERY_DAYS = 7           # an arm calibration is also the weekly assessment

# Rep state machine
ARM_START_HOLD_FRAMES = 5           # start posture held this many frames (debounce)
ARM_START_ELEVATION_MAX = 20.0      # "arm at your side" (proposed)
ARM_BENT_ELBOW_MIN = 60.0           # start of elbow extension: elbow bent at least this (proposed)
ARM_WRIST_START_MAX = 5.0           # wrist extension starts at or below neutral + this (proposed)
ARM_HAND_DOWN_TORSO = 0.5           # hand in the lap / hanging: wrist this far below the shoulder (torso lengths, proposed)
NOSE_ROUND_TIMEOUT_S = 60           # finger to nose: a round ends after this, touches or not (proposed)
# Hand to head: the wrist reaches the ear. benchmarks.json proposes 0.5
# shoulder widths, but side-on the shoulders overlap, so the distance is
# divided by the torso length instead: 0.35 torso lengths is about the same.
HAND_TO_EAR_TORSO = 0.35

# Finger joints (grip and release): MediaPipe finger angles are not
# validated, so "opened wider" needs a mean change of the provisional 20
# degrees per joint (benchmarks.json) or a rise in each of this many sessions.
TREND_SESSIONS = 3

# Progression (plan 9.4, proposed): per session, never during one
LEVEL_RAISE_AT = 0.80               # clean success, two sessions in a row
LEVEL_LOWER_AT = 0.50
MISSED_SESSION_DAYS = 3             # 2 missed daily sessions: restart one level lower

# Reaching (Levin 2004 Reaching Performance Scale; thresholds proposed)
REACH_PRIMARY_TOLERANCE = 0.05      # hand movement, in arm lengths
REACH_FAR_ARRIVED = 0.60            # far target: hand moved at least this many arm lengths
REACH_ELBOW_ALMOST_STRAIGHT = 20.0  # degrees of elbow flexion left at the far target

# Finger to nose (FMA 31-33, adapted: eyes open)
NOSE_TOUCHES = 5
NOSE_AWAY = 2.0                     # finger this far from the nose (eye distances) = away again
REST_ZONE_TORSO = 0.3               # wrist within this share of the torso above the hips = in the lap
TREMOR_MAX_PEAKS = 2                # speed peaks per approach for "no tremor" (proposed)

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

# Questions at the end of a session (1-5, keys or fingers held up), for
# measuring how the coach is experienced. Guests are also asked "ease".
RATING_QUESTIONS = ("exertion", "enjoyment")
GUEST_RATING_QUESTIONS = ("exertion", "enjoyment", "ease")
RATING_HOLD_S = 1.5
RATING_TIMEOUT_S = 25

# Stop / "I don't feel well" (S key): the safety screen can show a person to
# call. Empty = not shown. The app never calls anyone itself.
HELPER_NAME = ""
HELPER_PHONE = ""
EMERGENCY_NUMBER = "112"

# Finger piano notes
NOTE_VOLUME = 0.35

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
