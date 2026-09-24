import numpy as np
import pytest

from rehab.exercises import create
from rehab.exercises.finger_abduction import FingerAbduction
from rehab.exercises.finger_tapping import FingerTapping
from rehab.exercises.grip_release import GripRelease
from rehab.exercises.grip_squeeze import GripSqueeze
from rehab.exercises.thumb_flexion import ThumbFlexion
from rehab.exercises.thumb_opposition import ThumbOpposition
from helpers import Clock, calibrate, feat, lerp, ramp, run, texts

# Eleanor's range: she can only open part way and close part way.
HER_OPEN = (25, 35, 20)
HER_CLOSED = (60, 80, 50)


@pytest.fixture
def grip_cal():
    return calibrate(GripRelease, [dict(flex=HER_OPEN), dict(flex=HER_CLOSED)])


def grip_rep(ex, clock, open_flex=HER_OPEN, hold=2.5):
    said = ramp(ex, 1.0, clock, lambda u: dict(flex=lerp(HER_CLOSED, open_flex, u)))
    said += run(ex, hold, clock, flex=open_flex)
    said += ramp(ex, 1.0, clock, lambda u: dict(flex=lerp(open_flex, HER_CLOSED, u)))
    said += run(ex, hold, clock, flex=HER_CLOSED)
    return said


# --- grip and release ---------------------------------------------------------

def test_grip_counts_reps_against_her_own_range(grip_cal):
    ex = GripRelease(calibration=grip_cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    for _ in range(3):
        grip_rep(ex, clock)
    assert len(ex.reps) == 3
    rec = ex.reps[0]
    assert rec.range_high == pytest.approx(1.0, abs=0.1)
    assert rec.range_low == pytest.approx(0.0, abs=0.1)
    assert 0.2 < rec.movement_time < 1.5
    assert rec.smoothness_peaks == 1
    assert rec.hold_stability < 0.02
    assert rec.compensation == []


def test_grip_partial_opening_below_threshold_does_not_count(grip_cal):
    ex = GripRelease(calibration=grip_cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    halfway = lerp(HER_CLOSED, HER_OPEN, 0.5)
    run(ex, 3.0, clock, flex=halfway)
    assert ex.display["value"] == pytest.approx(0.5, abs=0.1)     # she sees how far she got
    run(ex, 3.0, clock, flex=HER_CLOSED)
    assert ex.reps == []


def test_grip_hold_broken_early_needs_another_hold(grip_cal):
    ex = GripRelease(calibration=grip_cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    said = run(ex, 1.0, clock, flex=HER_OPEN)            # too short
    said += run(ex, 1.0, clock, flex=HER_CLOSED)
    assert ex.reps == []
    assert "Hold it a little longer." in texts(said)
    assert ex.display["phase_label"] == "OPEN"


def test_grip_counts_aloud_and_names_rep(grip_cal):
    ex = GripRelease(calibration=grip_cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    said = texts(grip_rep(ex, clock))
    assert "And hold." in said and "two" in said
    assert "Now close your hand." in said
    assert "That's one." in said


def test_grip_hint_after_stall_names_lagging_finger(grip_cal):
    ex = GripRelease(calibration=grip_cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    # ring finger stays bent while the others open most of the way
    said = run(ex, 9.0, clock, flex=lerp(HER_CLOSED, HER_OPEN, 0.8), finger_flex={"ring": HER_CLOSED})
    assert "Try to open your ring finger a little more." in texts(said)
    assert ex.display["finger_colors"] == {"ring": "lagging"}


def test_grip_detects_wrist_rotation(grip_cal):
    ex = GripRelease(calibration=grip_cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    run(ex, 1.0, clock, flex=HER_CLOSED)
    said = []
    for i in range(60):
        t = clock.tick()
        f = feat(t, flex=HER_OPEN)
        # rotate the palm normal by 40 degrees
        a = np.radians(40 * i / 59)
        f.palm_normal_image = np.array([np.sin(a), 0, -np.cos(a)])
        said += ex.update(f, t)
    assert "Try not to turn your hand." in texts(said)


def test_grip_fatigue(grip_cal):
    ex = GripRelease(calibration=grip_cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    for _ in range(3):
        grip_rep(ex, clock, open_flex=(5, 10, 5))           # wider than calibrated
    for _ in range(3):
        grip_rep(ex, clock, open_flex=lerp(HER_CLOSED, HER_OPEN, 0.8))
    assert ex.fatigue


def test_grip_logs_gesture_disagreement(grip_cal):
    ex = GripRelease(calibration=grip_cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    for _ in range(int(5.5 * 30)):
        t = clock.tick()
        f = feat(t, flex=HER_OPEN)
        f.gesture = "Closed_Fist"          # classifier disagrees with the measurement
        ex.update(f, t)
    run(ex, 3.5, clock, flex=HER_CLOSED)
    assert ex.reps and ex.reps[0].extra["gesture_disagreement"] > 0.5


# --- finger abduction ---------------------------------------------------------------

def test_abduction_reps_and_pause_when_fingers_bent():
    cal = calibrate(FingerAbduction, [dict(spread=2), dict(spread=12)])
    ex = FingerAbduction(calibration=cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    for _ in range(2):
        run(ex, 3.0, clock, spread=12)
        run(ex, 3.0, clock, spread=2)
    assert len(ex.reps) == 2
    assert ex.frame_problem(feat(flex=(60, 80, 50))) is not None
    assert ex.frame_problem(feat()) is None


def test_abduction_names_small_gap():
    cal = calibrate(FingerAbduction, [dict(spread=2), dict(spread=12)])
    ex = FingerAbduction(calibration=cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    said = []
    for _ in range(300):
        t = clock.tick()
        f = feat(t, spread=9)
        f.spread["ring_pinky"] = 3.0
        said += ex.update(f, t)
    assert "Spread your ring and little finger a bit more." in texts(said)


# --- thumb flexion --------------------------------------------------------------------

def test_thumb_flexion_reps_and_other_fingers_check():
    cal = calibrate(ThumbFlexion, [dict(thumb_out=0.0), dict(thumb_out=1.0)])
    ex = ThumbFlexion(calibration=cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    for _ in range(2):
        run(ex, 3.0, clock, thumb_out=0.0)
        run(ex, 3.0, clock, thumb_out=1.0)
    assert len(ex.reps) == 2
    assert ex.reps[0].compensation == []
    said = run(ex, 3.0, clock, thumb_out=0.0, flex=(50, 60, 40))
    assert "Try to move only your thumb." in texts(said)


# --- thumb opposition ----------------------------------------------------------------------

def opposition_cal():
    return calibrate(ThumbOpposition, [dict(), dict(thumb_touch="index")])


def touch(ex, clock, finger):
    said = run(ex, 0.8, clock, thumb_touch=finger)
    said += run(ex, 0.8, clock)
    return said


def test_opposition_guided_round():
    ex = ThumbOpposition(calibration=opposition_cal(), level=1)
    clock = Clock()
    ex.start_set(1, clock.t)
    said = run(ex, 0.5, clock)
    assert "Touch your index finger." in texts(said)
    for finger in ["index", "middle", "ring", "pinky", "ring", "middle", "index"]:
        touch(ex, clock, finger)
    assert len(ex.reps) == 1
    rec = ex.reps[0]
    assert rec.extra["correct"] == 7 and rec.extra["wrong"] == 0
    assert rec.range_high == 1.0


def test_opposition_wrong_finger_is_neutral():
    ex = ThumbOpposition(calibration=opposition_cal(), level=1)
    clock = Clock()
    ex.start_set(1, clock.t)
    run(ex, 0.5, clock)
    said = texts(touch(ex, clock, "middle"))
    assert "That was your middle finger. Let's try the index finger." in said
    assert not any("wrong" in s.lower() for s in said)


def test_opposition_memory_level_hides_sequence_and_levels_up():
    ex = ThumbOpposition(calibration=opposition_cal(), level=1,
                         params={"rounds_to_level_up": 1})
    clock = Clock()
    ex.start_set(1, clock.t)
    run(ex, 0.5, clock)
    said = []
    for finger in ["index", "middle", "ring", "pinky", "ring", "middle", "index"]:
        said = texts(touch(ex, clock, finger))
    assert ex.level == 2 and ex.mode == "memory" and ex.length == 3
    assert "You're doing so well. Now let's try one from memory." in said
    assert any(s.startswith("Remember:") for s in said)
    assert ex.display["hidden"] is False
    run(ex, 5.0, clock)
    assert ex.display["hidden"] is True
    for finger in ex._round["seq"]:
        touch(ex, clock, finger)
    assert len(ex.reps) == 2 and ex.reps[1].extra["mode"] == "memory"


# --- finger tapping ---------------------------------------------------------------------------

def tapping_cal():
    return calibrate(FingerTapping, [dict(), dict(finger_flex={"index": (-25, 0, 0)})])


def lift(ex, clock, finger, others=None):
    pose = {finger: (-25, 0, 0)}
    pose.update(others or {})
    said = run(ex, 0.8, clock, finger_flex=pose)
    said += run(ex, 0.8, clock)
    return said


def test_tapping_in_order_with_isolation():
    ex = FingerTapping(calibration=tapping_cal())
    clock = Clock()
    ex.start_set(1, clock.t)
    said = texts(run(ex, 0.5, clock))
    assert "Lift your index finger." in said
    for finger in ["index", "middle", "ring", "pinky", "ring", "middle", "index"]:
        said += texts(lift(ex, clock, finger))
    assert len(ex.reps) == 1
    assert ex.reps[0].extra["isolation"] > 0.9
    assert "Nice, only your index finger moved." in said


def test_tapping_poor_isolation_hint():
    ex = FingerTapping(calibration=tapping_cal())
    clock = Clock()
    ex.start_set(1, clock.t)
    run(ex, 0.5, clock)
    said = texts(lift(ex, clock, "index", others={"middle": (-12, 0, 0), "ring": (-12, 0, 0),
                                                  "pinky": (-12, 0, 0)}))
    assert "Try to keep the other fingers resting on the table." in said


def test_tapping_called_out_mode_is_random_fingers():
    ex = FingerTapping(calibration=tapping_cal(), params={"mode": "called_out", "called_out_count": 4},
                       rng=np.random.default_rng(3))
    clock = Clock()
    ex.start_set(1, clock.t)
    run(ex, 0.3, clock)
    seq = list(ex._round["seq"])
    assert len(seq) == 4
    for finger in seq:
        lift(ex, clock, finger)
    assert len(ex.reps) == 1 and ex.reps[0].extra["mode"] == "called_out"


# --- grip squeeze ------------------------------------------------------------------------------

def test_squeeze_hold_and_relax():
    cal = calibrate(GripSqueeze, [dict(flex=(45, 60, 40)), dict(flex=(60, 80, 55))])
    ex = GripSqueeze(calibration=cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    for _ in range(2):
        run(ex, 5.0, clock, flex=(60, 80, 55))
        run(ex, 5.0, clock, flex=(45, 60, 40))
    assert len(ex.reps) == 2
    rec = ex.reps[0]
    assert rec.extra["relaxed_fully"] is True
    assert rec.raw_high >= 4.0            # seconds squeezed


def test_create_uses_profile_thresholds():
    ex = create("grip_release", {}, {"grip_release": {"high": 0.8}})
    assert ex.hyst.high == 0.8


# --- benchmarks for the hand (plan 7.2: the "Bloom" and "pinch" exercises) ------------------

def test_grip_scores_fma_style_and_bain_ranges(grip_cal):
    """Her partial range: FMA 25/24-style 1, not yet Bain's functional open or grasp."""
    ex = GripRelease(calibration=grip_cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    grip_rep(ex, clock)
    extra = ex.reps[-1].extra
    assert extra["fma25_style"] == 1 and not extra["bain_functional_open"]
    assert "milestones_reached" not in extra
    assert extra["aperture_open"] > extra["aperture_closed"]


def test_grip_full_range_reaches_the_milestones_once(grip_cal):
    from datetime import date
    from rehab import storage
    from rehab.progress import SessionProgress
    ex = GripRelease(calibration=calibrate(GripRelease, [dict(flex=(0, 0, 0)),
                                                         dict(flex=(85, 100, 75))]))
    clock = Clock()
    ex.start_set(1, clock.t)
    for _ in range(2):
        ramp(ex, 1.0, clock, lambda u: dict(flex=lerp((85, 100, 75), (0, 0, 0), u)))
        run(ex, 2.5, clock, flex=(0, 0, 0))
        ramp(ex, 1.0, clock, lambda u: dict(flex=lerp((0, 0, 0), (85, 100, 75), u)))
        run(ex, 2.5, clock, flex=(85, 100, 75))
    extra = ex.reps[0].extra
    assert extra["fma25_style"] == 2 and extra["bain_functional_open"]
    assert extra["bain_functional_grasp"]
    assert {"open_hand", "grasp", "fma25"} <= set(extra["milestones_reached"])

    class Log:
        today = date.today()
        session_id = "s"

        def normal_history(self, name):
            return []

    profile = storage.new_profile()
    progress = SessionProgress(profile, Log(), therapist={})
    first = progress.rep_events(ex, ex.reps[0])
    assert sum(e.type == "BenchmarkMilestone" for e in first) == 3
    second = progress.rep_events(ex, ex.reps[1])
    assert not any(e.type == "BenchmarkMilestone" for e in second)     # announced once


def test_opposition_scores_the_index_pinch():
    cal = calibrate(ThumbOpposition, [dict(thumb_out=1.0), dict(thumb_touch="index")])
    ex = ThumbOpposition(calibration=cal, level=1)
    clock = Clock()
    ex.start_set(1, clock.t)
    run(ex, 0.5, clock, thumb_out=1.0)
    for finger in ["index", "middle", "ring", "pinky", "ring", "middle", "index"]:
        run(ex, 1.0, clock, thumb_touch=finger)
        run(ex, 1.0, clock, thumb_out=1.0)
    rec = ex.reps[-1]
    assert rec.extra["fma28_style"] == 1                  # never 2: no tug on a pencil
    assert rec.extra["pinch_gap_min"] < 0.12


def test_bubble_pinch_scores_the_pincer_grasp():
    from rehab.exercises.bubble_pinch import BubblePinch
    cal = calibrate(BubblePinch, [dict(thumb_out=1.0), dict(thumb_touch="index")])
    ex = BubblePinch(calibration=cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    for _ in range(2):
        run(ex, 3.0, clock, thumb_touch="index")
        run(ex, 2.0, clock, thumb_out=1.0)
    rec = ex.reps[-1]
    assert rec.extra["fma28_style"] == 1 and rec.extra["pinch_gap_min"] < 0.12
