"""
Lift one finger at a time with the hand flat, back to the camera: the hand's
side from the knuckle triangle, a resting position taken from the live
hand, lifts compared between fingers, and lifts that never get stuck.
"""

from dataclasses import replace

import numpy as np
import pytest

from rehab import config, features, storage
from rehab.Act import SilentSpeaker
from rehab.exercises.finger_tapping import FingerTapping
from rehab.Think import QUALITY_TEXT, Coach, SessionManager
from helpers import FPS, Clock, answer, calibrate, returning_profile
from synthetic_hand import IMAGE_SIZE, hand

BACK = dict(back_to_camera=True)
INDEX_UP = {"index": (-25, 0, 0)}


def back(t, **pose):
    """Her left hand flat, back to the camera."""
    return features.extract(hand(**BACK, **pose), t, IMAGE_SIZE)


def right_hand_back_up():
    """A right hand with its back to the camera: the left hand's mirror image."""
    obs = hand(handedness="Right", **BACK)
    image = obs.image * np.array([-1.0, 1.0, 1.0]) + np.array([1.0, 0.0, 0.0])
    world = obs.world * np.array([-1.0, 1.0, 1.0])
    return replace(obs, image=image, world=world)


def tapping(**params):
    cal = calibrate(FingerTapping, [BACK, dict(BACK, finger_flex=INDEX_UP)])
    ex = FingerTapping(calibration=cal, params=params or None)
    events = []
    ex.trace = lambda kind, **data: events.append((kind, data))
    clock = Clock()
    ex.start_set(1, clock.t)
    return ex, clock, events


def feed(ex, clock, seconds, make=None, **pose):
    said = []
    for _ in range(int(seconds * FPS)):
        t = clock.tick()
        f = make(t) if make else back(t, **pose)
        said += ex.update(f, t)
    return said


def starts(events):
    return [d["finger"] for kind, d in events if kind == "detect" and d["event"] == "start"]


def ends(events):
    return [(d["finger"], d["reason"]) for kind, d in events
            if kind == "detect" and d["event"] == "end"]


# --- the hand ---------------------------------------------------------------------------------

def test_knuckle_triangle_tells_the_side_of_a_hand_seen_from_its_back():
    assert back(0.0).dorsal_side == "Left" and back(0.0).dorsal_match is True
    palm = features.extract(hand(), 0.0, IMAGE_SIZE)
    assert palm.dorsal_side == "Right" and palm.dorsal_match is False
    right = features.extract(right_hand_back_up(), 0.0, IMAGE_SIZE)
    assert right.dorsal_side == "Right"


def test_flickering_label_is_not_the_wrong_hand_when_the_knuckles_say_left():
    f = features.extract(hand(handedness="Right", **BACK), 0.0, IMAGE_SIZE)
    assert not f.correct_hand
    assert f.quality_problem(palm_down=True) is None
    assert f.quality_problem() == "wrong_hand"          # other exercises: unchanged


def test_a_real_right_hand_back_up_is_still_the_wrong_hand():
    f = features.extract(right_hand_back_up(), 0.0, IMAGE_SIZE)
    assert f.quality_problem(palm_down=True) == "wrong_hand"


def test_palm_to_the_camera_is_not_flat():
    f = features.extract(hand(), 0.0, IMAGE_SIZE)
    assert f.correct_hand
    assert f.quality_problem(palm_down=True) == "not_flat"
    assert "flat on the table" in QUALITY_TEXT["not_flat"]


def test_palm_down_prefers_the_hand_whose_knuckles_match_over_a_ghost():
    ghost = hand()                                          # labelled Left, turned the wrong way
    real = hand(handedness="Right", **BACK)                 # labelled Right, knuckles say Left
    assert features.choose_hand([ghost, real], "Left", palm_down=True) is real
    assert features.choose_hand([ghost, real], "Left") is ghost


def test_a_foreshortened_flat_hand_is_not_too_far():
    obs = hand(**BACK)
    wrist = obs.image[features.WRIST]
    # small in the picture, and seen from a low camera: the palm looks short
    image = wrist + (obs.image - wrist) * np.array([0.2, 0.05, 0.2])
    f = features.extract(replace(obs, image=image), 0.0, IMAGE_SIZE)
    assert f.palm_size_image < config.MIN_PALM_SIZE_IMAGE
    assert not f.too_small and f.quality_problem(palm_down=True) is None
    # a hand that really is far away is still too far
    far = wrist + (obs.image - wrist) * 0.1
    assert features.extract(replace(obs, image=far), 0.0, IMAGE_SIZE).too_small


# --- calibration --------------------------------------------------------------------------------

def test_an_old_format_calibration_is_measured_again():
    old = {"flat": {f"{m}_{k}": 0.1 for k in features.FINGERS for m in ("h", "mcp")},
           "lift": {"h_index": 0.4, "mcp_index": 30.0}}
    assert not FingerTapping.calibration_valid(old)
    new = calibrate(FingerTapping, [BACK, dict(BACK, finger_flex=INDEX_UP)])
    assert FingerTapping.calibration_valid(new)
    no_lift = {"flat": new["flat"], "lift": {"r_index": new["flat"]["r_index"] + 0.01}}
    assert not FingerTapping.calibration_valid(no_lift)


# --- detection ----------------------------------------------------------------------------------

def test_a_stale_calibration_does_not_make_a_lift_at_the_start():
    ex, clock, events = tapping()
    # today her middle and ring fingers rest higher than when she was measured
    resting = {"middle": (-14, 0, 0), "ring": (-14, 0, 0)}
    feed(ex, clock, 2.0, finger_flex=resting)
    assert starts(events) == []
    feed(ex, clock, 0.8, finger_flex=dict(resting, **INDEX_UP))
    assert starts(events) == ["index"]


def test_a_whole_hand_shift_is_not_a_lift():
    ex, clock, events = tapping()
    feed(ex, clock, 1.0)

    def shifted(t):
        f = back(t)
        f.tip_rise = {k: v + 0.3 for k, v in f.tip_rise.items()}
        return f

    feed(ex, clock, 2.0, make=shifted)
    feed(ex, clock, 1.0)
    assert starts(events) == [] and ex._lifted is None


def test_a_right_label_burst_does_not_stop_a_lift():
    ex, clock, events = tapping()
    speaker = SilentSpeaker()
    coach = Coach(ex, speaker, None, hand="Left")
    coach.start_set(1, clock.t)
    quality = []
    coach.trace = lambda kind, **data: quality.append(data.get("problem"))

    def step(label="Left", **pose):
        t = clock.tick()
        f = features.extract(hand(handedness=label, **BACK, **pose), t, IMAGE_SIZE)
        coach.update(f, t)

    for _ in range(int(1.0 * FPS)):
        step()
    for _ in range(4):                                      # the index starts to rise ...
        step(finger_flex=INDEX_UP)
    for _ in range(5):                                      # ... MediaPipe calls it a right hand
        step("Right", finger_flex=INDEX_UP)
    for _ in range(int(0.6 * FPS)):
        step(finger_flex=INDEX_UP)
    for _ in range(int(0.6 * FPS)):
        step()
    assert "wrong_hand" not in quality and coach.quality is None
    assert "Please use your left hand." not in [m.text for m in speaker.spoken]
    assert ex._round["correct"] == 1 and ex._round["wrong"] == 0
    assert ex._label == "Left"


def test_a_label_that_stays_different_is_accepted():
    ex, clock, events = tapping()
    feed(ex, clock, 1.0)
    feed(ex, clock, 1.5, make=lambda t: features.extract(hand(handedness="Right", **BACK), t,
                                                         IMAGE_SIZE))
    assert ex._label == "Right"
    assert ("baseline", {"exercise": "finger_tapping", "reason": "label"}) in events
    feed(ex, clock, 0.8, make=lambda t: features.extract(
        hand(handedness="Right", finger_flex=INDEX_UP, **BACK), t, IMAGE_SIZE))
    assert starts(events) == ["index"]


def test_a_jump_of_the_hand_measures_the_resting_position_again():
    ex, clock, events = tapping()
    feed(ex, clock, 1.0)

    def moved(t):
        obs = hand(**BACK)
        return features.extract(replace(obs, image=obs.image + np.array([0.3, 0.0, 0.0])), t,
                                IMAGE_SIZE)

    feed(ex, clock, 0.2, make=moved)
    assert any(kind == "baseline" and d["reason"] == "jump" for kind, d in events)


def test_a_stuck_lift_ends_after_max_lift_s():
    ex, clock, events = tapping()
    feed(ex, clock, 1.0)
    feed(ex, clock, 7.0, finger_flex=INDEX_UP)              # she keeps it up
    assert starts(events) == ["index"]
    assert ends(events) == [("index", "too_long")]
    assert ex._lifted is None
    feed(ex, clock, 1.0)                                    # down, then the next finger counts
    feed(ex, clock, 0.8, finger_flex={"middle": (-25, 0, 0)})
    assert starts(events) == ["index", "middle"]


def test_a_clearly_higher_second_finger_takes_over():
    ex, clock, events = tapping()
    feed(ex, clock, 1.0)
    feed(ex, clock, 0.8, finger_flex=INDEX_UP)
    # the index stays half up (its lift has not ended) while the middle finger goes up high
    feed(ex, clock, 1.0, finger_flex={"index": (-14, 0, 0), "middle": (-30, 0, 0)})
    assert starts(events) == ["index", "middle"]
    assert ends(events) == [("index", "switch")]
    assert ex._round["correct"] == 2 and ex._round["wrong"] == 0


def test_nothing_is_detected_while_the_resting_position_is_measured():
    ex, clock, events = tapping()
    feed(ex, clock, 0.2, finger_flex=INDEX_UP)
    assert ex._base is None and starts(events) == []


# --- session ------------------------------------------------------------------------------------

@pytest.fixture
def log(tmp_path):
    return storage.SessionLog(tmp_path / "reps.csv", tmp_path / "history.csv", session_id="s1")


def test_calibration_screen_waits_before_showing_a_problem(log):
    s = SessionManager(SilentSpeaker(), returning_profile(), log)
    clock = Clock()
    s.update(back(clock.t), clock.t)
    s.on_key(" ", clock.t)
    answer(s, clock.t)
    s.on_key(str(config.SESSION_ORDER.index("finger_tapping") + 1), clock.t)

    def run(seconds, make=back):
        for _ in range(int(seconds * FPS)):
            t = clock.tick()
            s.update(make(t), t)

    run(15)
    assert s.stage == "calibrating" and s.palm_down
    palm_up = lambda t: features.extract(hand(), t, IMAGE_SIZE)     # noqa: E731
    run(0.2, palm_up)
    assert s.view()["quality"] is None                     # a short glitch shows nothing
    run(2.0, palm_up)
    assert s.view()["quality"] == "Please rest your left hand flat on the table."
    run(0.2)
    assert s.view()["quality"] is None
