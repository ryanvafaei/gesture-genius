"""
Finger counting for the ratings, touch and lift detection (ring finger,
neighbours, drift), short sessions, Skip and the menu toolbar.
"""

import sys
import time

import cv2
import numpy as np
import pytest

from rehab import config, features, storage
from rehab.Act import Display, SilentSpeaker
from rehab.exercises.finger_tapping import FingerTapping
from rehab.exercises.thumb_opposition import ThumbOpposition
from rehab.Think import FingerCount, SessionManager
from helpers import OPPOSITION_POSES, Clock, FPS, answer, calibrate, feat, returning_profile, run
from tools.report_job import ReportJob

FIST = dict(flex=(80, 95, 60), thumb_out=0.0)
STRAIGHT = (0, 0, 0)


@pytest.fixture
def log(tmp_path):
    return storage.SessionLog(tmp_path / "reps.csv", tmp_path / "history.csv", session_id="s1")


def menu_session(log, **kw):
    s = SessionManager(SilentSpeaker(), returning_profile(), log, **kw)
    clock = Clock()
    s.update(feat(clock.t), clock.t)
    s.on_key(" ", clock.t)                    # after the greeting
    answer(s, clock.t)                        # check-in
    assert s.stage == "menu"
    return s, clock


# --- finger counting ------------------------------------------------------------------------

def test_count_with_straight_fingers_that_are_a_little_bent():
    # the old rule (openness > 0.75) counted none of these: flexion adds up to 65-75 degrees
    for flex in [(20, 25, 20), (25, 30, 20)]:
        assert features.count_extended(feat(flex=flex, thumb_out=0.0)) == 4
        assert features.count_extended(feat(flex=flex)) == 5


def test_count_two_three_four_and_half_bent_fingers():
    up = (15, 20, 15)
    count = lambda fingers: features.count_extended(      # noqa: E731
        feat(**FIST, finger_flex={k: up for k in fingers}))
    assert count(["index", "middle"]) == 2
    assert count(["index", "middle", "ring"]) == 3
    assert count(["index", "middle", "ring", "pinky"]) == 4
    assert features.count_extended(feat(flex=(45, 60, 40), thumb_out=0.0)) == 0


def test_a_thumb_bent_across_the_palm_is_not_counted():
    assert features.count_extended(feat(**FIST, finger_flex={"index": STRAIGHT})) == 1


def test_finger_count_ignores_a_single_missed_frame():
    fc = FingerCount(hold_s=1.0)
    three = feat(**FIST, finger_flex={"index": STRAIGHT, "middle": STRAIGHT, "ring": STRAIGHT})
    two = feat(**FIST, finger_flex={"index": STRAIGHT, "middle": STRAIGHT})
    t = 0.0
    fc.update(feat(**FIST), t)                                 # hand down: armed
    got = None
    for i in range(int(1.5 * FPS)):
        t += 1 / FPS
        f = two if i in (10, 20) else three                    # the ring finger lost for a frame
        got = fc.update(f, t) or got
    assert got == 3


# --- touch your fingertips ------------------------------------------------------------------

def opposition():
    ex = ThumbOpposition(calibration=calibrate(ThumbOpposition, OPPOSITION_POSES), level=1)
    clock = Clock()
    ex.start_set(1, clock.t)
    run(ex, 0.5, clock)
    return ex, clock


def run_with(ex, seconds, clock, change, **pose):
    """Like helpers.run, with change(features) applied to every frame."""
    for _ in range(int(seconds * FPS)):
        t = clock.tick()
        f = feat(t, **pose)
        change(f)
        ex.update(f, t)


def test_opposition_needs_a_touch_on_every_finger_in_calibration():
    cal = calibrate(ThumbOpposition, OPPOSITION_POSES)
    assert ThumbOpposition.calibration_valid(cal)
    old = {"open": cal["open"], "touch": cal["touch"]}          # only the index touch
    assert not ThumbOpposition.calibration_valid(old)


def test_ring_touch_counts_when_the_middle_fingertip_is_close_too():
    ex, clock = opposition()
    for finger in ["index", "middle"]:
        run(ex, 0.8, clock, thumb_touch=finger)
        run(ex, 0.8, clock)
    assert ex._round["seq"][ex._round["step"]] == "ring"

    def middle_near(f):
        # tracking puts the thumb nearly as close to the middle fingertip
        f.thumb_tip_dist["middle"] = f.thumb_tip_dist["ring"] + 0.05
        f.thumb_tip_dist_image["middle"] = f.thumb_tip_dist_image["ring"] + 0.05
    run_with(ex, 0.8, clock, middle_near, thumb_touch="ring")
    run(ex, 0.8, clock)
    r = ex._round
    assert r["correct"] == 3 and r["wrong"] == 0


def test_a_clear_wrong_finger_is_still_wrong():
    ex, clock = opposition()
    run(ex, 0.8, clock, thumb_touch="ring")                     # asked for the index
    run(ex, 0.8, clock)
    assert ex._round["wrong"] == 1 and ex._round["correct"] == 0


def test_little_finger_touch_is_found():
    ex, clock = opposition()
    for finger in ["index", "middle", "ring", "pinky"]:
        run(ex, 0.8, clock, thumb_touch=finger)
        run(ex, 0.8, clock)
    assert ex._round["correct"] == 4 and ex._round["wrong"] == 0


# --- lift one finger at a time --------------------------------------------------------------

# the hand lies flat with its back to the camera
BACK = dict(back_to_camera=True)


def tapping(**params):
    cal = calibrate(FingerTapping, [BACK, dict(BACK, finger_flex={"index": (-25, 0, 0)})])
    ex = FingerTapping(calibration=cal, params=params or None)
    clock = Clock()
    ex.start_set(1, clock.t)
    run(ex, 0.5, clock, **BACK)
    return ex, clock


def lift(ex, clock, pose):
    run(ex, 0.8, clock, finger_flex=pose, **BACK)
    run(ex, 0.8, clock, **BACK)


def test_ring_finger_has_a_lower_threshold():
    ex, clock = tapping()
    assert ex.thresholds["ring"] < ex.thresholds["middle"] < ex.thresholds["index"]
    for finger in ["index", "middle"]:
        lift(ex, clock, {finger: (-25, 0, 0)})
    # a small ring lift (40% of the index lift): too small for the index threshold
    lift(ex, clock, {"ring": (-10, 0, 0)})
    r = ex._round
    assert r["correct"] == 3 and r["wrong"] == 0


def test_neighbour_rising_with_the_prompted_finger_does_not_take_the_lift():
    ex, clock = tapping()
    lift(ex, clock, {"index": (-25, 0, 0)})
    # middle finger asked; the ring finger comes up with it and even scores higher
    lift(ex, clock, {"middle": (-25, 0, 0), "ring": (-20, 0, 0)})
    r = ex._round
    assert r["correct"] == 2 and r["wrong"] == 0


def test_hand_settling_on_the_table_is_not_a_lift():
    ex, clock = tapping()
    n = int(30 * FPS)
    for i in range(n):
        a = -12.0 * i / (n - 1)                 # every finger drifts by 12 degrees in 30 s
        t = clock.tick()
        ex.update(feat(t, finger_flex={k: (a, 0, 0) for k in features.FINGERS}, **BACK), t)
    run(ex, 2.0, clock, finger_flex={k: (-12, 0, 0) for k in features.FINGERS}, **BACK)
    r = ex._round
    assert r["correct"] == 0 and r["wrong"] == 0 and ex._lifted is None


def test_coupled_neighbours_do_not_spoil_isolation():
    ex, clock = tapping()
    said = []
    for _ in range(int(0.8 * FPS)):
        t = clock.tick()
        said += ex.update(feat(t, finger_flex={"index": (-25, 0, 0), "middle": (-6, 0, 0)}, **BACK), t)
    said += run(ex, 0.8, clock, **BACK)
    assert "Try to keep the other fingers resting on the table." not in [m.text for m in said]


# --- short sessions ---------------------------------------------------------------------------

def test_short_session_values():
    assert config.SHORT_SESSION["sets"] == 1 and config.SHORT_SESSION["reps"] == 3
    assert config.SHORT_SESSION["rest_between_exercises_s"] == 5


def test_short_session_rests_five_seconds_between_exercises(log):
    s, clock = menu_session(log, short=True)
    s.on_key(" ", clock.t)                                      # all of today's
    s.on_key(" ", clock.t)                                      # past today's plan
    assert s.stage == "intro"
    assert s.exercise.sets == 1 and s.exercise.reps_per_set == 3
    s._exercise_finished(clock.t)
    assert s.stage == "rest" and s._rest_s == 5


# --- skip -----------------------------------------------------------------------------------

def test_skip_an_exercise_goes_to_the_next_one(log):
    s, clock = menu_session(log)
    s.on_key(" ", clock.t)
    s.on_key(" ", clock.t)
    first = s.name
    assert s.stage == "intro" and s.view()["skip"]["label"] == "Skip exercise"
    s.on_key("k", clock.t)
    assert s.stage == "intro" and s.name != first and s.skipped == [first]
    s.on_click("skip", clock.t)                                 # the button does the same
    assert len(s.skipped) == 2


def test_skip_the_only_exercise_shows_the_summary(log):
    s, clock = menu_session(log)
    s.on_key("1", clock.t)
    s.on_key("k", clock.t)
    assert s.stage == "summary"
    assert "Skipped: grip_release." in s._session_row(None)["note"]


def test_skip_a_rest(log):
    s, clock = menu_session(log)
    s.on_key(" ", clock.t)
    s.on_key(" ", clock.t)
    first = s.name
    s._rest(clock.t, 30, then="next_exercise")
    assert s.view()["skip"]["label"] == "Skip rest"
    s.on_key("k", clock.t)
    assert s.stage == "intro" and s.name != first and s.skipped == []


def test_no_skip_in_the_menu(log):
    s, clock = menu_session(log)
    assert "skip" not in s.view()
    s.on_key("k", clock.t)
    assert s.stage == "menu"


# --- toolbar --------------------------------------------------------------------------------

class FakeJob:
    def __init__(self):
        self.calls = 0

    def status(self):
        self.calls += 1
        return "running", "Making the report..."


def test_toolbar_is_hidden_and_only_in_the_menu(log):
    s, clock = menu_session(log)
    v = s.view()
    assert v["toolbar"]["open"] is False
    s.on_key("t", clock.t)                                      # closed: t does nothing
    assert s.short is False
    s.on_key("i", clock.t)
    assert s.view()["toolbar"]["open"] is True
    s.on_key("1", clock.t)                                      # an exercise: no toolbar
    assert s.stage == "intro" and "toolbar" not in s.view() and s.toolbar_open is False


def test_toolbar_short_toggle_changes_the_next_exercise(log):
    s, clock = menu_session(log)
    s.on_click("toolbar", clock.t)
    s.on_click("short", clock.t)
    assert s.short is True
    assert [i["on"] for i in s.view()["toolbar"]["items"] if i["id"] == "short"] == [True]
    s.on_key("1", clock.t)
    assert s.exercise.sets == 1 and s.exercise.reps_per_set == config.SHORT_SESSION["reps"]


def test_toolbar_new_guest_ends_the_session_for_main(log):
    s, clock = menu_session(log)
    s.on_key("i", clock.t)
    s.on_key("b", clock.t)                                      # not a guest: nothing to go back from
    assert s.switch_user is None and not s.done
    s.on_key("g", clock.t)
    assert s.switch_user == "new_guest" and s.done


def test_toolbar_report(log):
    jobs = []
    s, clock = menu_session(log, make_report=lambda: jobs.append(FakeJob()) or jobs[-1])
    s.on_key("i", clock.t)
    s.on_key("o", clock.t)
    s.on_key("o", clock.t)                                      # still running: not again
    assert len(jobs) == 1
    items = {i["id"]: i for i in s.view()["toolbar"]["items"]}
    assert items["report"]["status"] == "Making the report..."


def test_display_maps_clicks_to_buttons(log):
    s, clock = menu_session(log)
    s.on_key("i", clock.t)
    display = Display(screen=(1512, 982))
    frame = np.full((720, 1280, 3), 90, np.uint8)
    display.render(frame, s.view(), feat(clock.t))
    spots = dict((action, rect) for rect, action in display.hotspots)
    assert set(spots) == {"toolbar", "short", "new_guest", "report"}
    x0, y0, x1, y1 = spots["short"]
    display._on_mouse(cv2.EVENT_LBUTTONUP, (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
    assert display.pop_click() == "short"
    assert display.pop_click() is None
    display._on_mouse(cv2.EVENT_LBUTTONUP, 5, 500, 0, None)    # not on a button
    assert display.pop_click() is None


def wait(job, seconds=60):
    end = time.time() + seconds
    while job.status()[0] == "running" and time.time() < end:
        time.sleep(0.1)
    return job.status()


def test_report_job_runs_the_report_tool(tmp_path):
    job = ReportJob(tmp_path, open_when_done=False)
    state, text = wait(job)
    assert state == "done" and "report" in text
    assert (tmp_path / "report" / "report.md").is_file()


def test_report_job_failure(tmp_path):
    job = ReportJob(tmp_path, open_when_done=False,
                    command=[sys.executable, "-c", "raise SystemExit(3)"])
    assert wait(job)[0] == "failed"
