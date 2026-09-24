from datetime import date, timedelta

import pytest

from rehab import config, storage
from rehab.Act import SilentSpeaker
from rehab.exercises.base import RepRecord
from rehab.exercises.grip_release import GripRelease
from rehab.Think import SessionManager
from helpers import Clock, answer, feat, returning_profile

OPEN, CLOSED = (25, 35, 20), (60, 80, 50)


@pytest.fixture
def log(tmp_path):
    return storage.SessionLog(tmp_path / "reps.csv", tmp_path / "history.csv", session_id="s1")


def test_profile_round_trip(tmp_path):
    p = storage.new_profile()
    p["calibration"]["grip_release"] = {"date": "2024-01-01", "steps": {"open": {"mean": 0.6}}}
    storage.save_profile(p, tmp_path / "profile.json")
    back = storage.load_profile(tmp_path / "profile.json")
    assert storage.calibrations(back)["grip_release"]["open"]["mean"] == 0.6


def test_progress_message_last_week(tmp_path):
    old = storage.SessionLog(tmp_path / "r.csv", tmp_path / "h.csv", session_id="old",
                             today=date.today() - timedelta(days=7))
    rec = RepRecord("grip_release", 1, 1, 0, 1, range_high=0.8, raw_high=0.40)
    old.save_summary(old.summarize("grip_release", [rec], 1, 10))

    new = storage.SessionLog(tmp_path / "r.csv", tmp_path / "h.csv", session_id="new")
    # 0.40 -> 0.64 openness = 20 degrees less flexion per joint: big enough to claim
    rec2 = RepRecord("grip_release", 1, 1, 0, 1, range_high=0.9, raw_high=0.64)
    row = new.summarize("grip_release", [rec2], 1, 10)
    earlier, when = new.comparison("grip_release")
    assert when == "last week"
    msg = storage.progress_message(GripRelease, row, earlier, when)
    assert msg == "Your hand opened 60% wider than last week."


def test_finger_progress_needs_20_degrees_or_a_trend(tmp_path):
    """Finger angles are not validated: a small gain is not claimed, unless it keeps rising."""
    for days, raw in ((21, 0.50), (14, 0.51), (7, 0.52)):
        old = storage.SessionLog(tmp_path / "r.csv", tmp_path / "h.csv", session_id=f"d{days}",
                                 today=date.today() - timedelta(days=days))
        old.save_summary(old.summarize("grip_release",
                                       [RepRecord("grip_release", 1, 1, 0, 1, raw_high=raw)], 1, 10))
    new = storage.SessionLog(tmp_path / "r.csv", tmp_path / "h.csv", session_id="new")
    earlier, when = new.comparison("grip_release")
    rising = new.summarize("grip_release", [RepRecord("grip_release", 1, 1, 0, 1, raw_high=0.56)], 1, 10)
    history = new.normal_history("grip_release")
    msg = storage.progress_message(GripRelease, rising, earlier, when, history=history)
    assert msg == "Your hand opened 8% wider than last week."
    # the same gain after a dip is noise
    flat = history[:1] + [dict(history[1], mean_raw_high="0.53")] + history[2:]
    msg = storage.progress_message(GripRelease, rising, earlier, when, history=flat)
    assert "%" not in msg


def test_progress_message_never_negative(log):
    rec = RepRecord("grip_release", 1, 1, 0, 1, raw_high=0.6)
    log.save_summary(log.summarize("grip_release", [rec], 1, 10))
    new = storage.SessionLog(log.rep_path, log.history_path, session_id="s2")
    row = new.summarize("grip_release", [RepRecord("grip_release", 1, 1, 0, 1, raw_high=0.5)], 1, 10)
    earlier, when = new.comparison("grip_release")
    msg = storage.progress_message(GripRelease, row, earlier, when)
    assert "%" not in msg


def test_every_other_day(log):
    rec = RepRecord("grip_squeeze", 1, 1, 0, 1, raw_high=4.0)
    yesterday = storage.SessionLog(log.rep_path, log.history_path, session_id="y",
                                   today=date.today() - timedelta(days=1))
    yesterday.save_summary(yesterday.summarize("grip_squeeze", [rec], 1, 8))
    assert not storage.should_do_today("grip_squeeze", log)
    assert storage.should_do_today("grip_release", log)
    assert storage.should_do_today("grip_squeeze", log, today=date.today() + timedelta(days=1))


def test_full_session_one_exercise(tmp_path, log, monkeypatch):
    """First session ever, from choosing the coach's name to the garden and goodbye."""
    monkeypatch.setitem(config.EXERCISES, "grip_release",
                        dict(config.EXERCISES["grip_release"], sets=2, reps=2))
    monkeypatch.setattr(config, "REST_BETWEEN_SETS_S", 2)
    profile = storage.new_profile()
    garden = storage.new_garden()
    saved, gardens = [], []
    speaker = SilentSpeaker()
    s = SessionManager(speaker, profile, log, exercises=["grip_release"],
                       save_profile=lambda p: saved.append(True), garden=garden,
                       save_garden=lambda g: gardens.append(dict(g)))
    clock = Clock()

    def run(seconds, **pose):
        for _ in range(int(seconds * 30)):
            t = clock.tick()
            s.update(feat(t, **pose), t)

    run(0.5)
    assert s.stage == "setup_name"
    s.on_key("n", clock.t)                    # not the first name ...
    s.on_key("y", clock.t)                    # ... the second
    assert profile["coach_name"] == "Robin"
    assert s.stage == "greeting"
    assert "I'm Robin" in speaker.last_text
    run(1.5)
    assert s.stage == "setup_activities"
    for yes in (True, False, True, True):     # three favourites
        answer(s, clock.t, yes)
    assert profile["chosen_activities"] == ["gardening", "cooking", "reading"]
    assert s.stage == "check_in"
    answer(s, clock.t)
    assert s.stage == "today_plan"
    run(1.5)
    assert s.stage == "intro"
    assert s.activity == "gardening"          # her favourite comes first
    run(10)
    assert s.stage == "calibrating"           # no calibration yet
    run(5.5, flex=OPEN)
    run(5.5, flex=CLOSED)
    assert s.stage == "exercise"
    assert "grip_release" in profile["calibration"]

    def rep():
        run(3, flex=OPEN)
        run(3, flex=CLOSED)

    rep(); rep()
    assert s.stage == "rest"
    s.on_key(" ", clock.t)                    # skip rest
    assert s.stage == "exercise" and s.set_no == 2

    # pause stops counting
    s.on_key(" ", clock.t)
    run(3, flex=OPEN)
    s.on_key(" ", clock.t)
    assert s.exercise.reps_this_set == 0

    # hand disappears: quality message after the grace period
    for _ in range(90):
        t = clock.tick()
        s.update(feat(t).__class__(t=t), t)
    assert s.view()["quality"] == "Please show your left hand to the camera."

    rep(); rep()
    assert s.stage == "summary"
    assert log.history("grip_release")[0]["reps_done"] == "4"
    assert len(log.rep_path.read_text().strip().splitlines()) == 5
    assert "I've saved today as your starting point." in [m.text for m in speaker.spoken]
    assert speaker.chimes == 4
    s.on_key(" ", clock.t)
    assert s.stage == "plant_choice"          # the first seed: she picks the plant
    s.on_key("y", clock.t)
    assert s.stage == "garden"
    assert garden["plots"] == [{"plant": "rose", "stage": 0}] and gardens
    assert "rose seed" in speaker.last_text
    s.on_key(" ", clock.t)
    assert s.stage == "goodbye"
    assert "keep your rose growing" in speaker.last_text
    s.on_key(" ", clock.t)
    assert s.done

    session = log.sessions()[0]
    assert session["total_reps"] == "4" and session["difficult_day"] == "no"
    assert session["check_in"] == "good" and session["garden"] == "rose:0"
    last = profile["last_session"]
    assert last["date"] == date.today().isoformat() and last["exercises"] == ["grip_release"]
    assert profile["targets"]["grip_release"]["high"] > config.EXERCISES["grip_release"]["high"]


def test_quit_mid_session_keeps_the_data(log, monkeypatch):
    monkeypatch.setitem(config.EXERCISES, "grip_release",
                        dict(config.EXERCISES["grip_release"], sets=2, reps=2))
    profile = returning_profile()
    profile["calibration"]["grip_release"] = {
        "date": date.today().isoformat(),
        "steps": {"open": {"index": .8, "middle": .8, "ring": .8, "pinky": .8, "mean": .8},
                  "closed": {"index": .3, "middle": .3, "ring": .3, "pinky": .3, "mean": .3}},
    }
    s = SessionManager(SilentSpeaker(), profile, log, exercises=["grip_release"])
    s.index = 0
    s._build_exercise()
    s.stage = "exercise"
    s.set_no = 1
    s.exercise.reps = [RepRecord("grip_release", 1, 1, 0, 1, range_high=0.9, raw_high=0.6)]
    s.stop(1.0)                               # "q" in the middle of the first set
    assert log.history("grip_release")[0]["reps_done"] == "1"
    assert log.sessions()[0]["exercises"] == "grip_release"
    assert profile["last_session"]["exercises"] == ["grip_release"]
    assert s.garden["plots"] == []            # nothing completed: the garden just waits


def test_recalibration_offer_times_out_to_exercise(log):
    profile = returning_profile()
    profile["calibration"]["grip_release"] = {
        "date": date.today().isoformat(),
        "steps": {"open": {"index": .8, "middle": .8, "ring": .8, "pinky": .8, "mean": .8},
                  "closed": {"index": .3, "middle": .3, "ring": .3, "pinky": .3, "mean": .3}},
    }
    s = SessionManager(SilentSpeaker(), profile, log, exercises=["grip_release"])
    clock = Clock()
    s.update(feat(clock.t), clock.t)
    s.on_key(" ", clock.t)                    # greeting
    answer(s, clock.t)                        # check-in
    s.on_key(" ", clock.t)                    # today's plan
    assert s.stage == "intro"
    for _ in range(int((10 + config.RECALIBRATION_OFFER_S + 1) * 30)):
        t = clock.tick()
        s.update(feat(t), t)
    assert s.stage == "exercise"


def _set_of(s, successes, failures=0, hold=0.8):
    """Finish a set with the given numbers of successful and hinted reps."""
    ex = s.exercise
    s.set_no += 1
    ex.set_no = s.set_no
    reps = [RepRecord("grip_release", s.set_no, i, 0, 1, range_high=hold + 0.05,
                      hold_value=hold, hold_raw=0.5, success=i <= successes)
            for i in range(1, successes + failures + 1)]
    ex.reps += reps
    s.stage = "exercise"
    s._set_finished(0.0)


def test_target_changes_once_per_set(log):
    s = SessionManager(SilentSpeaker(), returning_profile(), log, exercises=["grip_release"])
    s.index = 0
    s._build_exercise()
    start = s.exercise.target
    assert start == config.EXERCISES["grip_release"]["high"]    # first time: the default
    _set_of(s, 5)                             # 100%: up one step
    assert s.exercise.target == pytest.approx(start + config.TARGET_STEP)
    assert s.exercise.hyst.high == s.exercise.target
    _set_of(s, 3, 2)                          # 60%: keep
    assert s.exercise.target == pytest.approx(start + config.TARGET_STEP)


def test_target_lowered_silently_and_saved_at_session_end(log):
    profile = returning_profile()
    speaker = SilentSpeaker()
    s = SessionManager(speaker, profile, log, exercises=["grip_release"])
    s.index = 0
    s._build_exercise()
    start = s.exercise.target
    _set_of(s, 1, 3)                          # 25%: down one step, nothing said about it
    assert s.exercise.target == pytest.approx(start - config.TARGET_STEP)
    said = [m.text for m in speaker.spoken]
    assert not any("stretch" in t or "further" in t for t in said)
    s._record_summary()
    s.end_session()
    assert profile["targets"]["grip_release"]["high"] == pytest.approx(start - config.TARGET_STEP)


def _menu_session(log, **kw):
    s = SessionManager(SilentSpeaker(), returning_profile(), log, **kw)
    clock = Clock()
    s.update(feat(clock.t), clock.t)
    s.on_key(" ", clock.t)                    # after the greeting
    answer(s, clock.t)                        # check-in
    return s, clock


def test_menu_number_picks_one_exercise(log):
    s, clock = _menu_session(log)
    assert s.stage == "menu"
    items = s.view()["menu"]
    assert items[0]["text"] == "All of today's exercises" and items[0]["selected"]
    assert [i["key"] for i in items] == [str(i) for i in range(len(config.SESSION_ORDER) + 4)]
    assert items[-3]["text"] == "Arm exercises"
    assert items[-2]["text"] == "Finish for today" and items[-1]["text"] == "My profile"
    number = config.SESSION_ORDER.index("thumb_flexion") + 1
    s.on_key(str(number), clock.t)
    assert s.stage == "intro" and s.plan == ["thumb_flexion"]
    assert s.view()["title"] == "Bend and stretch your thumb"


def test_menu_arrows_and_space(log):
    s, clock = _menu_session(log)
    s.on_key("up", clock.t)                   # wraps to the last item
    assert s.view()["menu"][-1]["selected"]
    s.on_key("down", clock.t)
    s.on_key("down", clock.t)
    s.on_key(" ", clock.t)
    assert s.plan == [config.SESSION_ORDER[0]]


def test_menu_space_starts_todays_session(log):
    s, clock = _menu_session(log)
    s.on_key(" ", clock.t)
    assert s.stage == "today_plan" and s.plan == s.today
    assert s.view()["can"] == {"sections": len(s.today), "filled": 0}
    s.on_key(" ", clock.t)
    assert s.name == config.SESSION_ORDER[0]


def test_menu_back_and_after_summary(log, monkeypatch):
    monkeypatch.setitem(config.EXERCISES, "grip_release",
                        dict(config.EXERCISES["grip_release"], sets=1, reps=1))
    s, clock = _menu_session(log)
    s.on_key("2", clock.t)
    s.on_key("m", clock.t)                    # changed her mind
    assert s.stage == "menu" and s.calibration is None

    s.on_key("1", clock.t)
    for _ in range(int(10 * 30)):
        t = clock.tick()
        s.update(feat(t), t)
    for pose in (OPEN, CLOSED, OPEN, CLOSED):
        for _ in range(int(5.5 * 30)):
            t = clock.tick()
            s.update(feat(t, flex=pose), t)
    assert s.stage == "summary"
    assert log.history("grip_release")[0]["reps_done"] == "1"
    s.on_key(" ", clock.t)
    assert s.stage == "menu" and not s.done


# --- profile ---------------------------------------------------------------------

def test_profile_screen_shows_what_is_remembered(log):
    s, clock = _menu_session(log)
    s.on_key("p", clock.t)
    assert s.stage == "profile"
    v = s.view()
    assert v["screen"] == "profile"
    rows = dict(v["profile_rows"])
    assert rows["Your coach"] == "Iris"
    assert rows["Favourite activities"] == "making tea, gardening and reading"
    s.on_key(" ", clock.t)                    # back
    assert s.stage == "menu"
    s.on_key(str(len(s.menu) - 1), clock.t)   # the menu item
    assert s.stage == "profile"


def test_profile_delete_needs_the_y_key(log):
    deleted = []
    s, clock = _menu_session(log, delete_profile=lambda: deleted.append(True))
    s.on_key("p", clock.t)
    s.on_key("d", clock.t)
    assert s.stage == "profile_delete"
    s.on_key("n", clock.t)                    # keep it
    assert s.stage == "profile" and not deleted
    s.on_key("d", clock.t)
    for _ in range(60):                       # a thumbs up does not delete ...
        t = clock.tick()
        s.update(feat(t), t, gestures=[(config.YES_GESTURE, 0.9)])
    assert s.stage == "profile_delete" and not deleted
    s.update(feat(clock.tick()), clock.t, gestures=[])       # hand down
    for _ in range(30):                       # ... a thumbs down keeps it
        t = clock.tick()
        s.update(feat(t), t, gestures=[(config.NO_GESTURE, 0.9)])
    assert s.stage == "profile" and not deleted


def test_profile_delete_restarts_without_saving(log):
    saved, deleted = [], []
    s, clock = _menu_session(log, save_profile=lambda p: saved.append(dict(p)),
                             delete_profile=lambda: deleted.append(True))
    s.on_key("p", clock.t)
    s.on_key("d", clock.t)
    s.on_key("y", clock.t)
    assert deleted == [True] and s.restart and s.stage == "profile_deleted"
    before = len(saved)
    for _ in range(90):
        t = clock.tick()
        s.update(feat(t), t)
    assert s.done
    s.stop(clock.t)                           # nothing is written back after deleting
    assert len(saved) == before
    assert log.sessions() == []


def test_delete_profile_files(tmp_path):
    files = [tmp_path / "profile.json", tmp_path / "garden.json", tmp_path / "history.csv"]
    for f in files:
        f.write_text("{}")
    backup = storage.delete_profile(files + [tmp_path / "missing.csv"], keep_backup=True,
                                    backup_dir=tmp_path / "deleted")
    assert not any(f.exists() for f in files)
    assert sorted(p.name for p in backup.iterdir()) == ["garden.json", "history.csv",
                                                       "profile.json"]
    files[0].write_text("{}")
    assert storage.delete_profile(files, keep_backup=False) is None
    assert not files[0].exists()
    assert storage.load_profile(files[0])["coach_name"] is None      # a first day again


def test_profile_overview_first_day():
    from rehab import memory
    rows = dict(memory.profile_overview(storage.new_profile(), storage.new_garden(), [],
                                        storage.load_content("activities")))
    assert rows["Your coach"] == "Not chosen yet"
    assert rows["Sessions"] == "None yet" and rows["Garden"] == "No plants yet"
