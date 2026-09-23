from datetime import date, timedelta

import pytest

from rehab import config, storage
from rehab.Act import SilentSpeaker
from rehab.exercises.base import RepRecord
from rehab.exercises.grip_release import GripRelease
from rehab.Think import SessionManager
from helpers import Clock, feat

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
    rec = RepRecord("grip_release", 1, 1, 0, 1, range_high=0.8, raw_high=0.50)
    old.save_summary(old.summarize("grip_release", [rec], 1, 10))

    new = storage.SessionLog(tmp_path / "r.csv", tmp_path / "h.csv", session_id="new")
    rec2 = RepRecord("grip_release", 1, 1, 0, 1, range_high=0.9, raw_high=0.56)
    row = new.summarize("grip_release", [rec2], 1, 10)
    earlier, when = new.comparison("grip_release")
    assert when == "last week"
    msg = storage.progress_message(GripRelease, row, earlier, when)
    assert msg == "Your hand opened 12% wider than last week."


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
    monkeypatch.setitem(config.EXERCISES, "grip_release",
                        dict(config.EXERCISES["grip_release"], sets=2, reps=2))
    monkeypatch.setattr(config, "REST_BETWEEN_SETS_S", 2)
    profile = storage.new_profile()
    saved = []
    speaker = SilentSpeaker()
    s = SessionManager(speaker, profile, log, exercises=["grip_release"],
                       save_profile=lambda p: saved.append(True))
    clock = Clock()

    def run(seconds, **pose):
        for _ in range(int(seconds * 30)):
            t = clock.tick()
            s.update(feat(t, **pose), t)

    run(0.5)
    assert s.stage == "greeting"
    s.on_key(" ", clock.t)
    assert s.stage == "intro"
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
    assert "Well done" in speaker.last_text
    s.on_key(" ", clock.t)
    assert s.done


def test_recalibration_offer_times_out_to_exercise(log):
    profile = storage.new_profile()
    profile["calibration"]["grip_release"] = {
        "date": date.today().isoformat(),
        "steps": {"open": {"index": .8, "middle": .8, "ring": .8, "pinky": .8, "mean": .8},
                  "closed": {"index": .3, "middle": .3, "ring": .3, "pinky": .3, "mean": .3}},
    }
    s = SessionManager(SilentSpeaker(), profile, log, exercises=["grip_release"])
    clock = Clock()
    s.update(feat(clock.t), clock.t)
    s.on_key(" ", clock.t)
    for _ in range(int((10 + config.RECALIBRATION_OFFER_S + 1) * 30)):
        t = clock.tick()
        s.update(feat(t), t)
    assert s.stage == "exercise"


def test_progression_raises_target_after_good_session(log, monkeypatch):
    monkeypatch.setitem(config.EXERCISES, "grip_release",
                        dict(config.EXERCISES["grip_release"], sets=1, reps=2))
    profile = storage.new_profile()
    s = SessionManager(SilentSpeaker(), profile, log, exercises=["grip_release"])
    s.index = 0
    s._build_exercise()
    s.set_no = 1
    s.exercise.reps = [RepRecord("grip_release", 1, i, 0, 1, range_high=0.95, raw_high=0.6)
                       for i in (1, 2)]
    s._record_summary()
    assert profile["thresholds"]["grip_release"]["high"] == pytest.approx(0.73)
