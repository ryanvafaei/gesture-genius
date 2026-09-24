"""
Bubble pinch, two-hand match, memory pairs, the finger piano, the demo hand,
finger counting, ratings, Stop, repeat, the menu with nine exercises, guest
data folders and the report tool.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from rehab import config, demo, features, sound, storage
from rehab.Act import SilentSpeaker
from rehab.calibration import CalibrationRoutine
from rehab.exercises import EXERCISES
from rehab.exercises.bubble_pinch import BubblePinch
from rehab.exercises.memory_pairs import MemoryPairs
from rehab.exercises.thumb_opposition import ThumbOpposition
from rehab.exercises.two_hand_match import TwoHandMatch
from rehab.Think import FingerCount, SessionManager
from helpers import Clock, answer, calibrate, feat, returning_profile, run, texts
from synthetic_hand import IMAGE_SIZE, hand

FIST = dict(flex=(80, 95, 60), thumb_out=0.0)


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

def test_count_extended_fingers():
    count = lambda **k: features.count_extended(feat(**k))       # noqa: E731
    assert count() == 5
    assert count(**FIST) == 0
    assert count(**FIST, finger_flex={"index": (0, 0, 0)}) == 1
    assert count(**FIST, finger_flex={"index": (0, 0, 0), "middle": (0, 0, 0),
                                      "ring": (0, 0, 0)}) == 3
    assert count(thumb_out=0.0) == 4
    assert features.count_extended(None) == 0


def test_finger_count_needs_the_hand_down_first_and_a_steady_hold():
    fc = FingerCount(hold_s=1.0)
    three = feat(**FIST, finger_flex={"index": (0, 0, 0), "middle": (0, 0, 0), "ring": (0, 0, 0)})
    assert fc.update(three, 0.0) is None and fc.update(three, 2.0) is None   # not armed yet
    assert fc.update(feat(**FIST), 2.1) is None                             # hand down: armed
    assert fc.update(three, 2.2) is None
    assert fc.update(three, 3.3) == 3


# --- bubble pinch ----------------------------------------------------------------------------

PINCH = dict(flex=(15, 20, 10), thumb_out=0.6, thumb_touch="index")


def test_bubble_pinch_counts_pinches_and_shows_the_bubble():
    cal = calibrate(BubblePinch, [dict(), PINCH])
    assert BubblePinch.calibration_valid(cal)
    ex = BubblePinch(calibration=cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    assert texts(ex.first_messages()) == ["Pinch your thumb and finger together."]
    run(ex, 0.5, clock)
    for _ in range(3):
        run(ex, 2.6, clock, **PINCH)
        assert ex.display["bubble"]["pinching"] or ex.display["phase_label"] == "OPEN"
        run(ex, 1.6, clock)
    assert len(ex.reps) == 3
    assert all(r.success for r in ex.reps)
    assert 0.0 < ex.reps[0].raw_high <= 1.0               # pinch closure, higher is better
    assert ex.display["bubble"]["popped_t"] is not None
    assert ex.display["demo_key"] in ("pinch", "release")


def test_bubble_pinch_rejects_a_swapped_calibration():
    assert not BubblePinch.calibration_valid({"open": {"dist": 0.2}, "pinch": {"dist": 1.0}})


# --- two-hand match ------------------------------------------------------------------------

def both(t, affected=None, other=None):
    f = features.extract(hand(**(affected or {})), t, IMAGE_SIZE)
    if other is not False:
        f.other = features.extract(hand(handedness="Right", **(other or {})), t, IMAGE_SIZE)
    return f


def two_hand_cal():
    routine = CalibrationRoutine(TwoHandMatch)
    clock = Clock()
    routine.start(clock.t)
    for pose in ({}, FIST):
        name = routine.step.name
        while not routine.done and routine.step.name == name:
            t = clock.tick()
            routine.update(both(t, pose, pose), t)
    assert routine.done
    return routine.result


def test_two_hand_match_needs_both_hands():
    f = both(0.0, other=False)
    assert f.quality_problem(True, need_both=True) == "no_other_hand"
    assert both(0.0).quality_problem(True, need_both=True) is None


def test_two_hand_match_logs_how_well_the_hands_move_together():
    cal = two_hand_cal()
    assert TwoHandMatch.calibration_valid(cal)
    ex = TwoHandMatch(calibration=cal, hand="Left")
    clock = Clock()
    ex.start_set(1, clock.t)
    for together in (True, False):
        for pose in ({}, FIST):
            other = pose if together else ({} if pose else FIST)
            for _ in range(int(1.6 * 30)):
                t = clock.tick()
                ex.update(both(t, pose, other), t)
    assert len(ex.reps) == 2
    sym_together, sym_apart = ex.reps[0].extra["symmetry"], ex.reps[1].extra["symmetry"]
    assert sym_together > 0.8 and sym_apart < sym_together
    assert ex.reps[0].raw_high == sym_together
    assert set(ex.display["pair_values"]) == {"left", "right"}


# --- memory pairs ----------------------------------------------------------------------------

def pointer_at(card_rect):
    x0, y0, x1, y1 = card_rect
    pts = np.zeros((21, 2))
    pts[8] = ((x0 + x1) / 2 * IMAGE_SIZE[0], (y0 + y1) / 2 * IMAGE_SIZE[1])
    return SimpleNamespace(present=True, image_points=pts, image_size=IMAGE_SIZE,
                           correct_hand=True)


def pairs_of(ex):
    by_icon = {}
    for i, c in enumerate(ex._board["cards"]):
        by_icon.setdefault(c["icon"], []).append(i)
    return list(by_icon.values())


def test_memory_pairs_by_pointing_and_keys():
    ex = MemoryPairs(rng=np.random.default_rng(1), params={"reps": 1})
    clock = Clock()
    ex.start_set(1, clock.t)
    said = ex.update(None, clock.t)
    assert texts(said) == ["Find the 2 pairs."]
    (a1, a2), (b1, b2) = pairs_of(ex)
    # a wrong pair: both stay up for a moment, then turn back; no failure words
    said = ex.select(a1, clock.tick()) + ex.select(b1, clock.tick())
    assert "Not a pair. Try to remember where they are." in texts(said)
    assert not any("wrong" in s.lower() for s in texts(said))
    for _ in range(int(2.2 * 30)):
        ex.update(None, clock.tick())
    assert all(c["state"] == "down" for c in ex._board["cards"])
    # pointing and holding still turns a card over
    rect = ex._board["cards"][a1]["rect"]
    for _ in range(int(1.7 * 30)):
        said = ex.update(pointer_at(rect), clock.tick())
        if ex._board["cards"][a1]["state"] == "up":
            break
    assert ex._board["cards"][a1]["state"] == "up"
    said = ex.select(a2, clock.tick())
    assert said[0].kind == "chime"
    said = ex.select(b1, clock.tick()) + ex.select(b2, clock.tick())
    assert "Well done, you found all the pairs." in texts(said)
    assert len(ex.reps) == 1 and ex.set_done
    rec = ex.reps[0]
    assert rec.extra["turns"] == 3 and rec.extra["pairs"] == 2
    assert rec.range_high == pytest.approx(2 / 3)


def test_memory_pairs_levels_up_after_good_boards():
    ex = MemoryPairs(rng=np.random.default_rng(2), params={"reps": 5, "boards_to_level_up": 2})
    clock = Clock()
    ex.start_set(1, clock.t)
    for _ in range(2):
        ex.update(None, clock.tick())
        for a, b in pairs_of(ex):
            ex.select(a, clock.tick())
            ex.select(b, clock.tick())
    assert ex.level == 2 and ex.pairs == 3


def test_memory_pairs_starts_without_measuring_and_takes_number_keys(log):
    s, clock = menu_session(log)
    s.on_key(str(config.SESSION_ORDER.index("memory_pairs") + 1), clock.t)
    for _ in range(int(12 * 30)):
        t = clock.tick()
        s.update(feat(t), t)
        if s.stage == "exercise":
            break
    assert s.stage == "exercise" and s.calibration is None      # no measuring for a game
    s.update(feat(clock.t), clock.tick())
    s.on_key("1", clock.t)
    assert s.exercise._board["cards"][0]["state"] == "up"


# --- finger piano ----------------------------------------------------------------------------

def test_finger_piano_plays_a_note_for_correct_touches_only():
    cal = calibrate(ThumbOpposition, [dict(), dict(thumb_touch="index")])
    ex = ThumbOpposition(calibration=cal, level=1)
    clock = Clock()
    ex.start_set(1, clock.t)
    run(ex, 0.5, clock)
    said = run(ex, 0.8, clock, thumb_touch="middle") + run(ex, 0.8, clock)      # wrong
    assert not [m for m in said if m.kind == "note"]
    said = run(ex, 0.8, clock, thumb_touch="index") + run(ex, 0.8, clock)       # right
    said += run(ex, 0.8, clock, thumb_touch="middle") + run(ex, 0.8, clock)
    assert [m.note for m in said if m.kind == "note"] == [0, 1]


def test_notes_are_played_at_once_never_queued():
    from rehab.exercises.base import Say
    sp = SilentSpeaker()
    sp.say(Say("", "note", note=3))
    assert sp.notes == [3] and sp.spoken == []


def test_note_files(tmp_path):
    samples = sound.tone(60)
    assert samples.dtype == np.int16 and len(samples) == int(sound.NOTE_S * sound.RATE)
    assert np.abs(samples).max() < 32767 * config.NOTE_VOLUME + 1
    notes = sound.Notes(folder=tmp_path, enabled=False)
    assert notes.file(sound.TUNE[0]).is_file()
    notes.play(0)                              # no player: nothing happens


# --- demo hand -------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(EXERCISES))
def test_every_exercise_has_a_demo(name):
    box = (100, 50, 500, 450)
    for t in (0.0, 0.7, 2.1, 5.3):
        for pts in demo.demo_points(name, t, box):
            # a hand, or the seated figure of an arm exercise
            assert pts.shape in ((21, 2), (demo.ARM_POINTS, 2)) and np.isfinite(pts).all()
            assert (pts[:, 0] >= box[0] - 5).all() and (pts[:, 0] <= box[2] + 5).all()
            assert (pts[:, 1] >= box[1] - 5).all() and (pts[:, 1] <= box[3] + 5).all()
    assert demo.POSES.get(name), "a still picture for each position"


def test_demo_keys_match_the_exercise_phases():
    for name, cls in EXERCISES.items():
        for phase in getattr(cls, "phases", ()) or ():
            assert phase.key in demo.POSES[name], (name, phase.key)
    assert demo.phase_points("grip_release", "open", (0, 0, 100, 100))
    assert demo.phase_points("grip_release", "nope", (0, 0, 100, 100)) == []
    assert len(demo.demo_points("two_hand_match", 0.0, (0, 0, 400, 200))) == 2


# --- menu, plan, short sets ------------------------------------------------------------------

def test_daily_plan_and_menu_keys(log):
    s, clock = menu_session(log)
    assert "memory_pairs" in s.today
    assert "bubble_pinch" not in s.today and "two_hand_match" not in s.today
    keys = [i["key"] for i in s.view()["menu"]]
    assert keys[:10] == [str(i) for i in range(10)] and keys[-2:] == ["E", "P"]
    s.on_key("8", clock.t)
    assert s.plan == ["two_hand_match"]
    s.on_key("m", clock.t)
    s.on_key("e", clock.t)                   # finish for today
    assert s.stage in ("goodbye", "garden", "plant_choice")


def test_short_session_uses_one_short_set(log):
    s, clock = menu_session(log, short=True)
    s.on_key("1", clock.t)
    assert s.exercise.sets == 1 and s.exercise.reps_per_set == config.SHORT_SESSION["reps"]
    s.on_key("m", clock.t)
    s.on_key(str(config.SESSION_ORDER.index("thumb_opposition") + 1), clock.t)
    assert s.exercise.reps_per_set == config.SHORT_SESSION["rounds"]


def test_step_label_during_a_plan(log):
    s, clock = menu_session(log)
    s.on_key("0", clock.t)
    s.on_key(" ", clock.t)                   # past "today we'll do ..."
    assert s.stage == "intro"
    v = s.view()
    assert v["step_label"] == f"Exercise 1 of {len(s.plan)}"
    assert v["demo"]["exercise"] == s.plan[0]
    assert v["stop_hint"] and "R: repeat" in v["footer"]


# --- stop and repeat -------------------------------------------------------------------------

def test_stop_shows_the_safety_screen_and_is_logged(log):
    s, clock = menu_session(log)
    s.on_key("1", clock.t)
    s.on_key("s", clock.t)
    assert s.stage == "safety"
    v = s.view()
    assert v["screen"] == "safety" and v["card_event"].type == "SafetyStop"
    said = texts(s.speaker.spoken)
    assert "Let's stop here. Please sit down and rest." in said
    assert any("call 112" in t for t in said)
    for _ in range(int(60 * 30)):           # never moves on by itself
        s.update(feat(), clock.tick())
    assert s.stage == "safety"
    s.on_key(" ", clock.t)                   # I feel fine
    assert s.stage == "menu"
    s.stop(clock.t)
    row = log.sessions()[0]
    assert row["safety_stop"] == "yes" and "Stop" in row["note"]


def test_repeat_says_the_screen_again(log):
    s, clock = menu_session(log)
    s.on_key("s", clock.t)
    before = texts(s.speaker.spoken).count("Let's stop here. Please sit down and rest.")
    s.on_key("r", clock.t)
    after = texts(s.speaker.spoken).count("Let's stop here. Please sit down and rest.")
    assert after == before + 1
    s.on_key(" ", clock.t)
    s.on_key("r", clock.t)
    assert s.speaker.spoken[-1].text.startswith("Press a number to choose")


# --- ratings -------------------------------------------------------------------------------

def test_ratings_with_fingers_keys_and_no_answer(log):
    s, clock = menu_session(log, rating_questions=("exertion", "enjoyment", "ease"))
    s.progress.completed.append("grip_release")
    s.on_key("e", clock.t)
    assert s.stage == "rating" and s.view()["screen"] == "rating"
    assert s.view()["card_event"].get("key") == "exertion"
    three = dict(**FIST, finger_flex={"index": (0, 0, 0), "middle": (0, 0, 0), "ring": (0, 0, 0)})
    for pose, seconds in ((FIST, 0.3), (three, 2.0)):
        for _ in range(int(seconds * 30)):
            s.update(feat(**pose), clock.tick())
    assert s.ratings["exertion"] == 3
    assert s.view()["card_event"].get("key") == "enjoyment"
    s.on_key("4", clock.t)
    assert s.ratings["enjoyment"] == 4
    for _ in range(int((config.RATING_TIMEOUT_S + 1) * 30)):     # no answer: left blank
        s.update(feat(**FIST), clock.tick())
        if s.stage != "rating":
            break
    assert s.ratings["ease"] == "" and s.stage != "rating"


# --- guests and the report -------------------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path):
    old = config.DATA_DIR
    config.use_data_dir(tmp_path / "data")
    yield tmp_path / "data"
    config.use_data_dir(old)


def test_use_data_dir_keeps_files_apart(data_dir):
    p = storage.new_profile()
    p["coach_name"] = "Robin"
    storage.save_profile(p)
    assert (data_dir / "profile.json").is_file()
    assert storage.load_profile()["coach_name"] == "Robin"
    assert storage.SessionLog().rep_path == data_dir / "reps.csv"
    assert all(str(path).startswith(str(data_dir)) for path in storage.profile_files())


def test_report_tool_writes_tables_and_charts(data_dir, tmp_path):
    from rehab.exercises.base import RepRecord
    from tools import report

    def one_session(folder, sid, exertion):
        log = storage.SessionLog(folder / "reps.csv", folder / "history.csv", session_id=sid)
        recs = [RepRecord("grip_release", 1, i, 0, 1, range_high=0.8, raw_high=0.6, hints=0)
                for i in range(1, 4)]
        log.save_summary(log.summarize("grip_release", recs, 1, 10))
        log.save_session({"session_id": sid, "date": "2026-09-24", "total_reps": 3,
                          "exertion": exertion, "enjoyment": 4, "safety_stop": "no"})

    one_session(data_dir, "a", 2)
    one_session(data_dir, "b", 3)
    one_session(data_dir / "guests" / "20260924-100000", "g", 4)
    out = report.build(data_dir, tmp_path / "report")
    summary = (out / "summary.csv").read_text()
    assert "grip_release" in summary and "guests" in summary and "eleanor" in summary
    md = (out / "report.md").read_text()
    assert "Ratings" in md and "| eleanor | exertion |" in md
    pytest.importorskip("matplotlib")
    assert (out / "ratings.png").is_file() and (out / "progress_grip_release.png").is_file()
