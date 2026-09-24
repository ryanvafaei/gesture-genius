"""
Motivation features with fake histories (no camera, no speaker):
greetings, personal bests, adaptive targets, difficult days, praise,
activities and milestones, the garden, and data safety.
"""

import collections
import json
import random
from datetime import date, timedelta

import numpy as np
import pytest

from rehab import config, garden as garden_model, memory, storage
from rehab.Act import SilentSpeaker, enqueue, interrupts, next_message
from rehab.events import event
from rehab.exercises.base import RepRecord, Say
from rehab.feedback import Feedback
from rehab.progress import PersonalBests, SessionProgress, grow_range, next_target
from rehab.Think import SessionManager, YesNo
from helpers import Clock, answer, feat, returning_profile

TODAY = date(2026, 9, 23)
GRIP_CAL = {"open": {"index": .8, "middle": .8, "ring": .8, "pinky": .8, "mean": .8},
            "closed": {"index": .3, "middle": .3, "ring": .3, "pinky": .3, "mean": .3}}


def log_at(tmp_path, day=TODAY, sid="s1"):
    return storage.SessionLog(tmp_path / "reps.csv", tmp_path / "history.csv",
                              session_id=sid, today=day)


def rep(n=1, set_no=1, hold=0.8, raw=0.5, success=True, stability=0.05, hints=0, fingers=None):
    return RepRecord("grip_release", set_no, n, 0, 1, range_high=hold + 0.05, hold_value=hold,
                     hold_raw=raw, success=success, hold_stability=stability, hints=hints,
                     extra={"finger_hold": fingers} if fingers else {})


class FakeExercise:
    """Just enough of a TwoPhaseExercise for SessionProgress."""
    name = "grip_release"
    range_steps = ("closed", "open")

    def __init__(self, target=0.7, set_no=1, reps=None, cal=None):
        self.target = target
        self.set_no = set_no
        self.reps = reps or []
        self.cal = cal or {}


def session_with_reps(tmp_path, profile, day, values, sid, difficult=False):
    """A past session: one history row and bests recorded from these hold values."""
    log = log_at(tmp_path, day, sid)
    prog = SessionProgress(profile, log, today=day)
    reps = [rep(i + 1, raw=v) for i, v in enumerate(values)]
    for r in reps:
        prog.bests.record("grip_release", "range", r.hold_raw)
    row = log.summarize("grip_release", reps, 1, len(reps), difficult_day=difficult)
    log.save_summary(row)
    return log


# ---------------------------------------------------------------------------
# Greetings and coach memory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("days, kind", [(None, "first"), (0, "same_day"), (1, "one_day"),
                                        (5, "few_days"), (10, "long_gap")])
def test_greeting_by_days_since_last_session(days, kind):
    profile = returning_profile()
    if days is not None:
        profile["last_session"] = {"date": (TODAY - timedelta(days=days)).isoformat(),
                                   "highlight": {"type": "PersonalBest", "exercise": "thumb_flexion",
                                                 "metric": "range", "level": "week"}}
    ev = memory.greeting(profile, TODAY)
    assert ev.get("kind") == kind
    text = " ".join(m.text for m in Feedback(rng=random.Random(1)).words(ev))
    assert "Eleanor" in text
    assert "days" not in text and "missed" not in text      # a gap is never a failure
    if kind == "one_day":
        assert text == "Good to see you, Eleanor. Yesterday you reached a new best with your thumb."
    if kind == "few_days":
        assert text == "Hello Eleanor. Last time you reached a new best with your thumb."
    if kind == "long_gap":
        assert "start gently" in text


def test_greeting_after_a_difficult_day_and_first_meeting():
    profile = returning_profile()
    profile["last_session"] = {"date": (TODAY - timedelta(days=2)).isoformat(),
                               "difficult_day": True}
    assert memory.greeting(profile, TODAY).get("kind") == "after_difficult"
    first = memory.greeting(returning_profile(), TODAY)
    assert [m.text for m in Feedback().words(first)] == [
        "Hello Eleanor, I'm Iris. I'll be your practice partner."]


def test_only_stored_facts_one_at_most():
    profile = returning_profile()
    profile["last_session"] = {"date": (TODAY - timedelta(days=1)).isoformat(), "highlight": None}
    text = " ".join(m.text for m in Feedback().words(memory.greeting(profile, TODAY)))
    assert text == "Good to see you, Eleanor."
    # a best of that day only is not recalled as "a new best"
    profile["last_session"]["highlight"] = {"type": "PersonalBest", "exercise": "grip_release",
                                            "metric": "range", "level": "today"}
    text = " ".join(m.text for m in Feedback().words(memory.greeting(profile, TODAY)))
    assert text == "Good to see you, Eleanor."


def test_unreadable_profile_starts_as_a_first_session(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{ this is not json")
    profile = storage.load_profile(path)
    assert profile["last_session"] is None and profile["coach_name"] is None
    assert memory.greeting(profile, TODAY).get("kind") == "first"
    assert list(tmp_path.glob("profile.json.broken-*"))         # kept for inspection
    # wrongly typed fields fall back to their default
    path.write_text(json.dumps({"last_session": {"date": "yesterday"}, "targets": [1, 2],
                                "thresholds": {"grip_release": {"high": 0.8}}}))
    profile = storage.load_profile(path)
    assert profile["last_session"] is None and profile["targets"] == {}
    path.write_text(json.dumps({"thresholds": {"grip_release": {"high": 0.8}}, "user": "Ellie"}))
    profile = storage.load_profile(path)                        # older profile file
    assert profile["targets"] == {"grip_release": {"high": 0.8}} and profile["name"] == "Ellie"


# ---------------------------------------------------------------------------
# Personal bests
# ---------------------------------------------------------------------------

def test_first_session_is_the_starting_point(tmp_path):
    profile = returning_profile()
    prog = SessionProgress(profile, log_at(tmp_path), today=TODAY)
    ex = FakeExercise()
    for i, v in enumerate((0.5, 0.6, 0.7), start=1):
        r = rep(i, raw=v)
        ex.reps.append(r)
        assert not [e for e in prog.rep_events(ex, r) if e.type == "PersonalBest"]
    row = log_at(tmp_path).summarize("grip_release", ex.reps, 1, 3)
    events = prog.exercise_finished(ex, row)
    assert [e.type for e in events] == ["StartingPoint"]
    assert profile["personal_bests"]["grip_release"]["range"]["value"] == 0.7


def test_bests_levels_and_noise_threshold(tmp_path):
    profile = returning_profile()
    session_with_reps(tmp_path, profile, TODAY - timedelta(days=20), [0.60], "old")
    session_with_reps(tmp_path, profile, TODAY - timedelta(days=3), [0.50], "recent")
    pb = PersonalBests(profile, TODAY)
    assert pb.level("grip_release", "range", 0.505) is None     # within noise of 0.50
    assert pb.level("grip_release", "range", 0.52) == "week"    # beats this week's 0.50
    assert pb.level("grip_release", "range", 0.62) == "all_time"
    pb.record("grip_release", "range", 0.53)
    assert pb.level("grip_release", "range", 0.54) is None      # only 2% above today's 0.53
    pb.record("grip_release", "range", 0.40)
    assert pb.level("grip_release", "range", 0.55) == "week"


def test_today_level_below_this_weeks_best(tmp_path):
    profile = returning_profile()
    session_with_reps(tmp_path, profile, TODAY - timedelta(days=2), [0.80], "a")
    pb = PersonalBests(profile, TODAY)
    pb.record("grip_release", "range", 0.60)
    assert pb.level("grip_release", "range", 0.70) == "today"


def test_at_most_one_best_per_set_and_fake_bests_from_noise(tmp_path):
    profile = returning_profile()
    session_with_reps(tmp_path, profile, TODAY - timedelta(days=1), [0.50], "y")
    prog = SessionProgress(profile, log_at(tmp_path), today=TODAY)
    ex = FakeExercise()
    rng = np.random.default_rng(0)
    announced = []
    # jitter around yesterday's best never makes a best
    for i in range(10):
        r = rep(i + 1, raw=0.50 + rng.normal(0, 0.004))
        ex.reps.append(r)
        announced += [e for e in prog.rep_events(ex, r) if e.type == "PersonalBest"]
    assert announced == []
    # a real improvement: announced once in this set, not again
    for i, v in enumerate((0.56, 0.60, 0.65)):
        r = rep(11 + i, raw=v)
        ex.reps.append(r)
        announced += [e for e in prog.rep_events(ex, r) if e.type == "PersonalBest"]
    assert len(announced) == 1 and announced[0].get("level") == "all_time"
    ex.set_no = 2
    r = rep(1, set_no=2, raw=0.70)
    ex.reps.append(r)
    assert [e.type for e in prog.rep_events(ex, r)].count("PersonalBest") == 1


def test_bests_recorded_on_a_difficult_day(tmp_path):
    profile = returning_profile()
    session_with_reps(tmp_path, profile, TODAY - timedelta(days=1), [0.50], "y")
    prog = SessionProgress(profile, log_at(tmp_path), today=TODAY)
    prog.switch_difficult("check_in")
    ex = FakeExercise(reps=[rep(1, raw=0.60)])
    events = prog.rep_events(ex, ex.reps[0])
    assert "PersonalBest" in [e.type for e in events]
    assert profile["personal_bests"]["grip_release"]["range"]["value"] == 0.6


# ---------------------------------------------------------------------------
# Praise
# ---------------------------------------------------------------------------

def test_rep_praise_events(tmp_path):
    prog = SessionProgress(returning_profile(), log_at(tmp_path), today=TODAY)
    ex = FakeExercise()
    fingers = {"index": .8, "middle": .8, "ring": .7, "pinky": .8}
    for i in range(3):
        ex.reps.append(rep(i + 1, hold=0.75, stability=0.05, fingers=fingers))
        prog.rep_events(ex, ex.reps[-1])
    better = dict(fingers, ring=0.85)
    r = rep(4, hold=0.8, stability=0.02, fingers=better)
    ex.reps.append(r)
    types = {e.type: e for e in prog.rep_events(ex, r)}
    assert types["Improvement"].get("finger") == "ring"
    assert "SteadyHold" in types
    r = rep(5, hold=0.75, success=False, hints=1, fingers=fingers)
    ex.reps.append(r)
    assert "RecoveredAfterHint" in [e.type for e in prog.rep_events(ex, r)]


def test_praise_is_only_what_was_detected_and_not_too_often():
    fb = Feedback(rng=random.Random(3))

    def words(*events):
        return [m for m in fb.words([event("RepCompleted", success=True)] + list(events))]

    said = [words() for _ in range(6)]
    spoken = [[m.text for m in w if m.kind != "chime"] for w in said]
    assert all(any(m.kind == "chime" for m in w) for w in said)    # a chime every rep
    assert [bool(s) for s in spoken] == [False, False, True, False, False, True]
    out = words(event("Improvement", exercise="grip_release", finger="ring"),
                event("SteadyHold", exercise="grip_release"))
    texts = [m.text for m in out if m.kind != "chime"]
    assert len(texts) == 1 and "ring finger" in texts[0]           # one phrase, the true one
    assert len(texts[0].split()) <= 8


def test_no_phrase_repeats_within_three():
    fb = Feedback(rng=random.Random(7))
    picks = [fb.pick("SteadyHold") for _ in range(40)]
    for i in range(len(picks)):
        assert picks[i] not in picks[max(0, i - 3):i]           # none of the last 3


def test_effort_praise_on_a_difficult_day():
    fb = Feedback(rng=random.Random(1))
    for _ in range(2):
        fb.words([event("RepCompleted", success=True, difficult=True)])
    out = [m.text for m in fb.words([event("RepCompleted", success=True, difficult=True)])
           if m.text]
    assert len(out) == 1
    assert out[0] in [p.format(name="Eleanor") for p in fb.phrases["EffortPraise"]]


# ---------------------------------------------------------------------------
# Adaptive targets
# ---------------------------------------------------------------------------

def test_target_rules_with_floor_and_ceiling():
    assert next_target(0.70, 0.9) == (pytest.approx(0.73), "raised")
    assert next_target(0.70, 0.6) == (0.70, None)
    assert next_target(0.70, 0.4) == (pytest.approx(0.67), "lowered")
    assert next_target(config.TARGET_CEILING, 1.0) == (config.TARGET_CEILING, None)
    assert next_target(config.TARGET_FLOOR, 0.0) == (config.TARGET_FLOOR, None)


def test_warm_up_start_below_last_time(tmp_path):
    profile = returning_profile()
    profile["targets"]["grip_release"] = {"high": 0.80, "date": (TODAY - timedelta(days=2)).isoformat()}
    prog = SessionProgress(profile, log_at(tmp_path), today=TODAY)
    assert prog.start_target("grip_release") == pytest.approx(0.80 - config.TARGET_STEP)
    profile["targets"]["grip_release"]["date"] = (TODAY - timedelta(days=10)).isoformat()
    assert prog.start_target("grip_release") == pytest.approx(0.80 - 2 * config.TARGET_STEP)
    assert prog.start_target("thumb_opposition") is None           # levels, not a target


def test_set_raises_keeps_and_lowers(tmp_path):
    prog = SessionProgress(returning_profile(), log_at(tmp_path), today=TODAY)
    for successes, expected, change in ((5, 0.73, "TargetRaised"), (3, 0.70, None),
                                        (1, 0.67, "TargetLowered")):
        reps = [rep(i + 1, success=i < successes) for i in range(5)]
        events, target = prog.set_finished(FakeExercise(0.70, reps=reps))
        assert target == pytest.approx(expected)
        types = [e.type for e in events]
        assert (change in types) if change else not any(t.startswith("Target") for t in types)


def test_range_grows_beyond_her_calibrated_maximum(tmp_path):
    profile = returning_profile()
    profile["calibration"]["grip_release"] = {"date": TODAY.isoformat(), "steps": dict(GRIP_CAL)}
    prog = SessionProgress(profile, log_at(tmp_path), today=TODAY)
    ex = FakeExercise(0.9, reps=[rep(i + 1, hold=1.2) for i in range(4)])
    events, target = prog.set_finished(ex)
    assert "RangeGrew" in [e.type for e in events]
    steps = profile["calibration"]["grip_release"]["steps"]
    assert steps["open"]["mean"] == pytest.approx(0.3 + 0.5 * 1.2)
    assert ex.cal is steps
    assert target == pytest.approx(min(config.TARGET_CEILING, 0.9 / 1.2 + config.TARGET_STEP))
    # also for a range that runs downwards (thumb flexion angle)
    grown = grow_range({"in": {"flexion": 120.0}, "out": {"flexion": 20.0}}, ("in", "out"), 1.1)
    assert grown["out"]["flexion"] == pytest.approx(10.0)


def test_difficult_days_do_not_move_saved_targets(tmp_path):
    profile = returning_profile()
    profile["targets"]["grip_release"] = {"high": 0.80, "date": TODAY.isoformat()}
    prog = SessionProgress(profile, log_at(tmp_path), today=TODAY)
    prog.answer_check_in(False)
    start = prog.start_target("grip_release")
    assert start == pytest.approx((0.80 - config.TARGET_STEP) * config.DIFFICULT_TARGET_FACTOR)
    reps = [rep(i + 1, success=False) for i in range(5)]
    prog.set_finished(FakeExercise(start, reps=reps))
    prog.final_targets["grip_release"] = 0.5
    prog.end_session(["grip_release"])
    assert profile["targets"]["grip_release"]["high"] == 0.80


# ---------------------------------------------------------------------------
# Difficult day mode
# ---------------------------------------------------------------------------

def test_low_warm_up_switches_on_difficult_day(tmp_path):
    profile = returning_profile()
    for d in range(1, 6):
        session_with_reps(tmp_path, profile, TODAY - timedelta(days=d), [0.60] * 3, f"d{d}")
    prog = SessionProgress(profile, log_at(tmp_path), today=TODAY)
    assert prog.baseline("grip_release") == pytest.approx(0.60)
    events, _ = prog.set_finished(FakeExercise(reps=[rep(i + 1, raw=0.45) for i in range(5)]))
    assert "DifficultDayStarted" in [e.type for e in events] and prog.difficult
    assert prog.difficult_reason == "low_warm_up"


def test_difficult_sessions_left_out_of_the_baseline(tmp_path):
    profile = returning_profile()
    for d in range(1, 4):
        session_with_reps(tmp_path, profile, TODAY - timedelta(days=d), [0.60], f"n{d}")
    session_with_reps(tmp_path, profile, TODAY - timedelta(days=4), [0.20], "bad", difficult=True)
    prog = SessionProgress(profile, log_at(tmp_path), today=TODAY)
    assert prog.baseline("grip_release") == pytest.approx(0.60)


def test_low_success_two_sets_in_a_row(tmp_path):
    prog = SessionProgress(returning_profile(), log_at(tmp_path), today=TODAY)
    bad = lambda s: FakeExercise(set_no=s, reps=[rep(i + 1, set_no=s, success=i == 0)
                                                 for i in range(5)])
    events, _ = prog.set_finished(bad(1))
    assert not prog.difficult
    events, target = prog.set_finished(bad(2))
    assert prog.difficult and prog.difficult_reason == "low_success"
    assert target == pytest.approx((0.70 - config.TARGET_STEP) * config.DIFFICULT_TARGET_FACTOR)
    assert prog.switch_difficult("fatigue") is None            # stays on, said once


def _session(log, profile=None, **kw):
    profile = profile or returning_profile()
    profile["calibration"]["grip_release"] = {"date": date.today().isoformat(), "steps": GRIP_CAL}
    speaker = SilentSpeaker()
    s = SessionManager(speaker, profile, log, **kw)
    clock = Clock()
    s.update(feat(clock.t), clock.t)
    s.on_key(" ", clock.t)                    # greeting
    return s, speaker, clock


def test_thumbs_down_makes_the_session_easier(tmp_path):
    s, speaker, clock = _session(log_at(tmp_path, date.today()))
    assert s.stage == "check_in"
    answer(s, clock.t, yes=False)
    assert s.progress.difficult
    said = [m.text for m in speaker.spoken]
    assert said.count("Thank you for telling me. Let's take it easier today.") == 1
    assert "grip_squeeze" not in s.today and s.stage == "menu"
    s.on_key(" ", clock.t)                    # all of today's exercises
    s.on_key(" ", clock.t)
    assert s.name == "grip_release"
    assert s.exercise.sets == config.EXERCISES["grip_release"]["sets"] - 1
    assert s.exercise.target == pytest.approx(
        config.EXERCISES["grip_release"]["high"] * config.DIFFICULT_TARGET_FACTOR)
    assert s._rest_seconds(30) == 45


def test_difficult_summary_praises_effort_and_hides_comparisons():
    fb = Feedback()
    ev = event("Summary", difficult=True, comparisons=["Your hand opened 12% wider than last week."],
               highlight={"type": "PersonalBest", "exercise": "grip_release", "level": "week"},
               activities=["tea"])
    lines = fb.summary_lines(ev)
    assert lines[0] == "You kept your routine going."
    assert not any("%" in line or "best" in line for line in lines)


def test_repeated_difficult_days_leave_a_note(tmp_path):
    log = log_at(tmp_path, date.today())
    for i in range(3):
        log.session_id = f"old{i}"
        log.save_session({"session_id": f"old{i}", "difficult_day": "yes" if i else "no"})
    log.session_id = "now"
    s = SessionManager(SilentSpeaker(), returning_profile(), log, exercises=["grip_release"])
    s.progress.answer_check_in(False)
    s.end_session()
    row = [r for r in log.sessions() if r["session_id"] == "now"][0]
    assert row["difficult_day"] == "yes" and "therapist" in row["note"]


# ---------------------------------------------------------------------------
# Activities and milestones
# ---------------------------------------------------------------------------

def test_her_chosen_activities_come_first():
    activities = storage.load_content("activities")
    profile = returning_profile()
    profile["chosen_activities"] = ["gardening"]
    assert memory.activity_for("grip_release", profile, activities) == "gardening"
    profile["chosen_activities"] = ["reading"]
    assert memory.activity_for("grip_release", profile, activities) == "tea"   # first linked
    assert len(activities["activities"]) == 6


def test_milestones_at_thresholds(tmp_path):
    profile = returning_profile()
    profile["practice"]["tea"] = 45
    prog = SessionProgress(profile, log_at(tmp_path), today=TODAY)
    ex = FakeExercise(reps=[rep(i + 1) for i in range(10)])
    row = log_at(tmp_path).summarize("grip_release", ex.reps, 1, 10)
    events = prog.exercise_finished(ex, row, activity="tea")
    milestone = [e for e in events if e.type == "MilestoneReached"]
    assert milestone and milestone[0].get("count") == 50
    assert profile["practice"]["tea"] == 55
    again = prog.exercise_finished(ex, row, activity="tea")         # 55 -> 65: no threshold
    assert not [e for e in again if e.type == "MilestoneReached"]
    text = Feedback().words(milestone[0])[0].text
    assert text == "That's 50 repetitions of practice for holding your cup."


# ---------------------------------------------------------------------------
# Garden
# ---------------------------------------------------------------------------

def test_garden_grows_and_never_goes_backwards():
    g = storage.new_garden()
    stages = []
    for i in range(config.GARDEN_STAGES * config.GARDEN_PLOTS + 2):
        ev = garden_model.water(g, bee=i == 3, butterflies=1 if i == 7 else 0)
        stages.append((g["season"], len(g["plots"]), g["plots"][-1]["stage"]))
        assert ev.get("stage") == g["plots"][-1]["stage"]
    # each watering moves one stage on; a flower means a new seed next time
    assert stages[:6] == [(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 2, 0)]
    assert g["season"] == 2 and g["plots"][0]["stage"] == 1      # a new season
    assert g["bees"] == 1 and g["butterflies"] == 1
    assert all(p["stage"] == config.GARDEN_STAGES - 1 for p in g["plots"][:-1])


def test_garden_waits_on_missed_days_and_grows_on_difficult_days(tmp_path):
    g = storage.new_garden()
    g["plots"] = [{"plant": "tulip", "stage": 2}]
    s = SessionManager(SilentSpeaker(), returning_profile(), log_at(tmp_path, date.today()),
                       exercises=["grip_release"], garden=g)
    s.progress.answer_check_in(False)                # difficult day
    s.progress.completed.append("grip_release")
    s.summaries.append(("grip_release", {"reps_done": 5, "success_rate": 1.0,
                                         "mean_range_high": 0.8}, None))
    grew = s.end_session()
    assert grew.get("stage") == 3 and g["plots"] == [{"plant": "tulip", "stage": 3}]
    assert s.end_session() is None                   # once per session


def test_garden_round_trip_and_bad_file(tmp_path):
    path = tmp_path / "garden.json"
    g = storage.new_garden()
    garden_model.water(g, "lavender")
    storage.save_garden(g, path)
    assert storage.load_garden(path)["plots"] == [{"plant": "lavender", "stage": 0}]
    path.write_text('{"plots": [{"plant": "rose", "stage": 99}], "bees": -3}')
    back = storage.load_garden(path)
    assert back["plots"] == [] and back["bees"] == 0


# ---------------------------------------------------------------------------
# Speech queue, yes / no, data files
# ---------------------------------------------------------------------------

def test_instructions_jump_ahead_of_old_praise_but_not_their_own_batch():
    items = collections.deque()
    enqueue(items, Say("That's three.", "praise", tag="rep_done"), 4, now=10.0)
    enqueue(items, Say("Very steady hold.", "praise", priority=4), 4, now=10.0)
    enqueue(items, Say("Open your hand wide."), 4, now=10.0)
    assert [m.text for m in items] == ["That's three.", "Very steady hold.", "Open your hand wide."]
    items.clear()
    enqueue(items, Say("A new best. Well done.", "praise", priority=3), 4, now=10.0)
    enqueue(items, Say("Please show your left hand to the camera.", "quality"), 4, now=11.0)
    assert items[0].kind == "quality"
    # praise waiting too long is dropped, an instruction never
    assert next_message(items, 12.0).kind == "quality"
    assert next_message(items, 14.5) is None
    items.append(Say("Now close your hand.", queued_at=0.0))
    assert next_message(items, 100.0).text == "Now close your hand."


def test_instruction_cuts_off_older_praise_only():
    praise = Say("That's your best this week, Eleanor.", "praise", priority=3, queued_at=10.0)
    assert interrupts(praise, Say("Now close your hand."), 11.0)
    assert not interrupts(praise, Say("Now close your hand."), 10.05)   # same batch
    assert not interrupts(Say("Now close your hand.", queued_at=10.0),
                          Say("Please use your left hand.", "quality"), 11.0)
    assert not interrupts(praise, Say("Good.", "praise"), 11.0)


def test_thumbs_must_be_held_and_released():
    yn = YesNo()
    up = [("Thumb_Up", 0.9)]
    assert yn.update(up, 0.0) is None                # still up from before: not armed
    assert yn.update([], 0.1) is None
    assert yn.update(up, 0.2) is None
    assert yn.update(up, 0.5) is None                # too short
    assert yn.update(up, 0.2 + config.GESTURE_HOLD_S) == "yes"
    assert yn.update(up, 2.0) is None                # the same gesture does not answer twice
    yn.update([], 2.1)
    yn.update([("Thumb_Down", 0.9)], 2.2)
    assert yn.update([("Thumb_Down", 0.9)], 3.0) == "no"
    assert yn.update([("Thumb_Up", 0.3)], 5.0) is None   # unsure: ignored


def test_gesture_answers_the_check_in(tmp_path):
    s, speaker, clock = _session(log_at(tmp_path, date.today()))
    assert s.stage == "check_in"
    for g in ([], [("Thumb_Down", 0.9)] * 1):
        for _ in range(int(1.0 * 30)):
            t = clock.tick()
            s.update(feat(t), t, gestures=g)
    assert s.progress.difficult and s.stage == "menu"


def test_old_rep_log_is_upgraded_to_the_new_columns(tmp_path):
    path = tmp_path / "reps.csv"
    path.write_text("session_id,timestamp,exercise,set,rep\nold,2024-01-01,grip_release,1,1\n")
    log = storage.SessionLog(path, tmp_path / "history.csv", session_id="new")
    log.log_rep(rep(1))
    rows = storage._read(path)
    assert [r["session_id"] for r in rows] == ["old", "new"]
    assert rows[1]["success"] == "yes" and rows[1]["hold_raw"] == "0.5"
    assert rows[0]["success"] == ""                                 # old rows keep their values
    assert list(rows[0]) == storage.REP_FIELDS


def test_screens_render(tmp_path):
    """Every screen draws without errors (with Pillow text)."""
    from rehab.Act import Display
    display = Display(character=storage.load_content("character"))
    frame = np.zeros((720, 1280, 3), np.uint8)
    g = storage.new_garden()
    grew = garden_model.water(g, "sunflower", bee=True)
    views = [
        {"screen": "card", "card_event": event("CheckIn"), "can": {"sections": 3, "filled": 1}},
        {"screen": "card", "card_event": event("ActivityQuestion", activity="phone", first=True)},
        {"screen": "card", "card_event": event("PlantQuestion", first="rose", second="tulip")},
        {"screen": "summary", "summary_event": event("Summary", activities=["tea"]),
         "garden": {"state": g}},
        {"screen": "garden", "garden_event": grew, "garden": {"state": g, "grow": {
            "plot": 0, "prev_stage": None, "progress": 0.5, "new_bees": 1}}},
        {"stage": "rest", "countdown": 10, "progress": 0.5, "activity": "gardening",
         "can": {"sections": 3, "filled": 2}, "mood": "happy"},
    ]
    for v in views:
        out = display.render(frame.copy(), v)
        assert out.shape == (720, 1280 + 420, 3) and out.std() > 0


def test_screens_take_the_screen_shape():
    """Drawn in the screen's shape, so the window can scale it without cutting anything off."""
    from rehab.Act import PANEL_W, Display
    frame = np.zeros((720, 1280, 3), np.uint8)
    for screen, size in (((1710, 1107), (1700, 1101)), ((1920, 1080), (1700, 956)),
                         ((3440, 1440), (1720, 720))):
        display = Display(character=storage.load_content("character"), screen=screen)
        assert display.canvas_size(frame.shape) == size
        for v in ({"stage": "rest", "countdown": 3, "progress": 0.5},
                  {"screen": "card", "card_event": event("CheckIn"),
                   "gesture": {"hand": True, "answer": "yes", "progress": 0.5}},
                  {"screen": "profile", "title": "Your profile",
                   "profile_rows": [("Name", "Eleanor")] * 10,
                   "answer_labels": ("Thumbs up or Space: back", "D: delete my profile")}):
            out = display.render(frame.copy(), v)
            assert out.shape == (size[1], size[0], 3) and out.std() > 0
    # a camera with another resolution is scaled to the same design height
    display = Display()
    assert display.canvas_size((1080, 1920)) == (1280 + PANEL_W, 720)


def test_yes_no_state_for_the_preview():
    from rehab.Think import YesNo
    yn = YesNo(hold_s=1.0)
    assert yn.state(0.0) == (None, 0.0)
    yn.update([], 0.0)                               # hand down: ready
    yn.update([(config.YES_GESTURE, 0.9)], 0.1)
    yn.update([(config.YES_GESTURE, 0.9)], 0.6)
    assert yn.state(0.6) == ("yes", 0.5)
    assert yn.update([(config.YES_GESTURE, 0.9)], 1.2) == "yes"
    assert yn.state(1.2)[0] == "lower"               # still up after answering
