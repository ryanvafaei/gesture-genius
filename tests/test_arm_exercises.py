"""Arm exercises with a synthetic body: reps, form rules, cues, calibration and whole sessions."""

import csv
import json
from datetime import date, timedelta

import numpy as np
import pytest

from rehab import benchmarks, config, storage
from rehab.Act import Display, SilentSpeaker
from rehab.calibration import SideCalibration
from rehab.exercises import create
from rehab.exercises.arm_raise import ShoulderFlexionRaise
from rehab.exercises.elbow_bend import HandToMouth
from rehab.Think import SessionManager
from helpers import Clock, answer, returning_profile, texts
from synthetic_body import feat

FPS = 30.0
CAL = {"right": {"best": 150.0, "mean": 148.0, "n": 3, "start": 2.0},
       "left": {"best": 58.0, "mean": 55.0, "n": 3, "start": 2.0}}


def therapist(**exercises):
    t = benchmarks.load_therapist_profile()
    t = dict(t, exercises=dict(t["exercises"]))
    for name, settings in exercises.items():
        t["exercises"][name] = dict(t["exercises"].get(name, {}), **settings)
    return t


class Driver:
    """Feeds one exercise a synthetic body at 30 fps."""

    def __init__(self, ex, t=0.0):
        self.ex, self.t, self.said = ex, t, []

    def frame(self, **pose):
        self.t += 1 / FPS
        self.said += self.ex.update(feat(self.t, **pose), self.t)

    def hold(self, seconds, **pose):
        for _ in range(int(seconds * FPS)):
            self.frame(**pose)

    def ramp(self, key, a, b, seconds, **pose):
        n = int(seconds * FPS)
        for i in range(n):
            self.frame(**dict(pose, **{key: a + (b - a) * i / max(1, n - 1)}))

    def rep(self, key, start, peak, hold=1.3, **pose):
        self.hold(0.5, **dict(pose, **{key: start}))
        self.ramp(key, start, peak, 1.2, **pose)
        self.hold(hold, **dict(pose, **{key: peak}))
        self.ramp(key, peak, start, 1.2, **pose)
        self.hold(0.5, **dict(pose, **{key: start}))
        return self.ex.reps[-1] if self.ex.reps else None


def raise_ex(level=3, **params):
    t = therapist(shoulder_flexion_raise={"sets": 1, "reps": 5, "hold_s": 1})
    ex = create("shoulder_flexion_raise", {"shoulder_flexion_raise": CAL},
                params=params or None, level=level, therapist=t)
    ex.start_set(1, 0.0)
    return ex


# --- the rep machine ------------------------------------------------------------------------

def test_target_comes_from_the_ladder():
    ex = raise_ex(level=3)
    assert ex.ladder.ceiling == 150 and ex.target_deg == 58 + 3 * 5.5
    assert [m["deg"] for m in ex.ladder.milestones] == [71, 86, 90, 105, 108]


def test_clean_rep_scores_two_past_90():
    d = Driver(raise_ex())
    rec = d.rep("shoulder", 2, 100)
    assert rec is not None and rec.success
    assert rec.extra["fma_style_score"] == 2 and rec.extra["coaching_success"]
    assert rec.raw_high == pytest.approx(100, abs=0.5)
    assert "That's one." in texts(d.said) and "And hold." in texts(d.said)


def test_peak_in_the_uncertainty_band_scores_one():
    rec = Driver(raise_ex()).rep("shoulder", 2, 93)
    assert rec.extra["fma_style_score"] == 1 and "within_uncertainty_band" in rec.extra["reasons"]
    assert rec.success                          # above today's target


def test_bent_elbow_breaks_form_and_gets_one_cue():
    d = Driver(raise_ex())
    rec = d.rep("shoulder", 2, 100, elbow=30)
    assert rec.extra["fma_style_score"] in (0, 1) and not rec.success
    assert rec.extra["form_rules_broken"] == [] or "elbow_straight" in rec.extra["form_rules_broken"]
    # elbow bending during the lift, from a straight start
    d = Driver(raise_ex())
    d.hold(0.5, shoulder=2)
    d.ramp("shoulder", 2, 100, 1.2, elbow=0)
    d.hold(1.3, shoulder=100, elbow=30)
    d.ramp("shoulder", 100, 2, 1.2, elbow=30)
    d.hold(0.5, shoulder=2)
    rec = d.ex.reps[-1]
    assert "elbow_straight" in rec.extra["form_rules_broken"] and not rec.success
    assert "Keep your elbow straight." in texts(d.said)


def test_elbow_bent_at_the_start_scores_zero():
    rec = Driver(raise_ex()).rep("shoulder", 2, 100, elbow=25)
    assert rec.extra["fma_style_score"] == 0 and "start_position_not_obtained" in rec.extra["reasons"]


def test_trunk_lean_is_compensation_success_with_one_cue_per_set():
    d = Driver(raise_ex())
    d.hold(0.5, shoulder=2)
    d.ramp("shoulder", 2, 100, 1.2)
    d.hold(1.3, shoulder=100, trunk=15)
    d.ramp("shoulder", 100, 2, 1.2)
    d.hold(0.5, shoulder=2)
    rec = d.ex.reps[-1]
    assert rec.success and rec.extra["with_compensation"] and rec.compensation == ["trunk"]
    assert rec.extra["fma_style_score"] == 1 and rec.extra["trunk_max_change_deg"] >= 10
    assert "Keep your back against the chair." in texts(d.said)
    # a second compensated rep in the same set: no second technique cue
    before = len(d.said)
    d.hold(0.5, shoulder=2)
    d.ramp("shoulder", 2, 100, 1.2)
    d.hold(1.3, shoulder=100, trunk=15)
    d.ramp("shoulder", 100, 2, 1.2)
    d.hold(0.5, shoulder=2)
    assert "Keep your back against the chair." not in texts(d.said[before:])


def test_arm_leaving_the_plane_is_a_form_break():
    d = Driver(raise_ex())
    d.hold(0.5, shoulder=2)
    d.ramp("shoulder", 2, 100, 1.2)
    d.hold(1.3, shoulder=100, abduct_out_of_plane=40)
    d.ramp("shoulder", 100, 2, 1.2)
    d.hold(0.5, shoulder=2)
    assert "in_plane" in d.ex.reps[-1].extra["form_rules_broken"]


def test_below_target_gets_a_little_further():
    d = Driver(raise_ex(level=10))           # target 113
    rec = d.rep("shoulder", 2, 80)
    assert not rec.success and rec.extra["cue_given"] == "A little further next time, if you can."


def test_implausible_readings_are_rejected():
    ex = raise_ex()
    d = Driver(ex)
    d.hold(0.5, shoulder=2)
    ex.update(feat(d.t + 0.03, shoulder=2, wrist_at=(640, 0)), d.t + 0.03)
    d.t += 0.03
    upper, _ = ex.bounds
    assert upper == pytest.approx(188 + 5.5)
    ex.bounds = (50.0, None)
    d.frame(shoulder=80)
    assert ex._rep["rejects"] >= 1


def test_tracking_lost_is_the_cameras_fault():
    ex = raise_ex()
    f = feat(0.0, hidden=("left_wrist",))
    assert ex.quality_problem(f) == "arm_hidden"
    from rehab.Think import QUALITY_TEXT
    assert "I can't see your left arm" in QUALITY_TEXT["arm_hidden"].format(hand="left")


def test_no_movement_is_recorded_after_prompts():
    """She cannot lift at all: after a few prompts an attempt without movement is logged."""
    ex = raise_ex()
    d = Driver(ex)
    d.hold(3 * max(config.HINT_DELAY_S, config.HINT_REPEAT_S) + 2, shoulder=2)
    assert ex.reps and ex.reps[-1].extra["fma_style_score"] == 0 and not ex.reps[-1].success


# --- the other exercises ------------------------------------------------------------------------

def test_abduction_shoulder_hike():
    t = therapist(shoulder_abduction_raise={"sets": 1, "reps": 3, "hold_s": 1})
    ex = create("shoulder_abduction_raise", {"shoulder_abduction_raise": CAL}, level=5, therapist=t)
    ex.start_set(1, 0.0)
    d = Driver(ex)
    d.hold(0.5, shoulder=2, view="frontal")
    d.ramp("shoulder", 2, 100, 1.2, view="frontal")
    d.hold(1.3, shoulder=100, view="frontal", hike=45)
    d.ramp("shoulder", 100, 2, 1.2, view="frontal")
    d.hold(0.5, shoulder=2, view="frontal")
    rec = ex.reps[-1]
    assert "no_shoulder_hike" in rec.extra["form_rules_broken"]
    assert "Let your shoulder relax down." in texts(d.said)


def test_hand_to_mouth_logs_the_shoulder_too():
    cal = {"right": {"best": 140.0, "start": 5.0}, "left": {"best": 90.0, "start": 5.0}}
    ex = create("hand_to_mouth", {"hand_to_mouth": cal}, level=2, therapist=therapist())
    ex.start_set(1, 0.0)
    assert ex.fma_item is None and [m["deg"] for m in ex.ladder.milestones] == [100, 121]
    d = Driver(ex)
    d.hold(0.5, elbow=5)
    d.ramp("elbow", 5, 125, 1.2, shoulder=30)
    d.hold(1.3, elbow=125, shoulder=30)
    d.ramp("elbow", 125, 5, 1.2)
    d.hold(0.5, elbow=5)
    rec = ex.reps[-1]
    assert rec.success and rec.extra["fma_style_score"] is None
    assert rec.extra["shoulder_elevation_max"] == pytest.approx(30, abs=1)


def test_hand_to_head_needs_the_ear():
    cal = {"right": {"best": 150.0, "start": 5.0}, "left": {"best": 100.0, "start": 5.0}}
    ex = create("hand_to_head", {"hand_to_head": cal}, level=0, therapist=therapist())
    ex.start_set(1, 0.0)
    d = Driver(ex)
    rec = d.rep("elbow", 5, 120, shoulder=40)
    assert rec.extra["fma_style_score"] == 1 and "hand_not_at_ear" in rec.extra["reasons"]
    ear = feat(0.0).points[7]
    d.hold(0.5, elbow=5)
    d.ramp("elbow", 5, 120, 1.2, shoulder=40)
    d.hold(1.3, elbow=120, shoulder=40, wrist_at=ear + [0, 20])
    d.ramp("elbow", 120, 5, 1.2, shoulder=40)
    d.hold(0.5, elbow=5)
    assert ex.reps[-1].extra["fma_style_score"] == 2


def test_elbow_extension_goes_down_to_zero():
    cal = {"right": {"best": 2.0, "start": 90.0}, "left": {"best": 40.0, "start": 90.0}}
    ex = create("elbow_extension", {"elbow_extension": cal}, level=1, therapist=therapist())
    ex.start_set(1, 0.0)
    assert ex.target_deg == 35 and ex.ladder.ceiling == 2
    d = Driver(ex)
    rec = d.rep("elbow", 90, 3)
    assert rec.success and rec.extra["fma_style_score"] == 2
    rec = d.rep("elbow", 90, 45)
    assert not rec.success and rec.extra["fma_style_score"] == 1


def test_contracture_of_30_degrees_is_not_testable():
    t = dict(therapist(), passive_limits_deg={"left.elbow_extension_deficit": 30})
    from rehab.exercises import EXERCISES
    assert EXERCISES["elbow_extension"].not_testable(t)
    assert EXERCISES["shoulder_flexion_raise"].not_testable(t)
    t = dict(therapist(), passive_limits_deg={"left.elbow_extension_deficit": 20})
    assert not EXERCISES["elbow_extension"].not_testable(t)
    ex = create("elbow_extension", {"elbow_extension": {"left": {"best": 40.0}}}, therapist=t)
    assert ex.fma_threshold == 20.0


def test_wrist_extension_is_capped_at_one_and_needs_the_hand():
    cal = {"right": {"best": 60.0, "start": 0.0}, "left": {"best": 10.0, "start": 0.0}}
    ex = create("wrist_extension", {"wrist_extension": cal}, level=1, therapist=therapist())
    ex.start_set(1, 0.0)
    assert ex.quality_problem(feat(0.0, elbow=90)) == "hand_hidden"
    assert ex.change_threshold() == 10.0
    d = Driver(ex)
    rec = d.rep("hand_ext", 0, 40, elbow=90)
    assert rec.success and rec.extra["fma_style_score"] == 1 and rec.extra["wrist_cycle"] == 1


def test_reach_scores_the_trunk_share():
    ex = create("tabletop_reach", therapist=therapist(tabletop_reach={"sets": 1, "reps": 2}))
    ex.start_set(1, 0.0)
    assert ex.make_calibration() is None and ex.target_name == "near"
    d = Driver(ex)
    # the arm alone reaches: no trunk
    d.hold(0.5, shoulder=10, elbow=90, view="oblique")
    d.ramp("shoulder", 10, 40, 1.0, elbow=60, view="oblique")
    d.ramp("shoulder", 40, 10, 1.0, elbow=90, view="oblique")
    d.hold(0.5, shoulder=10, elbow=90, view="oblique")
    rec = ex.reps[-1]
    assert rec.extra["reach_target"] == "close" and rec.extra["rps_trunk_score"] == 3
    assert rec.success
    # the far lamp, done mostly by leaning the trunk
    d.hold(0.5, shoulder=10, elbow=90, view="oblique")
    d.ramp("trunk", 0, 35, 1.0, shoulder=12, elbow=88, view="oblique")
    d.ramp("trunk", 35, 0, 1.0, shoulder=10, elbow=90, view="oblique")
    d.hold(0.5, shoulder=10, elbow=90, view="oblique")
    rec = ex.reps[-1]
    assert rec.extra["reach_target"] == "far" and rec.extra["rps_trunk_score"] <= 1
    assert rec.compensation == ["trunk_reach"]
    assert "Let your arm do the reaching." in texts(d.said)


def test_finger_to_nose_times_both_arms():
    ex = create("finger_to_nose_timed", therapist=therapist())
    ex.start_set(1, 0.0)
    assert ex.round_side == "right" and ex.make_calibration() is None
    d = Driver(ex)
    nose = feat(0.0, view="frontal").points[0]

    def touches(side, n, seconds_each):
        d.hold(0.5, side=side, view="frontal", shoulder=5, elbow=60)
        for _ in range(n):
            d.hold(seconds_each / 2, side=side, view="frontal", shoulder=60, elbow=120,
                   index_at=nose + [3, 2])
            d.hold(seconds_each / 2, side=side, view="frontal", shoulder=5, elbow=60)

    touches("right", 5, 1.0)
    assert len(ex.reps) == 1 and ex.reps[0].extra["touches"] == 5
    assert ex.round_side == "left"
    touches("left", 5, 2.0)
    rec = ex.reps[-1]
    assert rec.extra["side"] == "left" and rec.extra["fma32_style"] == 2
    assert rec.extra["fma_style_score"] == 1                   # 5 s slower: 2.0-5.9 s -> 1
    assert ex.set_done


# --- calibration: right side first, then left ---------------------------------------------------------

def test_side_calibration_right_then_left(monkeypatch):
    monkeypatch.setattr(config, "BENCH_REST_BETWEEN_REPS_S", 0.5)
    routine = SideCalibration(ShoulderFlexionRaise, therapist=therapist())
    t = 0.0
    said = routine.start(t)

    def frame(**pose):
        nonlocal t
        t += 1 / FPS
        said.extend(routine.update(feat(t, **pose), t))

    def rep(side, peak):
        for _ in range(15):
            frame(side=side)
        for i in range(36):
            frame(side=side, shoulder=2 + (peak - 2) * i / 35)
        for i in range(36):
            frame(side=side, shoulder=peak - (peak - 2) * i / 35)
        for _ in range(30):
            frame(side=side)

    for _ in range(60):
        frame(side="right")                       # positioning
    assert routine.step.screen_text.startswith("Right arm")
    for peak in (100, 120, 145, 150, 148):         # 2 practice + 3 recorded
        rep("right", peak)
    assert routine.side == "left"
    for _ in range(60):
        frame(side="left")
    for peak in (55, 58, 50):
        rep("left", peak)
    assert routine.done
    steps = routine.as_profile_entry()["steps"]
    assert steps["right"]["best"] == pytest.approx(150, abs=0.5)
    assert steps["right"]["n"] == 3                 # practice reps are not recorded
    assert steps["left"]["best"] == pytest.approx(58, abs=0.5)
    assert steps["symmetry_index"] == pytest.approx(58 / 150, abs=0.01)
    assert ShoulderFlexionRaise.calibration_valid(steps)
    assert "First your right arm." in texts(said)
    assert "Good. That one was for practice." in texts(said)
    assert "Now the same with your left arm." in texts(said)


# --- whole sessions -------------------------------------------------------------------------------

@pytest.fixture
def log(tmp_path):
    return storage.SessionLog(tmp_path / "reps.csv", tmp_path / "history.csv", session_id="s1")


class Session:
    def __init__(self, log, profile=None, exercises=("shoulder_flexion_raise",), t=None, **kw):
        self.speaker = SilentSpeaker()
        self.t = t or therapist(shoulder_flexion_raise={"sets": 2, "reps": 2, "hold_s": 1})
        self.s = SessionManager(self.speaker, profile or returning_profile(), log,
                                exercises=list(exercises) if exercises else None,
                                therapist=self.t, **kw)
        self.clock = Clock()

    def run(self, seconds, **pose):
        for _ in range(int(seconds * FPS)):
            now = self.clock.tick()
            self.s.update(feat(now, **pose), now)

    def rep(self, peak, side="left", **pose):
        self.run(0.5, side=side, **pose)
        n = int(1.2 * FPS)
        for i in range(n):
            now = self.clock.tick()
            self.s.update(feat(now, side=side, shoulder=2 + (peak - 2) * i / (n - 1), **pose), now)
        self.run(1.3, side=side, shoulder=peak, **pose)
        for i in range(n):
            now = self.clock.tick()
            self.s.update(feat(now, side=side, shoulder=peak - (peak - 2) * i / (n - 1), **pose), now)
        self.run(0.5, side=side, **pose)

    def start(self):
        self.run(0.2)
        self.s.on_key(" ", self.clock.t)          # after the greeting
        answer(self.s, self.clock.t)              # check-in

    def said(self):
        return texts(self.speaker.spoken)


def test_full_arm_session(log, monkeypatch):
    monkeypatch.setattr(config, "BENCH_REST_BETWEEN_REPS_S", 0.5)
    monkeypatch.setattr(config, "REST_BETWEEN_SETS_S", 1)
    x = Session(log)
    s = x.s
    x.start()
    assert s.stage == "today_plan"
    x.run(1.5)
    assert s.stage == "intro"
    assert "Sit side-on to the screen, with your left arm nearest to it." in x.said()
    x.run(12)
    assert s.stage == "calibrating" and s.needs_body
    x.run(3, side="right")
    for peak in (140, 150, 145, 148, 150):
        x.rep(peak, side="right")
        x.run(0.8, side="right")
    x.run(3)
    for peak in (55, 58, 50):
        x.rep(peak)
        x.run(0.8)
    entry = s.profile["calibration"]["shoulder_flexion_raise"]
    assert entry["steps"]["baseline"] == pytest.approx(58, abs=0.5)
    assert s.profile["assessments"]["shoulder_flexion_raise"][0]["affected_best"] == pytest.approx(58, abs=0.5)
    x.run(2.5)
    assert s.stage == "exercise" and s.exercise.target_deg == pytest.approx(58, abs=0.5)
    x.rep(62)
    x.rep(75)                                  # passes 71: drinking from a cup
    assert s.stage == "rest"
    s.on_key(" ", x.clock.t)
    x.rep(64)
    x.rep(66)
    assert s.stage == "summary"
    said = x.said()
    assert any("arm lift" in t and "drink from your cup of tea" in t for t in said)   # Gates 2016: 71
    assert "You did all four, well done." in said          # no claim without an earlier session
    # logs
    rows = list(csv.DictReader(open(log.bench_rep_path)))
    assert rows[0]["exercise_id"] == "shoulder_flexion_raise" and len(rows) == 4
    assert rows[0]["fma_style_score"] == "1" and rows[0]["coaching_success"] == "yes"
    assert list(rows[0]) == storage.BENCH_REP_FIELDS
    extra = json.loads(log.history("shoulder_flexion_raise")[0]["extra"])
    assert extra["clean_success_rate"] == 1.0 and extra["best_peak"] == pytest.approx(75, abs=0.5)
    s.on_key(" ", x.clock.t)
    s.stop(x.clock.t)
    sessions = list(csv.DictReader(open(log.bench_session_path)))
    assert sessions[0]["exercises_done"] == "shoulder_flexion_raise"
    assert sessions[0]["camera_view_check_passed"] == "yes"
    assert json.loads(sessions[0]["milestones_crossed_json"])[0]["deg"] == 71


def _history_row(log, day, level, clean, best, success=1.0):
    old = storage.SessionLog(log.rep_path, log.history_path, session_id=f"d{day}",
                             today=date.today() - timedelta(days=day))
    old.save_summary({
        "session_id": f"d{day}", "date": old.today.isoformat(), "exercise": "shoulder_flexion_raise",
        "reps_done": 4, "reps_target": 4, "success_rate": success, "difficult_day": "no",
        "extra": json.dumps({"benchmark": True, "clean_success_rate": clean, "compensation_rate": 0.0,
                             "best_peak": best, "level": level}),
    })


def _calibrated_profile(days_ago=1):
    p = returning_profile()
    p["calibration"]["shoulder_flexion_raise"] = {
        "date": (date.today() - timedelta(days=days_ago)).isoformat(),
        "steps": dict(CAL, baseline=58.0, symmetry_index=0.39)}
    return p


def test_level_rises_after_two_clean_sessions_and_claims_use_the_mdc(log, monkeypatch):
    monkeypatch.setattr(config, "REST_BETWEEN_SETS_S", 1)
    _history_row(log, 1, 2, 0.9, 60.0)
    p = _calibrated_profile()
    p["levels"]["shoulder_flexion_raise"] = 2
    x = Session(log, profile=p)
    x.start()
    x.run(1.5)
    x.run(12)
    assert x.s.stage == "calibration_offer"
    x.run(config.RECALIBRATION_OFFER_S + 3)
    assert x.s.stage == "exercise" and x.s.exercise.level == 2
    assert x.s.exercise.target_deg == pytest.approx(58 + 2 * 5.5)
    for peak in (72, 76):
        x.rep(peak)
    x.s.on_key(" ", x.clock.t)
    for peak in (74, 75):
        x.rep(peak)
    assert x.s.stage == "summary"
    said = x.said()
    # 76 - 60 = 16 degrees >= MDC 13.19: a real improvement
    assert "Your arm lifted 16 degrees higher than last time." in said
    assert any("aim a" in t and "higher" in t for t in said)      # LevelRaised
    x.s.on_key(" ", x.clock.t)
    x.s.stop(x.clock.t)
    assert x.s.profile["levels"]["shoulder_flexion_raise"] == 3


def test_small_gain_is_not_claimed(log, monkeypatch):
    monkeypatch.setattr(config, "REST_BETWEEN_SETS_S", 1)
    _history_row(log, 1, 0, 0.5, 70.0, success=0.7)
    x = Session(log, profile=_calibrated_profile())
    x.start()
    x.run(13.5)
    x.run(config.RECALIBRATION_OFFER_S + 3)
    for peak in (72, 76):
        x.rep(peak)
    x.s.on_key(" ", x.clock.t)
    for peak in (74, 75):
        x.rep(peak)
    said = x.said()
    assert not any("degrees higher" in t for t in said)
    assert "You did all four, well done." in said


def test_two_missed_sessions_start_one_level_lower(log):
    _history_row(log, config.MISSED_SESSION_DAYS, 3, 0.9, 80.0)
    p = _calibrated_profile(days_ago=config.MISSED_SESSION_DAYS)
    p["levels"]["shoulder_flexion_raise"] = 3
    x = Session(log, profile=p)
    x.start()
    x.run(13.5)
    x.run(config.RECALIBRATION_OFFER_S + 3)
    assert x.s.exercise.level == 2
    assert not any("missed" in t.lower() for t in x.said())


def test_elbow_tasks_claim_only_from_the_assessment(log):
    cls = HandToMouth
    row = log.summarize("hand_to_mouth", [], 1, 4)
    row["extra"] = json.dumps({"best_peak": 130.0})
    earlier = dict(row, extra=json.dumps({"best_peak": 90.0}))
    assert storage.improvement_claimed(cls, row, earlier, 20.55) is None
    from rehab.progress import SessionProgress
    progress = SessionProgress(returning_profile(), log, therapist=therapist())
    first = {"steps": {"left": {"best": 90.0}, "right": {"best": 140.0}}}
    assert progress.assessment_done(cls, first)[0].type == "StartingPoint"
    second = {"steps": {"left": {"best": 115.0}, "right": {"best": 140.0}}}
    ev = progress.assessment_done(cls, second, first)
    assert ev[0].type == "AssessmentImproved" and ev[0].get("deg") == 25
    assert second["steps"]["baseline"] == 90.0          # the first best stays her baseline
    third = {"steps": {"left": {"best": 125.0}}}
    assert progress.assessment_done(cls, third, second) == []   # 10 < MDC 20.55


def test_setup_check_asks_her_to_turn(log, monkeypatch):
    x = Session(log, profile=_calibrated_profile())
    x.start()
    x.run(13.5)
    x.run(config.RECALIBRATION_OFFER_S + 0.5, view="frontal")
    assert x.s.stage == "setup_check"
    x.run(3, view="frontal")
    assert "Please turn your chair so your left arm is nearest the screen." in x.said()
    assert x.s.view()["setup"]["problem"] == "turn_side"
    x.run(2)
    assert x.s.stage == "exercise"
    assert "Good, I can see you well." in x.said()


def test_arm_menu(log):
    t = therapist(shoulder_flexion_raise={"enabled": True}, hand_to_head={"enabled": False})
    x = Session(log, exercises=None, t=t)
    x.start()
    assert x.s.stage == "menu"
    x.s.on_key(str(x.s.menu.index("arm")), x.clock.t)
    assert x.s.stage == "arm_menu"
    items = x.s.view()["menu"]
    assert items[0]["text"] == "Back"
    head = x.s.arm_menu.index("hand_to_head")
    assert items[head]["note"] == "with your therapist"
    x.s.on_key(str(head), x.clock.t)
    assert x.s.stage == "arm_menu"                          # not on her own
    assert "This one is for when your therapist is with you." in x.said()
    x.s.on_key("m", x.clock.t)
    assert x.s.stage == "menu"
    x.s.on_key(str(x.s.menu.index("arm")), x.clock.t)
    x.s.on_key(str(x.s.arm_menu.index("shoulder_flexion_raise")), x.clock.t)
    assert x.s.stage == "intro" and x.s.plan == ["shoulder_flexion_raise"]


def test_not_testable_exercise_is_skipped(log):
    t = dict(therapist(), passive_limits_deg={"left.elbow_extension_deficit": 35})
    x = Session(log, exercises=("elbow_extension",), t=t)
    x.start()
    x.run(1.5)
    assert x.s.stage == "summary"
    assert "We'll leave this one until your therapist has checked your elbow." in x.said()


def test_difficult_day_does_not_save_the_level(log, monkeypatch):
    monkeypatch.setattr(config, "REST_BETWEEN_SETS_S", 1)
    _history_row(log, 1, 2, 0.9, 60.0)
    p = _calibrated_profile()
    p["levels"]["shoulder_flexion_raise"] = 2
    x = Session(log, profile=p)
    x.run(0.2)
    x.s.on_key(" ", x.clock.t)
    answer(x.s, x.clock.t, yes=False)                  # thumbs down: a difficult day
    x.run(13.5)
    x.run(config.RECALIBRATION_OFFER_S + 3)
    assert x.s.exercise.level == 1                     # one lower today
    x.rep(80)
    x.s.stop(x.clock.t)
    assert x.s.profile["levels"]["shoulder_flexion_raise"] == 2


def test_arm_exercise_renders(log):
    x = Session(log, profile=_calibrated_profile())
    x.start()
    x.run(13.5)
    x.run(config.RECALIBRATION_OFFER_S + 3)
    x.run(0.6)
    f = feat(x.clock.t, shoulder=40)
    frame = np.zeros((720, 1280, 3), np.uint8)
    canvas = Display(screen=None).render(frame, x.s.view(), f)
    assert canvas.shape[0] == config.DESIGN_HEIGHT
    x.s.stage = "setup_check"
    x.s._setup = __import__("rehab.calibration", fromlist=["SetupCheck"]).SetupCheck(
        ["left_wrist"], "sagittal", "left")
    Display(screen=None).render(frame, x.s.view(), f)


def test_elbow_extension_is_capped_by_the_extension_deficit():
    t = dict(therapist(), passive_limits_deg={"left.elbow_extension_deficit": 15,
                                               "left.elbow_flexion": 120})
    cal = {"right": {"best": 2.0, "start": 90.0}, "left": {"best": 40.0, "start": 90.0}}
    ex = create("elbow_extension", {"elbow_extension": cal}, level=10, therapist=t)
    assert ex.ladder.ceiling == 15 and ex.target_deg == 15
    mouth = create("hand_to_mouth", {"hand_to_mouth": {"left": {"best": 90.0}, "right": {"best": 140.0}}},
                   therapist=t)
    assert mouth.ladder.ceiling == 120                 # the flexion limit caps bending
