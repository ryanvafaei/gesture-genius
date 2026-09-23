"""
Voice feedback with realistic speech timing.

Speech takes seconds, so these tests use TimedSpeaker and a patient who only
does what she hears. They check that what is said matches what she has to
do at the moment she hears it.
"""

import collections

from rehab import config, storage
from rehab.Act import enqueue
from rehab.calibration import CalibrationRoutine
from rehab.exercises.base import Say
from rehab.exercises.grip_release import GripRelease
from rehab.exercises.grip_squeeze import GripSqueeze
from rehab.Think import SessionManager
from helpers import Clock, TimedSpeaker, calibrate, feat, lerp, run, texts

OPEN, CLOSED = (25, 35, 20), (60, 80, 50)
MID = lerp(OPEN, CLOSED, 0.5)


class ListeningPatient:
    """Opens or closes her hand when she hears it, moving over about a second."""

    def __init__(self):
        self.pose = MID
        self.target = MID

    def hear(self, text):
        t = text.lower()
        if "open" in t and "close" not in t:
            self.target = OPEN
        elif "close" in t and "open" not in t:
            self.target = CLOSED

    def step(self):
        self.pose = lerp(self.pose, self.target, 0.1)
        return dict(flex=self.pose)


def session(log, profile=None, **kw):
    clock = Clock()
    patient = ListeningPatient()
    heard_in_phase = []       # (text, phase key when she heard it)
    s = None

    def on_speak(msg):
        patient.hear(msg.text)
        ex = s.exercise if s is not None and s.stage == "exercise" else None
        phase = ex.phases[ex._phase].key if ex is not None else None
        heard_in_phase.append((msg.text, phase))

    speaker = TimedSpeaker(clock, on_speak=on_speak)
    profile = profile or storage.new_profile()
    s = SessionManager(speaker, profile, log, exercises=["grip_release"], **kw)

    def go(seconds):
        for _ in range(int(seconds * 30)):
            t = clock.tick()
            speaker.advance()
            s.update(feat(t, **patient.step()), t)
            if s.stage == "greeting" and not speaker.busy:
                s.on_key(" ", t)

    return s, speaker, profile, heard_in_phase, go, clock


def _log(tmp_path):
    return storage.SessionLog(tmp_path / "reps.csv", tmp_path / "history.csv", session_id="s1")


def test_calibration_waits_for_the_spoken_prompt(tmp_path):
    s, speaker, profile, _, go, _ = session(_log(tmp_path))
    go(50)
    assert s.stage == "exercise"
    steps = profile["calibration"]["grip_release"]["steps"]
    # measured in the right order: "open" really is the open hand
    assert steps["open"]["mean"] > steps["closed"]["mean"] + 0.2
    assert GripRelease.calibration_valid(steps)


def test_prompts_match_the_phase_she_is_in(tmp_path, monkeypatch):
    monkeypatch.setitem(config.EXERCISES, "grip_release",
                        dict(config.EXERCISES["grip_release"], sets=1, reps=5))
    s, speaker, _, heard, go, _ = session(_log(tmp_path))
    go(120)
    assert len(s.exercise.reps) == 5            # she can follow the voice alone
    prompts = {"Open your hand wide.": "open", "Now close your hand.": "close"}
    said = [(text, phase) for text, phase in heard if text in prompts]
    assert said
    for text, phase in said:
        assert prompts[text] == phase, (text, phase)


def test_no_new_rep_asked_for_at_the_end_of_a_set(tmp_path, monkeypatch):
    monkeypatch.setitem(config.EXERCISES, "grip_release",
                        dict(config.EXERCISES["grip_release"], sets=2, reps=2))
    s, speaker, _, _, go, _ = session(_log(tmp_path))
    go(80)
    heard = [t for _, t in speaker.heard]
    end = heard.index("Well done. That was set one of two.")
    assert heard[end - 1] == "That's two."
    assert "Open your hand wide." not in heard[end - 2:end + 2]


def test_stored_reversed_calibration_is_measured_again(tmp_path):
    profile = storage.new_profile()
    fingers = ("index", "middle", "ring", "pinky", "mean")
    profile["calibration"]["grip_release"] = {
        "date": storage.date.today().isoformat(),
        "steps": {"open": {k: .3 for k in fingers}, "closed": {k: .8 for k in fingers}},
    }
    s, speaker, profile, _, go, _ = session(_log(tmp_path), profile=profile)
    go(20)
    assert s.stage == "calibrating"
    go(30)
    assert s.stage == "exercise"
    steps = profile["calibration"]["grip_release"]["steps"]
    assert steps["open"]["mean"] > steps["closed"]["mean"]


def test_reversed_calibration_is_repeated():
    clock = Clock()
    routine = CalibrationRoutine(GripRelease)
    said = texts(routine.start(clock.t))
    for pose in (CLOSED, OPEN):                  # she did the opposite
        name = routine.step.name
        while routine.step.name == name and routine._i >= 0 and routine.attempt == 1:
            t = clock.tick()
            said += texts(routine.update(feat(t, flex=pose), t))
    assert "That didn't quite work. Let's try once more." in said
    assert not routine.done and routine.step.name == "open"


def test_calibration_settle_starts_after_speech():
    clock = Clock()
    routine = CalibrationRoutine(GripRelease)
    routine.start(clock.t)
    for _ in range(int(6 * 30)):                  # 6 s of talking
        t = clock.tick()
        routine.update(feat(t, flex=MID), t, speaking=True)
    assert routine.step.name == "open" and routine.progress == 0.0


def test_pause_clears_old_speech_and_resume_repeats_the_prompt(tmp_path):
    s, speaker, _, _, go, clock = session(_log(tmp_path))
    go(50)
    assert s.stage == "exercise"
    speaker.say(Say("Now close your hand."))
    s.on_key(" ", clock.t)
    assert [m.text for m in speaker.items] == ["Paused. Press the space bar when you are ready."]
    s.on_key(" ", clock.t)
    queued = [m.text for m in speaker.items]
    assert queued[0] == "Let's continue."
    assert queued[1] == s.exercise.phases[s.exercise._phase].prompt


def test_prompt_repeated_when_the_hand_comes_back(tmp_path):
    s, speaker, _, _, go, clock = session(_log(tmp_path))
    go(50)
    assert s.stage == "exercise"
    for _ in range(90):                           # hand gone for 3 s
        t = clock.tick()
        speaker.advance()
        s.update(feat(t).__class__(t=t), t)
    speaker.clear()
    t = clock.tick()
    s.update(feat(t, flex=MID), t)
    assert s.exercise.phases[s.exercise._phase].prompt in [m.text for m in speaker.items]


def test_out_of_date_messages_are_skipped():
    clock = Clock()
    speaker = TimedSpeaker(clock)
    state = {"phase": "open"}
    speaker.say(Say("Let's open and close your hand."))
    speaker.say(Say("Open your hand wide.", valid=lambda: state["phase"] == "open"))
    speaker.advance()
    state["phase"] = "close"                      # she opened and held while we talked
    for _ in range(200):
        clock.tick()
        speaker.advance()
    assert [t for _, t in speaker.heard] == ["Let's open and close your hand."]


def test_full_queue_drops_hints_before_instructions():
    items = collections.deque()
    enqueue(items, Say("Now close your hand."), 3)
    enqueue(items, Say("Hold it a little longer.", "hint"), 3)
    enqueue(items, Say("Well done. That was set one of three."), 3)
    enqueue(items, Say("Let's rest for 30 seconds."), 3)
    assert [m.text for m in items] == ["Now close your hand.",
                                       "Well done. That was set one of three.",
                                       "Let's rest for 30 seconds."]


def test_squeeze_relax_is_not_counted_as_a_hold():
    cal = calibrate(GripSqueeze, [dict(flex=(40, 50, 35)), dict(flex=(60, 80, 50))])
    ex = GripSqueeze(calibration=cal)
    clock = Clock()
    ex.start_set(1, clock.t)
    run(ex, 5.0, clock, flex=(60, 80, 50))       # squeeze and hold
    assert ex.phases[ex._phase].key == "relax"
    said = texts(run(ex, 5.0, clock, flex=(40, 50, 35)))
    assert "And hold." not in said and "two" not in said
    assert "That's one." in said


def test_fatigue_rest_drops_the_next_prompt_but_later_reps_are_prompted(tmp_path):
    s = SessionManager(TimedSpeaker(Clock()), storage.new_profile(), _log(tmp_path),
                       exercises=["grip_release"])
    s.index = 0
    s._build_exercise()
    ex = s.exercise
    ex.start_set(1, 0.0)
    ex._phase = 1
    prompt = ex._prompt()                         # "Now close your hand." queued ...
    s.coach.interrupt(0.0)                        # ... then a fatigue rest starts
    assert not prompt.still_valid()
    # still tired after the one offered rest: the next rep is asked for as usual
    ex.fatigue = True
    ex._rep.update(values=[0.1], raw=[0.1], times=[1.0])
    assert texts(ex._phase_complete(None, 1.0))[-1] == "Open your hand wide."
