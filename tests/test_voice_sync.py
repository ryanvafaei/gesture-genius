"""
Voice and screen stay together.

When she moves on (answers, presses space, the next step starts) the coach
stops saying what belonged to the step she left and only says what belongs
to the new one; what the screen shows is the same text she hears; drawing a
frame stays fast.
"""

import threading
import time
import types

import cv2
import numpy as np

from rehab import Act, storage, tts_util, ui
from rehab.Act import Speaker, interrupts
from rehab.events import event
from rehab.exercises.base import Say
from rehab.feedback import Feedback
from rehab.Think import SessionManager
from rehab.tts_util import CommandTTS, Pyttsx3TTS
from helpers import Clock, TimedSpeaker, feat, returning_profile

PHRASES = storage.load_content("phrases")


def _log(tmp_path):
    return storage.SessionLog(tmp_path / "reps.csv", tmp_path / "history.csv", session_id="s1")


def _session(tmp_path):
    clock = Clock()
    speaker = TimedSpeaker(clock)
    s = SessionManager(speaker, returning_profile(), _log(tmp_path), exercises=["grip_release"])
    return s, speaker, clock


def _frames(s, speaker, clock, seconds):
    for _ in range(int(seconds * 30)):
        t = clock.tick()
        speaker.advance()
        s.update(feat(t), t)


def _heard_after(speaker, t):
    return [text for when, text in speaker.heard if when > t]


# --- moving on ---------------------------------------------------------------------

def test_going_on_during_the_greeting_cuts_it_off(tmp_path):
    s, speaker, clock = _session(tmp_path)
    _frames(s, speaker, clock, 0.2)
    assert s.stage == "greeting" and speaker.current is not None
    greeting = speaker.current.text
    t = clock.tick()
    s.on_key(" ", t)            # she goes on while the greeting is still being said
    assert s.stage == "check_in"
    assert greeting in speaker.cut_off
    for _ in range(300):
        clock.tick()
        speaker.advance()
    # from now on she only hears the question on the screen
    assert _heard_after(speaker, t)[:2] == [PHRASES["CheckIn"][0], PHRASES["CheckInHint"][0]]


def test_answering_drops_the_rest_of_the_question(tmp_path):
    s, speaker, clock = _session(tmp_path)
    _frames(s, speaker, clock, 0.2)
    s.on_key(" ", clock.tick())
    _frames(s, speaker, clock, 0.2)
    assert s.stage == "check_in"
    assert speaker.current.text == PHRASES["CheckIn"][0]
    t = clock.tick()
    s.on_key("y", t)            # answered before the hint was said
    assert s.stage == "today_plan"
    for _ in range(300):
        clock.tick()
        speaker.advance()
    heard = _heard_after(speaker, t)
    assert PHRASES["CheckInHint"][0] not in heard
    assert PHRASES["CheckIn"][0] not in heard


def test_pause_cuts_off_the_instructions(tmp_path):
    s, speaker, clock = _session(tmp_path)
    _frames(s, speaker, clock, 0.2)
    s.on_key(" ", clock.tick())                 # greeting -> check-in
    s.on_key("y", clock.tick())                 # check-in -> today's plan
    s.on_key(" ", clock.tick())                 # today's plan -> intro
    _frames(s, speaker, clock, 0.5)
    assert s.stage == "intro" and speaker.current is not None
    speaking = speaker.current.text
    t = clock.tick()
    s.on_key(" ", t)
    assert s.paused
    assert speaking in speaker.cut_off
    for _ in range(300):
        clock.tick()
        speaker.advance()
    assert _heard_after(speaker, t) == ["Paused. Press the space bar when you are ready."]


def test_what_is_said_for_the_new_step_is_kept(tmp_path):
    s, speaker, clock = _session(tmp_path)
    _frames(s, speaker, clock, 0.2)
    mark = speaker.mark()
    speaker.say(Say("Old words."))
    speaker.drop_before(speaker.mark())
    speaker.say(Say("New words."))
    speaker.drop_before(mark + 1)               # only what came before "New words." goes
    assert [m.text for m in speaker.items] == ["New words."]


def test_an_out_of_date_message_is_cut_off_by_the_next_instruction():
    stale = Say("Now close your hand.", valid=lambda: False)
    assert interrupts(stale, Say("Open your hand wide."), 0.0)
    assert not interrupts(Say("Now close your hand."), Say("Open your hand wide."), 0.0)
    assert not interrupts(stale, Say("That's two.", "count"), 0.0)


class SlowTTS:
    """Takes five seconds per text unless it is cancelled."""

    def __init__(self):
        self.started = []

    def speak(self, text, cancelled=None):
        self.started.append(text)
        end = time.monotonic() + 5
        while time.monotonic() < end and not (cancelled and cancelled()):
            time.sleep(0.01)

    def stop(self):
        pass


def _wait(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        time.sleep(0.01)
    return condition()


def test_speaker_stops_the_old_step_at_once(monkeypatch):
    tts = SlowTTS()
    monkeypatch.setattr(Act, "make_tts", lambda rate, voice=None: tts)
    speaker = Speaker(enabled=True)
    try:
        speaker.say(Say("Hello Eleanor. Last time you practised grip and release."))
        speaker.say(Say("Thumbs up to continue."))
        assert _wait(lambda: tts.started)
        mark = speaker.mark()
        speaker.say(Say("How is your hand feeling today?"))
        start = time.monotonic()
        speaker.drop_before(mark)
        assert _wait(lambda: len(tts.started) == 2)
        assert time.monotonic() - start < 1.0
        assert tts.started[1] == "How is your hand feeling today?"
        assert _wait(lambda: speaker.last_text == "How is your hand feeling today?")
    finally:
        speaker.close()


# --- text-to-speech backends can be stopped ------------------------------------------

class FakeProc:
    def __init__(self):
        self.done = threading.Event()

    def wait(self):
        self.done.wait(5)

    def terminate(self):
        self.done.set()


def test_say_command_is_not_started_when_cancelled(monkeypatch):
    started = []
    monkeypatch.setattr(tts_util.subprocess, "Popen",
                        lambda *a, **k: started.append(a) or FakeProc())
    CommandTTS(lambda text: ["say", text]).speak("Hello.", cancelled=lambda: True)
    assert started == []


def test_say_command_is_ended_by_stop(monkeypatch):
    proc = FakeProc()
    monkeypatch.setattr(tts_util.subprocess, "Popen", lambda *a, **k: proc)
    tts = CommandTTS(lambda text: ["say", text])
    thread = threading.Thread(target=tts.speak, args=("Hello.",))
    thread.start()
    assert _wait(lambda: tts._proc is not None)
    tts.stop()
    thread.join(1.0)
    assert not thread.is_alive()


def test_pyttsx3_stops_at_the_next_word(monkeypatch):
    class Engine:
        def __init__(self):
            self.words = []
            self.callback = None
            self.stopped = False

        def setProperty(self, name, value):
            pass

        def connect(self, topic, callback):
            self.callback = callback

        def say(self, text):
            self.text = text

        def stop(self):
            self.stopped = True

        def runAndWait(self):
            for i, word in enumerate(self.text.split()):
                self.callback("utt", i, len(word))
                if self.stopped:
                    return
                self.words.append(word)

    engine = Engine()
    monkeypatch.setitem(__import__("sys").modules, "pyttsx3",
                        types.SimpleNamespace(init=lambda: engine))
    tts = Pyttsx3TTS(140)
    flag = {"cancel": False}

    def cancelled():
        if len(engine.words) == 2:
            flag["cancel"] = True
        return flag["cancel"]

    tts.speak("Open your hand wide and hold it there.", cancelled=cancelled)
    assert engine.words == ["Open", "your"]


# --- the screen shows what she hears -------------------------------------------------

def _varied_feedback():
    phrases = dict(PHRASES)
    phrases["GardenGrew.2"] = ["Your {plant} has leaves.", "Look, your {plant} grew.",
                               "Your {plant} is a little taller."]
    phrases["Summary.nothing"] = ["Thank you for practising.", "Good to see you today.",
                                  "Well done for showing up."]
    phrases["ExerciseDone"] = ["Well done. That's the exercise finished.",
                               "Lovely. That exercise is done.", "Good. All done with this one."]
    return Feedback(phrases=phrases)


def test_cards_show_the_words_she_heard():
    fb = _varied_feedback()
    for _ in range(10):
        ev = event("ExerciseDone")
        spoken = [m.text for m in fb.words(ev)]
        card = fb.card(ev)          # "Well done." as the title, the rest below it
        assert f"{card['title']} {card['message']}".strip() == spoken[0]


def test_summary_and_garden_show_the_words_she_heard():
    fb = _varied_feedback()
    for _ in range(10):
        summary = event("Summary", highlight=None, comparisons=[], activities=[], difficult=False)
        assert [m.text for m in fb.words(summary)] == fb.summary_lines(summary)
        grew = event("GardenGrew", plant="rose", stage=2)
        assert [m.text for m in fb.words(grew)] == [fb.garden_line(grew)]


def test_words_said_twice_are_new_messages():
    fb = _varied_feedback()
    ev = event("ExerciseDone")
    first, second = fb.words(ev)[0], fb.words(ev)[0]
    first.seq = 7
    assert first is not second and second.seq is None
    assert first.text == second.text


# --- drawing -----------------------------------------------------------------------

def test_kept_text_looks_the_same_as_drawing_it_with_pillow():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 700, 3), dtype=np.uint8)
    lines = [("Now open your hand wide.", (20, 60), 40, (255, 255, 255), True),
             ("That's two.", (30, 130), 30, (0, 220, 255), False),
             ("Cut at the edge", (600, 190), 34, (80, 200, 80), True)]
    layer = ui.TextLayer()
    for text, xy, size, color, bold in lines:
        layer.add(text, xy, size, color, bold=bold)
    ours = layer.apply(img.copy())

    pil = ui.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ui.ImageDraw.Draw(pil)
    for text, xy, size, color, bold in lines:
        draw.text(xy, text, font=ui._font(size, bold), fill=tuple(reversed(color)), anchor="ls")
    theirs = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    assert np.abs(ours.astype(int) - theirs.astype(int)).max() <= 1


def test_drawing_a_frame_is_fast():
    display = Act.Display()
    frame = np.zeros((720, 1280, 3), np.uint8)
    view = {"stage": "exercise", "title": "Open and close", "status": "Set 1 of 3   2 of 10",
            "instruction": "Now open your hand wide.", "subtitle": "That's two.",
            "footer": "Space: pause / continue   M: menu",
            "exercise_display": {"kind": "bar", "value": 0.6, "high": 0.8, "low": 0.2,
                                 "target_zone": "high", "phase_label": "Open",
                                 "finger_colors": {}},
            "can": {"sections": 3, "filled": 1}}
    display.render(frame.copy(), view)
    start = time.perf_counter()
    for _ in range(20):
        display.render(frame.copy(), view)
    per_frame = (time.perf_counter() - start) / 20
    assert per_frame < 0.010, f"{per_frame * 1000:.1f} ms per frame"
