"""Choosing the text-to-speech backend by operating system (no audio is played)."""

import sys
import types

from rehab import tts_util
from rehab.tts_util import CommandTTS, Pyttsx3TTS, detect_os, make_tts


class FakeEngine:
    def __init__(self):
        self.props = {}
        self.spoken = []

    def setProperty(self, name, value):
        self.props[name] = value

    def say(self, text):
        self.spoken.append(text)

    def runAndWait(self):
        pass


def fake_pyttsx3(monkeypatch, works=True):
    engine = FakeEngine()

    def init():
        if not works:
            raise RuntimeError("no voices")
        return engine

    monkeypatch.setitem(sys.modules, "pyttsx3", types.SimpleNamespace(init=init))
    return engine


def no_pyttsx3(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyttsx3", None)      # import raises ImportError


def commands(monkeypatch, *available):
    monkeypatch.setattr(tts_util.shutil, "which",
                        lambda exe: f"/usr/bin/{exe}" if exe in available else None)


def test_detect_os():
    assert detect_os("darwin") == "macos"
    assert detect_os("win32") == "windows"
    assert detect_os("cygwin") == "windows"
    assert detect_os("linux") == "linux"
    assert detect_os("freebsd13") == "other"


def test_macos_uses_say(monkeypatch):
    fake_pyttsx3(monkeypatch)
    commands(monkeypatch, "say")
    tts = make_tts(140, "macos")
    assert isinstance(tts, CommandTTS)
    assert tts._argv("Open your hand wide.") == ["say", "-r", "140", "Open your hand wide."]


def test_macos_without_say_falls_back_to_pyttsx3(monkeypatch):
    fake_pyttsx3(monkeypatch)
    commands(monkeypatch)
    assert isinstance(make_tts(140, "macos"), Pyttsx3TTS)


def test_windows_uses_pyttsx3_at_the_slow_rate(monkeypatch):
    engine = fake_pyttsx3(monkeypatch)
    commands(monkeypatch, "say", "espeak")
    tts = make_tts(140, "windows")
    assert isinstance(tts, Pyttsx3TTS)
    assert engine.props["rate"] == 140
    tts.speak("Now close your hand.")
    assert engine.spoken == ["Now close your hand."]


def test_windows_without_pyttsx3_has_no_speech(monkeypatch):
    no_pyttsx3(monkeypatch)
    commands(monkeypatch, "espeak")
    assert make_tts(140, "windows") is None


def test_linux_prefers_pyttsx3(monkeypatch):
    fake_pyttsx3(monkeypatch)
    commands(monkeypatch, "espeak")
    assert isinstance(make_tts(140, "linux"), Pyttsx3TTS)


def test_linux_falls_back_to_espeak(monkeypatch):
    fake_pyttsx3(monkeypatch, works=False)
    commands(monkeypatch, "espeak")
    tts = make_tts(140, "linux")
    assert isinstance(tts, CommandTTS)
    assert tts._argv("Hello.") == ["espeak", "-s", "140", "Hello."]


def test_linux_without_any_voice_has_no_speech(monkeypatch):
    no_pyttsx3(monkeypatch)
    commands(monkeypatch)
    assert make_tts(140, "linux") is None
