"""
Text-to-speech backend, chosen by operating system.

macOS     the built-in `say` command (`say -r 140`), as before.
Windows   pyttsx3 (SAPI5 voices).
Linux     pyttsx3 (eSpeak voices) when it is installed and works,
          otherwise the `espeak-ng` / `espeak` command.

Every backend speaks one text and blocks until it has been spoken; the
Speaker thread in Act.py decides what is said and when. Create the backend
in the thread that will speak with it: pyttsx3 engines (COM objects on
Windows) should stay in the thread that made them.
"""

import shutil
import subprocess
import sys


def detect_os(platform=None):
    """'macos', 'windows', 'linux' or 'other'."""
    platform = platform or sys.platform
    if platform == "darwin":
        return "macos"
    if platform.startswith("win") or platform == "cygwin":
        return "windows"
    if platform.startswith("linux"):
        return "linux"
    return "other"


class CommandTTS:
    """Speaks by running a command line, e.g. `say -r 140 text`."""

    def __init__(self, argv):
        self._argv = argv           # text -> list of arguments
        self._proc = None

    def speak(self, text):
        self._proc = subprocess.Popen(self._argv(text),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._proc.wait()
        self._proc = None

    def stop(self):
        proc = self._proc
        if proc is not None:
            proc.terminate()


class Pyttsx3TTS:
    """pyttsx3; raises when it is not installed or has no working voice."""

    def __init__(self, rate):
        import pyttsx3
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)

    def speak(self, text):
        self._engine.say(text)
        self._engine.runAndWait()

    def stop(self):
        # runAndWait() blocks the speaking thread, which is a daemon: nothing to do
        pass


def _say(rate):
    if shutil.which("say"):
        return CommandTTS(lambda text: ["say", "-r", str(rate), text])
    return None


def _pyttsx3(rate):
    try:
        return Pyttsx3TTS(rate)
    except Exception:
        return None


def _espeak(rate):
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            return CommandTTS(lambda text, exe=exe: [exe, "-s", str(rate), text])
    return None


# Tried in order for each system; the first one that works is used.
BACKENDS = {
    "macos": (_say, _pyttsx3),
    "windows": (_pyttsx3,),
    "linux": (_pyttsx3, _espeak),
    "other": (_pyttsx3, _espeak),
}


def make_tts(rate, os_name=None):
    """The text-to-speech backend for this system, or None when there is none."""
    for make in BACKENDS[os_name or detect_os()]:
        tts = make(rate)
        if tts is not None:
            return tts
    return None
