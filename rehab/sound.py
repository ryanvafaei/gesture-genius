"""
Act: soft musical notes for the "finger piano" (thumb opposition).

Every correct fingertip touch plays the next note of a familiar tune
("Twinkle, Twinkle, Little Star"). A wrong touch plays nothing: there is
never a "wrong" sound.

The notes are made here with numpy (a sine with a quiet overtone that fades
out, like a soft piano), written once as small WAV files to a temporary
folder, and played in the background with the system's own player, the same
way as the chime: afplay on macOS, paplay or aplay on Linux, winsound on
Windows. pygame is not used on purpose: its SDL library clashes with the
copy inside OpenCV on macOS. Without a player the notes are silently skipped.
"""

import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from rehab import config

RATE = 22050
NOTE_S = 0.6
# "Twinkle, Twinkle, Little Star" as MIDI note numbers (C4 = 60)
TUNE = (60, 60, 67, 67, 69, 69, 67,
        65, 65, 64, 64, 62, 62, 60,
        67, 67, 65, 65, 64, 64, 62,
        67, 67, 65, 65, 64, 64, 62)


def tone(midi, seconds=NOTE_S, rate=RATE, volume=config.NOTE_VOLUME):
    """A soft piano-like note as int16 samples."""
    t = np.arange(int(seconds * rate)) / rate
    freq = 440.0 * 2 ** ((midi - 69) / 12)
    wave_ = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
    attack = np.clip(t / 0.01, 0.0, 1.0)               # no click at the start
    envelope = attack * np.exp(-t * 5.0)
    samples = wave_ / 1.3 * envelope * volume
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)


def _player():
    """Command that plays a WAV file, "winsound" on Windows, or None."""
    if sys.platform.startswith("win"):
        return "winsound"
    for cmd in (["afplay"], ["paplay"], ["aplay", "-q"]):
        if shutil.which(cmd[0]):
            return cmd
    return None


class Notes:
    """Plays note i of the tune in the background (files are made on first use)."""

    def __init__(self, folder=None, enabled=True):
        self.folder = Path(folder) if folder else Path(tempfile.gettempdir()) / "hand_coach_notes"
        self.player = _player() if enabled else None
        self._files = {}

    def file(self, midi):
        path = self._files.get(midi)
        if path is None:
            self.folder.mkdir(parents=True, exist_ok=True)
            path = self.folder / f"note_{midi}.wav"
            if not path.is_file():
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(RATE)
                    w.writeframes(tone(midi).tobytes())
            self._files[midi] = path
        return path

    def play(self, i):
        if self.player is None:
            return
        try:
            path = str(self.file(TUNE[int(i) % len(TUNE)]))
            if self.player == "winsound":
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                subprocess.Popen(self.player + [path], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except (OSError, RuntimeError, ImportError):
            self.player = None
