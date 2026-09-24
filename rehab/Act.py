"""
Act: speech and drawing.

Speaker   non-blocking text-to-speech in its own thread, slow rate
          (`say -r 145` on macOS, pyttsx3 on Windows and Linux; see tts_util),
          with a priority queue: instructions may jump ahead of waiting
          praise and cut off praise being spoken, but praise never delays
          or interrupts an instruction, so she never misses what to do next.
          One message at a time; praise waiting longer than
          PRAISE_MAX_WAIT_S is dropped instead of played late. Counts are
          dropped when speech is busy, and every message is checked again
          just before it is spoken: one that is out of date by then (she
          already did what it asks) is skipped, and one being spoken is cut
          off by the next instruction. When she moves on (next step, an
          answer, pause) everything said for the step she left is dropped
          and cut off, so she only hears what belongs to what she sees.
          Events from Think are worded by feedback.Feedback; a successful
          rep plays a soft chime.
Display   camera image with the hand skeleton plus a side panel:
          a large bar with the target line, per-finger detail, the
          finger sequence, and big high-contrast text; full-screen cards,
          the summary and the garden (see ui.py). Whatever is said is also
          shown as a subtitle.
"""

import collections
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time

import cv2
import numpy as np

from rehab import config, ui
from rehab.events import Event
from rehab.exercises.base import FINGER_WORDS, Say
from rehab.features import FINGER_LANDMARKS, GAP_NAMES, THUMB, WRIST
from rehab.feedback import Feedback
from rehab.tts_util import make_tts

# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------


# Dropped first when the queue is full; instructions and session messages last.
DROP_FIRST = ("count", "praise", "hint")
# Messages queued within this time of each other belong together (e.g. the
# rep count, its praise and the next prompt) and keep their order.
BATCH_S = 0.1


def enqueue(items, msg, max_queue, now=0.0):
    """
    Add msg to the deque `items`: making room by dropping the least important
    old message, and placing an instruction ahead of older, less important
    messages (praise) that are still waiting.
    """
    msg.queued_at = now
    # out of date, or the same words again (e.g. a prompt repeated after a lost hand)
    for m in [m for m in items if not m.still_valid() or m.text == msg.text]:
        items.remove(m)
    while len(items) >= max_queue:
        droppable = [m for m in items if m.ephemeral or m.kind in DROP_FIRST]
        # the least important droppable one (the oldest of those), else the oldest
        victim = max(droppable, key=lambda m: m.priority) if droppable else items[0]
        items.remove(victim)
    i = len(items)
    if msg.priority <= 2:
        while (i > 0 and items[i - 1].priority > msg.priority
               and items[i - 1].queued_at is not None
               and items[i - 1].queued_at < now - BATCH_S):
            i -= 1
    items.insert(i, msg)


def next_message(items, now, max_wait=config.PRAISE_MAX_WAIT_S):
    """The next message worth saying (popped), or None."""
    while items:
        msg = items.popleft()
        if not msg.still_valid():
            continue
        if msg.kind == "praise" and msg.queued_at is not None and now - msg.queued_at > max_wait:
            continue            # too late now: she has moved on
        return msg
    return None


def interrupts(current, msg, now):
    """
    True when msg (an instruction) should cut off the message being spoken:
    praise from an earlier batch, or a message that is out of date by now
    (she already did what it asks).
    """
    if current is None or msg.priority > 2:
        return False
    if not current.still_valid():
        return True
    return (current.kind == "praise" and current.queued_at is not None
            and current.queued_at < now - BATCH_S)


def drop_older(items, mark):
    """Remove the waiting messages that were given to the speaker before mark."""
    for m in [m for m in items if m.seq is not None and m.seq < mark]:
        items.remove(m)


class SpeechBase:
    """
    Shared by all speakers: Think's events are worded here, chimes are played.

    Every message is numbered in the order it is given. Think takes a mark()
    before handling a frame or a key; when she moved on (next step, an
    answer, pause) it calls drop_before(mark): what was said before is no
    longer about what she sees or does, so it is dropped, and cut off if it
    is being spoken. What was said for the new step (after the mark) stays.
    """

    feedback = None
    mood = "neutral"
    _seq = 0

    def say(self, msg):
        if isinstance(msg, (Event, list, tuple)):
            self.say_events(msg)
            return
        if isinstance(msg, str):
            msg = Say(msg)
        if msg.kind == "chime":
            self.chime()
            return
        msg.seq = self._seq
        self._seq += 1
        self._say(msg)

    def mark(self):
        """A point in what was said, for drop_before()."""
        return self._seq

    def drop_before(self, mark):
        """She moved on: forget, and stop saying, everything said before mark."""

    def say_all(self, messages):
        for m in messages or []:
            self.say(m)

    def say_events(self, events):
        if self.feedback is None:
            self.feedback = Feedback()
        for m in self.feedback.words(events):
            self.say(m)

    def _say(self, msg):
        raise NotImplementedError

    def chime(self):
        pass


def _chime_command():
    if sys.platform == "darwin" and shutil.which("afplay"):
        sound = "/System/Library/Sounds/Tink.aiff"
        if os.path.exists(sound):
            return ["afplay", "-v", "0.4", sound]
    for player, sound in (("paplay", "/usr/share/sounds/freedesktop/stereo/message.oga"),
                          ("aplay", "/usr/share/sounds/alsa/Front_Center.wav")):
        if shutil.which(player) and os.path.exists(sound):
            return [player, sound]
    return None


class Speaker(SpeechBase):

    def __init__(self, enabled=config.SPEECH_ENABLED, rate=config.SPEECH_RATE, max_queue=6,
                 voice=config.VOICE, feedback=None):
        self.enabled = enabled
        self.rate = int(rate)
        self.voice = voice
        self.max_queue = max_queue
        self.feedback = feedback
        self.last_text = ""
        self.last_time = 0.0
        self._items = collections.deque()
        self._cond = threading.Condition()
        self._speaking = False
        self._current = None
        self._cancelled = None          # the message being cut off
        self._stop = threading.Event()
        self._tts = None
        self._chime = _chime_command() if enabled else None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def busy(self):
        with self._cond:
            return self._speaking or bool(self._items)

    def _say(self, msg):
        now = time.monotonic()
        with self._cond:
            if msg.ephemeral and (self._speaking or self._items):
                return
            if interrupts(self._current, msg, now):
                self._cut_off()
            enqueue(self._items, msg, self.max_queue, now)
            self._cond.notify()

    def chime(self):
        if self._chime:
            try:
                subprocess.Popen(self._chime, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                self._chime = None

    def clear(self):
        """Forget everything not yet spoken (e.g. on pause)."""
        with self._cond:
            self._items.clear()

    def drop_before(self, mark):
        with self._cond:
            drop_older(self._items, mark)
            current = self._current
            if current is not None and current.seq is not None and current.seq < mark:
                self._cut_off()
                self.last_text = ""         # no subtitle for words she no longer hears

    def _cut_off(self):
        """Stop the message being spoken (called with self._cond held)."""
        self._cancelled = self._current
        if self._tts is not None:
            try:
                self._tts.stop()
            except Exception:
                pass

    def _run(self):
        if self.enabled:
            # made here: a pyttsx3 engine stays in the thread that created it
            self._tts = make_tts(self.rate, voice=self.voice)
            self.enabled = self._tts is not None
        while not self._stop.is_set():
            with self._cond:
                if not self._items:
                    self._cond.wait(timeout=0.1)
                    continue
                msg = next_message(self._items, time.monotonic())
                if msg is None:
                    continue
                self._speaking = True       # set together with the pop: busy never blinks off
                self._current = msg
                self.last_text = msg.text
                self.last_time = time.monotonic()
                self.mood = msg.mood or "neutral"
            try:
                self._speak(msg)
            finally:
                with self._cond:
                    self._speaking = False
                    self._current = None

    def _speak(self, msg):
        def cancelled():
            return self._cancelled is msg or self._stop.is_set()

        text = msg.text
        if not self.enabled:
            print(f"[coach] {text}")
            # roughly paced, keeps timings realistic
            end = time.monotonic() + 0.05 * len(text.split()) + 0.2
            while time.monotonic() < end and not cancelled():
                time.sleep(0.02)
            return
        try:
            self._tts.speak(text, cancelled=cancelled)
        except Exception:
            print(f"[coach] {text}")
            self.enabled = False

    def close(self):
        self._stop.set()
        if self._tts is not None:
            self._tts.stop()


class SilentSpeaker(SpeechBase):
    """Prints instead of speaking and never blocks (for tests and validation)."""

    def __init__(self, echo=False, feedback=None):
        self.echo = echo
        self.feedback = feedback
        self.spoken = []
        self.chimes = 0
        self.last_text = ""
        self.last_time = 0.0

    @property
    def busy(self):
        return False

    def _say(self, msg):
        if not msg.still_valid():
            return
        self.spoken.append(msg)
        self.last_text = msg.text
        self.mood = msg.mood or "neutral"
        if self.echo:
            print(f"[coach] {msg.text}")

    def chime(self):
        self.chimes += 1

    def clear(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK = (35, 35, 35)
GREY = (140, 140, 140)
YELLOW = (0, 220, 255)
GREEN = (80, 200, 80)
BLUE = (230, 160, 60)
ORANGE = (0, 140, 255)
CYAN = (230, 230, 0)
FONT = cv2.FONT_HERSHEY_DUPLEX

FINGER_STATE_COLORS = {"lagging": ORANGE, "target": CYAN, "active": GREEN}
GAP_COLORS = {"index_middle": (80, 180, 255), "middle_ring": (120, 220, 120), "ring_pinky": (255, 150, 200)}

PANEL_W = 420


# Text goes through a ui.TextLayer (Atkinson Hyperlegible via Pillow), drawn
# once per frame; positions are the left end of the baseline, as in cv2.
_LAYER = {"text": None, "offsets": {}}


def _text(img, text, org, scale=1.0, color=WHITE, thickness=2):
    layer = _LAYER["text"]
    if layer is None:
        cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)
        return
    dx = _LAYER["offsets"].get(id(img), 0)
    layer.add(text, (org[0] + dx, org[1]), int(scale * ui.FONT_PX_PER_SCALE), color,
              bold=thickness >= 2)


def _band(img, y0, y1, alpha=0.6):
    """Dark translucent band so text stays readable on any camera image."""
    band = img[max(0, y0):max(0, y1 + 1)]   # darken only the band, not a copy of the whole image
    band[:] = cv2.addWeighted(band, 1 - alpha, band, 0, 0)


def _wrapped(img, text, x, y, width_px, scale=1.0, color=WHITE, thickness=2, line_gap=1.45):
    layer = _LAYER["text"]
    if layer is not None:
        size = int(scale * ui.FONT_PX_PER_SCALE)
        lines = layer.wrap(text, size, thickness >= 2, width_px)
        h = size * 0.72
        for i, line in enumerate(lines):
            _text(img, line, (x, int(y + i * h * line_gap)), scale, color, thickness)
        return y + len(lines) * h * line_gap
    char_w = cv2.getTextSize("M", FONT, scale, thickness)[0][0] * 0.8
    per_line = max(8, int(width_px / char_w))
    h = cv2.getTextSize("Mg", FONT, scale, thickness)[0][1]
    for i, line in enumerate(textwrap.wrap(text, per_line)):
        _text(img, line, (x, int(y + i * h * line_gap)), scale, color, thickness)
    return y + len(textwrap.wrap(text, per_line)) * h * line_gap


class Display:

    def __init__(self, window="Hand coach", character=None, feedback=None):
        self.window = window
        self.character = character or {}
        self.feedback = feedback or Feedback()
        self._cache = {}

    def _worded(self, kind, ev, fn):
        """Words for an event, worked out once (the same text she hears)."""
        hit = self._cache.get(kind)
        if hit is None or hit[0] is not ev:
            hit = (ev, fn(ev))
            self._cache[kind] = hit
        return hit[1]

    def _words(self, view):
        """Think's events in the view -> the text Act shows."""
        v = dict(view)
        if v.get("card_event") is not None:
            v["card"] = self._worded("card", v["card_event"], self.feedback.card)
        if v.get("summary_event") is not None:
            lines = self._worded("summary", v["summary_event"], self.feedback.summary_lines)
            v["summary_lines"] = list(lines) + list(view.get("summary_lines", []))
        if v.get("garden_event") is not None:
            line = self._worded("garden", v["garden_event"], self.feedback.garden_line)
            v["garden"] = dict(v.get("garden") or {}, line=line)
        if v.get("activity"):
            v["activity_icon"] = self.feedback.activity_info(v["activity"]).get("icon")
        return v

    # --- skeleton -----------------------------------------------------------

    def draw_hand(self, frame, f, finger_colors=None):
        if f is None or not f.present or f.image_points is None:
            return
        pts = f.image_points.astype(int)
        finger_colors = finger_colors or {}
        chains = {"thumb": (WRIST,) + THUMB}
        for name, lm in FINGER_LANDMARKS.items():
            chains[name] = (WRIST,) + lm if name in ("index", "pinky") else lm
        cv2.polylines(frame, [pts[[5, 9, 13, 17]]], False, WHITE, 3, cv2.LINE_AA)
        for name, chain in chains.items():
            state = finger_colors.get(name)
            color = FINGER_STATE_COLORS.get(state, WHITE)
            width = 8 if state else 3
            for a, b in zip(chain[:-1], chain[1:]):
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, width, cv2.LINE_AA)
            if state == "target":
                cv2.circle(frame, tuple(pts[chain[-1]]), 22, CYAN, 4, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), 4, BLUE, -1, cv2.LINE_AA)

    # --- panel widgets ------------------------------------------------------

    def _bar(self, panel, d, top, height=360):
        x0, x1 = 60, 170
        y_top, y_bot = top, top + height
        span = y_bot - y_top

        def y_of(v):
            return int(y_bot - np.clip(v, 0.0, 1.2) / 1.2 * span)

        cv2.rectangle(panel, (x0, y_top), (x1, y_bot), GREY, 2)
        value = d.get("value", 0.0)
        target_zone = d.get("target_zone")
        reached = (value >= d["high"]) if target_zone == "high" else (value <= d["low"])
        fill = GREEN if reached else BLUE
        cv2.rectangle(panel, (x0 + 3, y_of(value)), (x1 - 3, y_bot - 3), fill, -1)
        # target lines
        for level, key in ((d["high"], "high"), (d["low"], "low")):
            y = y_of(level)
            active = key == target_zone
            cv2.line(panel, (x0 - 25, y), (x1 + 25, y), YELLOW if active else GREY, 5 if active else 2)
        if d.get("best") is not None:
            y = y_of(d["best"])
            cv2.line(panel, (x1 + 5, y), (x1 + 30, y), WHITE, 2)
            _text(panel, "best", (x1 + 34, y + 8), 0.6, WHITE, 1)
        # hold progress as a filling ring
        cx, cy, r = 300, top + 90, 60
        cv2.circle(panel, (cx, cy), r, GREY, 6)
        if d.get("holding"):
            cv2.ellipse(panel, (cx, cy), (r, r), -90, 0, 360 * d.get("hold_progress", 0.0), GREEN, 10)
        _text(panel, d.get("phase_label", ""), (cx - 70, cy + r + 45), 1.1, YELLOW, 2)
        _text(panel, f"{int(round(value * 100))}%", (x0 + 5, y_bot + 45), 1.1, WHITE, 2)
        return y_bot + 70

    def _finger_bars(self, panel, values, top, colors, labels):
        x = 30
        for key, v in values.items():
            h = int(np.clip(v, 0, 1.2) / 1.2 * 90)
            color = colors.get(key, GREY)
            cv2.rectangle(panel, (x, top + 90 - h), (x + 60, top + 90), color, -1)
            cv2.rectangle(panel, (x, top), (x + 60, top + 90), GREY, 1)
            _text(panel, labels.get(key, key)[:6], (x, top + 115), 0.55, WHITE, 1)
            x += 85
        return top + 130

    def _sequence(self, panel, d, top):
        seq, step = d.get("sequence", []), d.get("step", 0)
        y = top
        for i, finger in enumerate(seq):
            done = i < step
            current = i == step
            label = "?" if d.get("hidden") and not done else FINGER_WORDS[finger].replace(" finger", "")
            filled = done or (current and not d.get("hidden"))
            color = GREEN if done else (CYAN if filled else GREY)
            cv2.rectangle(panel, (40, y), (PANEL_W - 40, y + 50), color, -1 if filled else 2)
            cv2.putText(panel, label.upper(), (60, y + 37), FONT, 1.0,
                        BLACK if filled else WHITE, 2, cv2.LINE_AA)
            y += 60
        if "level" in d:
            _text(panel, f"Level {d['level']}", (40, y + 35), 0.9, WHITE, 2)
            y += 50
        return y

    def _ring(self, panel, progress, center, radius=70, label=""):
        cv2.circle(panel, center, radius, GREY, 8)
        cv2.ellipse(panel, center, (radius, radius), -90, 0, 360 * progress, GREEN, 12)
        if label:
            size = cv2.getTextSize(label, FONT, 1.4, 3)[0]
            _text(panel, label, (center[0] - size[0] // 2, center[1] + size[1] // 2), 1.4, WHITE, 3)

    def _menu(self, frame, items):
        """The exercise list, large, over the camera image."""
        h, w = frame.shape[:2]
        top, bottom = 90, h - 150
        _band(frame, top - 10, bottom + 10, 0.7)
        row = int(min(72, (bottom - top) / max(1, len(items))))
        scale = row / 55
        for i, item in enumerate(items):
            y = top + i * row
            if item["selected"]:
                cv2.rectangle(frame, (30, y + 4), (w - 30, y + row - 4), CYAN, -1)
            color = BLACK if item["selected"] else WHITE
            base = y + int(row * 0.7)
            _text(frame, item["key"], (60, base), scale, YELLOW if not item["selected"] else BLACK, 3)
            _text(frame, item["text"], (60 + int(70 * scale), base), scale, color, 2)
            if item.get("note"):
                size = cv2.getTextSize(item["note"], FONT, scale * 0.7, 1)[0]
                _text(frame, item["note"], (w - 60 - size[0], base), scale * 0.7,
                      BLACK if item["selected"] else GREY, 1)

    # --- whole screen -------------------------------------------------------

    def render(self, frame, view, features=None):
        h, w = frame.shape[:2]
        view = self._words(view)
        text = ui.TextLayer() if ui.pillow_available() else None
        screen = view.get("screen", "exercise")
        if screen in ("card", "summary", "garden"):
            draw = {"card": ui.card_screen, "summary": ui.summary_screen,
                    "garden": ui.garden_screen}[screen]
            layer = text or ui.TextLayer()
            canvas = draw((w + PANEL_W, h), view, layer, self.character)
            return layer.apply(canvas)

        panel = np.full((h, PANEL_W, 3), DARK, dtype=np.uint8)
        _LAYER["text"] = text
        _LAYER["offsets"] = {id(frame): 0, id(panel): w}
        try:
            canvas = self._render_camera(frame, panel, view, features)
        finally:
            _LAYER["text"] = None
            _LAYER["offsets"] = {}
        return text.apply(canvas) if text is not None else canvas

    def _render_camera(self, frame, panel, view, features):
        h, w = frame.shape[:2]
        ex = view.get("exercise_display") or {}

        self.draw_hand(frame, features, ex.get("finger_colors"))
        if view.get("menu"):
            self._menu(frame, view["menu"])

        # title and counters
        _wrapped(panel, view.get("title", ""), 20, 45, PANEL_W - 40, 1.0, WHITE, 2)
        if view.get("status"):
            _text(panel, view["status"], (20, 120), 0.9, YELLOW, 2)
        top = 150

        stage = view.get("stage")
        if stage == "exercise" and ex.get("kind") == "bar":
            detail = ex.get("finger_values") or ex.get("gap_values")
            room = h - top - 70 - 80 - (140 if detail else 0)
            top = self._bar(panel, ex, top + 10, height=int(np.clip(room, 150, 400)))
            if ex.get("finger_values"):
                colors = {k: (ORANGE if ex["finger_colors"].get(k) == "lagging" else BLUE)
                          for k in ex["finger_values"]}
                labels = {k: FINGER_WORDS[k].replace(" finger", "") for k in ex["finger_values"]}
                top = self._finger_bars(panel, ex["finger_values"], top, colors, labels)
            elif ex.get("gap_values"):
                labels = {g: g.replace("_", "-").replace("pinky", "little") for g in GAP_NAMES}
                top = self._finger_bars(panel, ex["gap_values"], top, GAP_COLORS, labels)
        elif stage == "exercise" and ex.get("kind") == "sequence":
            top = self._sequence(panel, ex, top + 10)
        elif stage == "calibrating":
            self._ring(panel, view.get("progress", 0.0), (PANEL_W // 2, top + 130))
        elif stage == "rest":
            self._ring(panel, view.get("progress", 0.0), (PANEL_W // 2, top + 130),
                       label=str(view.get("countdown", "")))
        elif stage == "summary":
            y = top + 20
            for line in view.get("summary_lines", []):
                y = _wrapped(panel, line, 20, y, PANEL_W - 40, 0.75, WHITE, 1) + 10

        # small reminders at the right edge of the camera image: the coach,
        # what this practises for, and the watering can (exercises done today)
        ui.draw_face(frame, (w - 70, 150), 45, view.get("mood", "neutral"), self.character)
        if view.get("activity_icon"):
            cv2.circle(frame, (w - 70, 265), 48, (60, 60, 60), -1, cv2.LINE_AA)
            ui.draw_icon(frame, view["activity_icon"], (w - 70, 265), 64, WHITE)
        can = view.get("can")
        if can and can.get("sections") and view.get("stage") != "menu":
            ui.draw_can(frame, (w - 80, 375), 80, can["sections"], can.get("filled", 0))

        _text(panel, view.get("footer", ""), (20, h - 25), 0.7, GREY, 1)

        # big instruction at the bottom of the camera image
        instruction = view.get("instruction") or ex.get("prompt") or ""
        if instruction:
            _band(frame, h - 130, h)
            _wrapped(frame, instruction, 30, h - 80, w - 60, 1.6, WHITE, 3)

        # quality problem in yellow at the top
        if view.get("quality"):
            cv2.rectangle(frame, (0, 0), (w, 80), BLACK, -1)
            _text(frame, view["quality"], (30, 55), 1.3, YELLOW, 3)
        elif view.get("subtitle"):
            _band(frame, 0, 70, 0.45)
            _text(frame, view["subtitle"], (30, 48), 0.9, WHITE, 2)

        if view.get("paused"):
            if _LAYER["text"] is not None:
                _LAYER["text"].drop(0, w)       # covered by the pause screen
            _band(frame, 0, h)
            _text(frame, "Paused", (w // 2 - 110, h // 2 - 20), 2.2, WHITE, 4)
            _text(frame, "Press space to continue", (w // 2 - 250, h // 2 + 50), 1.2, YELLOW, 2)

        return np.hstack([frame, panel])

    def show(self, canvas):
        cv2.imshow(self.window, canvas)

    def close(self):
        cv2.destroyAllWindows()
