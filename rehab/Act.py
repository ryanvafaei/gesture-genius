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
          rep plays a soft chime, and in thumb opposition every correct
          touch plays the next note of a tune (sound.py).
Display   camera image with the hand skeleton plus a side panel:
          a large bar with the target line, per-finger detail, the
          finger sequence, and big high-contrast text. Arm exercises show
          the body skeleton (the trained arm highlighted), a box she should
          sit in, and a bar in degrees with today's target (and its
          tolerance band) and the daily-task milestones; full-screen cards,
          the summary, the garden and her profile (see ui.py). Cards, the
          summary and the profile show a small camera image, so she can see
          that her hand (and her thumbs up) is in view. Whatever is said is
          also shown as a subtitle.
          The window opens full screen and every screen is drawn in the
          shape of the screen (see canvas_size), so nothing falls off the edge;
          the window scales it to fit.
          Also drawn here: a still demo hand of the position to reach, the
          bubble between thumb and finger (bubble pinch), two bars for two
          hands, the memory game's cards over the camera image, and the
          "S: Stop" hint at the bottom right of every screen.
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

from rehab import body, config, demo, ui
from rehab.body import BodyFeatures
from rehab.events import Event
from rehab.exercises.base import FINGER_WORDS, Say
from rehab.features import FINGER_LANDMARKS, GAP_NAMES, THUMB, WRIST
from rehab.feedback import Feedback
from rehab.sound import Notes
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
    on_spoken = None        # callable(text) when a line is really spoken (verbose log)

    def _spoken(self, text):
        if self.on_spoken is not None:
            self.on_spoken(text)

    def say(self, msg):
        if isinstance(msg, (Event, list, tuple)):
            self.say_events(msg)
            return
        if isinstance(msg, str):
            msg = Say(msg)
        if msg.kind == "chime":
            self.chime()
            return
        if msg.kind == "note":
            self.note(msg.note or 0)       # finger piano: at once, never queued
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

    def note(self, i):
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
        self._notes = Notes(enabled=enabled)
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

    def note(self, i):
        self._notes.play(i)

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
            self._spoken(msg.text)
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
        self.notes = []
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
        self._spoken(msg.text)
        self.mood = msg.mood or "neutral"
        if self.echo:
            print(f"[coach] {msg.text}")

    def chime(self):
        self.chimes += 1

    def note(self, i):
        self.notes.append(i)

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
SKIP_BOTTOM = 105       # the Skip button's bottom edge, from the bottom of the panel
# full-screen pictures (ui.py); the camera screen is drawn here
SCREENS = {"card": ui.card_screen, "summary": ui.summary_screen, "garden": ui.garden_screen,
           "profile": ui.profile_screen, "rating": ui.rating_screen, "safety": ui.safety_screen}
# pair bars (two-hand match): the affected hand in blue, the leading hand grey
PAIR_COLORS = {"left": BLUE, "right": (170, 170, 170)}
CARD_BACK = (96, 150, 76)
POINTER = CYAN


def screen_size():
    """
    (width, height) of the main screen, or None when it cannot be found.
    Asked in a separate process: a second GUI toolkit in this process could
    clash with OpenCV's window. Only the shape is used, so logical (HiDPI)
    sizes are fine.
    """
    commands = [[sys.executable, "-c",
                 "import tkinter as t; r = t.Tk(); r.withdraw(); "
                 "print(r.winfo_screenwidth(), r.winfo_screenheight())"]]
    if sys.platform == "darwin":
        # no Dock icon, unlike tkinter
        commands.insert(0, ["osascript", "-l", "JavaScript", "-e",
                            'ObjC.import("AppKit"); var f = $.NSScreen.mainScreen.frame; '
                            'f.size.width + " " + f.size.height'])
    for cmd in commands:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.split()
            w, h = int(float(out[0])), int(float(out[1]))
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            continue
        if w >= 320 and h >= 240:
            return w, h
    return None


def fit_rect(frame_shape, w, h):
    """Where fit() puts the frame inside (w, h): (x0, y0, width, height)."""
    fh, fw = frame_shape[:2]
    k = min(w / fw, h / fh)
    nw, nh = max(1, int(round(fw * k))), max(1, int(round(fh * k)))
    return (w - nw) // 2, (h - nh) // 2, nw, nh


def fit(frame, w, h, background=DARK):
    """frame scaled to fit (w, h) without distortion, centred on the background."""
    fh, fw = frame.shape[:2]
    k = min(w / fw, h / fh)
    nw, nh = max(1, int(round(fw * k))), max(1, int(round(fh * k)))
    if (nw, nh) != (fw, fh):
        frame = cv2.resize(frame, (nw, nh),
                           interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    out = np.full((h, w, 3), background, np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = frame
    return out


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


def _fit_scale(text, width_px, scale, thickness=2):
    """scale, made smaller when one line of text would be wider than width_px."""
    layer = _LAYER["text"]
    if layer is not None:
        width = layer.measure(text, int(scale * ui.FONT_PX_PER_SCALE), thickness >= 2)
    else:
        width = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
    return scale if width <= width_px else scale * width_px / width


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


class _PanelText:
    """ui.draw_stop_hint's text calls, routed to the frame's text layer at the panel's offset."""

    def measure(self, text, size, bold=False):
        return _LAYER["text"].measure(text, size, bold)

    def add(self, text, xy, size, color, bold=False):
        panel_x = max(_LAYER["offsets"].values(), default=0)
        _LAYER["text"].add(text, (xy[0] + panel_x, xy[1]), size, color, bold=bold)


class Display:

    def __init__(self, window="Hand coach", character=None, feedback=None, screen=None,
                 fullscreen=config.FULLSCREEN):
        """screen: (width, height) whose shape every screen is drawn in; None = natural shape."""
        self.window = window
        self.character = character or {}
        self.feedback = feedback or Feedback()
        self.screen = screen
        self.fullscreen = fullscreen
        self._opened = False
        self._cache = {}
        self.hotspots = []          # [((x0, y0, x1, y1), action)] of the last picture
        self._click = None
        self._seen_open = False     # the window was seen open (closed())

    def canvas_size(self, frame_shape):
        """
        (width, height) to draw in: the camera image at DESIGN_HEIGHT plus the
        panel, grown in one direction to the screen's shape (the camera image
        gets dark space around it instead of being cut). The window then scales
        it to the screen, so it always fits.
        """
        fh, fw = frame_shape[:2]
        h = config.DESIGN_HEIGHT
        w = int(round(h * fw / fh)) + PANEL_W
        if not self.screen:
            return w, h
        aspect = self.screen[0] / self.screen[1]
        if aspect >= w / h:
            return int(round(h * aspect)), h
        return w, int(round(w / aspect))

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
        if v.get("stop_hint"):
            v["stop_hint"] = self.feedback.first("StopHint") or "S: Stop"
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

    def draw_body(self, frame, f, side=None):
        """
        Trunk and arms; the trained arm thick and labelled ("LEFT ARM"), so
        she can see which arm the coach watches. Points the camera does not
        really see (guessed by the model: the far arm side-on, hips under
        the table) are not drawn, so the picture never shows an arm that
        is not being tracked. Hands seen are drawn as well.
        """
        if f is None or not f.present or f.points is None:
            return
        pts = f.points.astype(int)

        def p(name):
            return tuple(pts[body.POSE[name]])

        def line(a, b, color, width):
            if f.seen(a) and f.seen(b):
                cv2.line(frame, p(a), p(b), color, width, cv2.LINE_AA)

        for a, b in body.TRUNK_LINES:
            line(a, b, WHITE, 3)
        for s in body.SIDES:
            active = s == side
            for a, b in body.ARM_LINES:
                line(f"{s}_{a}", f"{s}_{b}", CYAN if active else GREY, 8 if active else 3)
            for j in ("shoulder", "elbow", "wrist"):
                if f.seen(f"{s}_{j}"):
                    cv2.circle(frame, p(f"{s}_{j}"), 7 if active else 4, BLUE, -1, cv2.LINE_AA)
        if side in body.SIDES and f.seen(f"{side}_shoulder"):
            x, y = p(f"{side}_shoulder")
            label = f"{side.upper()} ARM"
            scale = max(0.6, frame.shape[0] / 900)
            (tw, th), _ = cv2.getTextSize(label, FONT, scale, 2)
            org = (int(min(max(x - tw // 2, 5), frame.shape[1] - tw - 5)), int(max(y - 40, th + 5)))
            cv2.putText(frame, label, org, FONT, scale, BLACK, 6, cv2.LINE_AA)
            cv2.putText(frame, label, org, FONT, scale, CYAN, 2, cv2.LINE_AA)
        for hand in f.hands.values():
            self.draw_hand(frame, hand)

    def draw_box(self, frame, ok):
        """The box she should sit in: the whole arm inside it (plan 3.1)."""
        h, w = frame.shape[:2]
        m = config.SETUP_EDGE_MARGIN
        cv2.rectangle(frame, (int(m * w), int(m * h)), (int((1 - m) * w), int((1 - m) * h)),
                      GREEN if ok else YELLOW, 4, cv2.LINE_AA)

    def draw_overlay(self, frame, f, ex):
        """Pictures tied to her hand, in the camera's pixels (the bubble between thumb and finger)."""
        bubble = ex.get("bubble")
        if not bubble or f is None or not f.present or f.image_points is None:
            return
        tip_t, tip_i = f.image_points[4], f.image_points[8]
        centre = tuple(int(v) for v in (tip_t + tip_i) / 2)
        r0 = max(18.0, 0.45 * f.palm_size_px)
        now, popped = bubble.get("now"), bubble.get("popped_t")
        if popped is not None and now is not None and 0 <= now - popped < 0.6:
            # the pop: a ring that grows and a few sparkles
            u = (now - popped) / 0.6
            cv2.circle(frame, centre, int(r0 * (0.6 + u)), (250, 230, 200), 2, cv2.LINE_AA)
            for k in range(8):
                a = k * np.pi / 4
                p0 = (int(centre[0] + np.cos(a) * r0 * (0.8 + u)), int(centre[1] + np.sin(a) * r0 * (0.8 + u)))
                p1 = (int(centre[0] + np.cos(a) * r0 * (1.1 + u)), int(centre[1] + np.sin(a) * r0 * (1.1 + u)))
                cv2.line(frame, p0, p1, (250, 230, 200), 3, cv2.LINE_AA)
            return
        if not bubble.get("pinching"):
            return
        r = int(r0 * (1.0 - 0.6 * float(bubble.get("hold_progress", 0.0))))
        layer = frame.copy()
        cv2.circle(layer, centre, r, (245, 225, 190), -1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.35, frame, 0.65, 0, frame)
        cv2.circle(frame, centre, r, (250, 235, 205), 3, cv2.LINE_AA)
        cv2.circle(frame, (centre[0] - r // 3, centre[1] - r // 3), max(3, r // 6), WHITE, -1,
                   cv2.LINE_AA)

    def _cards(self, frame, ex, rect):
        """The memory game's cards over the camera image; rect = where the image lies."""
        x0, y0, iw, ih = rect
        under = frame.copy()                # the cards let her hand show through a little
        for card in ex.get("cards", []):
            cx0, cy0, cx1, cy1 = card["rect"]
            p0 = (int(x0 + cx0 * iw), int(y0 + cy0 * ih))
            p1 = (int(x0 + cx1 * iw), int(y0 + cy1 * ih))
            centre = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
            size = int(min(p1[0] - p0[0], p1[1] - p0[1]) * 0.6)
            if card["state"] == "down":
                cv2.rectangle(frame, p0, p1, CARD_BACK, -1, cv2.LINE_AA)
                cv2.rectangle(frame, p0, p1, WHITE, 4, cv2.LINE_AA)
                _text(frame, card["key"], (p0[0] + 14, p0[1] + 40), 1.0, WHITE, 2)
                if card.get("dwell"):
                    r = size // 3
                    cv2.circle(frame, centre, r, WHITE, 4, cv2.LINE_AA)
                    cv2.ellipse(frame, centre, (r, r), -90, 0, 360 * card["dwell"], YELLOW, 10,
                                cv2.LINE_AA)
            else:
                cv2.rectangle(frame, p0, p1, WHITE, -1, cv2.LINE_AA)
                border = GREEN if card["state"] == "matched" else GREY
                cv2.rectangle(frame, p0, p1, border, 8 if card["state"] == "matched" else 3,
                              cv2.LINE_AA)
                ui.draw_icon(frame, card["icon"], centre, size, ui.INK)
        cv2.addWeighted(frame, 0.82, under, 0.18, 0, frame)
        if ex.get("pointer") is not None:
            px, py = ex["pointer"]
            cv2.circle(frame, (int(x0 + px * iw), int(y0 + py * ih)), 16, POINTER, 4, cv2.LINE_AA)

    def _still_demo(self, frame, view, ex):
        """What to do now as a small picture, at the right of the camera image."""
        key, name = ex.get("demo_key"), view.get("exercise")
        if not key or not name:
            return
        h, w = frame.shape[:2]
        size = 170 if name not in demo.TWO_HANDS else 150
        width = size if name not in demo.TWO_HANDS else 2 * size
        box = (w - width - 15, 430, w - 15, min(h - 140, 430 + size))
        if box[3] - box[1] < 100:
            return
        points = demo.phase_points(name, key, (box[0] + 10, box[1] + 10, box[2] - 10, box[3] - 10),
                                   view.get("hand", "Left"))
        if points:
            ui.draw_demo_hand(frame, points, box=box)

    # --- panel widgets ------------------------------------------------------

    def _angle(self, panel, d, top, height=360):
        """Degrees: the value, today's target with its tolerance band, milestones, her best."""
        x0, x1 = 60, 170
        y_top, y_bot = top, top + height
        lo, hi = d.get("lo", 0.0), d.get("hi", 180.0)
        degrees = d.get("unit", "deg") == "deg"

        def y_of(v):
            u = (np.clip(v, lo, hi) - lo) / max(hi - lo, 1e-6)
            return int(y_bot - u * (y_bot - y_top))

        cv2.rectangle(panel, (x0, y_top), (x1, y_bot), GREY, 2)
        value, target, tol = d.get("value"), d.get("target"), d.get("tolerance", 0.0)
        sgn = 1.0 if d.get("direction", "increase") == "increase" else -1.0
        reached = value is not None and target is not None and sgn * (value - target) >= -tol
        if target is not None:
            band = panel[y_of(target + tol):y_of(target - tol) + 1, x0 + 3:x1 - 2]
            band[:] = (band * 0.5 + np.array(YELLOW) * 0.25).astype(np.uint8)
        if value is not None:
            cv2.rectangle(panel, (x0 + 3, y_of(value)), (x1 - 3, y_bot - 3),
                          GREEN if reached else BLUE, -1)
        last_label_y = None
        for m in d.get("milestones", []):
            y = y_of(m["deg"])
            cv2.line(panel, (x1, y), (x1 + 18, y), WHITE, 2)
            # close milestones (86 and 90) share one label so the numbers stay readable
            if degrees and (last_label_y is None or abs(y - last_label_y) >= 18):
                _text(panel, f"{m['deg']:.0f}", (x1 + 22, y + 7), 0.5, WHITE, 1)
                last_label_y = y
        if d.get("ceiling") is not None:
            y = y_of(d["ceiling"])
            cv2.line(panel, (x0 - 10, y), (x1 + 10, y), GREY, 1)
        if target is not None:
            y = y_of(target)
            cv2.line(panel, (x0 - 25, y), (x1 + 25, y), YELLOW, 5)
        if d.get("best") is not None:
            y = y_of(d["best"])
            cv2.line(panel, (x0 - 30, y), (x0 - 8, y), WHITE, 2)
            _text(panel, "best", (x0 - 55, y - 8), 0.5, WHITE, 1)
        cx, cy, r = 300, top + 90, 60
        cv2.circle(panel, (cx, cy), r, GREY, 6)
        if d.get("holding"):
            cv2.ellipse(panel, (cx, cy), (r, r), -90, 0, 360 * d.get("hold_progress", 0.0), GREEN, 10)
        _text(panel, d.get("phase_label", ""), (cx - 60, cy + r + 45), 1.1, YELLOW, 2)
        if value is not None:
            shown = f"{value:.0f}°" if degrees else f"{int(round(value * 100))}%"
            _text(panel, shown, (x0 + 5, y_bot + 45), 1.1, WHITE, 2)
        label = d.get("label", "")
        if d.get("side"):
            label = f"{label}, {d['side']}"
        _text(panel, label, (x0 - 40, y_bot + 85), 0.7, GREY, 1)
        return y_bot + 100

    def _count(self, panel, d, top):
        """A count of touches: "3 of 5" (finger to nose)."""
        _text(panel, d.get("label", ""), (40, top + 40), 1.0, WHITE, 2)
        _text(panel, f"{d.get('count', 0)} of {d.get('of', 0)}", (40, top + 140), 2.2, YELLOW, 4)
        _text(panel, d.get("phase_label", ""), (40, top + 200), 1.0, CYAN, 2)
        return top + 230

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

    def _sequence(self, panel, d, top, bottom=None):
        seq, step = d.get("sequence", []), d.get("step", 0)
        y = top
        row = 60
        if bottom is not None and seq:
            row = int(np.clip((bottom - top - 50) / len(seq), 36, 60))
        for i, finger in enumerate(seq):
            done = i < step
            current = i == step
            label = "?" if d.get("hidden") and not done else FINGER_WORDS[finger].replace(" finger", "")
            filled = done or (current and not d.get("hidden"))
            color = GREEN if done else (CYAN if filled else GREY)
            cv2.rectangle(panel, (40, y), (PANEL_W - 40, y + row - 10), color, -1 if filled else 2)
            cv2.putText(panel, label.upper(), (60, y + int(row * 0.62)), FONT, min(1.0, row / 60),
                        BLACK if filled else WHITE, 2, cv2.LINE_AA)
            y += row
        if "level" in d:
            _text(panel, f"Level {d['level']}", (40, y + 35), 0.9, WHITE, 2)
            y += 50
        return y

    def _ring(self, panel, progress, center, radius=70, label=""):
        cv2.circle(panel, center, radius, GREY, 8)
        if progress > 0:
            cv2.ellipse(panel, center, (radius, radius), -90, 0, 360 * progress, GREEN, 12)
        if label:
            size = cv2.getTextSize(label, FONT, 1.4, 3)[0]
            _text(panel, label, (center[0] - size[0] // 2, center[1] + size[1] // 2), 1.4, WHITE, 3)

    def _menu(self, frame, items, rect, hand=None):
        """
        The menu, large, over the camera image: each item where Think looks for
        her fingertip (rect: where the image lies). An item fills up while she
        points at it or holds up its number; the fingertip gets a ring.
        """
        x0, y0, iw, ih = rect
        boxes = [(int(x0 + a * iw), int(y0 + b * ih), int(x0 + c * iw), int(y0 + d * ih))
                 for a, b, c, d in (item["rect"] for item in items)]
        if not boxes:
            return
        _band(frame, min(b[1] for b in boxes) - 10, max(b[3] for b in boxes) + 10, 0.7)
        row = min(b[3] - b[1] for b in boxes)
        size = min(row / 55, 1.3)
        key_w = int(48 * size)
        # one text size for the whole menu: the longest item decides
        scale = min((_fit_scale(item["text"], (b[2] - b[0]) - key_w - 45, size)
                     for item, b in zip(items, boxes)), default=size)

        def baseline(top, bottom, s):
            """The baseline that centres text of scale s between top and bottom."""
            return (top + bottom) // 2 + int(0.36 * s * ui.FONT_PX_PER_SCALE)
        for item, (bx0, by0, bx1, by1) in zip(items, boxes):
            progress = float(item.get("progress") or 0.0)
            if item["selected"]:
                cv2.rectangle(frame, (bx0, by0), (bx1, by1), CYAN, -1)
            else:
                cv2.rectangle(frame, (bx0, by0), (bx1, by1), (90, 90, 90), 2, cv2.LINE_AA)
            if progress > 0:
                # fills from the left while she points or holds up the number
                cv2.rectangle(frame, (bx0, by0), (bx0 + int((bx1 - bx0) * progress), by1),
                              GREEN, -1)
                cv2.rectangle(frame, (bx0, by0), (bx1, by1), YELLOW, 4, cv2.LINE_AA)
            dark = item["selected"] or progress > 0
            _text(frame, item["key"], (bx0 + 18, baseline(by0, by1, size)), size,
                  BLACK if dark else YELLOW, 3)
            _text(frame, item["text"], (bx0 + 25 + key_w, baseline(by0, by1, scale)), scale,
                  BLACK if dark else WHITE, 2)
            if item.get("note"):
                note = scale * 0.6
                width = cv2.getTextSize(item["note"], FONT, note, 1)[0][0]
                _text(frame, item["note"], (bx1 - 15 - width, by0 + int(0.3 * (by1 - by0))), note,
                      BLACK if dark else GREY, 1)
        hand = hand or {}
        if hand.get("pointer") is not None:
            px, py = hand["pointer"]
            cv2.circle(frame, (int(x0 + px * iw), int(y0 + py * ih)), 18,
                       POINTER if hand.get("armed") else GREY, 4, cv2.LINE_AA)

    # --- whole screen -------------------------------------------------------

    def render(self, frame, view, features=None):
        """The whole screen for this frame, in the shape of canvas_size()."""
        cw, ch = self.canvas_size(frame.shape)
        view = self._words(view)
        ex = view.get("exercise_display") or {}
        if isinstance(features, BodyFeatures):                      # in the camera's pixels
            setup = view.get("setup") or {}
            self.draw_box(frame, bool(features.present) and not setup.get("problem")
                          and not view.get("quality"))
            self.draw_body(frame, features, ex.get("side") or setup.get("side"))
        else:
            self.draw_hand(frame, getattr(features, "other", None))     # the other hand, when seen
            self.draw_hand(frame, features, ex.get("finger_colors"))
            self.draw_overlay(frame, features, ex)
        text = ui.TextLayer() if ui.pillow_available() else None
        screen = view.get("screen", "exercise")
        if screen in SCREENS:
            if screen != "garden":
                view["camera"] = frame
            layer = text or ui.TextLayer()
            canvas = SCREENS[screen]((cw, ch), view, layer, self.character)
            return self._controls(layer.apply(canvas), view, card=True)

        camera = fit(frame, cw - PANEL_W, ch)
        view["camera_rect"] = fit_rect(frame.shape, cw - PANEL_W, ch)
        panel = np.full((ch, PANEL_W, 3), DARK, dtype=np.uint8)
        _LAYER["text"] = text
        _LAYER["offsets"] = {id(camera): 0, id(panel): cw - PANEL_W}
        try:
            canvas = self._render_camera(camera, panel, view)
        finally:
            _LAYER["text"] = None
            _LAYER["offsets"] = {}
        return self._controls(text.apply(canvas) if text is not None else canvas, view)

    # --- buttons that can be clicked (toolbar, skip) ----------------------------

    def _controls(self, canvas, view, card=False):
        """
        The toolbar (menu only) and the Skip button, on top of the finished
        picture. Kept at the right, like the Stop hint, never at the left
        edge. Remembers where each button is for clicks.
        """
        self.hotspots = []
        h, w = canvas.shape[:2]
        layer = ui.TextLayer()
        toolbar = view.get("toolbar")
        if toolbar is not None:
            icon = (w - 72, 12, w - 14, 70)
            ui.draw_toolbar_icon(canvas, icon, toolbar.get("open"))
            self.hotspots.append((icon, "toolbar"))
            if toolbar.get("open"):
                items = toolbar.get("items", [])
                bw, gap, y0, y1 = 300, 14, 10, 96
                # an opaque strip (the text under it would show through)
                cv2.rectangle(canvas, (0, 0), (icon[0] - 1, y1 + 10), DARK, -1)
                x1 = icon[0] - gap
                for item in reversed(items):
                    rect = (x1 - bw, y0, x1, y1)
                    if "on" in item:
                        label = f"{item['key']}  {item['label']}"
                        note = "On" if item["on"] else "Off"
                        fill = GREEN if item["on"] else (90, 90, 90)
                    else:
                        label, note = f"{item['key']}  {item['label']}", item.get("status") or ""
                        fill = BLUE
                    ui.draw_button(canvas, layer, rect, label, fill, WHITE, 26, note or None)
                    self.hotspots.append((rect, item["id"]))
                    x1 -= bw + gap
        skip = view.get("skip")
        if skip:
            label = f"{skip['key']}  {skip['label']}"
            if card:
                rect = (w - 300, 14, w - 20, 72)
            else:
                # the panel, above the Stop hint (_render_camera leaves room)
                rect = (w - PANEL_W + 20, h - SKIP_BOTTOM - 50, w - 20, h - SKIP_BOTTOM)
            ui.draw_button(canvas, layer, rect, label, ORANGE, WHITE, 26)
            self.hotspots.append((rect, "skip"))
        return layer.apply(canvas)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self._click = (x, y)

    def pop_click(self):
        """The action of the button clicked since the last call, or None."""
        click, self._click = self._click, None
        if click is None:
            return None
        x, y = click
        for (x0, y0, x1, y1), action in self.hotspots:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return action
        return None

    def _render_camera(self, frame, panel, view):
        """frame: the camera column (the camera image with dark space around it)."""
        h, w = frame.shape[:2]
        ex = view.get("exercise_display") or {}

        if view.get("menu"):
            self._menu(frame, view["menu"], view.get("camera_rect") or (0, 0, w, h),
                       view.get("menu_hand"))

        # title and counters
        _wrapped(panel, view.get("title", ""), 20, 45, PANEL_W - 40, 1.0, WHITE, 2)
        if view.get("status"):
            _text(panel, view["status"], (20, 120), 0.9, YELLOW, 2)
        top = 150
        if view.get("step_label"):
            _text(panel, view["step_label"], (20, 155), 0.75, GREY, 1)
            top = 175
        # the stop hint sits at the bottom of the panel, above the key help,
        # and the Skip button (drawn later, _controls) above the stop hint
        stop_top = h - 100 if view.get("stop_hint") else h - 45
        if view.get("skip"):
            stop_top = h - SKIP_BOTTOM - 60

        stage = view.get("stage")
        if stage == "exercise" and ex.get("kind") == "bar":
            detail = ex.get("finger_values") or ex.get("gap_values") or ex.get("pair_values")
            room = stop_top - top - 70 - 35 - (140 if detail else 0)
            top = self._bar(panel, ex, top + 10, height=int(np.clip(room, 150, 400)))
            if ex.get("pair_values"):
                colors = {k: PAIR_COLORS.get(k, GREY) for k in ex["pair_values"]}
                labels = {k: k for k in ex["pair_values"]}
                self._finger_bars(panel, ex["pair_values"], top, colors, labels)
                if ex.get("symmetry") is not None:
                    _text(panel, f"Together: {int(round(ex['symmetry'] * 100))}%", (210, top + 60),
                          0.8, WHITE, 2)
            elif ex.get("finger_values"):
                colors = {k: (ORANGE if ex["finger_colors"].get(k) == "lagging" else BLUE)
                          for k in ex["finger_values"]}
                labels = {k: FINGER_WORDS[k].replace(" finger", "") for k in ex["finger_values"]}
                top = self._finger_bars(panel, ex["finger_values"], top, colors, labels)
            elif ex.get("gap_values"):
                labels = {g: g.replace("_", "-").replace("pinky", "little") for g in GAP_NAMES}
                top = self._finger_bars(panel, ex["gap_values"], top, GAP_COLORS, labels)
        elif stage in ("exercise", "calibrating") and ex.get("kind") == "angle":
            room = h - top - 70 - 80 - 60
            top = self._angle(panel, ex, top + 10, height=int(np.clip(room, 150, 400)))
        elif stage == "exercise" and ex.get("kind") == "count":
            top = self._count(panel, ex, top + 10)
        elif stage == "exercise" and ex.get("kind") == "sequence":
            top = self._sequence(panel, ex, top + 10, bottom=stop_top - 10)
        elif stage == "exercise" and ex.get("kind") == "cards":
            _text(panel, f"Level {ex.get('level', 1)}", (40, top + 50), 1.0, WHITE, 2)
            _text(panel, f"Pairs found: {ex.get('found', 0)} of {ex.get('pairs', 0)}",
                  (40, top + 110), 0.9, YELLOW, 2)
            _wrapped(panel, "Point and hold still, or press the number on the card.", 40, top + 170,
                     PANEL_W - 80, 0.7, GREY, 1)
            self._cards(frame, ex, view.get("camera_rect") or (0, 0, w, h))
        elif stage in ("calibrating", "setup_check"):
            self._ring(panel, view.get("progress", 0.0), (PANEL_W // 2, top + 130))
        elif stage == "rest":
            self._ring(panel, view.get("progress", 0.0), (PANEL_W // 2, top + 130),
                       label=str(view.get("countdown", "")))
        elif view.get("menu"):
            _wrapped(panel, "Point at your choice and hold still, or hold up its number of "
                     "fingers. Both hands count.", 20, top + 20, PANEL_W - 40, 0.7, GREY, 1)
            fingers = (view.get("menu_hand") or {}).get("fingers")
            if fingers:
                # the number she is holding up fills its ring
                self._ring(panel, view["menu_hand"].get("progress", 0.0),
                           (PANEL_W // 2, top + 220), label=str(fingers))
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
        if can and can.get("sections") and not view.get("menu"):
            ui.draw_can(frame, (w - 80, 375), 80, can["sections"], can.get("filled", 0))
        if stage == "exercise" and ex.get("kind") != "cards":
            self._still_demo(frame, view, ex)

        _text(panel, view.get("footer", ""), (20, h - 25), _fit_scale(view.get("footer", ""),
                                                                      PANEL_W - 60, 0.7, 1), GREY, 1)
        if view.get("stop_hint") and _LAYER["text"] is not None:
            # drawn in canvas coordinates: the panel starts where the camera column ends
            ui.draw_stop_hint(panel, _PanelText(), view["stop_hint"], PANEL_W - 15, h - 48, size=22)

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
            _text(frame, view["subtitle"], (30, 48), _fit_scale(view["subtitle"], w - 60, 0.9),
                  WHITE, 2)

        if view.get("paused"):
            if _LAYER["text"] is not None:
                _LAYER["text"].drop(0, w)       # covered by the pause screen
            _band(frame, 0, h)
            _text(frame, "Paused", (w // 2 - 110, h // 2 - 20), 2.2, WHITE, 4)
            _text(frame, "Press space to continue", (w // 2 - 250, h // 2 + 50), 1.2, YELLOW, 2)

        return np.hstack([frame, panel])

    def _open(self):
        """A window that scales its picture to fit (never cut off), full screen by default."""
        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
        cv2.namedWindow(self.window, flags)
        # clicks arrive in picture coordinates, whatever the window's size
        cv2.setMouseCallback(self.window, self._on_mouse)
        if self.screen:
            # the size it has when not full screen: most of the screen
            k = min(0.9 * self.screen[0] / self._size[0], 0.85 * self.screen[1] / self._size[1])
            cv2.resizeWindow(self.window, int(self._size[0] * k), int(self._size[1] * k))
        self._opened = True
        self.set_fullscreen(self.fullscreen)

    def set_fullscreen(self, on):
        self.fullscreen = on
        if self._opened:
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)

    def toggle_fullscreen(self):
        self.set_fullscreen(not self.fullscreen)

    def show(self, canvas):
        if not self._opened:
            self._size = (canvas.shape[1], canvas.shape[0])
            self._open()
        cv2.imshow(self.window, canvas)

    def closed(self):
        """
        True once the window was closed with its close button (the app then
        quits as with q). Only after it has been seen open: a backend that
        cannot tell never closes the app by mistake.
        """
        if not self._opened:
            return False
        try:
            visible = cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return False
        if visible >= 1:
            self._seen_open = True
        return self._seen_open and visible < 1

    def close(self):
        cv2.destroyAllWindows()
