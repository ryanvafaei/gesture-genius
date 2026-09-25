"""
Think: the Coach runs one exercise, the SessionManager runs the whole session.

Think decides what happened; Act decides how to say it. Motivation (bests,
praise, targets, greetings, the garden) leaves Think as events
(rehab/events.py) that the speaker words with the phrase bank.

Coach
  * checks tracking quality first (hand visible, correct hand, close enough,
    palm facing the camera when needed) and asks her to adjust, calmly and
    not too often;
  * feeds good frames to the active exercise and passes its messages to the
    speaker;
  * logs every repetition, and after it the events for that rep (a best, a
    steady hold, recovering after a hint ...), spoken after the rep count
    and before the next prompt.

SessionManager
  (first time: choose the coach's name and her favourite activities)
  greeting (remembers one fact) -> check-in (thumbs up / down) ->
  menu (1 today's routine, 2 hand, 3 arm and 4 memory exercises, each a menu
  of its own, 5 end for today, P her profile) or, with exercises given up
  front, today's plan -> for each exercise: activity card -> calibration
  check -> sets with rests (the target is adapted after every set) ->
  summary (one highlight, compared with her own history) -> ... ->
  garden (grows from showing up) -> goodbye.
  Arm exercises (menu "Arm exercises", body tracking) measure angles
  in degrees against published benchmarks (rehab/benchmarks.py). Their
  calibration measures the right arm, then the left, and is repeated every
  week as an assessment; before the sets a setup check makes sure the camera
  sees the arm from the right direction. Their target is a level on a
  ladder from her own baseline, changed once per session (rehab/progress.py).
  A difficult day (thumbs down, a low first set, tiredness) makes the rest
  of the session easier: lower targets, one set fewer, longer rests, no
  strength exercise, praise for effort, no comparisons.

Yes / no answers: thumbs up / thumbs down (either hand), with space or "y"
and "n" as a backup. Space starts, pauses and continues; m goes back to the
menu. In a menu an item is chosen with its number key (up/down and space
move and choose), by pointing at it with the index finger and holding
still, or by holding up its number of fingers, both hands adding up
(MenuPicker). The hand has to come down between two choices.

Profile (menu item "My profile", or p in the menu): what the coach remembers
about her. d asks whether to delete it; only the y key confirms (a thumbs up
there could be an accident). After deleting, the session ends with
`restart` set, and main starts again as on her first day.

Always available
  s  Stop / "I don't feel well": everything stops, the safety screen shows
     the stroke warning signs and the emergency number; thumbs up (or
     space) when she feels fine goes back to the menu. Logged for the
     therapist. The app never decides it is an emergency and never calls.
  r  repeat: says what belongs to this screen again.
  k  skip (or the Skip button): during an exercise (from its card to the
     last set) go on to the next exercise, keeping the reps done; during a
     rest, end the rest.

Toolbar (menu only, hidden until i or the icon at the top right is
pressed): short sessions on / off (t), new guest (g: the session ends and
a new one starts for a new guest with a unique id and its own data
folder, rehab/guests.py), back to her own profile (b, while a guest is
active), and make the report with its charts (o, tools/report.py; a
guest's own report while a guest is active).

Verbose logging (main --verbose): `trace` (a callable(kind, **data)) gets
every stage change, key, click, calibration, detection, rep, rating and
skip, and debug_state() gives the inner state of each frame (rehab/verbose.py).

Before goodbye she rates the session from 1 to 5 (how hard, how enjoyable,
and for guests how easy to use), with keys 1-5 or by holding up 1-5 fingers.
"""

import collections
from datetime import datetime

from rehab import benchmarks, config, memory
from rehab import garden as garden_model
from rehab.features import count_extended, finger_raised
from rehab.calibration import SetupCheck, is_stale
from rehab.events import event
from rehab.exercises import ARM_EXERCISES, EXERCISES, create, is_arm
from rehab.exercises.base import Say
from rehab.progress import SessionProgress, finite
from rehab.storage import (calibrations, improvement_claimed, load_content, new_garden,
                           progress_message, should_do_today)

QUALITY_TEXT = {
    "no_hand": "Please show your {hand} hand to the camera.",
    "wrong_hand": "Please use your {hand} hand.",
    "too_far": "Please move your hand a little closer.",
    "palm_away": "Please turn your palm towards the camera.",
    "not_flat": "Please rest your {hand} hand flat on the table.",
    "no_other_hand": "Please show both hands to the camera.",
    # arm exercises: the camera's fault, never hers (plan 10.5)
    "no_body": "I can't see you. Please sit where the camera can see you.",
    "arm_hidden": "I can't see your {hand} arm. Please move into the box.",
    "hand_hidden": "I can't see your {hand} hand. Please move it into the box.",
}

INTRO_S_PER_SENTENCE = 2.5
SUMMARY_S = 6.0

# the main menu
ALL = "all"             # 1. continue to today's routine
HAND = "hand"           # 2. opens the hand exercise menu
ARM = "arm"             # 3. opens the arm exercise menu
MEMORY = "memory"       # 4. opens the memory exercise menu
FINISH = "finish"       # 5. end for today
PROFILE = "profile"     # P (or six fingers)
MAIN_MENU = (ALL, HAND, ARM, MEMORY, FINISH, PROFILE)
BACK = "back"           # the last item of each group's menu
# each group's menu and its stage
GROUP_STAGES = {HAND: "hand_menu", ARM: "arm_menu", MEMORY: "memory_menu"}
MENU_STAGES = ("menu",) + tuple(GROUP_STAGES.values())
MENU_TEXT = {ALL: "Continue to your daily routine", HAND: "Hand exercises",
             ARM: "Arm exercises", MEMORY: "Memory exercises", FINISH: "End for today",
             PROFILE: "My profile", BACK: "Back"}

QUESTION_STAGES = ("setup_name", "setup_activities", "check_in", "plant_choice")
CARD_STAGES = QUESTION_STAGES + ("greeting", "today_plan", "intro", "goodbye")
# stages where a thumbs up means "carry on"
CONTINUE_STAGES = ("greeting", "today_plan", "intro", "rest", "summary", "garden", "goodbye")
# her profile and the question whether to delete it
PROFILE_STAGES = ("profile", "profile_delete")
# stages where an arm exercise needs body tracking
BODY_STAGES = ("setup_check", "calibration_offer", "calibrating", "exercise")
# stages where thumbs up / down are listened to
YES_NO_STAGES = QUESTION_STAGES + CONTINUE_STAGES + PROFILE_STAGES + ("safety",)
# stages where the exercise number ("Exercise 2 of 7") is shown
STEP_STAGES = ("intro", "calibration_offer", "calibrating", "setup_check", "exercise", "rest")
# stages of one exercise that Skip (k) leaves for the next exercise
SKIP_STAGES = ("intro", "calibration_offer", "calibrating", "setup_check", "exercise")
# toolbar keys (while it is open in the menu)
TOOLBAR_KEYS = {"t": "short", "g": "new_guest", "b": "main_profile", "o": "report"}


class YesNo:
    """
    Thumbs up / thumbs down, held for GESTURE_HOLD_S -> "yes" / "no".
    Either hand counts. After an answer (or a new question) the hand has to
    come down first, so one gesture never answers two questions.
    """

    def __init__(self, hold_s=config.GESTURE_HOLD_S, min_score=config.GESTURE_MIN_SCORE):
        self.hold_s = hold_s
        self.min_score = min_score
        self._seen = None
        self.reset()

    def reset(self):
        self._armed = False
        self._label = None
        self._since = None

    def update(self, gestures, now):
        labels = [g for g, score in gestures or []
                  if g in (config.YES_GESTURE, config.NO_GESTURE) and (score or 0) >= self.min_score]
        label = labels[0] if labels else None
        self._seen = label
        if label is None:
            self._armed = True
            self._label = None
            return None
        if not self._armed:
            return None
        if label != self._label:
            self._label, self._since = label, now
            return None
        if now - self._since >= self.hold_s:
            self.reset()
            return "yes" if label == config.YES_GESTURE else "no"
        return None

    def state(self, now):
        """
        For the camera preview: (answer, progress 0..1). answer is "yes" / "no"
        while a thumb is being held, "lower" when the hand has to come down
        before the next answer, else None.
        """
        if self._seen is None:
            return None, 0.0
        if not self._armed:
            return "lower", 0.0
        if self._label is None:
            return None, 0.0
        answer = "yes" if self._label == config.YES_GESTURE else "no"
        return answer, min(1.0, max(0.0, (now - self._since) / self.hold_s))


class FingerCount:
    """
    A number from 1 to 5 shown with raised fingers (either hand), held
    steady for hold_s -> that number. Like YesNo, the hand has to come down
    (no fingers raised) before the next answer counts.

    The count of each frame is voted over the last window_s: one frame in
    which a finger is missed (tracking noise) does not restart the hold.
    """

    def __init__(self, hold_s=config.RATING_HOLD_S, window_s=config.RATING_VOTE_S):
        self.hold_s = hold_s
        self.window_s = window_s
        self._seen = 0
        self.raw = 0
        self.reset()

    def reset(self):
        self._armed = False
        self._value = None
        self._since = None
        self._recent = collections.deque()

    def _vote(self, n, now):
        """The most frequent count of the last window_s (the newest one on a tie)."""
        self._recent.append((now, n))
        while self._recent and now - self._recent[0][0] > self.window_s:
            self._recent.popleft()
        counts = collections.Counter(v for _, v in self._recent)
        best = max(counts.values())
        return next(v for _, v in reversed(self._recent) if counts[v] == best)

    @staticmethod
    def count(f):
        if f is None:
            return 0
        return max(count_extended(f), count_extended(getattr(f, "other", None)))

    def update(self, f, now):
        self.raw = self.count(f)
        n = self._vote(self.raw, now)
        self._seen = n
        if n == 0:
            self._armed = True
            self._value = None
            return None
        if not self._armed:
            return None
        if n != self._value:
            self._value, self._since = n, now
            return None
        if now - self._since >= self.hold_s:
            self.reset()
            return n
        return None

    def state(self, now):
        """For the rating screen: (fingers seen, progress 0..1 of the hold, armed)."""
        if not self._armed or self._value is None:
            return self._seen, 0.0, self._armed
        return self._value, min(1.0, max(0.0, (now - self._since) / self.hold_s)), True


def menu_layout(n, area=config.MENU_AREA, gap=0.02):
    """
    Rectangles (x0, y0, x1, y1) of n menu items in camera image fractions:
    one column of up to six, else two columns (1-5 left, 6-10 right).
    """
    cols = 1 if n <= 6 else 2
    rows = max(1, -(-n // cols))
    x0, y0, x1, y1 = area
    w = (x1 - x0 - gap * (cols - 1)) / cols
    h = (y1 - y0) / rows
    rects = []
    for i in range(n):
        c, r = divmod(i, rows)
        left = x0 + c * (w + gap)
        rects.append((left, y0 + r * h + gap / 2, left + w, y0 + (r + 1) * h - gap / 2))
    return rects


class MenuPicker:
    """
    Choosing a menu item without a keyboard, with either hand:

      * point at it: only the index finger raised, its tip on the item for
        dwell_s (the item may be left for a frame or two of tracking noise);
      * show its number: that many fingers raised, held steady for hold_s.
        Both hands add up, so 6 to 10 are shown with two hands. The count of
        each frame is voted over window_s, as in FingerCount.

    One finger held up over an item is taken as pointing at that item; held
    up anywhere else it is the number 1. Like FingerCount, the hand has to
    come down (no finger raised) before anything counts: after entering a
    menu and after each choice, so pointing at item 2 never also chooses
    item 2 of the next menu.
    """

    def __init__(self, dwell_s=config.MENU_DWELL_S, hold_s=config.MENU_HOLD_S,
                 window_s=config.RATING_VOTE_S):
        self.dwell_s = dwell_s
        self.hold_s = hold_s
        self.window_s = window_s
        self.reset()

    def reset(self):
        self._armed = False
        self._dwell = None          # (item, since)
        self._number = None         # (fingers, since)
        self._recent = collections.deque()
        self._now = 0.0
        self.pointer = None         # index fingertip in image fractions, while pointing
        self.raw = 0                # fingers raised in this frame (both hands)
        self.voted = 0

    def _vote(self, n, now):
        self._recent.append((now, n))
        while self._recent and now - self._recent[0][0] > self.window_s:
            self._recent.popleft()
        counts = collections.Counter(v for _, v in self._recent)
        best = max(counts.values())
        return next(v for _, v in reversed(self._recent) if counts[v] == best)

    @staticmethod
    def hands(f):
        return [h for h in (f, getattr(f, "other", None))
                if h is not None and getattr(h, "present", False)]

    @staticmethod
    def _pointer(hands, counts):
        """The index fingertip (image fractions) when only an index finger is raised."""
        if sum(counts) != 1:
            return None
        h = hands[counts.index(1)]
        if not finger_raised(h, "index") or h.image_points is None:
            return None
        w, ht = h.image_size or (1.0, 1.0)
        tip = h.image_points[8]
        return float(tip[0] / w), float(tip[1] / ht)

    @staticmethod
    def item_at(rects, point):
        if point is None:
            return None
        x, y = point
        return next((i for i, (x0, y0, x1, y1) in enumerate(rects)
                     if x0 <= x <= x1 and y0 <= y <= y1), None)

    def _chosen(self, i):
        self._armed = False
        self._dwell = self._number = None
        return i

    def update(self, f, rects, now):
        """One frame; rects: menu_layout() of the menu. Returns the chosen index or None."""
        self._now = now
        hands = self.hands(f)
        counts = [count_extended(h) for h in hands]
        self.raw = sum(counts)
        n = self.voted = self._vote(self.raw, now)
        self.pointer = self._pointer(hands, counts)
        item = self.item_at(rects, self.pointer)
        if n == 0 and self.pointer is None:
            # the hand is down (or away): the next choice may start
            self._armed = True
            self._dwell = self._number = None
            return None
        if not self._armed:
            return None
        if item is not None:
            self._number = None
            if self._dwell is None or self._dwell[0] != item:
                self._dwell = (item, now)
                return None
            return self._chosen(item) if now - self._dwell[1] >= self.dwell_s else None
        if self.pointer is None and n == 1 and self._dwell is not None:
            return None             # the fingertip was missed for a moment
        self._dwell = None
        if not 1 <= n <= len(rects):
            self._number = None
            return None
        if self._number is None or self._number[0] != n:
            self._number = (n, now)
            return None
        return self._chosen(n - 1) if now - self._number[1] >= self.hold_s else None

    def state(self):
        """For the menu screen: where she points, the item and the number being held."""
        def progress(since, hold):
            return min(1.0, max(0.0, (self._now - since) / hold))

        return {
            "armed": self._armed,
            "pointer": self.pointer,
            "fingers": self.voted,
            "dwell": (self._dwell[0], progress(self._dwell[1], self.dwell_s))
            if self._dwell and self._armed else None,
            "number": (self._number[0], progress(self._number[1], self.hold_s))
            if self._number and self._armed else None,
        }


def _no_trace(kind, **data):
    pass


class Coach:

    def __init__(self, exercise, speaker, log, hand=config.AFFECTED_HAND, progress=None,
                 trace=None):
        self.exercise = exercise
        self.trace = trace or _no_trace
        self.speaker = speaker
        self.log = log
        self.progress = progress
        self.hand = hand.lower()
        self.quality = None            # message on screen, or None
        self._problem = None
        self._problem_since = None
        self._interrupted = False
        self._last_quality_say = {}

    def start_set(self, set_no, now):
        self.exercise.start_set(set_no, now)
        self.speaker.say_all(self.exercise.first_messages())

    def interrupt(self, now):
        self.exercise.interrupt(now)

    def resume(self, now):
        """After a pause or rest: cancel holds and say again what to do now."""
        self.exercise.interrupt(now)
        self.speaker.say_all(self.exercise.resume_messages())

    @property
    def side_word(self):
        """The side the exercise watches now ("left"), for the quality messages."""
        ex = self.exercise
        return getattr(ex, "round_side", None) or getattr(ex, "side", None) or self.hand

    def update(self, f, now):
        ex = self.exercise
        ex.speaking = self.speaker.busy
        # hand exercises: hand in view (both, either); arm exercises: the arm's landmarks
        problem = ex.quality_problem(f)
        problem_say = None
        if problem:
            problem_say = Say(QUALITY_TEXT[problem].format(hand=self.side_word), "quality")
        else:
            problem_say = self.exercise.frame_problem(f)
            problem = problem_say.text if problem_say else None

        if problem:
            self.exercise.skip_frame(now)
            if problem != self._problem:
                self.trace("quality", problem=problem)
                self._problem, self._problem_since = problem, now
            grace = getattr(self.exercise, "quality_grace_s", config.QUALITY_GRACE_S)
            if now - self._problem_since >= grace:
                # short drop-outs are ignored; only now cancel holds and speak
                if not self._interrupted:
                    self.exercise.interrupt(now)
                    self._interrupted = True
                self.quality = problem_say.text
                last = self._last_quality_say.get(problem, -1e9)
                if now - last >= config.QUALITY_MESSAGE_REPEAT_S:
                    self._last_quality_say[problem] = now
                    # not worth saying any more once she has fixed it
                    problem_say.valid = lambda p=problem: self._problem == p
                    self.speaker.say(problem_say)
            return

        was_interrupted = self._interrupted
        if self._problem is not None:
            self.trace("quality", problem=None)
        self._problem = None
        self.quality = None
        self._interrupted = False
        if was_interrupted:
            # the hold was cancelled: tell her again what to do
            self.speaker.say_all(self.exercise.resume_messages())
        says = self.exercise.update(f, now)
        reps = list(self.exercise.new_reps)
        self.exercise.new_reps.clear()
        difficult = bool(self.progress and self.progress.difficult)
        for rec in reps:
            self.trace("rep", **vars(rec))
            if self.log is not None:
                self.log.log_rep(rec, difficult_day=difficult)
        if not reps or self.progress is None:
            self.speaker.say_all(says)
            return
        # praise for the rep goes right after "That's three." and before the next prompt
        cut = max((i + 1 for i, m in enumerate(says) if m.tag == "rep_done"), default=len(says))
        self.speaker.say_all(says[:cut])
        for rec in reps:
            self.speaker.say(self.progress.rep_events(self.exercise, rec))
        self.speaker.say_all(says[cut:])


class SessionManager:

    def __init__(self, speaker, profile, log, exercises=None, save_profile=None,
                 garden=None, save_garden=None, character=None, activities=None,
                 delete_profile=None, short=False, rating_questions=config.RATING_QUESTIONS,
                 therapist=None, guest_id=None, make_report=None, trace=None):
        self.speaker = speaker
        self.profile = profile
        self.log = log
        self.save_profile = save_profile or (lambda p: None)
        self.garden = garden if garden is not None else new_garden()
        self.save_garden = save_garden or (lambda g: None)
        self.delete_profile = delete_profile or (lambda: None)
        self.character = character if character is not None else load_content("character")
        self.activities = activities if activities is not None else load_content("activities")
        self.therapist = therapist if therapist is not None else benchmarks.load_therapist_profile()
        # the trained side follows her profile (e.g. a guest started with --hand Right)
        hand = str(profile.get("affected_hand") or "").lower()
        if hand in ("left", "right") and hand != self.therapist.get("affected_side"):
            self.therapist = dict(self.therapist, affected_side=hand,
                                  unaffected_side="right" if hand == "left" else "left")
        self.progress = SessionProgress(profile, log, therapist=self.therapist)
        self.yes_no = YesNo()
        self.finger_count = FingerCount()
        self.menu_picker = MenuPicker()     # menus: pointing or fingers held up
        self.short = short                  # one short set per exercise (guests); toolbar toggle
        self.guest_id = guest_id            # a guest's unique id (rehab/guests.py), None: her own
        self.switch_user = None             # toolbar: "new_guest" / "main_profile"; main starts it
        self.trace = trace or _no_trace     # verbose log (main --verbose)
        self.make_report = make_report      # () -> job with status() -> (state, text), or None
        self._report_job = None
        self.toolbar_open = False
        self.skipped = []                   # exercises left with Skip
        self.rating_questions = tuple(rating_questions or ())
        self.ratings = {}                   # question -> 1..5 ("" when skipped)
        self._rating_index = None           # None: not asked yet
        self._safety = False                # Stop / "I don't feel well" was pressed
        # today's plan, the menu's "all" item
        self.today = [n for n in config.DAILY_PLAN if should_do_today(n, log)]
        # chosen explicitly: no day schedule and no menu
        self.fixed = bool(exercises)
        self.plan = list(exercises) if exercises else []
        self.menus = {
            "menu": list(MAIN_MENU),
            "hand_menu": [n for n in config.HAND_MENU if n in EXERCISES] + [BACK],
            "arm_menu": [n for n in config.ARM_EXERCISES if n in ARM_EXERCISES] + [BACK],
            "memory_menu": [n for n in config.MEMORY_MENU if n in EXERCISES] + [BACK],
        }
        self.menu_index = 0
        self.index = -1
        self.stage = "start"
        self.paused = False
        self.done = False
        self.restart = False            # her profile was deleted: start again from the beginning
        self._profile_rows = []
        self._hand_seen = False
        self.summaries = []
        self.summary_lines = []
        self.exercise = None
        self.coach = None
        self.calibration = None
        self.activity = None
        self.set_no = 0
        self._t = 0.0
        self._stage_t = 0.0
        self._quiet_t = 0.0             # when the speaker last went quiet
        self._rest_s = 0
        self._after_rest = None
        self._fatigue_offered = False
        self._instruction = ""
        self._quality = None
        self._cal_problem = None
        self._cal_problem_since = 0.0
        self._cal_last_say = {}
        self._from_all = self.fixed
        self._target_start = float("nan")
        self._card_event = None
        self._summary_event = None
        self._garden_event = None
        self._name_index = 0
        self._activity_cards = list(self.activities.get("activities", {}))
        self._activity_index = 0
        self._chosen = []
        self._plant_options = None
        self._practised = []
        self._ended = False
        self._started_at = datetime.now()
        self._setup = None
        self.setup_checks = []          # (exercise, passed) for the benchmark session log

    # --- helpers --------------------------------------------------------------

    @property
    def name(self):
        return self.plan[self.index] if 0 <= self.index < len(self.plan) else None

    @property
    def needs_body(self):
        """True while an arm exercise is watched: main then runs body tracking."""
        return bool(self.name) and is_arm(self.name) and self.stage in BODY_STAGES

    @property
    def palm_down(self):
        """True while an exercise with the hand flat, back up, is watched: main then trusts
        the knuckle triangle over the Left/Right label when picking the hand."""
        cls = EXERCISES.get(self.name) if self.name else None
        return (bool(cls) and getattr(cls, "palm_down", False)
                and self.stage in ("calibrating", "exercise"))

    @property
    def side(self):
        """The trained side as a word ("left")."""
        return self.therapist.get("affected_side") or \
            self.profile.get("affected_hand", config.AFFECTED_HAND).lower()

    @property
    def guest(self):
        return self.guest_id is not None

    @property
    def menu_items(self):
        """The items of the menu on screen (the main menu when none is)."""
        return self.menus.get(self.stage, self.menus["menu"])

    def _enter(self, stage, now):
        if stage != "menu":
            self.toolbar_open = False       # only on the menu page
        if stage != self.stage:
            self.trace("stage", stage=stage, exercise=self.name, set=self.set_no)
        self.stage = stage
        self._stage_t = now
        self._quiet_t = now
        self.yes_no.reset()

    def _say(self, *texts):
        for t in texts:
            self.speaker.say(t if isinstance(t, Say) else Say(t))

    def _card(self, ev, stage, now):
        """Show a full-screen card for ev, say it, and wait in stage."""
        self._card_event = ev
        self.speaker.say(ev)
        self._enter(stage, now)

    def _quiet_for(self, now):
        """Seconds since the stage started and the coach stopped talking."""
        return now - max(self._stage_t, self._quiet_t)

    def _build_exercise(self):
        cls = EXERCISES[self.name]
        hand = self.profile.get("affected_hand", config.AFFECTED_HAND)
        kwargs = {}
        if cls.uses_level and self.profile["levels"].get(self.name):
            kwargs["level"] = self.profile["levels"][self.name]
        if cls.need_both_hands:
            kwargs["hand"] = hand
        if is_arm(self.name):
            kwargs["level"] = self.progress.start_level(self.name)
            kwargs["therapist"] = self.therapist
        target = self.progress.start_target(self.name)
        thresholds = {self.name: {"high": target}} if target is not None else None
        if is_arm(self.name):
            sets = benchmarks.exercise_settings(self.therapist, self.name)["sets"]
        else:
            sets = config.EXERCISES.get(self.name, {}).get("sets", 3)
        params = {"sets": self.progress.sets_for(self.name, sets)}
        if self.short:
            short = config.SHORT_SESSION
            params["sets"] = min(params["sets"], short["sets"])
            # range and arm exercises count reps; sequences and games count rounds
            counts_reps = getattr(cls, "range_steps", ()) or is_arm(self.name)
            params["reps"] = short["reps"] if counts_reps else short["rounds"]
        self.exercise = create(self.name, calibrations(self.profile), thresholds,
                               params=params, **kwargs)
        self.exercise.trace = self.trace
        self.trace("exercise", exercise=self.name, params=self.exercise.params,
                   calibration=calibrations(self.profile).get(self.name),
                   detector=self.exercise.debug_settings())
        self._target_start = target if target is not None else float("nan")
        self.activity = memory.activity_for(self.name, self.profile, self.activities)
        self.coach = Coach(self.exercise, self.speaker, self.log, hand, progress=self.progress,
                           trace=self.trace)

    def _gestures(self, f, gestures):
        if gestures is not None:
            return gestures
        if f is not None and getattr(f, "present", False) and f.gesture:
            return [(f.gesture, f.gesture_score)]
        return []

    def _rest_seconds(self, base):
        factor = config.DIFFICULT_REST_FACTOR if self.progress.difficult else 1.0
        return int(round(base * factor))

    # --- speech follows the step she is on ---------------------------------------

    def _step(self):
        """Where she is; when this changes, what was said before is about a step she left."""
        return self.stage, self.index, self.set_no, self.paused

    def _handle(self, fn, *args):
        """
        Run fn (a frame or a key). When it moved her to another step, drop
        and cut off what the coach was still saying for the old one; what fn
        itself said (the new step's words) is kept.
        """
        mark = self.speaker.mark() if hasattr(self.speaker, "mark") else None
        before = self._step()
        try:
            fn(*args)
        finally:
            if mark is not None and self._step() != before:
                self.speaker.drop_before(mark)

    # --- keys -------------------------------------------------------------------

    def on_key(self, key, now):
        """key: " ", "up", "down", "m", "p", "d", "y", "n", "e", "s", "r" or a digit."""
        self.trace("key", key=key, stage=self.stage)
        self._handle(self._on_key, key, now)

    def on_click(self, action, now):
        """A click on a button Act drew: "toolbar", "skip" or a toolbar item (TOOLBAR_KEYS)."""
        self.trace("click", action=action, stage=self.stage)
        self._handle(self._on_click, action, now)

    def _on_click(self, action, now):
        if action == "skip":
            if self.stage in SKIP_STAGES + ("rest",):
                self._skip(now)
        elif self.stage == "menu":
            if action == "toolbar":
                self.toolbar_open = not self.toolbar_open
            elif self.toolbar_open and action in TOOLBAR_KEYS.values():
                self._toolbar(action, now)

    # --- toolbar (menu) -------------------------------------------------------------

    def _toolbar(self, item, now):
        self.trace("toolbar", item=item)
        if item == "short":
            self.short = not self.short
            self.speaker.clear()
            self._say("Short sessions are on." if self.short else "Short sessions are off.")
        elif item == "new_guest" or (item == "main_profile" and self.guest):
            # main saves this session and starts a new one: a new guest, or her own
            self.switch_user = item
            self.done = True
        elif item == "report":
            if self.make_report is None:
                return
            if self._report_job is not None and self._report_job.status()[0] == "running":
                return
            self._report_job = self.make_report()
            self.speaker.clear()
            self._say("I'm making the report.")

    def _toolbar_view(self):
        report = self._report_job.status()[1] if self._report_job is not None else ""
        now = self.guest_id or self.profile.get("name") or "her profile"
        items = [
            {"id": "short", "key": "T", "label": "Short sessions", "on": self.short},
            {"id": "new_guest", "key": "G", "label": "New guest", "status": f"Now: {now}"},
        ]
        if self.guest:
            items.append({"id": "main_profile", "key": "B", "label": "Main profile",
                          "status": f"back to {config.USER_NAME}"})
        items.append({"id": "report", "key": "O", "label": "Make report",
                      "status": report or ("this guest" if self.guest else "everyone")})
        return {"open": self.toolbar_open, "user": now, "items": items}

    # --- skip ---------------------------------------------------------------------

    def _skip(self, now):
        """Rest: end it. Exercise: keep what was done and go on to the next one."""
        self.trace("skip", stage=self.stage, exercise=self.name, set=self.set_no)
        self.speaker.clear()
        if self.stage == "rest":
            self._end_rest(now)
            return
        self._record_unfinished()
        if self.name:
            self.skipped.append(self.name)
        self.paused = False
        self.calibration = None
        self._setup = None
        self._quality = None
        self._after_rest = None
        self._say("Let's skip this one.")
        self._next_exercise(now)

    def _on_key(self, key, now):
        if key == "s" and self.stage not in ("start", "safety", "profile_deleted"):
            self._open_safety(now)
            return
        if key == "r":
            self._repeat(now)
            return
        if self.stage == "safety":
            if key in (" ", "y"):
                self._after_safety(now)
            return
        if self.stage == "rating":
            if key in "12345" and len(key) == 1:
                self._rate(int(key), now)
            elif key == " ":
                self._rate(None, now)
            return
        if (self.stage == "exercise" and key.isdigit() and not self.paused
                and hasattr(self.exercise, "select")):
            # memory game: keys 1-8 turn a card over
            self.speaker.say_all(self.exercise.select(int(key) - 1, now, affected=False))
            return
        if self.stage in MENU_STAGES:
            self._menu_key(key, now)
            return
        if self.stage in PROFILE_STAGES:
            self._profile_key(key, now)
            return
        if self.stage == "profile_deleted":
            return
        if self.stage in QUESTION_STAGES:
            if key in (" ", "y", "n"):
                self._answer(key != "n", now)
            return
        if key == "k" and self.stage in SKIP_STAGES + ("rest",):
            self._skip(now)
            return
        if key == "m" and not self.fixed and not self._ended and self.stage not in (
                "start", "greeting", "today_plan"):
            self._back_to_menu(now)
            return
        if key == "y" and self.stage in CONTINUE_STAGES:
            self._continue(now)
            return
        if key != " ":
            return
        if self.stage in ("greeting", "today_plan", "rest", "summary", "garden", "goodbye"):
            self._continue(now)
        elif self.stage == "calibration_offer":
            self._start_calibration(now)
        elif self.stage in ("exercise", "calibrating", "intro", "setup_check"):
            self.paused = not self.paused
            if self.paused:
                self.speaker.clear()        # nothing old after "Paused"
                self._say("Paused. Press the space bar when you are ready.")
            else:
                self.speaker.clear()
                self._say("Let's continue.")
                if self.stage == "exercise" and self.coach:
                    self.coach.resume(now)
                elif self.stage == "calibrating" and self.calibration:
                    self.speaker.say_all(self.calibration.resume(now))
                elif self.stage == "intro":
                    self._say(*self._instructions(EXERCISES[self.name]))
                    self._stage_t = now

    def _continue(self, now):
        """Space or thumbs up where there is nothing to answer: go on."""
        stage = self.stage
        if stage == "greeting":
            self._after_greeting(now)
        elif stage == "today_plan":
            self._next_exercise(now)
        elif stage == "intro":
            if not self.speaker.busy:
                self._calibration_check(now)
        elif stage == "rest":
            self._end_rest(now)
        elif stage == "summary":
            self._after_summary(now)
        elif stage == "garden":
            self._goodbye(now)
        elif stage == "goodbye":
            self.done = True

    # --- main update --------------------------------------------------------------

    def update(self, f, now, gestures=None):
        """One frame. gestures: [(label, score)] of all hands, for thumbs up / down."""
        self._handle(self._update, f, now, gestures)

    def _update(self, f, now, gestures):
        self._t = now
        seen = self._gestures(f, gestures)
        self._hand_seen = bool(seen) or bool(f is not None and getattr(f, "present", False))
        if self.paused or self.done:
            return
        if self.speaker.busy:
            self._quiet_t = now
        answer = None
        if self.stage in YES_NO_STAGES:
            answer = self.yes_no.update(seen, now)
        stage = self.stage
        if stage == "start":
            self._begin(now)
        elif stage == "safety":
            # stays until she says she feels fine; never moves on by itself
            if answer == "yes":
                self._after_safety(now)
        elif stage in MENU_STAGES:
            # point at an item, or show its number with the fingers
            i = self.menu_picker.update(f, menu_layout(len(self.menu_items)), now)
            if i is not None:
                how = "pointing" if self.menu_picker.pointer is not None else "fingers"
                self.trace("menu_gesture", menu=self.stage, item=self.menu_items[i], how=how)
                self._pick(i, now)
        elif stage == "rating":
            fingers = self.finger_count.update(f, now)
            if fingers:
                self._rate(min(5, fingers), now)
            elif self._quiet_for(now) >= config.RATING_TIMEOUT_S:
                self._rate(None, now)
        elif stage in QUESTION_STAGES:
            timeout = (config.CHECK_IN_TIMEOUT_S if stage == "check_in"
                       else config.QUESTION_TIMEOUT_S)
            if answer:
                self._answer(answer == "yes", now)
            elif self._quiet_for(now) >= timeout:
                self._answer(None, now)
        elif stage == "profile":
            if answer == "yes":
                self._open_menu(now)
        elif stage == "profile_delete":
            # only the y key deletes; thumbs down or no answer keeps her profile
            if answer == "no" or self._quiet_for(now) >= config.QUESTION_TIMEOUT_S:
                self._keep_profile(now)
        elif stage == "profile_deleted":
            if self._quiet_for(now) >= config.CARD_PAUSE_S:
                self.done = True
        elif answer == "yes":
            self._continue(now)
        elif stage in ("greeting", "today_plan"):
            if self._quiet_for(now) >= config.CARD_PAUSE_S:
                self._continue(now)
        elif stage == "intro":
            n = len(self.exercise.instructions) + 1
            wait = max(config.INTRO_CARD_S, INTRO_S_PER_SENTENCE * n)
            if now - self._stage_t >= wait and not self.speaker.busy:
                self._calibration_check(now)
        elif stage == "calibration_offer":
            if self.speaker.busy:
                self._stage_t = now         # her time to answer starts after the question
            elif now - self._stage_t >= config.RECALIBRATION_OFFER_S:
                self._begin_sets(now)
        elif stage == "calibrating":
            problem = self.calibration.quality_problem(f)
            self.speaker.say_all(self.calibration.update(f, now, quality_ok=problem is None,
                                                         speaking=self.speaker.busy))
            self._calibration_quality(problem, now)
            if self.calibration.done:
                self._calibration_done(now)
        elif stage == "setup_check":
            self.speaker.say_all(self._setup.update(f, now, speaking=self.speaker.busy))
            if self._setup.passed:
                self.setup_checks.append((self.name, True))
                if self._setup.reported:
                    self._say("Good, I can see you well.")
                self._setup = None
                self._start_exercise(now)
        elif stage == "exercise":
            self.coach.update(f, now)
            if self.exercise.fatigue and not self._fatigue_offered:
                self._fatigue_offered = True
                self.exercise.fatigue = False
                self.coach.interrupt(now)       # drops the prompt for the next rep
                self._say("You've worked hard. Let's rest for a minute.")
                self._difficult(self.progress.switch_difficult("fatigue"))
                self._rest(now, config.FATIGUE_REST_S, then="resume")
            elif self.exercise.set_done:
                self._set_finished(now)
        elif stage == "rest":
            if now - self._stage_t >= self._rest_s:
                self._end_rest(now)
        elif stage == "summary":
            if self._quiet_for(now) >= SUMMARY_S:
                self._after_summary(now)
        elif stage == "garden":
            if self._quiet_for(now) >= config.GARDEN_SCREEN_S:
                self._goodbye(now)
        elif stage == "goodbye":
            if self._quiet_for(now) >= config.GOODBYE_S:
                self.done = True

    # --- opening: name, greeting, activities, check-in --------------------------------

    def _begin(self, now):
        fixed_name = self.character.get("name")
        if fixed_name and not self.profile.get("coach_name"):
            self.profile["coach_name"] = fixed_name
        if memory.needs_name(self.profile, self.character):
            self._ask_name(now)
        else:
            self._greet(now)

    def _ask_name(self, now):
        options = self.character["name_options"]
        self._card(event("NameQuestion", option=options[self._name_index],
                         first=self._name_index == 0), "setup_name", now)

    def _greet(self, now, named_now=False):
        ev = memory.greeting(self.profile, self.progress.today)
        ev.details["named_now"] = named_now
        self._card(ev, "greeting", now)

    def _after_greeting(self, now):
        if memory.needs_activities(self.profile, self.activities):
            self._activity_index = 0
            self._chosen = []
            self._ask_activity(now)
        else:
            self._check_in(now)

    def _ask_activity(self, now):
        self._card(event("ActivityQuestion", activity=self._activity_cards[self._activity_index],
                         first=self._activity_index == 0), "setup_activities", now)

    def _check_in(self, now):
        self._card(event("CheckIn"), "check_in", now)

    def _answer(self, yes, now):
        """yes: True, False, or None when she did not answer in time."""
        stage = self.stage
        if stage == "setup_name":
            options = self.character["name_options"]
            if yes:
                self.profile["coach_name"] = options[self._name_index]
            else:
                self._name_index += 1
                if yes is False and self._name_index < len(options):
                    self._ask_name(now)
                    return
                self.profile["coach_name"] = options[0]
            self.save_profile(self.profile)
            self._greet(now, named_now=True)
        elif stage == "setup_activities":
            if yes:
                self._chosen.append(self._activity_cards[self._activity_index])
            self._activity_index += 1
            if (yes is not None and len(self._chosen) < config.ACTIVITY_CHOICES
                    and self._activity_index < len(self._activity_cards)):
                self._ask_activity(now)
                return
            if self._chosen:
                self.profile["chosen_activities"] = list(self._chosen)
                self.save_profile(self.profile)
                self.speaker.say(event("ActivitiesChosen", activities=list(self._chosen)))
            self._check_in(now)
        elif stage == "check_in":
            ev = self.progress.answer_check_in(yes)
            if yes:
                self.speaker.say(event("CheckInAnswer", good=True))
            self._difficult(ev)
            self._after_check_in(now)
        elif stage == "plant_choice":
            first, second = self._plant_options
            self._water_and_end(now, second if yes is False else first)

    def _after_check_in(self, now):
        if self.fixed:
            self._today_plan(now)
        else:
            self._say("What would you like to do?")
            self._open_menu(now)

    def _today_plan(self, now):
        self._card(event("TodayPlan", count=len(self.plan)), "today_plan", now)

    # --- difficult day ------------------------------------------------------------

    def _difficult(self, ev):
        """Difficult day mode was just switched on: say so once, make the rest easier."""
        if ev is None:
            return
        self.speaker.say(ev)
        self.today = [n for n in self.today if not self.progress.skip_today(n)]
        ex = self.exercise
        if ex is not None and self.stage in ("exercise", "rest", "calibrating",
                                             "calibration_offer", "intro"):
            ex.params["sets"] = max(self.set_no, 1, ex.sets - config.DIFFICULT_FEWER_SETS)

    # --- menu -------------------------------------------------------------------------

    def _open_menu(self, now, stage="menu"):
        """The main menu, or with stage one group's menu ("hand_menu" ...)."""
        self.menu_index = 0
        self.menu_picker.reset()
        self._say(self._menu_prompt(stage))
        self._instruction = "Point, or show the number with your fingers"
        self._enter(stage, now)

    @staticmethod
    def _menu_prompt(stage):
        if stage == "menu":
            return "Point at what you'd like to do, or show its number with your fingers."
        return "Point at an exercise, or show its number with your fingers."

    def _menu_keys(self):
        """The key of each item on screen: 1, 2, 3 ..., and P for her profile."""
        return ["P" if item == PROFILE else str(i + 1) for i, item in enumerate(self.menu_items)]

    def _menu_key(self, key, now):
        items = self.menu_items
        main = self.stage == "menu"
        if main and key == "i":
            self.toolbar_open = not self.toolbar_open
            return
        if main and self.toolbar_open and key in TOOLBAR_KEYS:
            self._toolbar(TOOLBAR_KEYS[key], now)
            return
        if key == "up":
            self.menu_index = (self.menu_index - 1) % len(items)
        elif key == "down":
            self.menu_index = (self.menu_index + 1) % len(items)
        elif key == " ":
            self._pick(self.menu_index, now)
        elif key.upper() in self._menu_keys():
            self._pick(self._menu_keys().index(key.upper()), now)
        elif main and key == "e":
            self._pick(items.index(FINISH), now)
        elif not main and key in ("m", "0"):
            self._open_menu(now)

    def _pick(self, i, now):
        """Item i of the menu on screen was chosen (key, pointing or fingers)."""
        item = self.menu_items[i]
        self.menu_index = i
        if item == BACK:
            self._open_menu(now)
        elif item in GROUP_STAGES:
            self._open_menu(now, GROUP_STAGES[item])
        elif is_arm(item) and not self._arm_allowed(item):
            self._say("This one is for when your therapist is with you.")
            self.menu_picker.reset()        # the hand comes down before the next choice
        else:
            self._choose(item, now)

    def _arm_allowed(self, name):
        """Exercises the therapist has not cleared for her alone start only with --exercise."""
        return bool(benchmarks.exercise_settings(self.therapist, name)["enabled"])

    def _choose(self, item, now):
        if item == FINISH:
            self._finish_session(now)
            return
        if item == PROFILE:
            self._open_profile(now)
            return
        self._from_all = item == ALL
        self.plan = list(self.today) if item == ALL else [item]
        self.index = -1
        self.exercise = None
        self.coach = None
        if self._from_all:
            self._today_plan(now)
        else:
            self._next_exercise(now)

    def _back_to_menu(self, now):
        """Leave whatever is running; keep what was done."""
        self._record_unfinished()
        self.paused = False
        self.calibration = None
        self._setup = None
        self._quality = None
        self._after_rest = None
        self._say("Let's choose another exercise.")
        self._open_menu(now)

    # --- profile ----------------------------------------------------------------------

    def _open_profile(self, now):
        """What the coach remembers about her, with the option to delete it."""
        self._profile_rows = memory.profile_overview(self.profile, self.garden,
                                                     self.log.sessions(), self.activities)
        self.speaker.clear()
        self.speaker.say(event("ProfileOverview"))
        self._enter("profile", now)

    def _profile_key(self, key, now):
        if self.stage == "profile":
            if key == "d":
                self._card(event("DeleteProfileQuestion"), "profile_delete", now)
            elif key in (" ", "y", "m", "p"):
                self._open_menu(now)
        elif key == "y":
            self._delete(now)
        elif key in ("n", " ", "m"):
            self._keep_profile(now)

    def _keep_profile(self, now):
        self.speaker.clear()
        self.speaker.say(event("ProfileKept"))
        self._enter("profile", now)

    def _delete(self, now):
        """
        Delete her profile, garden and history. Nothing of this session is
        saved any more; the session ends and main starts again from the first
        questions.
        """
        self._ended = True
        self.save_profile = self.save_garden = lambda _: None
        self.delete_profile()
        self.restart = True
        self.speaker.clear()
        self._card(event("ProfileDeleted"), "profile_deleted", now)

    # --- flow -------------------------------------------------------------------------

    def _next_exercise(self, now):
        self.index += 1
        while (self._from_all and not self.fixed and self.index < len(self.plan)
               and self.progress.skip_today(self.plan[self.index])):
            self.index += 1
        if self.index >= len(self.plan):
            self._finish(now)
            return
        cls = EXERCISES[self.name]
        if is_arm(self.name) and cls.not_testable(self.therapist):
            # FMA: an elbow contracture of 30 degrees or more rules the item out
            self.speaker.say(event("NotTestable", exercise=self.name))
            self._next_exercise(now)
            return
        self._build_exercise()
        self.set_no = 0
        self._fatigue_offered = False
        # the card shows the title; she hears why (the activity) and how
        ev = event("ExerciseIntro", exercise=self.name, activity=self.activity, title=cls.title)
        self._card_event = ev
        self.speaker.say(ev)
        lines = self._instructions(cls)
        self._say(*lines)
        self._instruction = lines[-1] if lines else cls.title
        self._enter("intro", now)

    def _calibration_quality(self, problem, now):
        """
        Measuring pauses while the hand is not seen well; say why, calmly.
        Like during the exercise, a short drop-out (e.g. one frame with the
        wrong Left/Right label) shows nothing: only after QUALITY_GRACE_S.
        """
        if problem != self._cal_problem:
            self._cal_problem, self._cal_problem_since = problem, now
        if problem is None or now - self._cal_problem_since < config.QUALITY_GRACE_S:
            self._quality = None
            return
        side = getattr(self.calibration, "side", None) or self.coach.hand
        self._quality = QUALITY_TEXT[problem].format(hand=side)
        if now - self._cal_last_say.get(problem, -1e9) >= config.QUALITY_MESSAGE_REPEAT_S:
            self._cal_last_say[problem] = now
            self.speaker.say(Say(QUALITY_TEXT[problem].format(hand=side), "quality",
                                 valid=lambda: self._cal_problem == problem))

    def _calibration_check(self, now):
        cls = EXERCISES[self.name]
        if cls.make_calibration(self.therapist) is None:
            self._begin_sets(now)           # nothing to measure first (e.g. the memory game)
            return
        entry = self.profile["calibration"].get(self.name)
        # a wrong stored calibration (e.g. open/closed swapped) reverses every
        # prompt of the exercise, so it is measured again like a missing one;
        # an arm calibration is also the weekly assessment
        if (is_stale(entry, max_age_days=cls.calibration_max_age_days)
                or not cls.calibration_valid(entry["steps"])):
            self._start_calibration(now)
        else:
            part = "arm" if is_arm(self.name) else "hand"
            self._say(f"Press the space bar if you'd like to measure your {part} again.")
            self._instruction = f"Space bar: measure my {part} again"
            self._enter("calibration_offer", now)

    def _start_calibration(self, now):
        self.speaker.clear()                # e.g. the recalibration question
        self._cal_problem = None
        self.calibration = EXERCISES[self.name].make_calibration(self.therapist)
        if self.calibration is None:
            self._begin_sets(now)
            return
        self.speaker.say_all(self.calibration.start(now))
        self._enter("calibrating", now)

    def _calibration_done(self, now):
        """Store the calibration; for an arm exercise it is also the weekly assessment."""
        old = self.profile["calibration"].get(self.name)
        entry = self.calibration.as_profile_entry()
        self.trace("calibration", exercise=self.name, entry=entry,
                   attempt=getattr(self.calibration, "attempt", None))
        if is_arm(self.name):
            self.speaker.say(self.progress.assessment_done(EXERCISES[self.name], entry, old))
        self.profile["calibration"][self.name] = entry
        self.save_profile(self.profile)
        self.calibration = None
        self._quality = None
        self._build_exercise()
        self._begin_sets(now)

    def _begin_sets(self, now):
        """Arm exercises check the camera first (plan 3.1); then the first set."""
        if is_arm(self.name):
            ex = self.exercise
            self._setup = SetupCheck(ex.required(), ex.setup_view(),
                                     getattr(ex, "round_side", ex.side))
            self._instruction = ""
            self._enter("setup_check", now)
        else:
            self._start_exercise(now)

    def _instructions(self, cls):
        """What she hears before an exercise ({side} filled in for the arm exercises)."""
        if hasattr(cls, "instruction_lines"):
            return cls.instruction_lines(self.side)
        return list(cls.instructions)

    def _start_exercise(self, now):
        self.set_no += 1
        self._fatigue_offered = False
        self._instruction = ""
        self._enter("exercise", now)
        self.coach.start_set(self.set_no, now)

    def _set_finished(self, now):
        ex = self.exercise
        events, target = self.progress.set_finished(ex)
        if target is not None:
            ex.set_target(target)           # the next set's target, never during a set
        started = next((e for e in events if e.type == "DifficultDayStarted"), None)
        if started:
            ex.params["sets"] = max(self.set_no, 1, ex.sets - config.DIFFICULT_FEWER_SETS)
            self.today = [n for n in self.today if not self.progress.skip_today(n)]
        if self.set_no < ex.sets:
            self.speaker.say(event("SetCompleted", set_no=self.set_no, sets=ex.sets))
            if started:
                self.speaker.say(started)
            elif not self.progress.difficult:
                self.speaker.say([e for e in events if e.type == "TargetRaised"])
            rest = self._rest_seconds(config.REST_BETWEEN_SETS_S)
            self._say(f"Let's rest for {rest} seconds.")
            self._rest(now, rest, then="next_set")
        else:
            if started:
                self.speaker.say(started)
            self._exercise_finished(now)

    def _exercise_finished(self, now):
        events = self._record_summary(completed=True)
        self.speaker.say(event("ExerciseDone"))
        self.speaker.say(events)
        remaining = [n for n in self.plan[self.index + 1:]
                     if self.fixed or not self._from_all or not self.progress.skip_today(n)]
        if remaining:
            rest = (config.SHORT_SESSION["rest_between_exercises_s"] if self.short
                    else self._rest_seconds(config.REST_BETWEEN_EXERCISES_S))
            self._say(f"Let's rest for {rest} seconds.")
            self._rest(now, rest, then="next_exercise")
        else:
            self._finish(now)

    def _record_summary(self, completed=True):
        ex = self.exercise
        if ex is None or not ex.reps:
            return []
        target_end = ex.target if getattr(ex, "range_steps", ()) else float("nan")
        row = self.log.summarize(ex.name, ex.reps, self.set_no, ex.sets * ex.reps_per_set,
                                 target_start=self._target_start, target_end=target_end,
                                 difficult_day=self.progress.difficult)
        earlier, when = self.log.comparison(ex.name)
        threshold = ex.change_threshold() if hasattr(ex, "change_threshold") else None
        message = progress_message(type(ex), row, earlier, when,
                                   history=self.log.normal_history(ex.name), threshold=threshold)
        if improvement_claimed(type(ex), row, earlier, threshold) is not None:
            self.progress.claims[ex.name] = message
        self.log.save_summary(row)
        self.summaries.append((ex.name, row, message))
        events = self.progress.exercise_finished(ex, row, self.activity, completed)
        if self.activity and self.activity not in self._practised:
            self._practised.append(self.activity)
        if getattr(ex, "uses_level", False):
            self.profile["levels"][ex.name] = ex.level
        self.save_profile(self.profile)
        return events

    def _record_unfinished(self):
        """Keep what was done of an exercise that was left early."""
        if self.stage in ("exercise", "rest") and self.exercise and self.exercise.reps:
            if not any(s[0] == self.exercise.name and s[1]["reps_done"] == len(self.exercise.reps)
                       for s in self.summaries[-1:]):
                self._record_summary(completed=False)

    def _rest(self, now, seconds, then):
        self._rest_s = seconds
        self._after_rest = then
        self._instruction = "Rest your hand. Thumbs up or space bar: continue"
        self._enter("rest", now)

    def _end_rest(self, now):
        then = self._after_rest
        self._after_rest = None
        if then == "next_set":
            self._say("Let's continue.")
            self._start_exercise(now)
        elif then == "resume":
            self._say("Let's continue.")
            self._instruction = ""
            self._enter("exercise", now)
            self.coach.resume(now)
        else:
            self._next_exercise(now)

    def _finish(self, now):
        """The chosen exercises are done: one highlight and a comparison."""
        lines = []
        for name, row, message in self.summaries:
            lines.append(f"{EXERCISES[name].title}: {row['reps_done']} done")
        self.summary_lines = lines or ["No exercises done today."]
        best = self.progress.highlight()
        comparisons = [] if self.progress.difficult else [m for _, _, m in self.summaries if m]
        ev = event("Summary", highlight=best.as_dict() if best else None,
                   comparisons=comparisons[-1:], activities=list(self._practised),
                   difficult=self.progress.difficult)
        self._summary_event = ev
        self.speaker.say(ev)
        if self.fixed:
            self._instruction = "Thumbs up or space bar: see your garden"
        else:
            self._instruction = "Thumbs up or space bar: choose another exercise"
        self._enter("summary", now)

    def _after_summary(self, now):
        if self.fixed:
            self._finish_session(now)
        else:
            self._open_menu(now)

    # --- closing: garden and goodbye ---------------------------------------------------------

    def _finish_session(self, now):
        if self.progress.completed and self.rating_questions and self._rating_index is None:
            # how was it? (1-5), before the garden
            self._rating_index = 0
            self._ask_rating(now)
            return
        if self.progress.completed and garden_model.needs_new_plant(self.garden):
            self._plant_options = garden_model.plant_options(self.garden)
            first, second = self._plant_options
            self._card(event("PlantQuestion", first=first, second=second), "plant_choice", now)
        else:
            self._water_and_end(now, None)

    # --- ratings (1-5): how hard, how enjoyable, how easy to use ----------------------

    def _ask_rating(self, now):
        key = self.rating_questions[self._rating_index]
        self.finger_count.reset()
        self._card(event("RatingQuestion", key=key, first=self._rating_index == 0), "rating", now)

    def _rate(self, value, now):
        """value: 1..5, or None (skipped or no answer)."""
        key = self.rating_questions[self._rating_index]
        self.trace("rating", question=key, value=value)
        self.ratings[key] = value if value else ""
        if value:
            self.speaker.say(event("RatingThanks"))
        self._rating_index += 1
        if self._rating_index < len(self.rating_questions):
            self._ask_rating(now)
        else:
            self._finish_session(now)

    # --- Stop / "I don't feel well" ---------------------------------------------------

    def _open_safety(self, now):
        """Stop everything, keep what was done, and show the safety screen."""
        self._record_unfinished()
        self._safety = True
        self.paused = False
        self.calibration = None
        self._setup = None
        self._quality = None
        self._after_rest = None
        self.speaker.clear()
        self._card(event("SafetyStop"), "safety", now)

    def _after_safety(self, now):
        """She feels fine: back to the menu (or on to the summary / goodbye)."""
        self.speaker.clear()
        self._say("All right. Take your time.")
        self.exercise = None
        self.coach = None
        if self._ended:
            self._goodbye(now)
        elif self.fixed:
            self._finish(now)
        else:
            self._open_menu(now)

    # --- repeat ----------------------------------------------------------------------

    def _repeat(self, now):
        """Say again what belongs to the screen she is on (R key)."""
        stage = self.stage
        self.speaker.clear()
        if stage in CARD_STAGES + ("rating", "safety", "profile_delete", "profile_deleted"):
            if self._card_event is not None:
                self.speaker.say(self._card_event)
            if stage == "intro":
                self._say(*self._instructions(EXERCISES[self.name]))
        elif stage == "calibration_offer":
            part = "arm" if is_arm(self.name) else "hand"
            self._say(f"Press the space bar if you'd like to measure your {part} again.")
        elif stage == "setup_check" and self._setup:
            self._setup = SetupCheck(self._setup.required, self._setup.view, self._setup.side)
        elif stage in MENU_STAGES:
            self._say(self._menu_prompt(stage))
        elif stage == "calibrating" and self.calibration:
            self.speaker.say_all(self.calibration.resume(now))
        elif stage == "exercise" and self.exercise:
            self.speaker.say_all(self.exercise.resume_messages())
        elif stage == "rest":
            self._say("Rest your hand. Thumbs up or the space bar when you're ready.")
        elif stage == "summary" and self._summary_event is not None:
            self.speaker.say(self._summary_event)
        elif stage == "garden" and self._garden_event is not None:
            self.speaker.say(self._garden_event)
        elif stage == "profile":
            self.speaker.say(event("ProfileOverview"))

    def _water_and_end(self, now, plant):
        grew = self.end_session(plant)
        if grew is not None:
            self._garden_event = grew
            self.speaker.say(grew)
            self._enter("garden", now)
        else:
            self._goodbye(now)

    def _goodbye(self, now):
        self.end_session()
        plot = garden_model.current(self.garden)
        self._card(event("Goodbye", plant=plot["plant"] if plot else None), "goodbye", now)

    def end_session(self, plant=None):
        """
        Once per session: water the garden (when at least one exercise was
        completed), remember the session and save everything. Returns the
        GardenGrew event, or None.
        """
        if self._ended:
            return None
        self._ended = True
        grew = None
        if self.progress.completed:
            highlights = self.progress.highlights
            bee = any(e.type == "PersonalBest" for e in highlights)
            butterflies = sum(1 for e in highlights if e.type == "MilestoneReached")
            grew = garden_model.water(self.garden, plant, bee=bee, butterflies=butterflies)
            self.save_garden(self.garden)
        if self.summaries:
            # "when she last practised" - only when she did
            self.progress.session_best()
            self.progress.end_session([name for name, _, _ in self.summaries])
        if self.progress.difficult:
            self.log.mark_difficult()
        bench = self.progress.benchmark_session_row(self.summaries, self.setup_checks)
        if bench is not None:
            self.log.save_bench_session(bench)
        row = self._session_row(grew)
        if self.progress.repeated_difficult_days(self.log.sessions()):
            row["note"] = " ".join(n for n in (row["note"], (
                f"{config.DIFFICULT_REPEAT_COUNT} or more of the last "
                f"{config.DIFFICULT_REPEAT_WINDOW} sessions were difficult days. "
                "Please ask her therapist to take a look.")) if n)
        self.log.save_session(row)
        self.save_profile(self.profile)
        return grew

    def _session_row(self, grew):
        rows = [row for _, row, _ in self.summaries]
        total = sum(int(r["reps_done"]) for r in rows)
        rates = [r["success_rate"] for r in rows if finite(r["success_rate"])]
        ranges = [r["mean_range_high"] for r in rows if finite(r["mean_range_high"])]
        best = self.progress.highlight()
        end = datetime.now()
        return {
            "session_id": self.log.session_id,
            "user_id": self.guest_id or "eleanor",
            "date": self.progress.today.isoformat(),
            "start": self._started_at.isoformat(timespec="seconds"),
            "end": end.isoformat(timespec="seconds"),
            "duration_min": round((end - self._started_at).total_seconds() / 60, 1),
            "exercises": ";".join(name for name, _, _ in self.summaries),
            "total_reps": total,
            "success_rate": sum(rates) / len(rates) if rates else float("nan"),
            "average_range": sum(ranges) / len(ranges) if ranges else float("nan"),
            "check_in": self.progress.check_in,
            "difficult_day": "yes" if self.progress.difficult else "no",
            "difficult_reason": self.progress.difficult_reason,
            "highlight": best.type + (f":{best.get('level')}" if best.get("level") else "")
            if best else "",
            "garden": f"{grew.get('plant')}:{grew.get('stage')}" if grew else "",
            "note": " ".join(n for n in (
                "She pressed Stop / I don't feel well." if self._safety else "",
                f"Skipped: {', '.join(self.skipped)}." if self.skipped else "") if n),
            "safety_stop": "yes" if self._safety else "no",
            **{key: self.ratings.get(key, "") for key in ("exertion", "enjoyment", "ease")},
        }

    def stop(self, now):
        """Quit (q, end of video): keep what was done and end the session properly."""
        self._record_unfinished()
        if (self.summaries or self.progress.check_in) and not self._ended:
            self.end_session(self._plant_options[0] if self._plant_options else None)
        self.save_profile(self.profile)

    # --- verbose log -------------------------------------------------------------------

    def debug_state(self):
        """What the session and the active detector are doing this frame (verbose log)."""
        d = {"stage": self.stage, "exercise": self.name, "set": self.set_no,
             "paused": self.paused, "short": self.short, "user": self.guest_id or "eleanor"}
        if self.stage in YES_NO_STAGES:
            answer, progress = self.yes_no.state(self._t)
            d["yes_no"] = {"answer": answer, "progress": progress}
        if self.stage == "rating":
            fingers, held, armed = self.finger_count.state(self._t)
            d["rating"] = {"question": self.rating_questions[self._rating_index],
                           "raw": self.finger_count.raw, "voted": fingers, "hold": held,
                           "armed": armed}
        if self.stage in MENU_STAGES:
            d["menu"] = dict(self.menu_picker.state(), raw=self.menu_picker.raw)
        if self.stage == "calibrating" and self.calibration:
            step = self.calibration.step
            d["calibration"] = {"step": step.name if step else None,
                                "stage": getattr(self.calibration, "_stage", None),
                                "progress": self.calibration.progress,
                                "attempt": getattr(self.calibration, "attempt", None),
                                "problem": self._cal_problem}
        if self.stage == "setup_check" and self._setup:
            d["setup"] = {"problem": self._setup.problem}
        if self.stage == "exercise" and self.exercise and self.coach:
            d["quality"] = self.coach._problem
            d["detector"] = self.exercise.debug_state()
        return d

    # --- what Act draws -----------------------------------------------------------------

    def _can(self):
        sections = len(self.plan) if self._from_all else len(self.today)
        filled = len(self.progress.completed)
        return {"sections": max(sections, filled), "filled": filled}

    def _menu_view(self):
        """The menu on screen: its items where they lie over the camera image, and the hand."""
        items, main = self.menu_items, self.stage == "menu"
        picker = self.menu_picker.state()
        dwell = picker["dwell"]
        number = picker["number"]
        rows = []
        for i, (item, key, rect) in enumerate(zip(items, self._menu_keys(),
                                                  menu_layout(len(items)))):
            note = ""
            if item in config.DAILY_PLAN and item not in self.today:
                note = "not today"          # every-other-day exercises not planned today
            elif item not in MENU_TEXT and is_arm(item) and not self._arm_allowed(item):
                note = "with your therapist"
            progress = max(dwell[1] if dwell and dwell[0] == i else 0.0,
                           number[1] if number and number[0] == i + 1 else 0.0)
            rows.append({"key": key, "text": MENU_TEXT.get(item) or EXERCISES[item].title,
                         "selected": i == self.menu_index, "note": note, "rect": rect,
                         "progress": progress})
        title = ("What would you like to do?" if main else
                 MENU_TEXT[next(g for g, st in GROUP_STAGES.items() if st == self.stage)])
        v = {"title": title, "menu": rows,
             "menu_hand": {"pointer": picker["pointer"], "armed": picker["armed"],
                           "fingers": number[0] if number else 0,
                           "progress": number[1] if number else 0.0}}
        if main:
            v["footer"] = "Keys 1-5, P: profile   E: end   I: toolbar"
            v["toolbar"] = self._toolbar_view()
        else:
            v["footer"] = f"Keys 1-{len(items)}   M or {len(items)}: back"
        return v

    def view(self):
        v = {
            "stage": self.stage,
            "paused": self.paused,
            "title": EXERCISES[self.name].title if self.name else "Your exercises",
            "instruction": self._instruction,
            "subtitle": getattr(self.speaker, "last_text", ""),
            "footer": "Space bar: start / pause / continue",
            "quality": None,
            "mood": getattr(self.speaker, "mood", "neutral"),
            "coach_name": self.profile.get("coach_name"),
            "can": self._can(),
            "activity": self.activity if self.stage in (
                "intro", "calibration_offer", "calibrating", "exercise", "rest",
                "setup_check") else None,
            "difficult_day": self.progress.difficult,
            "exercise": self.name,
            "hand": self.profile.get("affected_hand", config.AFFECTED_HAND),
        }
        if self.stage in STEP_STAGES and len(self.plan) > 1 and 0 <= self.index < len(self.plan):
            v["step_label"] = f"Exercise {self.index + 1} of {len(self.plan)}"
        if self.stage == "intro" and self.name:
            # the demo hand on the card loops through the movement
            v["demo"] = {"exercise": self.name, "t": self._t - self._stage_t}
        answer, progress = self.yes_no.state(self._t)
        v["gesture"] = {"hand": self._hand_seen, "answer": answer, "progress": progress}
        if not self.fixed and self.stage != "start" and self.stage not in MENU_STAGES:
            v["footer"] = "Space: pause / continue   M: menu"
        if self.stage in CARD_STAGES:
            v["screen"] = "card"
            v["card_event"] = self._card_event
            v["footer"] = ("Thumbs up / down, or Space = yes, N = no"
                           if self.stage in QUESTION_STAGES else "Thumbs up or space bar: continue")
            if self.stage == "intro":
                v["footer"] = "Thumbs up: start   Space: pause"
        if self.stage in SKIP_STAGES + ("rest",) and self.name:
            v["skip"] = {"key": "K", "label": "Skip rest" if self.stage == "rest" else "Skip exercise"}
        if self.stage == "exercise" and self.exercise:
            ex = self.exercise
            v["exercise_display"] = ex.display
            v["quality"] = self.coach.quality
            v["status"] = (f"Set {self.set_no} of {ex.sets}   "
                           f"{ex.reps_this_set} of {ex.reps_per_set}")
        elif self.stage == "calibrating" and self.calibration:
            step = self.calibration.step
            v["instruction"] = step.screen_text if step else "Thank you"
            v["progress"] = self.calibration.progress
            if is_arm(self.name):
                v["status"] = "Measuring your arms"
            else:
                v["status"] = (f"Measuring: step {self.calibration.index + 1} of "
                               f"{len(self.calibration.steps)}" if step else "Measuring your hand")
            v["quality"] = self._quality
            display = getattr(self.calibration, "display", None)
            if display:
                v["exercise_display"] = display
        elif self.stage == "setup_check" and self._setup:
            v["instruction"] = self._setup.screen_text
            v["status"] = "Checking the camera"
            v["setup"] = {"required": list(self._setup.required), "problem": self._setup.problem,
                          "view": self._setup.view, "side": self._setup.side}
        elif self.stage == "rest":
            left = max(0, int(round(self._rest_s - (self._t - self._stage_t))))
            v["countdown"] = left
            v["progress"] = 1.0 - left / max(1, self._rest_s)
            v["status"] = "Rest"
        elif self.stage == "summary":
            v["screen"] = "summary"
            v["title"] = "Today's session"
            v["summary_event"] = self._summary_event
            v["summary_lines"] = self.summary_lines
            v["garden"] = {"state": self.garden}
            v["footer"] = ("Thumbs up or space bar: continue" if self.fixed
                           else "Space: menu   Q: finish")
        elif self.stage == "garden":
            ev = self._garden_event
            v["screen"] = "garden"
            v["title"] = "Your garden"
            v["garden_event"] = ev
            p = (self._t - self._stage_t) / max(config.GARDEN_GROW_S, 1e-6)
            v["garden"] = {"state": self.garden, "grow": {
                "plot": ev.get("plot"), "prev_stage": ev.get("prev_stage"), "progress": p,
                "new_bees": 1 if ev.get("bee") else 0,
                "new_butterflies": ev.get("butterflies", 0)}}
            v["footer"] = "Thumbs up or space bar: continue"
        elif self.stage in MENU_STAGES:
            v.update(self._menu_view())
        elif self.stage == "profile":
            coach = self.profile.get("coach_name")
            v["screen"] = "profile"
            v["title"] = "Your profile"
            v["message"] = f"What {coach or 'your coach'} remembers about you."
            v["profile_rows"] = self._profile_rows
            v["answer_labels"] = ("Thumbs up or Space: back", "D: delete my profile")
            v["footer"] = "Space or M: back to the menu   D: delete my profile and start again"
        elif self.stage in ("profile_delete", "profile_deleted"):
            v["screen"] = "card"
            v["card_event"] = self._card_event
            v["footer"] = ("Y: delete   N, Space or thumbs down: keep"
                           if self.stage == "profile_delete" else "")
        elif self.stage == "rating":
            fingers, held, armed = self.finger_count.state(self._t)
            v["screen"] = "rating"
            v["card_event"] = self._card_event
            v["rating"] = {"fingers": fingers, "progress": held, "armed": armed,
                           "question": self._rating_index + 1, "questions": len(self.rating_questions)}
            v["footer"] = "Keys 1-5 or show fingers   Space: skip"
        elif self.stage == "safety":
            v["screen"] = "safety"
            v["card_event"] = self._card_event
            v["footer"] = "Thumbs up or Space: I feel fine   Q: finish for today"
        if v.get("skip"):
            v["footer"] = (v["footer"] + "   K: skip").strip()
        if self.stage not in ("start", "safety", "profile_delete", "profile_deleted"):
            v["footer"] = (v["footer"] + "   R: repeat").strip()
            v["stop_hint"] = True           # "S: Stop - I don't feel well" on every screen
        return v
