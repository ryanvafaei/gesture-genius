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
  menu (one exercise, all of today's, or finish) or, with exercises given up
  front, today's plan -> for each exercise: activity card -> calibration
  check -> sets with rests (the target is adapted after every set) ->
  summary (one highlight, compared with her own history) -> ... ->
  garden (grows from showing up) -> goodbye.
  A difficult day (thumbs down, a low first set, tiredness) makes the rest
  of the session easier: lower targets, one set fewer, longer rests, no
  strength exercise, praise for effort, no comparisons.

Yes / no answers: thumbs up / thumbs down (either hand), with space or "y"
and "n" as a backup. Space starts, pauses and continues; in the menu a
number picks an exercise and up/down move the highlight; m goes back to the
menu.

Profile (menu item "My profile", or p in the menu): what the coach remembers
about her. d asks whether to delete it; only the y key confirms (a thumbs up
there could be an accident). After deleting, the session ends with
`restart` set, and main starts again as on her first day.
"""

from datetime import datetime

from rehab import config, memory
from rehab import garden as garden_model
from rehab.calibration import CalibrationRoutine, is_stale
from rehab.events import event
from rehab.exercises import EXERCISES, create
from rehab.exercises.base import Say
from rehab.progress import SessionProgress, finite
from rehab.storage import (calibrations, load_content, new_garden, progress_message,
                           should_do_today)

QUALITY_TEXT = {
    "no_hand": "Please show your {hand} hand to the camera.",
    "wrong_hand": "Please use your {hand} hand.",
    "too_far": "Please move your hand a little closer.",
    "palm_away": "Please turn your palm towards the camera.",
}

INTRO_S_PER_SENTENCE = 2.5
SUMMARY_S = 6.0

ALL = "all"
FINISH = "finish"
PROFILE = "profile"

QUESTION_STAGES = ("setup_name", "setup_activities", "check_in", "plant_choice")
CARD_STAGES = QUESTION_STAGES + ("greeting", "today_plan", "intro", "goodbye")
# stages where a thumbs up means "carry on"
CONTINUE_STAGES = ("greeting", "today_plan", "intro", "rest", "summary", "garden", "goodbye")
# her profile and the question whether to delete it
PROFILE_STAGES = ("profile", "profile_delete")


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


class Coach:

    def __init__(self, exercise, speaker, log, hand=config.AFFECTED_HAND, progress=None):
        self.exercise = exercise
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

    def update(self, f, now):
        self.exercise.speaking = self.speaker.busy
        problem = f.quality_problem(self.exercise.need_palm_facing)
        problem_say = None
        if problem:
            problem_say = Say(QUALITY_TEXT[problem].format(hand=self.hand), "quality")
        else:
            problem_say = self.exercise.frame_problem(f)
            problem = problem_say.text if problem_say else None

        if problem:
            if problem != self._problem:
                self._problem, self._problem_since = problem, now
            if now - self._problem_since >= config.QUALITY_GRACE_S:
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
                 delete_profile=None):
        self.speaker = speaker
        self.profile = profile
        self.log = log
        self.save_profile = save_profile or (lambda p: None)
        self.garden = garden if garden is not None else new_garden()
        self.save_garden = save_garden or (lambda g: None)
        self.delete_profile = delete_profile or (lambda: None)
        self.character = character if character is not None else load_content("character")
        self.activities = activities if activities is not None else load_content("activities")
        self.progress = SessionProgress(profile, log)
        self.yes_no = YesNo()
        # today's plan, the menu's "all" item
        self.today = [n for n in config.SESSION_ORDER if should_do_today(n, log)]
        # chosen explicitly: no day schedule and no menu
        self.fixed = bool(exercises)
        self.plan = list(exercises) if exercises else []
        self.menu = [ALL] + list(config.SESSION_ORDER) + [FINISH, PROFILE]
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

    # --- helpers --------------------------------------------------------------

    @property
    def name(self):
        return self.plan[self.index] if 0 <= self.index < len(self.plan) else None

    def _enter(self, stage, now):
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
        kwargs = {}
        if self.name == "thumb_opposition" and self.profile["levels"].get(self.name):
            kwargs["level"] = self.profile["levels"][self.name]
        target = self.progress.start_target(self.name)
        thresholds = {self.name: {"high": target}} if target is not None else None
        sets = config.EXERCISES.get(self.name, {}).get("sets", 3)
        self.exercise = create(self.name, calibrations(self.profile), thresholds,
                               params={"sets": self.progress.sets_for(self.name, sets)}, **kwargs)
        self._target_start = target if target is not None else float("nan")
        self.activity = memory.activity_for(self.name, self.profile, self.activities)
        self.coach = Coach(self.exercise, self.speaker, self.log,
                           self.profile.get("affected_hand", config.AFFECTED_HAND),
                           progress=self.progress)

    def _gestures(self, f, gestures):
        if gestures is not None:
            return gestures
        if f is not None and getattr(f, "present", False) and f.gesture:
            return [(f.gesture, f.gesture_score)]
        return []

    def _rest_seconds(self, base):
        factor = config.DIFFICULT_REST_FACTOR if self.progress.difficult else 1.0
        return int(round(base * factor))

    # --- keys -------------------------------------------------------------------

    def on_key(self, key, now):
        """key: " ", "up", "down", "m", "p", "d", "y", "n" or a digit."""
        if self.stage == "menu":
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
        elif self.stage in ("exercise", "calibrating", "intro"):
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
                    self._say(*EXERCISES[self.name].instructions)
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
        self._t = now
        seen = self._gestures(f, gestures)
        self._hand_seen = bool(seen) or bool(f is not None and getattr(f, "present", False))
        if self.paused or self.done:
            return
        if self.speaker.busy:
            self._quiet_t = now
        answer = None
        if self.stage in QUESTION_STAGES + CONTINUE_STAGES + PROFILE_STAGES:
            answer = self.yes_no.update(seen, now)
        stage = self.stage
        if stage == "start":
            self._begin(now)
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
                self._start_exercise(now)
        elif stage == "calibrating":
            step = self.calibration.step
            need_palm = step.need_palm_facing if step else False
            problem = f.quality_problem(need_palm)
            self.speaker.say_all(self.calibration.update(f, now, quality_ok=problem is None,
                                                         speaking=self.speaker.busy))
            self._quality = (QUALITY_TEXT[problem].format(hand=self.coach.hand)
                             if problem else None)
            self._calibration_quality(problem, now)
            if self.calibration.done:
                self.profile["calibration"][self.name] = self.calibration.as_profile_entry()
                self.save_profile(self.profile)
                self.calibration = None
                self._quality = None
                self._build_exercise()
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
            self._say("Which exercise would you like to do?")
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

    def _open_menu(self, now):
        self.menu_index = 0
        self._say("Press a number to choose an exercise, "
                  "or press the space bar to do all of today's exercises.")
        self._instruction = "Press a number to choose"
        self._enter("menu", now)

    def _menu_key(self, key, now):
        if key == "up":
            self.menu_index = (self.menu_index - 1) % len(self.menu)
        elif key == "down":
            self.menu_index = (self.menu_index + 1) % len(self.menu)
        elif key == " ":
            self._choose(self.menu[self.menu_index], now)
        elif key.isdigit() and int(key) < len(self.menu):
            self.menu_index = int(key)
            self._choose(self.menu[self.menu_index], now)
        elif key == "p":
            self._choose(PROFILE, now)

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
        self._build_exercise()
        self.set_no = 0
        self._fatigue_offered = False
        cls = EXERCISES[self.name]
        # the card shows the title; she hears why (the activity) and how
        ev = event("ExerciseIntro", exercise=self.name, activity=self.activity, title=cls.title)
        self._card_event = ev
        self.speaker.say(ev)
        self._say(*cls.instructions)
        self._instruction = cls.instructions[-1] if cls.instructions else cls.title
        self._enter("intro", now)

    def _calibration_quality(self, problem, now):
        """Measuring pauses while the hand is not seen well; say why, calmly."""
        if problem != self._cal_problem:
            self._cal_problem, self._cal_problem_since = problem, now
        if problem is None or now - self._cal_problem_since < config.QUALITY_GRACE_S:
            return
        if now - self._cal_last_say.get(problem, -1e9) >= config.QUALITY_MESSAGE_REPEAT_S:
            self._cal_last_say[problem] = now
            self.speaker.say(Say(QUALITY_TEXT[problem].format(hand=self.coach.hand), "quality",
                                 valid=lambda: self._cal_problem == problem))

    def _calibration_check(self, now):
        entry = self.profile["calibration"].get(self.name)
        # a wrong stored calibration (e.g. open/closed swapped) reverses every
        # prompt of the exercise, so it is measured again like a missing one
        if is_stale(entry) or not EXERCISES[self.name].calibration_valid(entry["steps"]):
            self._start_calibration(now)
        else:
            self._say("Press the space bar if you'd like to measure your hand again.")
            self._instruction = "Space bar: measure my hand again"
            self._enter("calibration_offer", now)

    def _start_calibration(self, now):
        self.speaker.clear()                # e.g. the recalibration question
        self._cal_problem = None
        self.calibration = CalibrationRoutine(EXERCISES[self.name])
        self.speaker.say_all(self.calibration.start(now))
        self._enter("calibrating", now)

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
            rest = self._rest_seconds(config.REST_BETWEEN_EXERCISES_S)
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
        message = progress_message(type(ex), row, earlier, when)
        self.log.save_summary(row)
        self.summaries.append((ex.name, row, message))
        events = self.progress.exercise_finished(ex, row, self.activity, completed)
        if self.activity and self.activity not in self._practised:
            self._practised.append(self.activity)
        if ex.name == "thumb_opposition":
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
        if self.progress.completed and garden_model.needs_new_plant(self.garden):
            self._plant_options = garden_model.plant_options(self.garden)
            first, second = self._plant_options
            self._card(event("PlantQuestion", first=first, second=second), "plant_choice", now)
        else:
            self._water_and_end(now, None)

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
        row = self._session_row(grew)
        if self.progress.repeated_difficult_days(self.log.sessions()):
            row["note"] = (f"{config.DIFFICULT_REPEAT_COUNT} or more of the last "
                           f"{config.DIFFICULT_REPEAT_WINDOW} sessions were difficult days. "
                           "Please ask her therapist to take a look.")
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
            "note": "",
        }

    def stop(self, now):
        """Quit (q, end of video): keep what was done and end the session properly."""
        self._record_unfinished()
        if (self.summaries or self.progress.check_in) and not self._ended:
            self.end_session(self._plant_options[0] if self._plant_options else None)
        self.save_profile(self.profile)

    # --- what Act draws -----------------------------------------------------------------

    def _can(self):
        sections = len(self.plan) if self._from_all else len(self.today)
        filled = len(self.progress.completed)
        return {"sections": max(sections, filled), "filled": filled}

    def view(self):
        v = {
            "stage": self.stage,
            "paused": self.paused,
            "title": EXERCISES[self.name].title if self.name else "Hand exercises",
            "instruction": self._instruction,
            "subtitle": getattr(self.speaker, "last_text", ""),
            "footer": "Space bar: start / pause / continue",
            "quality": None,
            "mood": getattr(self.speaker, "mood", "neutral"),
            "coach_name": self.profile.get("coach_name"),
            "can": self._can(),
            "activity": self.activity if self.stage in (
                "intro", "calibration_offer", "calibrating", "exercise", "rest") else None,
            "difficult_day": self.progress.difficult,
        }
        answer, progress = self.yes_no.state(self._t)
        v["gesture"] = {"hand": self._hand_seen, "answer": answer, "progress": progress}
        if not self.fixed and self.stage not in ("start", "menu"):
            v["footer"] = "Space: pause / continue   M: menu"
        if self.stage in CARD_STAGES:
            v["screen"] = "card"
            v["card_event"] = self._card_event
            v["footer"] = ("Thumbs up / down, or Space = yes, N = no"
                           if self.stage in QUESTION_STAGES else "Thumbs up or space bar: continue")
            if self.stage == "intro":
                v["footer"] = "Thumbs up: start   Space: pause"
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
            v["status"] = "Measuring your hand"
            v["quality"] = self._quality
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
        elif self.stage == "menu":
            v["title"] = "Choose an exercise"
            v["footer"] = "Up/Down: choose  Space: start"
            v["menu"] = [{
                "key": str(i),
                "text": ("All of today's exercises" if item == ALL else
                         "Finish for today" if item == FINISH else
                         "My profile" if item == PROFILE else EXERCISES[item].title),
                "selected": i == self.menu_index,
                "note": "" if item in (ALL, FINISH, PROFILE) or item in self.today else "not today",
            } for i, item in enumerate(self.menu)]
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
        return v
