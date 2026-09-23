"""
Think: the Coach runs one exercise, the SessionManager runs the whole session.

Coach
  * checks tracking quality first (hand visible, correct hand, close enough,
    palm facing the camera when needed) and asks her to adjust, calmly and
    not too often;
  * feeds good frames to the active exercise and passes its messages to the
    speaker;
  * logs every repetition and notices fatigue.

SessionManager
  greeting -> menu (one exercise, or all of today's) -> for each exercise:
  introduction -> calibration check (calibrate, or offer a quick
  recalibration) -> sets with rests -> summary compared with earlier
  sessions -> back to the menu.
  When the exercises are given up front (--exercise) there is no menu.

One key does everything: space starts, pauses and continues. In the menu
a number picks an exercise and up/down move the highlight; m goes back to
the menu from anywhere.
"""

from rehab import config
from rehab.calibration import CalibrationRoutine, is_stale
from rehab.exercises import EXERCISES, create
from rehab.exercises.base import Say, TwoPhaseExercise, number_word
from rehab.storage import (calibrations, progress_message, should_do_today)

QUALITY_TEXT = {
    "no_hand": "Please show your {hand} hand to the camera.",
    "wrong_hand": "Please use your {hand} hand.",
    "too_far": "Please move your hand a little closer.",
    "palm_away": "Please turn your palm towards the camera.",
}

INTRO_S_PER_SENTENCE = 2.5

ALL = "all"


class Coach:

    def __init__(self, exercise, speaker, log, hand=config.AFFECTED_HAND):
        self.exercise = exercise
        self.speaker = speaker
        self.log = log
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

    def update(self, f, now):
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
                    self.speaker.say(problem_say)
            return

        self._problem = None
        self.quality = None
        self._interrupted = False
        self.speaker.say_all(self.exercise.update(f, now))
        for rec in self.exercise.new_reps:
            if self.log is not None:
                self.log.log_rep(rec)
        self.exercise.new_reps.clear()


class SessionManager:

    def __init__(self, speaker, profile, log, exercises=None, save_profile=None):
        self.speaker = speaker
        self.profile = profile
        self.log = log
        self.save_profile = save_profile or (lambda p: None)
        # today's plan, the menu's "all" item
        self.today = [n for n in config.SESSION_ORDER if should_do_today(n, log)]
        # chosen explicitly: no day schedule and no menu
        self.fixed = bool(exercises)
        self.plan = list(exercises) if exercises else []
        self.menu = [ALL] + list(config.SESSION_ORDER)
        self.menu_index = 0
        self.index = -1
        self.stage = "start"
        self.paused = False
        self.done = False
        self.summaries = []
        self.summary_lines = []
        self.exercise = None
        self.coach = None
        self.calibration = None
        self.set_no = 0
        self._t = 0.0
        self._stage_t = 0.0
        self._rest_s = 0
        self._after_rest = None
        self._fatigue_offered = False
        self._instruction = ""
        self._quality = None

    # --- helpers --------------------------------------------------------------

    @property
    def name(self):
        return self.plan[self.index] if 0 <= self.index < len(self.plan) else None

    def _enter(self, stage, now):
        self.stage = stage
        self._stage_t = now

    def _say(self, *texts):
        for t in texts:
            self.speaker.say(t if isinstance(t, Say) else Say(t))

    def _build_exercise(self):
        kwargs = {}
        if self.name == "thumb_opposition" and self.profile["levels"].get(self.name):
            kwargs["level"] = self.profile["levels"][self.name]
        self.exercise = create(self.name, calibrations(self.profile),
                               self.profile.get("thresholds"), **kwargs)
        self.coach = Coach(self.exercise, self.speaker, self.log,
                           self.profile.get("affected_hand", config.AFFECTED_HAND))

    # --- keys -------------------------------------------------------------------

    def on_key(self, key, now):
        """key: " ", "up", "down", "m" or a digit."""
        if self.stage == "menu":
            self._menu_key(key, now)
            return
        if key == "m" and not self.fixed and self.stage not in ("start", "greeting"):
            self._back_to_menu(now)
            return
        if key != " ":
            return
        if self.stage == "greeting":
            self._next_exercise(now)
        elif self.stage == "calibration_offer":
            self._start_calibration(now)
        elif self.stage == "rest":
            self._end_rest(now)
        elif self.stage == "summary":
            if self.fixed:
                self.done = True
            else:
                self._open_menu(now)
        elif self.stage in ("exercise", "calibrating", "intro"):
            self.paused = not self.paused
            if self.paused:
                self._say("Paused. Press the space bar when you are ready.")
            else:
                self._say("Let's continue.")
                if self.coach:
                    self.coach.interrupt(now)
                if self.calibration:
                    self.calibration.interrupt(now)

    # --- main update --------------------------------------------------------------

    def update(self, f, now):
        self._t = now
        if self.paused or self.done:
            return
        stage = self.stage
        if stage == "start":
            name = config.USER_NAME
            if self.fixed:
                self._say(f"Hello {name}. Let's do your hand exercises together.",
                          "Press the space bar when you are ready.")
                self._instruction = "Press the space bar to start"
                self._enter("greeting", now)
            else:
                self._say(f"Hello {name}. Which exercise would you like to do?")
                self._open_menu(now)
        elif stage == "intro":
            n = len(self.exercise.instructions) + 1
            if now - self._stage_t >= INTRO_S_PER_SENTENCE * n and not self.speaker.busy:
                self._calibration_check(now)
        elif stage == "calibration_offer":
            if now - self._stage_t >= config.RECALIBRATION_OFFER_S:
                self._start_exercise(now)
        elif stage == "calibrating":
            step = self.calibration.step
            need_palm = step.need_palm_facing if step else False
            problem = f.quality_problem(need_palm)
            self.speaker.say_all(self.calibration.update(f, now, quality_ok=problem is None))
            self._quality = (QUALITY_TEXT[problem].format(hand=self.coach.hand)
                             if problem else None)
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
                self._say("You've worked hard. Let's rest for a minute.")
                self._rest(now, config.FATIGUE_REST_S, then="resume")
            elif self.exercise.set_done:
                self._set_finished(now)
        elif stage == "rest":
            if now - self._stage_t >= self._rest_s:
                self._end_rest(now)

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

    def _choose(self, item, now):
        self.plan = list(self.today) if item == ALL else [item]
        self.index = -1
        self.summaries = []
        self.exercise = None
        self.coach = None
        self._next_exercise(now)

    def _back_to_menu(self, now):
        """Leave whatever is running; keep what was done."""
        self.stop(now)
        self.paused = False
        self.calibration = None
        self._quality = None
        self._after_rest = None
        self._say("Let's choose another exercise.")
        self._open_menu(now)

    # --- flow -------------------------------------------------------------------------

    def _next_exercise(self, now):
        self.index += 1
        if self.index >= len(self.plan):
            self._finish(now)
            return
        self._build_exercise()
        self.set_no = 0
        self._fatigue_offered = False
        cls = EXERCISES[self.name]
        self._say(cls.title + ".", *cls.instructions)
        self._instruction = cls.instructions[-1] if cls.instructions else cls.title
        self._enter("intro", now)

    def _calibration_check(self, now):
        entry = self.profile["calibration"].get(self.name)
        if is_stale(entry):
            self._start_calibration(now)
        else:
            self._say("Press the space bar if you'd like to measure your hand again.")
            self._instruction = "Space bar: measure my hand again"
            self._enter("calibration_offer", now)

    def _start_calibration(self, now):
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
        sets = self.exercise.sets
        if self.set_no < sets:
            self._say(f"Well done. That was set {number_word(self.set_no)} of {number_word(sets)}.",
                      f"Let's rest for {config.REST_BETWEEN_SETS_S} seconds.")
            self._rest(now, config.REST_BETWEEN_SETS_S, then="next_set")
        else:
            self._exercise_finished(now)

    def _exercise_finished(self, now):
        self._record_summary()
        self._say("Well done. You finished this exercise.")
        if self.index + 1 < len(self.plan):
            self._rest(now, config.REST_BETWEEN_EXERCISES_S, then="next_exercise")
        else:
            self._finish(now)

    def _record_summary(self):
        ex = self.exercise
        if ex is None or not ex.reps:
            return
        row = self.log.summarize(ex.name, ex.reps, self.set_no, ex.sets * ex.reps_per_set)
        earlier, when = self.log.comparison(ex.name)
        message = progress_message(type(ex), row, earlier, when)
        self.log.save_summary(row)
        self.summaries.append((ex.name, row, message))
        self._progression(ex)

    def _progression(self, ex):
        if ex.name == "thumb_opposition":
            self.profile["levels"][ex.name] = ex.level
        if config.AUTO_PROGRESSION and isinstance(ex, TwoPhaseExercise):
            high = ex.params["high"]
            done = len(ex.reps) >= ex.sets * ex.reps_per_set
            good = [r for r in ex.reps if r.range_high >= high + config.PROGRESSION_MARGIN]
            if done and len(good) >= config.PROGRESSION_GOOD_SHARE * len(ex.reps):
                new_high = min(config.PROGRESSION_MAX_HIGH, high + config.PROGRESSION_STEP)
                if new_high > high:
                    self.profile["thresholds"].setdefault(ex.name, {})["high"] = round(new_high, 3)
        self.save_profile(self.profile)

    def _rest(self, now, seconds, then):
        self._rest_s = seconds
        self._after_rest = then
        self._instruction = "Rest your hand. Space bar: continue"
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
            self.coach.interrupt(now)
            self.speaker.say_all(self.exercise.first_messages())
        else:
            self._next_exercise(now)

    def _finish(self, now):
        lines = []
        for name, row, message in self.summaries:
            lines.append(f"{EXERCISES[name].title}: {row['reps_done']} done")
            if message:
                lines.append(f"  {message}")
                self._say(message)
        self.summary_lines = lines or ["No exercises done today."]
        if self.fixed:
            self._say(f"That's all for today. Well done, {config.USER_NAME}.")
            self._instruction = "Well done! Press the space bar to finish"
        else:
            self._say(f"Well done, {config.USER_NAME}.",
                      "Press the space bar to choose another exercise.")
            self._instruction = "Space bar: choose another exercise"
        self._enter("summary", now)

    def stop(self, now):
        """Quit early: keep what was done."""
        if self.stage in ("exercise", "rest") and self.exercise and self.exercise.reps:
            if not any(s[0] == self.exercise.name for s in self.summaries):
                self._record_summary()
        self.save_profile(self.profile)

    # --- what Act draws -----------------------------------------------------------------

    def view(self):
        v = {
            "stage": self.stage,
            "paused": self.paused,
            "title": EXERCISES[self.name].title if self.name else "Hand exercises",
            "instruction": self._instruction,
            "subtitle": getattr(self.speaker, "last_text", ""),
            "footer": "Space bar: start / pause / continue",
            "quality": None,
        }
        if not self.fixed and self.stage not in ("start", "menu"):
            v["footer"] = "Space: pause / continue   M: menu"
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
            v["title"] = "Today's session"
            v["summary_lines"] = self.summary_lines
            if not self.fixed:
                v["footer"] = "Space: menu   Q: finish"
        elif self.stage == "menu":
            v["title"] = "Choose an exercise"
            v["footer"] = "Up/Down: choose  Space: start"
            v["menu"] = [{
                "key": str(i),
                "text": "All of today's exercises" if item == ALL else EXERCISES[item].title,
                "selected": i == self.menu_index,
                "note": "" if item == ALL or item in self.today else "not today",
            } for i, item in enumerate(self.menu)]
        return v
