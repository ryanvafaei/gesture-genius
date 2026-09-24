"""
Act: events -> words.

Think decides what happened (rehab/events.py); this module decides how to
say it, using the phrase bank in content/phrases.json and the exercise and
activity names in content/activities.json.

Rules
  * only true praise: a phrase is used only for an event that was detected;
  * per rep: a soft chime for every successful rep, and at most one spoken
    phrase: the most important praise event, or else a short generic phrase
    about every PRAISE_EVERY_N_REPS reps;
  * no repeats: the last PHRASE_NO_REPEAT phrases of a category are skipped;
  * praise is short, and is queued after the rep count and before the next
    prompt, so it never talks over an instruction.
"""

import random
from collections import defaultdict, deque
from dataclasses import replace

from rehab import config
from rehab.events import REP_PRAISE, Event
from rehab.exercises.base import FINGER_WORDS, Say, number_word
from rehab.storage import load_content

HAPPY = {"PersonalBest", "TargetRaised", "MilestoneReached", "Improvement", "SteadyHold",
         "GardenGrew", "StartingPoint", "RepCompleted", "BenchmarkMilestone", "LevelRaised",
         "AssessmentImproved"}
ENCOURAGING = {"RecoveredAfterHint", "EffortPraise", "DifficultDayStarted"}


class _Blank(dict):
    """format_map helper: unknown placeholders stay visible instead of crashing."""

    def __missing__(self, key):
        return "{" + key + "}"


def join_words(items):
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class Feedback:

    def __init__(self, phrases=None, activities=None, rng=None, name=config.USER_NAME,
                 coach=None):
        self.phrases = phrases if phrases is not None else load_content("phrases")
        self.activities = activities if activities is not None else load_content("activities")
        self.rng = rng or random.Random()
        self.name = name
        self.coach = coach
        self.recent = defaultdict(lambda: deque(maxlen=config.PHRASE_NO_REPEAT))
        self._reps_since_phrase = 0

    # --- phrase bank ------------------------------------------------------------

    def has(self, category):
        return bool(self.phrases.get(category))

    def pick(self, category, **values):
        """A variant of the category, not one of the last few used, filled in. None if missing."""
        variants = [v for v in self.phrases.get(category, []) if isinstance(v, str)]
        if not variants:
            return None
        recent = self.recent[category]
        fresh = [v for v in variants if v not in recent] or variants
        text = self.rng.choice(fresh)
        recent.append(text)
        values.setdefault("name", self.name)
        values.setdefault("coach", self.coach or "")
        return text.format_map(_Blank(values))

    def first(self, *categories, **values):
        for c in categories:
            if self.has(c):
                return self.pick(c, **values)
        return None

    # --- names from content ---------------------------------------------------------

    def exercise_info(self, exercise):
        return self.activities.get("exercises", {}).get(exercise, {})

    def activity_info(self, activity):
        return self.activities.get("activities", {}).get(activity, {})

    def activity_name(self, activity):
        return self.activity_info(activity).get("name", activity or "")

    def fact(self, highlight):
        """A remembered highlight as a sentence part ("you reached a new best with your thumb")."""
        if not isinstance(highlight, dict) or not highlight.get("type"):
            return None
        kind = highlight["type"]
        if kind == "PersonalBest" and highlight.get("level") == "today":
            return None          # a best of that day only is not worth recalling
        info = self.exercise_info(highlight.get("exercise"))
        activity = self.activity_info(highlight.get("activity"))
        metric = highlight.get("metric")
        return self.first(f"Fact.{kind}.{metric}", f"Fact.{kind}",
                          part=info.get("part", "hand"), what=info.get("what", "exercise"),
                          count=highlight.get("count", ""), deg=highlight.get("deg", ""),
                          goal=self.goal(highlight.get("task")),
                          practice_for=activity.get("practice_for",
                                                    self.activity_name(highlight.get("activity"))))

    def goal(self, task):
        """A daily task from the benchmarks (Gates 2016) as her goal: "drink from your cup of tea"."""
        if not task:
            return ""
        return self.first(f"Goal.{task}") or task.lower()

    # --- events -> Say --------------------------------------------------------------

    @staticmethod
    def _once(ev, key, make):
        """
        Words for one event, worked out once: the phrase bank picks at random,
        so the card, summary and garden screens must reuse what she heard.
        """
        memo = ev.__dict__.setdefault("_worded", {})
        if key not in memo:
            memo[key] = make()
        return memo[key]

    def words(self, events):
        """Say messages for one event, or for the events of one rep (a list)."""
        if isinstance(events, Event):
            if events.type != "RepCompleted":
                said = self._once(events, "words", lambda: self._words([events]))
                return [replace(m, seq=None, queued_at=None) for m in said]
            events = [events]
        return self._words(events)

    def _words(self, events):
        events = [e for e in events if e is not None]
        if any(e.type == "RepCompleted" for e in events):
            return self._rep_words(events)
        out = []
        for e in events:
            out += self._event_words(e)
        return out

    def _say(self, text, ev, kind="instruction"):
        if not text:
            return []
        mood = "happy" if ev.type in HAPPY else "encouraging" if ev.type in ENCOURAGING else None
        return [Say(text, kind, priority=ev.priority if kind == "praise" else None, mood=mood)]

    def _rep_words(self, events):
        done = next(e for e in events if e.type == "RepCompleted")
        out = []
        if done.get("success"):
            out.append(Say("", "chime"))
        praise = sorted((e for e in events if e.type in REP_PRAISE), key=lambda e: e.priority)
        spoken = False
        for ev in praise:
            text = self._praise_text(ev)
            if text:
                self._reps_since_phrase = 0
                out += self._say(text, ev, "praise")
                spoken = True
                break
        if not spoken:
            self._reps_since_phrase += 1
        if (not spoken and done.get("success")
                and self._reps_since_phrase >= config.PRAISE_EVERY_N_REPS):
            self._reps_since_phrase = 0
            generic = "EffortPraise" if done.get("difficult") else "RepCompleted"
            out += self._say(self.pick(generic), Event(generic), "praise")
        # anything else in the batch (e.g. a milestone) as usual
        for ev in events:
            if ev.type not in REP_PRAISE and ev.type != "RepCompleted":
                out += self._event_words(ev)
        return out

    def _praise_text(self, ev):
        info = self.exercise_info(ev.get("exercise"))
        what = info.get("what", "movement")
        t = ev.type
        if t == "PersonalBest":
            return self.pick(f"PersonalBest.{ev.get('metric')}.{ev.get('level')}", what=what)
        if t == "Improvement":
            finger = ev.get("finger")
            if finger:
                return self.pick("Improvement.finger", finger=FINGER_WORDS.get(finger, finger))
            return self.first(f"Improvement.{ev.get('exercise')}", "Improvement")
        if t == "MilestoneReached":
            return self._milestone(ev)
        if t == "BenchmarkMilestone":
            if ev.get("kind") == "fma":
                return self.first(f"BenchmarkMilestone.fma.{ev.get('exercise')}",
                                  "BenchmarkMilestone.fma", what=what)
            return self.first(f"BenchmarkMilestone.{ev.get('task')}", "BenchmarkMilestone",
                              what=what, goal=self.goal(ev.get("task")))
        if t == "AssessmentImproved":
            return self.pick(t, what=what, deg=ev.get("deg"))
        if t == "LevelRaised":
            return self.first("LevelRaised", "TargetRaised", what=what)
        return self.pick(t, what=what)

    def _milestone(self, ev):
        activity = ev.get("activity")
        return self.pick("MilestoneReached", count=ev.get("count"),
                         practice_for=self.activity_info(activity).get(
                             "practice_for", self.activity_name(activity)))

    def _event_words(self, ev):
        t = ev.type
        g = ev.get
        if t in REP_PRAISE:
            # news at the end of a set or exercise (a best, a raised target, a
            # milestone) is not dropped when it waits: only a rep's praise is
            return self._say(self._praise_text(ev), ev)
        if t == "Greeting":
            return self._greeting(ev)
        if t == "NameQuestion":
            text = self.pick("NameQuestion.first" if g("first") else "NameQuestion.next",
                             option=g("option"))
            return self._say(text, ev) + self._say(self.pick("YesNoHint"), ev)
        if t == "ActivityQuestion":
            out = self._say(self.pick("ActivityQuestion.first"), ev) if g("first") else []
            card = self.activity_info(g("activity")).get("card", g("activity"))
            return out + self._say(self.pick("ActivityQuestion", activity=card), ev)
        if t == "ActivitiesChosen":
            names = join_words([self.activity_name(a) for a in g("activities", [])])
            return self._say(self.pick("ActivitiesChosen", activities=names), ev) if names else []
        if t == "CheckIn":
            return self._say(self.pick("CheckIn"), ev) + self._say(self.pick("CheckInHint"), ev)
        if t == "CheckInAnswer":
            return self._say(self.pick("CheckInAnswer.good"), ev) if g("good") else []
        if t == "DifficultDayStarted":
            return self._say(self.first(f"DifficultDayStarted.{g('reason')}",
                                        "DifficultDayStarted"), ev)
        if t == "TodayPlan":
            n = g("count", 0)
            cat = "TodayPlan.one" if n == 1 else "TodayPlan"
            return self._say(self.pick(cat, count_word=number_word(n)), ev)
        if t == "ExerciseIntro":
            link = self.exercise_info(g("exercise")).get("links", {}).get(g("activity"), {})
            return self._say(link.get("intro"), ev)
        if t == "SetCompleted":
            return self._say(self.pick("SetCompleted", set_word=number_word(g("set_no")),
                                       sets_word=number_word(g("sets"))), ev)
        if t == "Summary":
            return [s for text in self.summary_lines(ev) for s in self._say(text, ev)]
        if t == "PlantQuestion":
            return self._say(self.pick("PlantQuestion", first=g("first"), second=g("second")), ev)
        if t == "GardenGrew":
            return self._say(self.garden_line(ev), ev)
        if t == "Goodbye":
            out = self._say(self.pick("Goodbye"), ev)
            nxt = (self.pick("Goodbye.next", plant=g("plant")) if g("plant")
                   else self.pick("Goodbye.next_no_garden"))
            return out + self._say(nxt, ev)
        if t in ("ExerciseDone", "StartingPoint", "ProfileOverview", "DeleteProfileQuestion",
                 "ProfileKept", "ProfileDeleted", "NotTestable"):
            return self._say(self.pick(t), ev)
        # TargetLowered, RangeGrew, NameChosen ...: said by nobody (lowering is silent)
        return []

    def _greeting(self, ev):
        kind = ev.get("kind")
        self.coach = ev.get("coach") or self.coach
        if kind == "first":
            if ev.get("named_now"):
                cat = "Greeting.first_named"
            else:
                cat = "Greeting.first" if self.coach else "Greeting.first_no_coach"
            return self._say(self.pick(cat), ev)
        fact = self.fact(ev.get("fact")) if kind in ("one_day", "few_days") else None
        if fact:
            return self._say(self.pick(f"Greeting.{kind}_fact", fact=fact), ev)
        return self._say(self.pick(f"Greeting.{kind}"), ev)

    def card(self, ev):
        """What a full-screen card shows for an event: title, message, icon, answers."""
        if ev is None:
            return {}
        t, g = ev.type, ev.get
        yes_no = (self.pick("Card.yes"), self.pick("Card.no"))
        if t == "NameQuestion":
            return {"title": self.pick("Card.NameQuestion", option=g("option")),
                    "message": self.pick("YesNoHint"), "yes_no": True, "answer_labels": yes_no}
        if t == "ActivityQuestion":
            info = self.activity_info(g("activity"))
            return {"title": info.get("card", g("activity")), "icon": info.get("icon"),
                    "message": self.pick("Card.ActivityQuestion"), "yes_no": True,
                    "answer_labels": yes_no}
        if t == "CheckIn":
            return {"title": self.pick("CheckIn"), "message": "", "yes_no": True,
                    "answer_labels": (self.pick("Card.good"), self.pick("Card.not_good"))}
        if t == "PlantQuestion":
            first, second = g("first"), g("second")
            return {"title": self.pick("Card.PlantQuestion"),
                    "message": f"{first.capitalize()} or {second}?", "yes_no": True,
                    "answer_labels": (self.pick("Card.plant_first", first=first),
                                      self.pick("Card.plant_second", second=second))}
        if t == "DeleteProfileQuestion":
            return {"title": self.pick("Card.DeleteProfileQuestion"),
                    "message": self.pick("Card.DeleteProfileMessage"), "yes_no": True,
                    "answer_labels": (self.pick("Card.delete_yes"), self.pick("Card.delete_no"))}
        if t == "ExerciseIntro":
            link = self.exercise_info(g("exercise")).get("links", {}).get(g("activity"), {})
            return {"title": g("title", ""), "message": link.get("sentence", ""),
                    "icon": self.activity_info(g("activity")).get("icon")}
        texts = [m.text for m in self.words(ev) if m.text]
        if len(texts) == 1 and ". " in texts[0]:
            # "Hello Eleanor. Last time you ..." -> short title, the rest below
            head, rest = texts[0].split(". ", 1)
            texts = [head + ".", rest]
        return {"title": texts[0] if texts else "", "message": " ".join(texts[1:])}

    def summary_lines(self, ev):
        """What the summary says: one highlight, a comparison, the activities."""
        return list(self._once(ev, "summary", lambda: self._summary_lines(ev)))

    def _summary_lines(self, ev):
        lines = []
        if ev.get("difficult"):
            lines.append(self.pick("Summary.difficult"))
        else:
            fact = self.fact(ev.get("highlight"))
            if fact:
                lines.append(self.pick("Summary.highlight", fact=fact))
            lines += [m for m in ev.get("comparisons", []) if m][:1]
        names = join_words([self.activity_name(a) for a in ev.get("activities", [])])
        if names:
            lines.append(self.pick("Summary.activities", activities=names))
        if not lines:
            lines.append(self.pick("Summary.nothing"))
        return [line for line in lines if line]

    def garden_line(self, ev):
        return self._once(ev, "garden", lambda: self._garden_line(ev))

    def _garden_line(self, ev):
        plant = ev.get("plant")
        if ev.get("new_season"):
            text = self.pick("GardenGrew.new_season", plant=plant)
        else:
            text = self.pick(f"GardenGrew.{ev.get('stage')}", plant=plant)
        extras = []
        if ev.get("bee"):
            extras.append(self.pick("GardenGrew.bee"))
        if ev.get("butterflies"):
            extras.append(self.pick("GardenGrew.butterfly"))
        return " ".join(t for t in [text] + extras if t)
