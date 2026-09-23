"""
Events: what happened, decided by Think, worded by Act.

Think (Coach, SessionManager, progress) never writes sentences for
motivation; it produces events such as
    Event("PersonalBest", exercise="grip_release", metric="range", level="week")
and Act (feedback.py) turns them into speech, text and sounds from the phrase
files in content/. So wording can change without touching logic, and logic
can be tested without a camera or a speaker.

The priority decides which event is spoken when several happen at once
(1 = most important).
"""

from dataclasses import dataclass, field

PRIORITY = {
    # 1: safety and setup
    "HandLost": 1, "WrongHand": 1,
    # 2: instructions and the session flow
    "StepInstruction": 2, "ExerciseIntro": 2,
    "NameQuestion": 2, "NameChosen": 2, "Greeting": 2, "ActivityQuestion": 2,
    "ActivitiesChosen": 2, "CheckIn": 2, "CheckInAnswer": 2, "DifficultDayStarted": 2,
    "TodayPlan": 2, "ExerciseDone": 2, "StartingPoint": 2, "Summary": 2,
    "PlantQuestion": 2, "GardenGrew": 2, "Goodbye": 2, "SetCompleted": 2,
    # 3: bests and milestones
    "PersonalBest": 3, "TargetRaised": 3, "MilestoneReached": 3,
    # 4: quality of this rep
    "Improvement": 4, "SteadyHold": 4, "RecoveredAfterHint": 4,
    # 5: routine
    "RepCompleted": 5, "TargetLowered": 5, "RangeGrew": 5,
    # 6: effort (difficult day)
    "EffortPraise": 6,
}

# Events about how a rep went. The rest belong to the session flow.
REP_PRAISE = {"PersonalBest", "TargetRaised", "MilestoneReached", "Improvement",
              "SteadyHold", "RecoveredAfterHint", "EffortPraise"}

# Order of "how good is this news" for choosing one highlight of a session.
HIGHLIGHT_ORDER = ("PersonalBest", "MilestoneReached", "TargetRaised",
                   "SteadyHold", "Improvement", "RecoveredAfterHint")
PB_LEVEL_ORDER = ("all_time", "week", "today")


@dataclass
class Event:
    type: str
    details: dict = field(default_factory=dict)

    @property
    def priority(self):
        return PRIORITY.get(self.type, 2)

    def get(self, key, default=None):
        return self.details.get(key, default)

    def as_dict(self):
        """For storing a highlight in profile.json."""
        return {"type": self.type, **self.details}


def event(type_, **details):
    return Event(type_, details)


def highlight_rank(ev):
    """Lower = better news. Used to pick one highlight of the session."""
    t = ev.type if isinstance(ev, Event) else ev.get("type")
    get = ev.get
    rank = HIGHLIGHT_ORDER.index(t) if t in HIGHLIGHT_ORDER else len(HIGHLIGHT_ORDER)
    level = get("level")
    sub = PB_LEVEL_ORDER.index(level) if level in PB_LEVEL_ORDER else len(PB_LEVEL_ORDER)
    return rank, sub
