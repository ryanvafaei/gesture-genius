"""
Think: what the coach remembers about her, and how a session opens.

Only facts stored in profile.json are used (when she last practised, what
she did, her highlight, whether it was a difficult day), never guesses.
A gap is never mentioned as a failure: after a long break the coach just
starts gently (and the targets start lower, see progress.start_target).
"""

from rehab import config
from rehab.events import event
from rehab.progress import days_since


def greeting(profile, today):
    """
    Greeting event. kind is one of
      first            no earlier session (or it could not be read)
      same_day         practised earlier today
      one_day          yesterday (mentions one fact)
      few_days         2-6 days ago (mentions one fact)
      long_gap         LONG_GAP_DAYS or more
      after_difficult  last time was a difficult day
    """
    last = profile.get("last_session")
    days = days_since(last, today)
    fact = None
    if days is None or days < 0:
        kind = "first"
    elif days >= config.LONG_GAP_DAYS:
        kind = "long_gap"
    elif last.get("difficult_day"):
        kind = "after_difficult"
    elif days == 0:
        kind = "same_day"
    else:
        kind = "one_day" if days == 1 else "few_days"
        fact = last.get("highlight") if isinstance(last.get("highlight"), dict) else None
    return event("Greeting", kind=kind, days=days, fact=fact,
                 name=profile.get("name", config.USER_NAME), coach=profile.get("coach_name"))


def needs_name(profile, character):
    """The coach's name is chosen at the first session unless it is fixed in content."""
    return not profile.get("coach_name") and not character.get("name") \
        and bool(character.get("name_options"))


def needs_activities(profile, activities):
    return not profile.get("chosen_activities") and bool(activities.get("activities"))


def activity_for(exercise, profile, activities):
    """The daily activity shown with an exercise: one of her favourites first."""
    linked = activities.get("exercises", {}).get(exercise, {}).get("activities", [])
    chosen = [a for a in linked if a in profile.get("chosen_activities", [])]
    return (chosen or linked or [None])[0]
