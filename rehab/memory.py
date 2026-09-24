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
from rehab.feedback import join_words
from rehab.storage import valid_date


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


def long_date(d):
    """date -> "24 September 2026" (same on every OS)."""
    return f"{d.day} {d.strftime('%B %Y')}"


def profile_overview(profile, garden, sessions, activities):
    """
    Everything the coach remembers about her, as (label, value) rows for the
    profile screen: her answers to the first-time questions first, then what
    was saved since. sessions: the rows of sessions.csv.
    """
    names = activities.get("activities", {})

    def activity(a):
        return names.get(a, {}).get("name", a)

    chosen = [activity(a) for a in profile.get("chosen_activities", [])]
    started = valid_date(profile.get("created"))
    last = profile.get("last_session") or {}
    last_date = valid_date(last.get("date"))
    count = len(sessions)
    if count:
        sessions_text = str(count) + (f", the last on {long_date(last_date)}" if last_date else "")
    else:
        sessions_text = "None yet"
    calibrated = profile.get("calibration", {})
    measured = sum(1 for n in calibrated if n in config.SESSION_ORDER)
    arms = sum(1 for n in calibrated if n in config.ARM_EXERCISES)
    assessed = [a[-1].get("date") for a in profile.get("assessments", {}).values()
                if isinstance(a, list) and a and isinstance(a[-1], dict)]
    last_assessed = max((valid_date(d) for d in assessed if valid_date(d)), default=None)
    bests = sum(len(m) for m in profile.get("personal_bests", {}).values() if isinstance(m, dict))
    practice = sorted(((activity(a), int(n)) for a, n in profile.get("practice", {}).items()
                       if isinstance(n, (int, float)) and n > 0), key=lambda p: -p[1])
    plots = garden.get("plots", [])
    garden_text = ("No plants yet" if not plots else
                   f"{len(plots)} plant{'s' if len(plots) != 1 else ''}"
                   f", flowering: {sum(1 for p in plots if p.get('stage') == config.GARDEN_STAGES - 1)}")
    if garden.get("season", 1) > 1:
        garden_text += f", season {garden['season']}"
    return [
        ("Name", profile.get("name") or config.USER_NAME),
        ("Hand we train", profile.get("affected_hand") or config.AFFECTED_HAND),
        ("Your coach", profile.get("coach_name") or "Not chosen yet"),
        ("Favourite activities", join_words(chosen) if chosen else "Not chosen yet"),
        ("Profile started", long_date(started) if started else "Unknown"),
        ("Sessions", sessions_text),
        ("Hand measured", f"For {measured} of {len(config.SESSION_ORDER)} exercises"
                          if measured else "Not yet"),
        ("Arms measured", (f"For {arms} exercise{'s' if arms != 1 else ''}"
                           + (f", last on {long_date(last_assessed)}" if last_assessed else ""))
                          if arms else "Not yet"),
        ("Personal bests", str(bests) if bests else "None yet"),
        ("Practice", ", ".join(f"{name}: {n} rep{'s' if n != 1 else ''}" for name, n in practice[:3])
                     if practice else "None yet"),
        ("Garden", garden_text),
    ]
