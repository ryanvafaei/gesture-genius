"""
Think: her progress over time.

  * personal bests (today, this week, all time), from the median of each
    hold rather than the best single frame, and only above a minimum
    improvement, so landmark jitter cannot make a fake best;
  * adaptive targets: once per set, from the share of successful reps;
    a warm-up start below last time; floor and ceiling; the calibrated
    range grows when she goes beyond it;
  * difficult day mode: switched on by her check-in, a low first set, or
    tiredness during the session, and then kept for the whole session;
  * the praise-worthy moments of a rep (a finger opened more than usual, a
    very steady hold, recovering after a hint);
  * practice milestones per daily activity.

It decides what happened, never how to say it: everything comes out as
events (rehab/events.py). It changes only the profile dict it is given and
reads the history through the SessionLog.
"""

from collections import Counter
from datetime import date

import numpy as np

from rehab import config
from rehab.events import HIGHLIGHT_ORDER, event, highlight_rank
from rehab.storage import to_float, valid_date


def finite(v):
    try:
        return v is not None and bool(np.isfinite(v))
    except TypeError:
        return False


def clamp_target(t):
    return round(float(np.clip(t, config.TARGET_FLOOR, config.TARGET_CEILING)), 3)


# ---------------------------------------------------------------------------
# Personal bests
# ---------------------------------------------------------------------------

# metric -> (higher is better, minimum relative improvement)
PB_METRICS = {
    "range": (True, config.PB_MIN_IMPROVEMENT),          # median during the target hold
    "steadiness": (False, config.PB_MIN_STEADINESS_IMPROVEMENT),   # mean hold wobble
    "clean_reps": (True, config.PB_MIN_IMPROVEMENT),     # reps without any hint
    "reps": (True, config.PB_MIN_IMPROVEMENT),           # reps in one session (all exercises)
}
# Speed is deliberately not a best: controlled, slower movement is the goal.

SESSION = "session"          # personal_bests key for whole-session metrics


class PersonalBests:
    """
    profile["personal_bests"][exercise][metric] =
        {"value": best ever, "date": when, "daily": {date: best that day}}
    """

    def __init__(self, profile, today):
        self.store = profile.setdefault("personal_bests", {})
        self.today = today
        # Only exercises that already had bests when the session started
        # announce new ones: the first session is her starting point.
        self.baseline = {ex for ex, metrics in self.store.items() if metrics}

    def has_baseline(self, exercise):
        return exercise in self.baseline

    def _beats(self, metric, value, ref):
        higher, rel = PB_METRICS[metric]
        if not finite(ref):
            return False
        if higher:
            return value > ref and value >= ref + abs(ref) * rel
        return value < ref and value <= ref - abs(ref) * rel

    def _best(self, metric, values):
        higher, _ = PB_METRICS[metric]
        values = [v for v in values if finite(v)]
        if not values:
            return None
        return max(values) if higher else min(values)

    def level(self, exercise, metric, value):
        """"all_time", "week", "today" or None: the highest level this value beats."""
        entry = self.store.get(exercise, {}).get(metric)
        if not entry or not finite(value):
            return None
        if self._beats(metric, value, entry.get("value")):
            return "all_time"
        today_key = self.today.isoformat()
        daily = entry.get("daily", {})
        week = [v for d, v in daily.items()
                if valid_date(d) and 0 <= (self.today - valid_date(d)).days < config.PB_WEEK_DAYS]
        week_best = self._best(metric, week)
        if week_best is not None and self._beats(metric, value, week_best):
            return "week"
        if today_key in daily and self._beats(metric, value, daily[today_key]):
            return "today"
        return None

    def record(self, exercise, metric, value):
        """Store the value (also on difficult days: a best is a best)."""
        if not finite(value):
            return
        value = round(float(value), 4)
        entry = self.store.setdefault(exercise, {}).setdefault(metric, {"daily": {}})
        daily = entry.setdefault("daily", {})
        key = self.today.isoformat()
        daily[key] = self._best(metric, [daily.get(key), value])
        if entry.get("value") is None or self._best(metric, [entry["value"], value]) == value:
            if entry.get("value") != value:
                entry["value"], entry["date"] = value, key
        for d in list(daily):
            day = valid_date(d)
            if day is None or (self.today - day).days >= config.PB_KEEP_DAYS:
                del daily[d]

    def check(self, exercise, metric, value):
        """Level reached (or None), then record the value."""
        level = self.level(exercise, metric, value)
        self.record(exercise, metric, value)
        return level


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def next_target(target, success_rate):
    """(new target, "raised" | "lowered" | None) for the next set."""
    if not finite(success_rate):
        return target, None
    if success_rate >= config.TARGET_RAISE_AT:
        new = clamp_target(target + config.TARGET_STEP)
    elif success_rate < config.TARGET_LOWER_AT:
        new = clamp_target(target - config.TARGET_STEP)
    else:
        return target, None
    if new > target:
        return new, "raised"
    if new < target:
        return new, "lowered"
    return target, None


def grow_range(steps, range_steps, factor):
    """
    Calibration with the upper end moved so that `factor` (e.g. 1.1 = she
    reached 110% of her range) becomes the new 100%. Works per key, also
    for ranges that run downwards (e.g. thumb flexion angle).
    """
    lo_name, hi_name = range_steps
    lo, hi = steps.get(lo_name, {}), steps.get(hi_name, {})
    grown = {k: (lo[k] + (v - lo[k]) * factor if k in lo else v) for k, v in hi.items()}
    return {**steps, hi_name: grown}


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------

class SessionProgress:
    """
    Everything about progress in one session. The SessionManager calls
      start_target()      when an exercise is built
      rep_events()        after every rep (via the Coach)
      set_finished()      after every set
      exercise_finished() after every exercise
      end_session()       once, at the end
    """

    def __init__(self, profile, log, today=None):
        self.profile = profile
        self.log = log
        self.today = today or log.today
        self.bests = PersonalBests(profile, self.today)
        self.difficult = False
        self.difficult_reason = ""
        self.check_in = ""               # "good", "not_good" or "" (no answer)
        self.highlights = []             # events worth remembering
        self.final_targets = {}
        self.completed = []              # exercises finished with all their sets
        self.total_reps = 0
        self._pb_per_set = Counter()     # (exercise, set) -> bests announced
        self._low_success_sets = 0
        self._starting_point = set()
        self._reduced = set()            # exercises whose target was made easier today

    # --- difficult day --------------------------------------------------------

    def switch_difficult(self, reason):
        """Switch difficult day mode on (once). Returns the event to announce, or None."""
        if self.difficult:
            return None
        self.difficult = True
        self.difficult_reason = reason
        return event("DifficultDayStarted", reason=reason)

    def answer_check_in(self, good):
        """good: True, False, or None (no answer)."""
        self.check_in = "" if good is None else ("good" if good else "not_good")
        if good is False:
            return self.switch_difficult("check_in")
        return None

    def baseline(self, exercise):
        """Median of the mean hold value of her last normal sessions, or None."""
        rows = self.log.normal_history(exercise)[-config.DIFFICULT_BASELINE_SESSIONS:]
        values = [to_float(r.get("mean_hold_raw")) for r in rows]
        values = [v for v in values if finite(v)]
        if len(values) < config.DIFFICULT_MIN_BASELINE_SESSIONS:
            return None
        return float(np.median(values))

    # --- targets --------------------------------------------------------------

    def start_target(self, exercise):
        """Target for the first set today, or None for exercises without one."""
        default = config.EXERCISES.get(exercise, {}).get("high")
        if default is None:
            return None
        stored = self.profile.setdefault("targets", {}).get(exercise) or {}
        target = to_float(stored.get("high"))
        if not finite(target):
            target = default
        elif config.AUTO_PROGRESSION:
            # warm-up: a little below where she ended last time
            last = valid_date(stored.get("date"))
            gap = (self.today - last).days if last else 0
            target -= config.TARGET_STEP * (2 if gap >= config.LONG_GAP_DAYS else 1)
        if self.difficult:
            target *= config.DIFFICULT_TARGET_FACTOR
            self._reduced.add(exercise)
        return clamp_target(target)

    def sets_for(self, exercise, sets):
        if self.difficult:
            return max(1, sets - config.DIFFICULT_FEWER_SETS)
        return sets

    def skip_today(self, exercise):
        return self.difficult and exercise in config.DIFFICULT_SKIP

    # --- rep ------------------------------------------------------------------

    def rep_events(self, ex, rec):
        """Events for one finished rep, most important first."""
        self.total_reps += 1
        events = [event("RepCompleted", exercise=ex.name, set_no=rec.set_no, rep_no=rec.rep_no,
                        success=rec.success, difficult=self.difficult)]

        # personal best: at most one per set, never in her first session
        if finite(rec.hold_raw):
            level = self.bests.check(ex.name, "range", rec.hold_raw)
            key = (ex.name, rec.set_no)
            if (level and self.bests.has_baseline(ex.name)
                    and self._pb_per_set[key] < config.PB_MAX_PER_SET):
                self._pb_per_set[key] += 1
                pb = event("PersonalBest", exercise=ex.name, metric="range", level=level)
                events.append(pb)
                self.highlights.append(pb)

        earlier = [r for r in ex.reps if r is not rec and finite(r.hold_value)]
        recent = earlier[-config.IMPROVEMENT_WINDOW:]
        if len(recent) >= 2 and finite(rec.hold_value):
            improvement = self._improvement(rec, recent)
            if improvement is not None:
                events.append(event("EffortPraise", exercise=ex.name) if self.difficult
                              else improvement)
            stabilities = [r.hold_stability for r in recent if finite(r.hold_stability)]
            if (rec.success and finite(rec.hold_stability) and stabilities
                    and rec.hold_stability <= config.STEADY_HOLD_MAX_STD
                    and rec.hold_stability <= 0.7 * float(np.median(stabilities))):
                steady = event("SteadyHold", exercise=ex.name)
                events.append(steady)
                self.highlights.append(steady)

        if not rec.success:
            events.append(event("RecoveredAfterHint", exercise=ex.name))
        return sorted(events, key=lambda e: e.priority)

    def _improvement(self, rec, recent):
        """A finger (or the whole movement) clearly further than her recent average."""
        fingers = rec.extra.get("finger_hold") or {}
        gains = {}
        for finger, value in fingers.items():
            before = [r.extra.get("finger_hold", {}).get(finger) for r in recent]
            before = [v for v in before if finite(v)]
            if len(before) >= 2 and finite(value):
                gains[finger] = value - float(np.mean(before))
        if gains:
            finger = max(gains, key=gains.get)
            if gains[finger] >= config.IMPROVEMENT_MARGIN:
                return event("Improvement", exercise=rec.exercise, finger=finger)
            return None
        gain = rec.hold_value - float(np.mean([r.hold_value for r in recent]))
        if gain >= config.IMPROVEMENT_MARGIN:
            return event("Improvement", exercise=rec.exercise)
        return None

    # --- set ------------------------------------------------------------------

    def set_finished(self, ex):
        """
        After a set: events and the target for the next set (None = no
        target for this exercise). Also checks for a difficult day and
        grows the calibrated range when she went beyond it.
        """
        events = []
        reps = [r for r in ex.reps if r.set_no == ex.set_no]
        rate = sum(1 for r in reps if r.success) / len(reps) if reps else float("nan")

        # tiredness: low success two sets in a row
        if finite(rate) and rate < config.TARGET_LOWER_AT:
            self._low_success_sets += 1
        else:
            self._low_success_sets = 0
        if self._low_success_sets >= config.DIFFICULT_LOW_SUCCESS_SETS:
            events.append(self.switch_difficult("low_success"))

        target = getattr(ex, "target", None) if getattr(ex, "range_steps", ()) else None
        if target is None:
            return [e for e in events if e], None

        # low warm-up: first set clearly below her usual
        if ex.set_no == 1:
            values = [r.hold_raw for r in reps if finite(r.hold_raw)]
            base = self.baseline(ex.name)
            if values and base and float(np.mean(values)) < (1 - config.DIFFICULT_DROP) * base:
                events.append(self.switch_difficult("low_warm_up"))

        # beyond her calibrated maximum: the range grows with her
        held = [r.hold_value for r in reps if finite(r.hold_value)]
        if held and float(np.median(held)) > 1.0 + config.CALIBRATION_GROW_MARGIN:
            factor = float(np.median(held))
            entry = self.profile["calibration"].get(ex.name)
            if entry and entry.get("steps"):
                entry["steps"] = grow_range(entry["steps"], ex.range_steps, factor)
                entry["grown"] = self.today.isoformat()
                ex.cal = entry["steps"]
                target = clamp_target(target / factor)     # same position, new scale
                events.append(event("RangeGrew", exercise=ex.name, factor=round(factor, 3)))

        new = target
        if config.AUTO_PROGRESSION:
            new, change = next_target(target, rate)
            if change == "raised":
                raised = event("TargetRaised", exercise=ex.name, target=new)
                events.append(raised)
                if not self.difficult:
                    self.highlights.append(raised)
            elif change == "lowered":
                events.append(event("TargetLowered", exercise=ex.name, target=new))
        if self.difficult and ex.name not in self._reduced:
            # switched on during this exercise: easier from the next set on
            self._reduced.add(ex.name)
            new = clamp_target(new * config.DIFFICULT_TARGET_FACTOR)
            events = [e for e in events if e is None or e.type != "TargetRaised"]
        return [e for e in events if e], new

    # --- exercise -------------------------------------------------------------

    def exercise_finished(self, ex, row, activity=None, completed=True):
        """Events at the end of an exercise: at most one best, starting point, milestone."""
        events = []
        first = not self.bests.has_baseline(ex.name)
        candidates = []
        clean = to_float(row.get("clean_reps"))
        level = self.bests.check(ex.name, "clean_reps", clean)
        if level:
            candidates.append(event("PersonalBest", exercise=ex.name, metric="clean_reps",
                                    level=level))
        steadiness = to_float(row.get("mean_hold_stability"))
        if getattr(ex, "range_steps", ()) and finite(steadiness):
            level = self.bests.check(ex.name, "steadiness", steadiness)
            if level:
                candidates.append(event("PersonalBest", exercise=ex.name, metric="steadiness",
                                        level=level))
        if candidates and not first:
            best = min(candidates, key=highlight_rank)
            events.append(best)
            self.highlights.append(best)
        if first and ex.name not in self._starting_point:
            self._starting_point.add(ex.name)
            events.append(event("StartingPoint", exercise=ex.name))

        reps = int(to_float(row.get("reps_done")) or 0)
        if activity and reps > 0:
            practice = self.profile.setdefault("practice", {})
            before = int(practice.get(activity, 0))
            after = before + reps
            practice[activity] = after
            crossed = [m for m in config.MILESTONES if before < m <= after]
            if crossed:
                milestone = event("MilestoneReached", activity=activity, count=max(crossed))
                events.append(milestone)
                self.highlights.append(milestone)

        if getattr(ex, "range_steps", ()):
            self.final_targets[ex.name] = ex.target
        if completed and ex.name not in self.completed:
            self.completed.append(ex.name)
        return events

    # --- session ---------------------------------------------------------------

    def highlight(self):
        """The one best moment of the session, or None."""
        found = [e for e in self.highlights if e.type in HIGHLIGHT_ORDER]
        return min(found, key=highlight_rank) if found else None

    def session_best(self):
        """A best for the number of reps in the whole session (or None)."""
        if self.total_reps <= 0:
            return None
        first = not self.bests.has_baseline(SESSION)
        level = self.bests.check(SESSION, "reps", self.total_reps)
        if level and not first:
            pb = event("PersonalBest", exercise=SESSION, metric="reps", level=level)
            self.highlights.append(pb)
            return pb
        return None

    def repeated_difficult_days(self, earlier_sessions):
        """True when this and enough recent sessions were difficult days."""
        if not self.difficult:
            return False
        recent = earlier_sessions[-(config.DIFFICULT_REPEAT_WINDOW - 1):]
        count = 1 + sum(1 for r in recent if r.get("difficult_day") == "yes")
        return count >= config.DIFFICULT_REPEAT_COUNT

    def end_session(self, exercises):
        """
        Store what the coach may remember next time, and the targets unless
        this was a difficult day (one bad day must not pull targets down).
        """
        if not self.difficult and config.AUTO_PROGRESSION:
            targets = self.profile.setdefault("targets", {})
            for name, target in self.final_targets.items():
                targets[name] = {"high": target, "date": self.today.isoformat()}
        best = self.highlight()
        self.profile["last_session"] = {
            "date": self.today.isoformat(),
            "exercises": list(exercises),
            "highlight": best.as_dict() if best else None,
            "difficult_day": self.difficult,
        }


def days_since(last_session, today=None):
    """Whole days since the last session, or None when there was none."""
    if not last_session:
        return None
    last = valid_date(last_session.get("date"))
    if last is None:
        return None
    return ((today or date.today()) - last).days
