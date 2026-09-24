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

Arm exercises (degrees, rehab/benchmarks.py) follow the benchmark plan
instead of the percentage targets:
  * a level on her target ladder, chosen at the start (one lower after two
    missed sessions or on a difficult day) and changed once per session
    from the clean success and compensation of the last two sessions;
  * a best or an "improved" message only when the change is at least the
    minimum detectable change (MDC), never for measurement noise;
  * named milestones (FMA thresholds, heights daily tasks need) announced
    once, the first time she reaches them;
  * the weekly assessment (the calibration) with its own MDC comparison.

It decides what happened, never how to say it: everything comes out as
events (rehab/events.py). It changes only the profile dict it is given and
reads the history through the SessionLog.
"""

import json
from collections import Counter
from datetime import date

import numpy as np

from rehab import benchmarks, config
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

    def _beats(self, metric, value, ref, min_abs=None):
        higher, rel = PB_METRICS[metric]
        if not finite(ref):
            return False
        if min_abs is not None:
            # degrees: only a change of at least the MDC is real (plan rule T4)
            return (value - ref if higher else ref - value) >= min_abs
        if higher:
            return value > ref and value >= ref + abs(ref) * rel
        return value < ref and value <= ref - abs(ref) * rel

    def _best(self, metric, values):
        higher, _ = PB_METRICS[metric]
        values = [v for v in values if finite(v)]
        if not values:
            return None
        return max(values) if higher else min(values)

    def level(self, exercise, metric, value, min_abs=None):
        """"all_time", "week", "today" or None: the highest level this value beats."""
        entry = self.store.get(exercise, {}).get(metric)
        if not entry or not finite(value):
            return None
        if self._beats(metric, value, entry.get("value"), min_abs):
            return "all_time"
        today_key = self.today.isoformat()
        daily = entry.get("daily", {})
        week = [v for d, v in daily.items()
                if valid_date(d) and 0 <= (self.today - valid_date(d)).days < config.PB_WEEK_DAYS]
        week_best = self._best(metric, week)
        if week_best is not None and self._beats(metric, value, week_best, min_abs):
            return "week"
        if today_key in daily and self._beats(metric, value, daily[today_key], min_abs):
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

    def check(self, exercise, metric, value, min_abs=None):
        """Level reached (or None), then record the value."""
        level = self.level(exercise, metric, value, min_abs)
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

    def __init__(self, profile, log, today=None, therapist=None):
        self.profile = profile
        self.log = log
        self.today = today or log.today
        self.therapist = therapist if therapist is not None else benchmarks.load_therapist_profile()
        self.final_levels = {}           # arm exercises: level for next time (saved at the end)
        self.level_changes = {}
        self.claims = {}                 # exercise -> "improved" sentence said today (MDC passed)
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

    def start_level(self, exercise):
        """
        Level on an arm exercise's target ladder for today: where she ended,
        one lower after two missed sessions (no fuss about it) and one lower
        on a difficult day.
        """
        stored = to_float(self.profile.setdefault("levels", {}).get(exercise))
        level = int(stored) if finite(stored) else 0
        last = self.log.last_done(exercise)
        days = (self.today - last).days if last else None
        level = benchmarks.start_level(level, days)
        if self.difficult:
            level = max(0, level - 1)
        return level

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
        if getattr(ex, "benchmark", False):
            return sorted(events + self._benchmark_rep_events(ex, rec), key=lambda e: e.priority)

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
        events += self._hand_milestones(ex, rec)
        return sorted(events, key=lambda e: e.priority)

    def _hand_milestones(self, ex, rec):
        """Bain 2015 functional open / grasp and a first FMA 25-style 2, each announced once."""
        events = []
        reached = self.profile.setdefault("benchmark_milestones", {}).setdefault(ex.name, [])
        for key in rec.extra.get("milestones_reached") or []:
            if key in reached:
                continue
            reached.append(key)
            ev = event("BenchmarkMilestone", exercise=ex.name, task=key, metric=key,
                       kind="fma" if key.startswith("fma") else "task")
            events.append(ev)
            self.highlights.append(ev)
        return events

    def _benchmark_rep_events(self, ex, rec):
        """
        An arm rep: a best only when it beats the old one by the MDC, a
        milestone the first time it is reached, "further than before" only
        when it is more than the measurement tolerance further.
        """
        events = []
        if rec.extra.get("mode") != "training" or not finite(rec.raw_high):
            return events
        sgn = 1.0 if getattr(ex, "direction", "increase") == "increase" else -1.0
        mdc = ex.change_threshold() if hasattr(ex, "change_threshold") else None
        if mdc is not None and finite(rec.hold_raw):
            level = self.bests.check(ex.name, "range", sgn * rec.hold_raw, min_abs=mdc)
            key = (ex.name, rec.set_no)
            if (level and self.bests.has_baseline(ex.name)
                    and self._pb_per_set[key] < config.PB_MAX_PER_SET):
                self._pb_per_set[key] += 1
                pb = event("PersonalBest", exercise=ex.name, metric="range", level=level)
                events.append(pb)
                self.highlights.append(pb)
        ladder = getattr(ex, "ladder", None)
        if ladder is not None:
            reached = self.profile.setdefault("benchmark_milestones", {}).setdefault(ex.name, [])
            for m in ladder.milestones:
                key = f"{m.get('task', '')}@{m['deg']:g}"
                if key in reached or sgn * (rec.raw_high - m["deg"]) < 0:
                    continue
                reached.append(key)
                ev = event("BenchmarkMilestone", exercise=ex.name, deg=m["deg"],
                           task=m.get("task", ""), kind=m.get("kind", "task"),
                           metric=m.get("kind", "task"))
                events.append(ev)
                self.highlights.append(ev)
        # "further than before" only on a ladder (reaching alternates near and far targets)
        earlier = [r.raw_high for r in ex.reps if r is not rec and finite(r.raw_high)
                   and r.set_no == rec.set_no]
        recent = earlier[-config.IMPROVEMENT_WINDOW:]
        if (ladder is not None and len(recent) >= 2
                and sgn * (rec.raw_high - float(np.mean(recent))) > ex.tolerance):
            events.append(event("EffortPraise", exercise=ex.name) if self.difficult
                          else event("Improvement", exercise=ex.name))
        return events

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
        if getattr(ex, "ladder", None) is not None and completed:
            events += self._next_level(ex, row)
        if completed and ex.name not in self.completed:
            self.completed.append(ex.name)
        return events

    def _next_level(self, ex, row):
        """
        Arm exercise: the level for next time from this and the previous
        normal session (plan 9.4, proposed). Kept until the end of the
        session, and not saved at all on a difficult day.
        """
        if not config.AUTO_PROGRESSION:
            return []
        sessions = [bench_stats(r) for r in self.log.normal_history(ex.name)[-1:]]
        sessions.append(bench_stats(row))
        level, change = benchmarks.next_level(ex.level, sessions)
        level = min(level, ex.ladder.max_level)
        if level == ex.level:
            change = None if change != "pause" else change
        self.final_levels[ex.name] = level
        self.level_changes[ex.name] = (ex.level, level)
        if change == "raised" and not self.difficult:
            ev = event("LevelRaised", exercise=ex.name, level=level)
            self.highlights.append(ev)
            return [ev]
        if change == "lowered":
            return [event("LevelLowered", exercise=ex.name, level=level)]
        return []

    # --- arm assessment ---------------------------------------------------------------

    def assessment_done(self, exercise_cls, entry, old_entry=None):
        """
        A new arm calibration, which is also the weekly assessment. Keeps her
        first affected-side best as the baseline of the ladder, stores the
        assessment, and says "improved" only for a change of at least the MDC
        against the previous assessment (plan rules T4 and T5).
        """
        steps = entry.setdefault("steps", {})
        side = self.therapist.get("affected_side", "left")
        affected = steps.get(side) or {}
        old_steps = (old_entry or {}).get("steps") or {}
        baseline = old_steps.get("baseline")
        if not finite(baseline):
            baseline = affected.get("best")
        if finite(baseline):
            steps["baseline"] = baseline
        other = "right" if side == "left" else "left"
        history = self.profile.setdefault("assessments", {}).setdefault(exercise_cls.name, [])
        previous = history[-1] if history else None
        history.append({
            "date": self.today.isoformat(),
            "affected_best": affected.get("best"),
            "unaffected_best": (steps.get(other) or {}).get("best"),
            "symmetry_index": steps.get("symmetry_index"),
            "fma_style_best": (steps.get("fma_style_best") or {}).get(side),
        })
        if previous is None:
            self._starting_point.add(exercise_cls.name)
            return [event("StartingPoint", exercise=exercise_cls.name)]
        bench = benchmarks.load()
        mdc = bench.mdc(exercise_cls.name, self.therapist.get("mdc_setting", "laboratory"))
        now, before = affected.get("best"), previous.get("affected_best")
        if benchmarks.claim_improvement(now, before, mdc, exercise_cls.direction):
            ev = event("AssessmentImproved", exercise=exercise_cls.name,
                       deg=int(round(abs(now - before))), metric=exercise_cls.metric)
            self.highlights.append(ev)
            return [ev]
        return []

    def benchmark_session_row(self, summaries, setup_checks=()):
        """One row for benchmark_sessions.csv (plan 12), or None when no arm exercise was done."""
        rows = [(name, row) for name, row, _ in summaries
                if json.loads(row.get("extra") or "{}").get("benchmark")]
        if not rows:
            return None
        stats = {name: bench_stats(row) for name, row in rows}
        reps = sum(int(to_float(row.get("reps_done")) or 0) for _, row in rows)
        clean = [s["clean_success_rate"] for s in stats.values() if finite(s["clean_success_rate"])]
        comp = [s["compensation_rate"] for s in stats.values() if finite(s["compensation_rate"])]
        extra = {name: json.loads(row.get("extra") or "{}") for name, row in rows}
        claims = {n: m for n, m in self.claims.items() if n in stats}
        milestones = [{"exercise": e.get("exercise"), "task": e.get("task"), "deg": e.get("deg")}
                      for e in self.highlights if e.type == "BenchmarkMilestone"]
        t = self.therapist
        return {
            "session_id": self.log.session_id,
            "date": self.today.isoformat(),
            "user_id": t.get("user_id", ""),
            "seat_type": t.get("seat_type", ""),
            "camera_view_check_passed": "yes" if setup_checks and all(p for _, p in setup_checks)
            else ("no" if setup_checks else ""),
            "supervisor_present": "yes" if t.get("supervisor_present") else "no",
            "assistance_given": "",
            "exercises_done": ";".join(stats),
            "reps_attempted": reps,
            "clean_success_rate": float(np.mean(clean)) if clean else float("nan"),
            "compensation_rate": float(np.mean(comp)) if comp else float("nan"),
            "best_values_json": json.dumps({n: e.get("best_peak") for n, e in extra.items()}),
            "mean_values_json": json.dumps({n: e.get("mean_peak") for n, e in extra.items()}),
            "improvement_claims_json": json.dumps(claims),
            "milestones_crossed_json": json.dumps(milestones),
            "level_changes_json": json.dumps({n: list(v) for n, v in self.level_changes.items()}),
            "notes": "difficult day" if self.difficult else "",
        }

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
            self.profile.setdefault("levels", {}).update(self.final_levels)
        best = self.highlight()
        self.profile["last_session"] = {
            "date": self.today.isoformat(),
            "exercises": list(exercises),
            "highlight": best.as_dict() if best else None,
            "difficult_day": self.difficult,
        }


def bench_stats(row):
    """Clean success, success and compensation rates of one history row (arm exercises)."""
    extra = row.get("extra") or "{}"
    try:
        extra = json.loads(extra) if isinstance(extra, str) else dict(extra)
    except ValueError:
        extra = {}
    return {
        "success_rate": to_float(row.get("success_rate")),
        "clean_success_rate": to_float(extra.get("clean_success_rate")),
        "compensation_rate": to_float(extra.get("compensation_rate")),
    }


def days_since(last_session, today=None):
    """Whole days since the last session, or None when there was none."""
    if not last_session:
        return None
    last = valid_date(last_session.get("date"))
    if last is None:
        return None
    return ((today or date.today()) - last).days
