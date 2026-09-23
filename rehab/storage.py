"""
Persistent data: the profile (calibration, thresholds) and the logs.

profile.json   affected hand, calibrated ranges per exercise, adjusted
               thresholds, opposition level
reps.csv       one row per repetition with its quality measures
history.csv    one row per exercise per session, used for progress
               messages ("Your hand opened 12% wider than last week")
               and by family or therapist to follow progress
"""

import csv
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np

from rehab import config

REP_FIELDS = [
    "session_id", "timestamp", "exercise", "set", "rep",
    "duration_s", "range_high", "range_low", "raw_high",
    "movement_time_s", "smoothness_peaks", "hold_stability",
    "compensation", "hints", "extra",
]

HISTORY_FIELDS = [
    "session_id", "date", "exercise", "sets_done", "reps_done", "reps_target",
    "mean_range_high", "best_range_high", "mean_range_low", "mean_raw_high",
    "best_raw_high", "mean_movement_time_s", "mean_smoothness_peaks",
    "mean_hold_stability", "compensation_reps", "hints", "extra",
]


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def new_profile():
    return {
        "user": config.USER_NAME,
        "affected_hand": config.AFFECTED_HAND,
        "created": date.today().isoformat(),
        "calibration": {},     # exercise -> {"date": ..., "steps": {...}}
        "thresholds": {},      # exercise -> {"high": ..., "low": ...}
        "levels": {},          # exercise -> level (thumb opposition)
    }


def load_profile(path=config.PROFILE_PATH):
    path = Path(path)
    if not path.is_file():
        return new_profile()
    with open(path) as fh:
        profile = json.load(fh)
    base = new_profile()
    base.update(profile)
    return base


def save_profile(profile, path=config.PROFILE_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(profile, fh, indent=2)
    tmp.replace(path)


def calibrations(profile):
    """{exercise: {step: {key: value}}} as the exercises expect it."""
    return {name: entry.get("steps", {}) for name, entry in profile.get("calibration", {}).items()}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else round(v, 4)
    return v


def _append(path, fields, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: _fmt(row.get(k, "")) for k in fields})


def _read(path):
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _nanmean(values):
    values = [v for v in values if v == v]
    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class SessionLog:

    def __init__(self, rep_path=config.REP_LOG_PATH, history_path=config.HISTORY_PATH,
                 session_id=None, today=None):
        self.rep_path = Path(rep_path)
        self.history_path = Path(history_path)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.today = today or date.today()

    def log_rep(self, rec):
        _append(self.rep_path, REP_FIELDS, {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "exercise": rec.exercise, "set": rec.set_no, "rep": rec.rep_no,
            "duration_s": rec.duration, "range_high": rec.range_high,
            "range_low": rec.range_low, "raw_high": rec.raw_high,
            "movement_time_s": rec.movement_time,
            "smoothness_peaks": rec.smoothness_peaks,
            "hold_stability": rec.hold_stability,
            "compensation": ";".join(rec.compensation),
            "hints": rec.hints,
            "extra": json.dumps(rec.extra, default=str),
        })

    def summarize(self, exercise, reps, sets_done, reps_target):
        """Summary row for one exercise of this session (not yet saved)."""
        extra = {}
        for key in ("isolation", "correct", "wrong", "level", "squeeze_hold_s", "relaxed_fully"):
            vals = [r.extra.get(key) for r in reps if isinstance(r.extra.get(key), (int, float))]
            if vals:
                extra[key] = round(float(np.mean(vals)), 3)
        raw = [r.raw_high for r in reps]
        return {
            "session_id": self.session_id,
            "date": self.today.isoformat(),
            "exercise": exercise,
            "sets_done": sets_done,
            "reps_done": len(reps),
            "reps_target": reps_target,
            "mean_range_high": _nanmean([r.range_high for r in reps]),
            "best_range_high": max((r.range_high for r in reps if r.range_high == r.range_high),
                                   default=float("nan")),
            "mean_range_low": _nanmean([r.range_low for r in reps]),
            "mean_raw_high": _nanmean(raw),
            "best_raw_high": max((v for v in raw if v == v), default=float("nan")),
            "mean_movement_time_s": _nanmean([r.movement_time for r in reps]),
            "mean_smoothness_peaks": _nanmean([float(r.smoothness_peaks) for r in reps]),
            "mean_hold_stability": _nanmean([r.hold_stability for r in reps]),
            "compensation_reps": sum(1 for r in reps if r.compensation),
            "hints": sum(r.hints for r in reps),
            "extra": json.dumps(extra),
        }

    def save_summary(self, row):
        _append(self.history_path, HISTORY_FIELDS, row)

    # --- history ------------------------------------------------------------

    def history(self, exercise=None):
        rows = _read(self.history_path)
        if exercise:
            rows = [r for r in rows if r["exercise"] == exercise]
        return rows

    def last_done(self, exercise):
        dates = [date.fromisoformat(r["date"]) for r in self.history(exercise)
                 if _float(r.get("reps_done")) > 0]
        return max(dates) if dates else None

    def comparison(self, exercise):
        """
        Earlier session to compare with: the most recent one at least 6 days
        ago ("last week"), otherwise the previous session ("last time").
        Returns (row, when) or (None, None).
        """
        rows = [r for r in self.history(exercise)
                if r["session_id"] != self.session_id and _float(r.get("reps_done")) > 0]
        if not rows:
            return None, None
        week = [r for r in rows if (self.today - date.fromisoformat(r["date"])).days >= 6]
        if week:
            return week[-1], "last week"
        return rows[-1], "last time"


def progress_message(exercise_cls, summary, earlier, when):
    """
    A progress sentence about her own history, never about a norm.
    Only improvements are put into numbers; otherwise a neutral encouragement.
    """
    now = _float(summary.get("mean_raw_high"))
    if earlier is None or now != now:
        return None
    before = _float(earlier.get("mean_raw_high"))
    if before != before or before <= 0:
        return None
    if exercise_cls.progress_lower_is_better:
        change = (before - now) / before
    else:
        change = (now - before) / before
    pct = int(round(change * 100))
    if pct >= 3:
        return exercise_cls.progress_phrase.format(pct=pct, when=when)
    return "You kept up your good work. Steady practice is what helps."


def should_do_today(exercise, log, today=None):
    """Every-other-day exercises are skipped when done today or yesterday."""
    if exercise not in config.EVERY_OTHER_DAY:
        return True
    last = log.last_done(exercise)
    today = today or log.today
    return last is None or (today - last).days >= 2
