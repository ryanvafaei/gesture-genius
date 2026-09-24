"""
Persistent data. Storage only reads and writes; it decides nothing.

profile.json   who she is and what the coach remembers: name, hand, voice,
               chosen activities, calibrated ranges, targets, personal
               bests, practice counts and the last session
reps.csv       one row per repetition with its quality measures, written
               after every rep so nothing is lost in a crash
history.csv    one row per exercise per session, used for progress
               messages ("Your hand opened 12% wider than last week")
               and by family or therapist to follow progress
sessions.csv   one row per session (check-in, difficult day, highlight, notes)
garden.json    the garden: plots, waterings, bees and butterflies
benchmark_reps.csv, benchmark_sessions.csv
               the movement benchmark logs (benchmarks/BENCHMARK_PLAN.md
               section 12): FMA-style score, coaching success, compensation,
               measurement quality per rep; setup, rates, bests, claims,
               milestones and level changes per session. They are the
               objective measures for the report's testing section.

delete_profile() removes all of these at once (moved to data/deleted/<time>/
unless KEEP_DELETED_PROFILE is off), so the coach starts as on a first day.

profile.json, garden.json and sessions.csv are written to a temporary file
first and then renamed, so a crash cannot leave a half-written file. A file
that cannot be read is set aside (renamed *.broken-<time>) and the program
starts as on a first day rather than "remembering" something wrong.
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
    "duration_s", "target_used", "range_reached", "hold_value", "hold_raw",
    "range_high", "range_low", "raw_high",
    "movement_time_s", "smoothness_peaks", "hold_stability",
    "compensation", "hints", "success", "difficult_day", "extra",
]

HISTORY_FIELDS = [
    "session_id", "date", "exercise", "sets_done", "reps_done", "reps_target",
    "mean_range_high", "best_range_high", "mean_range_low", "mean_raw_high",
    "best_raw_high", "mean_hold_raw", "mean_movement_time_s", "mean_smoothness_peaks",
    "mean_hold_stability", "compensation_reps", "hints", "clean_reps", "success_rate",
    "target_start", "target_end", "difficult_day", "extra",
]

SESSION_FIELDS = [
    "session_id", "date", "start", "end", "duration_min", "exercises", "total_reps",
    "success_rate", "average_range", "check_in", "difficult_day", "difficult_reason",
    "highlight", "garden", "note",
]

# benchmark logs, the columns of the plan's templates
BENCH_REP_FIELDS = [
    "session_id", "timestamp_iso", "mode", "exercise_id", "side", "rep_index", "level",
    "personal_target_deg", "fma_threshold_deg", "tolerance_deg", "start_value", "peak_value",
    "peak_minus_start", "hold_s", "duration_s", "fma_style_score", "coaching_success",
    "with_compensation", "form_rules_broken", "compensation_flags", "trunk_max_change_deg",
    "shoulder_girdle_elevation_max", "in_plane_ratio_min", "cue_given", "feedback_key",
    "mean_visibility", "invalid_frame_pct", "plausibility_rejects", "details",
]
BENCH_SESSION_FIELDS = [
    "session_id", "date", "user_id", "seat_type", "camera_view_check_passed",
    "camera_distance_m", "lighting_ok", "supervisor_present", "assistance_given",
    "pain_before_0_10", "fatigue_before_0_10", "pain_after_0_10", "fatigue_after_0_10",
    "exercises_done", "reps_attempted", "clean_success_rate", "compensation_rate",
    "best_values_json", "mean_values_json", "improvement_claims_json",
    "milestones_crossed_json", "level_changes_json", "notes",
]
# rep extras that have their own column (the rest go to "details")
_BENCH_COLUMNS = set(BENCH_REP_FIELDS)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def new_profile():
    return {
        "name": config.USER_NAME,
        "affected_hand": config.AFFECTED_HAND,
        "created": date.today().isoformat(),
        "coach_name": None,        # chosen at the first session
        "voice": config.VOICE,
        "voice_rate": config.SPEECH_RATE,
        "chosen_activities": [],   # her favourites, used first
        "calibration": {},         # exercise -> {"date": ..., "steps": {...}}
        "targets": {},             # exercise -> {"high": ..., "date": ...}
        "levels": {},              # exercise -> level (thumb opposition, arm exercise ladders)
        "personal_bests": {},      # exercise -> metric -> {"value", "date", "daily"}
        "practice": {},            # activity -> repetitions practised for it
        "last_session": None,      # date, exercises, highlight, difficult_day
        "assessments": {},         # arm exercise -> [weekly assessment: bests, symmetry, FMA-style]
        "benchmark_milestones": {},  # arm exercise -> milestones already announced
    }


# expected type of each profile field; anything else is replaced by the default
_PROFILE_TYPES = {
    "name": str, "affected_hand": str, "created": str, "coach_name": (str, type(None)),
    "voice": (str, type(None)), "voice_rate": (int, float), "chosen_activities": list,
    "calibration": dict, "targets": dict, "levels": dict, "personal_bests": dict,
    "practice": dict, "last_session": (dict, type(None)), "assessments": dict,
    "benchmark_milestones": dict,
}


def _set_aside(path):
    """Keep an unreadable file for inspection, out of the program's way."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        path.replace(path.with_name(f"{path.name}.broken-{stamp}"))
    except OSError:
        pass


def _read_json(path):
    """Parsed JSON object, or None when the file is missing or unreadable (then set aside)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("not an object")
        return data
    except (OSError, ValueError):
        _set_aside(path)
        return None


def _write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(path)


def valid_date(text):
    try:
        return date.fromisoformat(str(text))
    except ValueError:
        return None


def load_profile(path=config.PROFILE_PATH):
    """
    Her profile, checked field by field. Missing or wrongly typed fields get
    their default, so the coach only ever refers to facts that are really stored.
    """
    profile = _read_json(path) or {}
    base = new_profile()
    # older profiles
    if "user" in profile and "name" not in profile:
        profile["name"] = profile["user"]
    if "thresholds" in profile and "targets" not in profile:
        profile["targets"] = profile["thresholds"]
    profile.pop("user", None)
    profile.pop("thresholds", None)
    for key, value in profile.items():
        expected = _PROFILE_TYPES.get(key)
        if expected is None or isinstance(value, expected):
            base[key] = value
    last = base["last_session"]
    if last is not None and not valid_date(last.get("date")):
        base["last_session"] = None
    base["chosen_activities"] = [a for a in base["chosen_activities"] if isinstance(a, str)]
    return base


def save_profile(profile, path=config.PROFILE_PATH):
    _write_json(profile, path)


# everything that belongs to her: removed together when she starts over
PROFILE_FILES = (config.PROFILE_PATH, config.GARDEN_PATH, config.REP_LOG_PATH,
                 config.HISTORY_PATH, config.SESSIONS_PATH, config.BENCH_REP_LOG_PATH,
                 config.BENCH_SESSION_LOG_PATH)


def delete_profile(paths=PROFILE_FILES, keep_backup=config.KEEP_DELETED_PROFILE,
                   backup_dir=config.DELETED_PROFILES_DIR):
    """
    Forget her: profile, garden and history, so the next start is a first day.
    With keep_backup the files are moved to backup_dir/<time>/ (a therapist can
    still put them back); otherwise they are erased. Returns the backup folder,
    or None when nothing was kept.
    """
    existing = [Path(p) for p in paths if Path(p).is_file()]
    if not existing:
        return None
    if not keep_backup:
        for path in existing:
            path.unlink(missing_ok=True)
        return None
    folder = Path(backup_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    for path in existing:
        path.replace(folder / path.name)
    return folder


# ---------------------------------------------------------------------------
# Garden
# ---------------------------------------------------------------------------

def new_garden():
    return {"season": 1, "plots": [], "waterings": 0, "bees": 0, "butterflies": 0}


def load_garden(path=config.GARDEN_PATH):
    data = _read_json(path) or {}
    garden = new_garden()
    for key in ("season", "waterings", "bees", "butterflies"):
        if isinstance(data.get(key), int) and data[key] >= 0:
            garden[key] = data[key]
    garden["season"] = max(1, garden["season"])
    plots = data.get("plots")
    if isinstance(plots, list):
        garden["plots"] = [
            {"plant": p["plant"], "stage": int(p["stage"])}
            for p in plots
            if isinstance(p, dict) and isinstance(p.get("plant"), str)
            and isinstance(p.get("stage"), int) and 0 <= p["stage"] < config.GARDEN_STAGES
        ][:config.GARDEN_PLOTS]
    return garden


def save_garden(garden, path=config.GARDEN_PATH):
    _write_json(garden, path)


# ---------------------------------------------------------------------------
# Content (phrases, activities, coach character)
# ---------------------------------------------------------------------------

def load_content(name, content_dir=config.CONTENT_DIR):
    """content/<name>.json, or {} when it is missing or unreadable (never set aside)."""
    path = Path(content_dir) / f"{name}.json"
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


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


def _header(path):
    with open(path, newline="") as fh:
        return next(csv.reader(fh), [])


def _append(path, fields, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    if not new and _header(path) != fields:
        # written by an older version: rewrite once with the new columns
        _rewrite(path, fields, _read(path))
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: _fmt(row.get(k, "")) for k in fields})


def _rewrite(path, fields, rows):
    """Whole file to a temporary file, then renamed: never half written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in fields})
    tmp.replace(path)


def _read(path):
    path = Path(path)
    if not path.is_file():
        return []
    try:
        with open(path, newline="") as fh:
            return [{k: (v if v is not None else "") for k, v in r.items() if k is not None}
                    for r in csv.DictReader(fh)]
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def to_float(v):
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
                 session_id=None, today=None, sessions_path=None):
        self.rep_path = Path(rep_path)
        self.history_path = Path(history_path)
        # next to the history unless given, so test logs never touch data/
        self.sessions_path = Path(sessions_path) if sessions_path else \
            self.history_path.with_name("sessions.csv")
        self.bench_rep_path = self.history_path.with_name(config.BENCH_REP_LOG_PATH.name)
        self.bench_session_path = self.history_path.with_name(config.BENCH_SESSION_LOG_PATH.name)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.today = today or date.today()

    def log_rep(self, rec, difficult_day=False):
        _append(self.rep_path, REP_FIELDS, {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "exercise": rec.exercise, "set": rec.set_no, "rep": rec.rep_no,
            "duration_s": rec.duration, "target_used": rec.target,
            "range_reached": rec.hold_value if rec.hold_value == rec.hold_value else rec.range_high,
            "hold_value": rec.hold_value, "hold_raw": rec.hold_raw,
            "range_high": rec.range_high,
            "range_low": rec.range_low, "raw_high": rec.raw_high,
            "movement_time_s": rec.movement_time,
            "smoothness_peaks": rec.smoothness_peaks,
            "hold_stability": rec.hold_stability,
            "compensation": ";".join(rec.compensation),
            "hints": rec.hints,
            "success": "yes" if rec.success else "no",
            "difficult_day": "yes" if difficult_day else "no",
            "extra": json.dumps(rec.extra, default=str),
        })
        if rec.extra.get("benchmark"):
            self.log_bench_rep(rec)

    def log_bench_rep(self, rec):
        """One row of benchmark_reps.csv (plan section 12)."""
        e = rec.extra

        def yes(v):
            return "" if v is None else ("yes" if v else "no")

        row = {k: e.get(k) for k in BENCH_REP_FIELDS if k in e}
        row.update({
            "session_id": self.session_id,
            "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
            "exercise_id": rec.exercise, "rep_index": rec.rep_no,
            "duration_s": rec.duration,
            "coaching_success": yes(e.get("coaching_success")),
            "with_compensation": yes(e.get("with_compensation")),
            "form_rules_broken": ";".join(e.get("form_rules_broken") or []),
            "compensation_flags": ";".join(e.get("compensation_flags") or []),
            "details": json.dumps({k: v for k, v in e.items()
                                   if k not in _BENCH_COLUMNS and k != "benchmark"}, default=str),
        })
        _append(self.bench_rep_path, BENCH_REP_FIELDS,
                {k: ("" if v is None else v) for k, v in row.items()})

    def save_bench_session(self, row):
        rows = [r for r in _read(self.bench_session_path) if r.get("session_id") != self.session_id]
        _rewrite(self.bench_session_path, BENCH_SESSION_FIELDS, rows + [row])

    def summarize(self, exercise, reps, sets_done, reps_target, target_start=float("nan"),
                  target_end=float("nan"), difficult_day=False):
        """Summary row for one exercise of this session (not yet saved)."""
        extra = {}
        for key in ("isolation", "correct", "wrong", "level", "squeeze_hold_s", "relaxed_fully",
                    "fma28_style", "pinch_gap_min", "fma24_style", "fma25_style"):
            vals = [r.extra.get(key) for r in reps if isinstance(r.extra.get(key), (int, float))]
            if vals:
                extra[key] = round(float(np.mean(vals)), 3)
        extra.update(bench_summary(reps))
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
            "mean_hold_raw": _nanmean([r.hold_raw for r in reps]),
            "mean_movement_time_s": _nanmean([r.movement_time for r in reps]),
            "mean_smoothness_peaks": _nanmean([float(r.smoothness_peaks) for r in reps]),
            "mean_hold_stability": _nanmean([r.hold_stability for r in reps]),
            "compensation_reps": sum(1 for r in reps if r.compensation),
            "hints": sum(r.hints for r in reps),
            "clean_reps": sum(1 for r in reps if r.success and not r.hints),
            "success_rate": sum(1 for r in reps if r.success) / len(reps) if reps else float("nan"),
            "target_start": target_start,
            "target_end": target_end,
            "difficult_day": "yes" if difficult_day else "no",
            "extra": json.dumps(extra),
        }

    def save_summary(self, row):
        _append(self.history_path, HISTORY_FIELDS, row)

    def mark_difficult(self):
        """This session turned out to be a difficult day: flag its history rows too."""
        rows = _read(self.history_path)
        if any(r.get("session_id") == self.session_id for r in rows):
            for r in rows:
                if r.get("session_id") == self.session_id:
                    r["difficult_day"] = "yes"
            _rewrite(self.history_path, HISTORY_FIELDS, rows)

    def save_session(self, row):
        rows = [r for r in _read(self.sessions_path) if r.get("session_id") != self.session_id]
        _rewrite(self.sessions_path, SESSION_FIELDS, rows + [row])

    def sessions(self):
        return _read(self.sessions_path)

    # --- history ------------------------------------------------------------

    def history(self, exercise=None):
        rows = _read(self.history_path)
        if exercise:
            rows = [r for r in rows if r["exercise"] == exercise]
        return rows

    def last_done(self, exercise):
        dates = [valid_date(r.get("date")) for r in self.history(exercise)
                 if to_float(r.get("reps_done")) > 0 and valid_date(r.get("date"))]
        return max(dates) if dates else None

    def normal_history(self, exercise):
        """Earlier sessions of this exercise that were not difficult days."""
        return [r for r in self.history(exercise)
                if r.get("session_id") != self.session_id and to_float(r.get("reps_done")) > 0
                and r.get("difficult_day") != "yes" and valid_date(r.get("date"))]

    def comparison(self, exercise):
        """
        Earlier session to compare with: the most recent one at least 6 days
        ago ("last week"), otherwise the previous session ("last time").
        Difficult days are left out. Returns (row, when) or (None, None).
        """
        rows = self.normal_history(exercise)
        if not rows:
            return None, None
        week = [r for r in rows if (self.today - date.fromisoformat(r["date"])).days >= 6]
        if week:
            return week[-1], "last week"
        return rows[-1], "last time"


def bench_summary(reps):
    """History extras of the benchmark reps (training mode): rates, best and mean peak, FMA-style."""
    reps = [r for r in reps if r.extra.get("benchmark") and r.extra.get("mode", "training") == "training"]
    if not reps:
        return {}
    increase = reps[0].extra.get("direction", "increase") == "increase"
    peaks = [r.raw_high for r in reps if r.raw_high == r.raw_high]
    scores = [r.extra["fma_style_score"] for r in reps
              if isinstance(r.extra.get("fma_style_score"), int)]
    clean = [bool(r.extra.get("coaching_success")) and not r.extra.get("with_compensation")
             for r in reps]
    out = {
        "benchmark": True,
        "side": reps[0].extra.get("side"),
        "level": reps[0].extra.get("level"),
        "clean_success_rate": round(sum(clean) / len(reps), 3),
        "compensation_rate": round(sum(1 for r in reps if r.extra.get("with_compensation")) / len(reps), 3),
        "best_peak": round((max(peaks) if increase else min(peaks)), 2) if peaks else None,
        "mean_peak": round(float(np.mean(peaks)), 2) if peaks else None,
        "fma_style_best": max(scores) if scores else None,
    }
    return out


def _extra(row):
    try:
        return json.loads(row.get("extra") or "{}")
    except (TypeError, ValueError):
        return {}


def improvement_claimed(exercise_cls, summary, earlier, threshold):
    """
    Change in session best (degrees) against the earlier session, when it
    is at least the MDC (plan rule T4); None otherwise. Elbow tasks claim
    only from the weekly assessment (T5).
    """
    from rehab.benchmarks import claim_improvement
    if earlier is None or threshold is None or getattr(exercise_cls, "claims_from_assessment_only", False):
        return None
    now, before = to_float(_extra(summary).get("best_peak")), to_float(_extra(earlier).get("best_peak"))
    direction = getattr(exercise_cls, "direction", "increase")
    if claim_improvement(now, before, threshold, direction):
        return abs(now - before)
    return None


def progress_message(exercise_cls, summary, earlier, when, history=None, threshold=None):
    """
    A progress sentence about her own history, never about a norm.
    Only improvements are put into numbers; otherwise a neutral encouragement.

    Arm exercises (degrees) only claim an improvement of at least the
    minimum detectable change (threshold); otherwise she is praised for
    doing the reps. Grip and release (finger joints, not validated) needs a
    mean change of FINGER_CHANGE_DEG per joint or a steady rise over
    TREND_SESSIONS sessions (plan 7.2, proposed).
    """
    if getattr(exercise_cls, "benchmark", False):
        change = improvement_claimed(exercise_cls, summary, earlier, threshold)
        if change is not None:
            return exercise_cls.progress_phrase.format(deg=int(round(change)), when=when)
        done = int(to_float(summary.get("reps_done")) or 0)
        target = int(to_float(summary.get("reps_target")) or 0)
        if done and done >= target:
            from rehab.exercises.base import number_word
            return f"You did all {number_word(done)}, well done."
        return "You kept up your good work. Steady practice is what helps."
    now = to_float(summary.get("mean_raw_high"))
    if earlier is None or now != now:
        return None
    before = to_float(earlier.get("mean_raw_high"))
    if before != before or before <= 0:
        return None
    if exercise_cls.progress_lower_is_better:
        change = (before - now) / before
    else:
        change = (now - before) / before
    pct = int(round(change * 100))
    rule = getattr(exercise_cls, "claim_rule", None)
    if pct >= 3 and (rule is None or rule(now, before, history or [])):
        return exercise_cls.progress_phrase.format(pct=pct, when=when)
    return "You kept up your good work. Steady practice is what helps."


def should_do_today(exercise, log, today=None):
    """Every-other-day exercises are skipped when done today or yesterday."""
    if exercise not in config.EVERY_OTHER_DAY:
        return True
    last = log.last_done(exercise)
    today = today or log.today
    return last is None or (today - last).days >= 2
