"""
Think: movement benchmarks from published clinical and normative data.

The numbers live in benchmarks/benchmarks.json, built from the CSVs in
benchmarks/data (see benchmarks/BENCHMARK_PLAN.md). Every value there has a
source ID or is marked proposed. This module loads them once and holds the
logic that uses them; it never speaks and never draws.

  Layer 1  form (FMA-UE 2026)   score_rep(): an FMA-style 0/1/2 per rep with
                                the conservative uncertainty band ("a lower
                                score if uncertain"), form and compensation
                                rules debounced over consecutive frames
  Layer 2  range                normative ceilings and plausibility (Soucie
                                2011), daily-task milestones (Gates 2016 arm,
                                Bain 2015 fingers), reach compensation (Levin
                                2004 Reaching Performance Scale)
  Layer 3  personal             side_summary() / symmetry_index() from the
                                right-then-left calibration, TargetLadder,
                                progression rules (proposed)
  Tolerance                     within a rep: 95% limits of agreement (Lazem
                                2026); between sessions: only a change of at
                                least the minimum detectable change is called
                                an improvement (claim_improvement)

The FMA-style scores are for coaching and logs, not a clinical FMA-UE.
"""

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from rehab import config

# measurement_tolerance entry of each exercise's main joint (for the home MDC)
TOLERANCE_KEY = {
    "shoulder_flexion_raise": "forward_elevation.shoulder",
    "shoulder_abduction_raise": "abduction.shoulder",
    "hand_to_mouth": "hand_to_mouth.elbow",
    "hand_to_head": "hand_to_head.elbow",
    "elbow_extension": "elbow_flexion_simple",
    "wrist_extension": "wrist_flexion_extension",
    "grip_release": "finger_joints",
}

# Which normative motion caps which exercise (Soucie 2011 group maximum).
# Soucie has no abduction; flexion's maximum is used as the plausibility cap.
PLAUSIBILITY_MOTION = {
    "shoulder_flexion_raise": "shoulder_flexion",
    "shoulder_abduction_raise": "shoulder_flexion",
    "hand_to_mouth": "elbow_flexion",
    "hand_to_head": "elbow_flexion",
    "elbow_extension": "elbow_flexion",
}

# The plan's exercise ids for this app's hand exercises (Bloom = grip and release;
# the pinch position is scored in bubble pinch and in thumb opposition's index touch).
PLAN_ID = {"grip_release": "hand_open_close", "thumb_opposition": "pinch",
           "bubble_pinch": "pinch"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class Benchmarks:
    """benchmarks.json with accessors for what the exercises need."""

    def __init__(self, path=config.BENCHMARKS_PATH):
        with open(path, encoding="utf-8") as fh:
            self.raw = json.load(fh)
        self.pose = self.raw["landmarks"]["pose"]
        self.hand = self.raw["landmarks"]["hand"]
        self.tol = self.raw["measurement_tolerance"]
        self.ex = self.raw["exercises"]

    def exercise(self, name):
        return self.ex.get(PLAN_ID.get(name, name), {})

    # --- tolerance and detectable change --------------------------------------

    def tolerance(self, name):
        """Tolerance of the exercise's main measure, in degrees."""
        ex = self.exercise(name)
        if ex.get("tolerance_deg") is not None:
            return float(ex["tolerance_deg"])
        return float(self.tol["trunk"]["tolerance_deg"])

    def elbow_tolerance(self, name):
        """Elbow tolerance for "keep your elbow straight / still" (the widest elbow LoA if none is given)."""
        ex = self.exercise(name)
        if ex.get("elbow_tolerance_deg") is not None:
            return float(ex["elbow_tolerance_deg"])
        return float(self.tol["forward_elevation.elbow"]["tolerance_deg"])

    def mdc(self, name, setting="laboratory"):
        """
        Smallest change between sessions that may be called an improvement:
        the MDC (laboratory by default, home when the therapist profile says
        so), or the provisional threshold where no MDC was published.
        """
        entry = self.tol.get(TOLERANCE_KEY.get(name, ""), {})
        per_setting = entry.get("mdc") or {}
        value = per_setting.get(setting)
        if value is None:
            value = self.exercise(name).get("mdc_deg")
        if value is None:
            value = entry.get("mdc_default_deg")
        if value is None:
            value = (self.exercise(name).get("provisional_change_threshold_deg")
                     or entry.get("provisional_change_threshold_deg"))
        return float(value) if value is not None else None

    # --- layer 1 --------------------------------------------------------------

    @property
    def fma_numbers(self):
        return self.raw["layer1_fma"]["numeric_thresholds"]

    def fma_item(self, item):
        return self.raw["layer1_fma"]["items"].get(item, {})

    # --- layer 2 --------------------------------------------------------------

    def normative(self, motion, sex="female"):
        return self.raw["layer2_normative_rom"]["motions"][motion][sex]

    @property
    def active_shoulder_flexion(self):
        return float(self.raw["layer2_normative_rom"]["active_shoulder_flexion_reference_deg"]["value"])

    def plausible_max(self, name, sex="female"):
        """Largest believable reading: Soucie group maximum + tolerance (None = no cap)."""
        motion = PLAUSIBILITY_MOTION.get(name)
        if motion is None:
            return None
        return self.normative(motion, sex)["max_deg"] + self.tolerance(name)

    def plausible_min_elbow(self, sex="female"):
        """Most elbow hyperextension that is believable, as (negative) flexion."""
        return -self.normative("elbow_extension", sex)["max_deg"]

    def milestones(self, metric):
        """Gates 2016 daily-task values for a metric, as [{deg, task, source}]."""
        return list(self.raw["layer2_functional_rom"]["upper_limb_milestones"].get(metric, []))

    @property
    def fingers(self):
        return self.raw["layer2_functional_rom"]["finger_functional_rom"]

    def compensation(self, key):
        return self.raw["layer2_compensation"][key]

    @property
    def trunk_threshold(self):
        return float(self.tol["trunk"]["compensation_threshold_deg"])

    @property
    def reach(self):
        return self.raw["layer2_compensation"]["reach_trunk_share"]

    # --- signal processing and scoring -------------------------------------------

    def signal(self, key):
        v = self.raw["signal_processing"][key]
        return v["value"] if isinstance(v, dict) and "value" in v else v

    @property
    def max_cues_per_set(self):
        return int(self.raw["scoring"]["max_technique_cues_per_set"]["value"])


@lru_cache(maxsize=4)
def load(path=config.BENCHMARKS_PATH):
    """The benchmarks, read once."""
    return Benchmarks(Path(path))


# ---------------------------------------------------------------------------
# Therapist profile
# ---------------------------------------------------------------------------

THERAPIST_DEFAULTS = {
    "sex_for_norms": "female",
    "age": None,
    "affected_side": "left",
    "unaffected_side": "right",
    "seat_type": "",
    "supervisor_present": False,
    "mdc_setting": "laboratory",
    "passive_limits_deg": {},
    "contracture_over_quarter_range": {},
    "exercises": {},
}


def load_therapist_profile(path=config.THERAPIST_PROFILE_PATH):
    """Therapist settings merged over safe defaults (a missing file = defaults)."""
    data = {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        pass
    profile = {k: (dict(v) if isinstance(v, dict) else v) for k, v in THERAPIST_DEFAULTS.items()}
    if isinstance(data, dict):
        for key, value in data.items():
            # a section that should be a table but is not is ignored
            if isinstance(THERAPIST_DEFAULTS.get(key), dict) and not isinstance(value, dict):
                continue
            profile[key] = value
    if profile["mdc_setting"] not in ("laboratory", "home"):
        profile["mdc_setting"] = "laboratory"
    if profile["sex_for_norms"] not in ("female", "male"):
        profile["sex_for_norms"] = "female"
    return profile


def exercise_settings(therapist, name):
    """sets, reps, rest_s, hold_s, enabled, supervision for one arm exercise."""
    s = dict(therapist.get("exercises", {}).get(name) or {})
    s.setdefault("enabled", False)
    s.setdefault("supervision", "therapist_led")
    s.setdefault("sets", config.ARM_DEFAULT_SETS)
    s.setdefault("reps", 5)
    s.setdefault("rest_s", config.REST_BETWEEN_SETS_S)
    s.setdefault("hold_s", 1)
    return s


def passive_limit(therapist, side, metric):
    v = (therapist.get("passive_limits_deg") or {}).get(f"{side}.{metric}")
    return float(v) if isinstance(v, (int, float)) else None


def elbow_deficit(therapist, side="left"):
    """Elbow extension deficit from a contracture (degrees), or 0."""
    return passive_limit(therapist, side, "elbow_extension_deficit") or 0.0


def contracture_cap(therapist, name):
    """FMA: contracture over a quarter of the normal range -> the item scores at most 1."""
    flags = therapist.get("contracture_over_quarter_range") or {}
    return 1 if flags.get(name) or flags.get(PLAN_ID.get(name, name)) else 2


# ---------------------------------------------------------------------------
# Geometry helpers shared with the scoring
# ---------------------------------------------------------------------------

def finite(v):
    try:
        return v is not None and bool(np.isfinite(v))
    except TypeError:
        return False


def _sign(direction):
    return 1.0 if direction == "increase" else -1.0


def further(a, b, direction):
    """True when a is further along the movement than b."""
    return _sign(direction) * (a - b) > 0


def _runs(frames, min_frames):
    """First frame of the first run of at least min_frames consecutive frames, or None."""
    frames = sorted(set(int(i) for i in frames))
    start, prev, n = None, None, 0
    for i in frames:
        if prev is not None and i == prev + 1:
            n += 1
        else:
            start, n = i, 1
        prev = i
        if n >= min_frames:
            return start
    return None


# ---------------------------------------------------------------------------
# Layer 1: one repetition
# ---------------------------------------------------------------------------

@dataclass
class RepResult:
    fma_style_score: int
    coaching_success: bool
    with_compensation: bool
    peak: float
    reasons: list
    broken: list                  # form rules broken (debounced), first first
    compensations: list           # compensation flags (debounced)
    cue: str = None               # rule id for the one cue, or None
    feedback_key: str = "one_cue"
    moved: float = float("nan")


def score_rep(primary, direction, tolerance, personal_target=None, fma_threshold=None,
              form_frames=None, compensation_frames=None, start_ok=True,
              min_frames=config.ARM_START_HOLD_FRAMES, max_score=2, at_scale_end=False):
    """
    Score one repetition.

    primary              filtered main measure per frame (start ... peak ... return)
    direction            "increase" or "decrease"
    tolerance            measurement tolerance of that measure
    personal_target      today's target from the ladder (None: no coaching target)
    fma_threshold        the FMA item's number (e.g. 90), or None
    form_frames          {rule id: [frame indices where it was broken]}
    compensation_frames  {flag: [frame indices]}
    A rule counts only when broken for min_frames consecutive frames (debounce).
    max_score            2, or 1 for items the webcam cannot fully score
                         (resistance) or with a contracture flag
    at_scale_end         the threshold is the end of what can be measured
                         (0 degrees elbow extension: 2D flexion is never
                         below 0), so reaching it within tolerance counts

    FMA-style score (conservative, FMA 2026 "a lower score if uncertain"):
      0  moved less than the tolerance, or the start position was not obtained
      1  moved, but below the threshold, within +/- tolerance of it, a form
         rule broken before reaching it, or any compensation
      2  clearly past the threshold, every form rule held, no compensation
    Coaching success (benefit of the doubt): peak >= target - tolerance and
    every form rule held. A compensation does not cancel it: the rep counts,
    gets one cue and is logged.
    """
    form_frames = form_frames or {}
    compensation_frames = compensation_frames or {}
    sgn = _sign(direction)
    vals = [float(v) for v in primary if finite(v)]
    if not vals or not start_ok:
        return RepResult(0, False, False, float("nan"), ["start_position_not_obtained"],
                         [], [], "start", "encourage_and_demo")
    start = vals[0]
    peak = max(vals, key=lambda v: sgn * v)
    moved = sgn * (peak - start)
    if moved < tolerance:
        return RepResult(0, False, False, peak, ["no_movement_beyond_tolerance"],
                         [], [], None, "encourage_and_demo", moved)

    broken = {k: _runs(v, min_frames) for k, v in form_frames.items()}
    broken = {k: i for k, i in broken.items() if i is not None}
    comp = {k: _runs(v, min_frames) for k, v in compensation_frames.items()}
    comp = {k: i for k, i in comp.items() if i is not None}
    broken_ids = sorted(broken, key=broken.get)
    comp_ids = sorted(comp, key=comp.get)

    reasons = []
    score = 1
    if fma_threshold is not None:
        if at_scale_end:
            limit = fma_threshold + sgn * -tolerance      # e.g. <= 0 + 5 for extension
            cross = next((i for i, v in enumerate(vals) if sgn * (v - limit) >= 0), None)
            clear = cross is not None
        else:
            cross = next((i for i, v in enumerate(vals) if sgn * (v - fma_threshold) >= 0), None)
            clear = sgn * (peak - fma_threshold) > tolerance
        form_before = any(i <= (cross if cross is not None else len(vals)) for i in broken.values())
        if cross is None:
            reasons.append("below_fma_threshold")
        elif not clear:
            reasons.append("within_uncertainty_band")
        if form_before:
            reasons.append("form_broken_before_threshold")
        if comp:
            reasons.append("compensation")
        if cross is not None and clear and not form_before and not comp:
            score = 2
    elif comp or broken:
        reasons.append("compensation" if comp else "form_broken")
    score = min(score, max_score)

    if personal_target is None:
        reached = True
    else:
        reached = sgn * (peak - personal_target) >= -tolerance
    success = reached and not broken
    if not reached:
        reasons.append("below_personal_target")
    cue = None
    if broken_ids:
        cue = broken_ids[0]
    elif comp_ids:
        cue = comp_ids[0]
    elif not reached:
        cue = "more_range"
    if success and not comp:
        feedback = "praise_specific"
    elif success:
        feedback = "praise_plus_one_cue"
    else:
        feedback = "one_cue"
    return RepResult(score, success, bool(comp), peak, reasons, broken_ids, comp_ids,
                     cue, feedback, moved)


def claim_improvement(current_best, reference_best, threshold, direction="increase"):
    """True only when the change is at least the MDC (or provisional threshold)."""
    if threshold is None or not finite(current_best) or not finite(reference_best):
        return False
    return _sign(direction) * (current_best - reference_best) >= threshold


def is_plausible(value, upper=None, lower=None):
    """Readings outside the normative maximum + tolerance are tracking errors, never successes."""
    if not finite(value):
        return False
    if upper is not None and value > upper:
        return False
    if lower is not None and value < lower:
        return False
    return True


# ---------------------------------------------------------------------------
# Layer 3: calibration, target ladder, progression
# ---------------------------------------------------------------------------

def side_summary(peaks, direction="increase", starts=None):
    """Best (FMA: best performance counts) and mean of the recorded reps of one side."""
    peaks = [float(p) for p in peaks if finite(p)]
    if not peaks:
        return None
    best = max(peaks) if direction == "increase" else min(peaks)
    out = {"best": round(best, 2), "mean": round(float(np.mean(peaks)), 2), "n": len(peaks),
           "reps": [round(p, 2) for p in peaks]}
    starts = [float(s) for s in (starts or []) if finite(s)]
    if starts:
        out["start"] = round(float(np.mean(starts)), 2)
    return out


def symmetry_index(affected, unaffected, direction="increase"):
    """
    Affected best / unaffected best. For movements that go down (elbow
    extension) the ratio of the excursions from the start is used instead.
    """
    if not affected or not unaffected:
        return None
    if direction == "increase":
        a, u = affected.get("best"), unaffected.get("best")
    else:
        a = affected.get("start", float("nan")) - affected.get("best", float("nan"))
        u = unaffected.get("start", float("nan")) - unaffected.get("best", float("nan"))
    if not finite(a) or not finite(u) or abs(u) < 1e-6:
        return None
    return round(float(a / u), 3)


@dataclass
class TargetLadder:
    baseline: float
    ceiling: float
    step: float
    direction: str = "increase"
    milestones: list = field(default_factory=list)

    def target(self, level):
        t = self.baseline + _sign(self.direction) * max(0, int(level)) * self.step
        return float(min(self.ceiling, t) if self.direction == "increase" else max(self.ceiling, t))

    @property
    def max_level(self):
        return int(math.ceil(abs(self.ceiling - self.baseline) / max(self.step, 1e-9)))

    def milestones_crossed(self, previous_best, new_best):
        """Milestones between the old and the new best (each is announced once)."""
        sgn = _sign(self.direction)
        return [m for m in self.milestones
                if sgn * (m["deg"] - previous_best) > 0 and sgn * (new_best - m["deg"]) >= 0]


def build_ladder(baseline, unaffected_best=None, tolerance=5.0, milestones=(),
                 normative_ceiling=None, therapist_limit=None, direction="increase"):
    """
    target_k = baseline + k * step, up to the ceiling
    ceiling  = the nearest of (unaffected side, therapist limit, norm), never below the baseline
    step     = the tolerance: a smaller step cannot be detected (rule T3)
    """
    caps = [c for c in (unaffected_best, normative_ceiling, therapist_limit) if finite(c)]
    if direction == "increase":
        ceiling = max(min(caps), baseline) if caps else baseline
        inside = [m for m in milestones if baseline < m["deg"] <= ceiling]
        inside.sort(key=lambda m: m["deg"])
    else:
        ceiling = min(max(caps), baseline) if caps else baseline
        inside = [m for m in milestones if ceiling <= m["deg"] < baseline]
        inside.sort(key=lambda m: -m["deg"])
    return TargetLadder(float(baseline), float(ceiling), float(tolerance), direction, inside)


def start_level(level, days_since_last):
    """Two missed sessions (MISSED_SESSION_DAYS): restart one level lower, no fuss."""
    level = max(0, int(level or 0))
    if days_since_last is not None and days_since_last >= config.MISSED_SESSION_DAYS:
        return max(0, level - 1)
    return level


def next_level(level, sessions, pain_delta=None, new_shoulder_pain=False):
    """
    Level for the next session (plan 9.4, proposed project defaults).

    sessions  this exercise's recent normal sessions, oldest first, each
              {"success_rate", "clean_success_rate", "compensation_rate"}
    Returns (level, change) with change "raised", "lowered", "pause" or None.
    Pain is not asked in the app yet: pain_delta / new_shoulder_pain are for
    when it is.
    """
    level = max(0, int(level or 0))
    if new_shoulder_pain or (pain_delta is not None and pain_delta >= 2):
        return level, "pause"
    if not sessions:
        return level, None
    last = sessions[-1]
    success = last.get("success_rate")
    compensation = last.get("compensation_rate") or 0.0
    if finite(success) and (success < config.LEVEL_LOWER_AT or compensation > 0.5):
        return max(0, level - 1), "lowered" if level > 0 else None
    if len(sessions) >= 2:
        prev = sessions[-2]
        clean = [s.get("clean_success_rate") for s in sessions[-2:]]
        not_rising = (prev.get("compensation_rate") or 0.0) >= compensation
        if all(finite(c) and c >= config.LEVEL_RAISE_AT for c in clean) and not_rising:
            return level + 1, "raised"
    return level, None


# ---------------------------------------------------------------------------
# Specific scales
# ---------------------------------------------------------------------------

def fma33_time_score(t_affected_s, t_unaffected_s, completed_reps=config.NOSE_TOUCHES):
    """FMA item 33: affected arm < 2.0 s slower -> 2; 2.0-5.9 s -> 1; >= 6 s or < 5 reps -> 0."""
    if completed_reps < config.NOSE_TOUCHES or not finite(t_affected_s) or not finite(t_unaffected_s):
        return 0
    diff = t_affected_s - t_unaffected_s
    if diff < 2.0:
        return 2
    if diff < 6.0:
        return 1
    return 0


def fma32_dysmetria_score(errors, bench=None):
    """FMA item 32 from the touch errors (eye distances): on the nose 2, slight 1, else 0 (thresholds proposed)."""
    errors = [e for e in errors if finite(e)]
    if not errors:
        return 0
    cfg = (bench or load()).exercise("finger_to_nose_timed")["dysmetria_thresholds"]
    worst = max(errors)
    if worst <= cfg["on_nose_max"]:
        return 2
    if worst <= cfg["slight_max"]:
        return 1
    return 0


def rps_trunk_score(trunk_share, target, hand_arrived=True, elbow_almost_full=True, bench=None):
    """Levin 2004 trunk displacement component (0-3) from the trunk's share of the hand's movement."""
    cfg = (bench or load()).reach
    if not finite(trunk_share):
        return 0
    if target == "close":
        c = cfg["close_target"]
        if trunk_share >= c["score0_share"] - 1e-9:
            return 0
        if trunk_share > c["score1_min_share"]:
            return 1
        if trunk_share <= c["score3_max_share"]:
            return 3
        return 2
    c = cfg["far_target"]
    if trunk_share > c["score0_share"] and not hand_arrived:
        return 0
    if trunk_share >= c["score1_share"] - c["appropriate_tolerance"]:
        return 1
    if abs(trunk_share - c["appropriate_share"]) <= c["appropriate_tolerance"] and elbow_almost_full:
        return 3
    return 2


FINGER_ORDER = ("index", "middle", "ring", "pinky")


def bain_joints(joint_flexion):
    """features.joint_flexion (finger -> [MCP, PIP, DIP], thumb -> [CMC, MCP, IP]) -> named joints."""
    out = {f: {"MCP": float(v[0]), "PIP": float(v[1]), "DIP": float(v[2])}
           for f, v in joint_flexion.items() if f in FINGER_ORDER}
    if "thumb" in joint_flexion:
        t = joint_flexion["thumb"]
        out["thumb"] = {"MCP": float(t[1]), "IP": float(t[2])}
    return out


def hand_open_score(flex, tolerance, bench=None):
    """
    FMA 25-style mass extension (all five fingers straight within the
    tolerance) and Bain 2015 "open enough for 90% of daily tasks".
    """
    open_ = (bench or load()).fingers["open_hand_pre_grasp_targets_deg"]
    fingers = [f for f in ("index", "middle", "ring", "pinky", "thumb") if f in flex]
    straight = {f: all(v <= tolerance for v in flex[f].values()) for f in fingers}
    functional = all(flex[f]["MCP"] <= open_["MCP_max_flexion"] + tolerance
                     and flex[f]["PIP"] <= open_["PIP_max_flexion"] + tolerance
                     and flex[f]["DIP"] <= open_["DIP_max_flexion"] + tolerance
                     for f in FINGER_ORDER if f in flex)
    some = any(min(flex[f].values()) < 45 for f in fingers)     # proposed "some extension"
    score = 2 if fingers and all(straight.values()) else (1 if some else 0)
    return {"fma25_style": score, "bain_functional_open": bool(functional),
            "fingers_short": [f for f, ok in straight.items() if not ok]}


def hand_close_score(flex, tips_to_palm, thumb_outside, touch_max=None, bench=None):
    """FMA 24-style mass flexion (fingertips on the palm, thumb outside) and Bain 2015 functional grasp."""
    b = bench or load()
    close = b.fingers["closed_hand_grasp_targets_deg"]
    if touch_max is None:
        touch_max = b.exercise("grip_release")["close_phase"]["fma24_fingertip_to_palm_max"]
    touching = all(tips_to_palm.get(f, 9.0) <= touch_max for f in FINGER_ORDER)
    functional = all(flex[f]["MCP"] >= close["MCP_min_flexion"] and flex[f]["PIP"] >= close["PIP_min_flexion"]
                     and flex[f]["DIP"] >= close["DIP_min_flexion"] for f in FINGER_ORDER if f in flex)
    some = any(flex[f]["PIP"] > 20 for f in FINGER_ORDER if f in flex)   # proposed "some flexion"
    score = 2 if (touching and thumb_outside) else (1 if some else 0)
    return {"fma24_style": score, "bain_functional_grasp": bool(functional)}


def pinch_score(gap, other_gaps, bench=None):
    """
    FMA 28-style pincer grasp position: thumb and index pads together
    (gap <= contact, proposed) while the other fingertips stay away from the
    thumb. At most 1: the tug on the pencil cannot be felt by a webcam.
    """
    cfg = (bench or load()).exercise("thumb_opposition")
    contact = cfg["contact_threshold"]["value"]
    if not finite(gap):
        return 0
    others_away = all(g > contact for g in other_gaps if finite(g))
    return min(int(cfg.get("fma_max_score_webcam", 1)), 1 if gap <= contact and others_away else 0)
