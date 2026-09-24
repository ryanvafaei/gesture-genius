"""
What a verbose session log says about tracking, calibration and detection
(python main.py --verbose writes the logs, rehab/verbose.py).

    python -m tools.verbose_summary data/verbose/eleanor_20260925-143012
    python -m tools.verbose_summary --latest        # the newest log (hers or a guest's)

Writes summary.md and summary.json into the log folder and prints the
Markdown. It reads the frames one at a time, so long sessions fit in memory.

  Session      duration, frame rate, loop time, video frames dropped
  Tracking     how often a hand was seen (per kind of stage), the right hand,
               too far, palm turned away, handedness flips, drop-outs, and
               how long each quality message was on
  Noise floor  during the calibration holds (the hand held still): how much
               each measure and each landmark jitters
  Detection    per exercise and finger: prompts, found, wrong fingers (which
               for which), how strong the prompted finger's signal was
               compared with its threshold, the neighbours' signal, near
               misses, and suggested settings (e.g. finger_scale in
               rehab/config.py) where the numbers point to one
  Ratings      finger counts seen while a rating was asked, and how often a
               frame's count differed from the voted count (flicker)

Numbers from one session are a hint, not a verdict: look at a few sessions
before changing a setting.
"""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rehab import config

FINGERS = ("index", "middle", "ring", "pinky")
STAGE_GROUPS = {"exercise": "exercise", "calibrating": "calibration", "rating": "rating",
                "setup_check": "setup"}


def _read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    if not Path(path).is_file():
        return
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue            # a line cut off when the app was stopped


def _stats(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return None
    return {"n": int(len(v)), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}


def _r(v, digits=3):
    return "" if v is None else (round(v, digits) if isinstance(v, float) else v)


class Segment:
    """One prompt of a sequence exercise: from the prompt until the next one."""

    def __init__(self, exercise, finger):
        self.exercise, self.finger = exercise, finger
        self.own = []           # prompted finger's signal per frame (lift score / closeness)
        self.others = []        # strongest other finger per frame
        self.neighbours = []    # strongest neighbour per frame
        self.found = False
        self.wrong = []


def summarize(folder):
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text()) if (folder / "session.json").is_file() else {}
    cfg = meta.get("config", {})
    ex_cfg = cfg.get("EXERCISES", config.EXERCISES)

    # --- frames, streamed ---------------------------------------------------------
    n = 0
    t_first = t_last = None
    loop = []
    seen = defaultdict(lambda: [0, 0])          # stage group -> [frames, with a hand]
    quality = Counter()                         # problem -> frames
    track = defaultdict(list)                   # in exercise / calibration
    flips, last_hand = 0, None
    dropouts, run, prev_t = [], 0.0, None
    hold = defaultdict(lambda: defaultdict(list))   # (exercise, step) -> measure -> values
    world = defaultdict(list)                       # (exercise, step) -> world landmarks
    segments, seg = [], None
    rating_raw = defaultdict(Counter)
    rating_flicker = [0, 0]

    for row in _read_jsonl(folder / "frames.jsonl.gz"):
        n += 1
        t = row.get("t") or 0.0
        t_first = t if t_first is None else t_first
        dt = 0.0 if prev_t is None else max(0.0, t - prev_t)
        prev_t = t_last = t
        if row.get("loop_ms") is not None:
            loop.append(row["loop_ms"])
        st = row.get("state") or {}
        stage = st.get("stage")
        group = STAGE_GROUPS.get(stage, "other")
        hands = row.get("hands") or []
        f = row.get("features") or {}
        seen[group][0] += 1
        seen[group][1] += bool(hands)
        if st.get("quality"):
            quality[st["quality"]] += 1

        if group in ("exercise", "calibration") and not st.get("paused"):
            present = bool(f.get("present"))
            track["present"].append(present)
            if present:
                track["correct_hand"].append(bool(f.get("correct_hand")))
                track["too_small"].append(bool(f.get("too_small")))
                track["palm_facing"].append(f.get("palm_facing"))
                track["handedness_score"].append(f.get("handedness_score"))
                track["palm_size_image"].append(f.get("palm_size_image"))
                if last_hand is not None and f.get("handedness") != last_hand:
                    flips += 1
                last_hand = f.get("handedness")
                if run:
                    dropouts.append(run)
                run = 0.0
            else:
                run += dt

        # noise floor: calibration holds (the hand is held still)
        cal = st.get("calibration") or {}
        if stage == "calibrating" and cal.get("stage") == "hold" and f.get("present"):
            key = (st.get("exercise"), cal.get("step"))
            for measure in ("openness", "tip_height", "thumb_tip_dist", "thumb_tip_dist_image",
                            "tip_reach"):
                for finger, v in (f.get(measure) or {}).items():
                    hold[key][f"{measure}.{finger}"].append(v)
            if hands and len(world[key]) < 600:
                world[key].append(hands[0]["world"])

        # detection: one segment per prompt of a sequence exercise
        det = st.get("detector") or {}
        if stage == "exercise" and "prompted" in det:
            prompted = det.get("prompted")
            if seg is None or seg.finger != prompted or seg.exercise != st.get("exercise"):
                seg = Segment(st.get("exercise"), prompted) if prompted else None
                if seg is not None:
                    segments.append(seg)
            # the previous finger is often still up when the next is prompted: skip those frames
            busy = det.get("lifted") or det.get("touching")
            if seg is not None and busy in (None, prompted):
                values = det.get("scores") or det.get("closeness") or {}
                if prompted in values:
                    own = values[prompted]
                    others = {k: v for k, v in values.items() if k != prompted and v is not None}
                    near = [k for k in FINGERS if k in others
                            and abs(FINGERS.index(k) - FINGERS.index(prompted)) == 1]
                    lift = "scores" in det
                    pick = max if lift else min
                    seg.own.append(own)
                    if others:
                        seg.others.append(pick(others.values()))
                    if near:
                        seg.neighbours.append(pick(others[k] for k in near))

        rating = st.get("rating")
        if stage == "rating" and rating:
            rating_raw[rating.get("question")][rating.get("raw")] += 1
            rating_flicker[0] += 1
            rating_flicker[1] += rating.get("raw") != rating.get("voted")

    # --- events ---------------------------------------------------------------------
    calibrations, ratings, detections, skips = [], [], [], []
    seg_iter = iter(segments)
    for ev in _read_jsonl(folder / "events.jsonl"):
        kind = ev.get("type")
        if kind == "calibration":
            calibrations.append(ev)
        elif kind == "rating":
            ratings.append(ev)
        elif kind == "skip":
            skips.append(ev)
        elif kind == "detect" and ev.get("event") == "start" and ev.get("counted", True):
            detections.append(ev)
    # attach detections to their segments (same exercise and prompted finger, in order)
    open_by_key = defaultdict(list)
    for s in seg_iter:
        open_by_key[(s.exercise, s.finger)].append(s)
    used = defaultdict(int)
    for ev in detections:
        key = (ev.get("exercise"), ev.get("prompted"))
        lst = open_by_key.get(key, [])
        i = used[key]
        if i >= len(lst):
            continue
        if ev.get("correct"):
            lst[i].found = True
            used[key] += 1
        else:
            lst[i].wrong.append(ev.get("finger"))

    # --- results ----------------------------------------------------------------------
    duration = (t_last - t_first) if n else 0.0
    out = {"folder": str(folder), "user": meta.get("user"), "session_id": meta.get("session_id"),
           "started": meta.get("started"), "versions": meta.get("versions"),
           "session": {"frames": n, "duration_s": duration,
                       "frames_per_s": n / duration if duration > 0 else None,
                       "loop_ms": _stats(loop),
                       "slow_frames_over_50ms": int(sum(1 for x in loop if x > 50)),
                       "video": {k: (meta.get("totals") or {}).get(k)
                                 for k in ("video_frames_written", "video_frames_dropped")}}}

    out["tracking"] = {
        "hand_seen_share": {g: (v[1] / v[0] if v[0] else None) for g, v in seen.items()},
        "hand_present_share": _share(track["present"]),
        "correct_hand_share": _share(track["correct_hand"]),
        "too_far_share": _share(track["too_small"]),
        "palm_turned_away_share": _share([p is not None and p < cfg.get("MIN_PALM_FACING", 0.45)
                                          for p in track["palm_facing"]]),
        "handedness_score": _stats(track["handedness_score"]),
        "palm_size_image": _stats(track["palm_size_image"]),
        "handedness_flips": flips,
        "dropouts": {"count": len(dropouts), "longest_s": max(dropouts, default=0.0),
                     "over_0.5s": sum(1 for d in dropouts if d > 0.5)},
        "quality_message_frames": dict(quality),
    }

    noise = {}
    for (exercise, step), measures in hold.items():
        entry = {m: float(np.std(v)) for m, v in measures.items() if len(v) >= 5}
        if world.get((exercise, step)):
            w = np.array(world[(exercise, step)], dtype=float)       # frames x 21 x 3
            entry["world_landmark_jitter_mm"] = float(np.median(
                np.linalg.norm(w - w.mean(axis=0), axis=2).std(axis=0)) * 1000)
        noise[f"{exercise}/{step}"] = entry
    out["noise_floor"] = noise
    out["calibrations"] = [{"exercise": c.get("exercise"), "attempt": c.get("attempt"),
                            "steps": (c.get("entry") or {}).get("steps")} for c in calibrations]
    out["detection"] = _detection(segments, ex_cfg)
    out["ratings"] = {"answers": [{"question": r.get("question"), "value": r.get("value")}
                                  for r in ratings],
                      "raw_counts": {q: dict(c) for q, c in rating_raw.items()},
                      "flicker_share": (rating_flicker[1] / rating_flicker[0]
                                        if rating_flicker[0] else None)}
    out["skips"] = [{"stage": s.get("stage"), "exercise": s.get("exercise")} for s in skips]
    return out


def _share(values):
    return float(np.mean(values)) if len(values) else None


def _detection(segments, ex_cfg):
    """Per exercise and finger: how strong the prompted finger's signal was, and suggestions."""
    result = {}
    by = defaultdict(list)
    for s in segments:
        if s.own:
            by[(s.exercise, s.finger)].append(s)
    for (exercise, finger), segs in sorted(by.items()):
        lift = exercise == "finger_tapping"
        params = ex_cfg.get(exercise, {})
        # the prompted finger's best moment in each prompt: highest lift score, lowest closeness
        best = [max(s.own) if lift else min(s.own) for s in segs]
        # the strongest other finger at the same time
        others = [(max(s.others) if lift else min(s.others)) if s.others else None for s in segs]
        neighbours = [(max(s.neighbours) if lift else min(s.neighbours)) if s.neighbours else None
                      for s in segs]
        found = sum(s.found for s in segs)
        wrong = Counter(w for s in segs for w in s.wrong)
        entry = {"prompts": len(segs), "found": found, "wrong_fingers": dict(wrong),
                 "prompted_peak": _stats(best), "other_peak": _stats(others),
                 "neighbour_peak": _stats(neighbours)}
        if lift:
            keep = params.get("candidate_keep", 0.75)
            entry["near_misses"] = sum(1 for s, b in zip(segs, best) if not s.found and keep <= b < 1.0)
            entry["suggestion"] = _suggest_lift(finger, best, params)
        else:
            touch = params.get("touch_factor", 0.35)
            entry["near_misses"] = sum(1 for s, b in zip(segs, best) if not s.found and touch <= b < touch * 1.6)
            entry["suggestion"] = _suggest_touch(finger, best, others, params)
        result.setdefault(exercise, {})[finger] = entry
    return result


def _suggest_lift(finger, peaks, params):
    """finger_scale that puts her median lift at about 1.5 x the threshold."""
    peaks = [p for p in peaks if p is not None]
    if len(peaks) < 3:
        return None
    scale = (params.get("finger_scale") or {}).get(finger, 1.0)
    median = float(np.median(peaks))
    if 1.2 <= median <= 2.5:
        return None
    new = round(float(np.clip(scale * median / 1.5, 0.3, 1.5)), 2)
    why = "too weak to count reliably" if median < 1.2 else "far above the threshold (noise could count)"
    return (f"{finger}: median lift {median:.2f} x threshold, {why}; "
            f"try finger_scale[{finger!r}] = {new} (now {scale})")


def _suggest_touch(finger, closest, others, params):
    closest = [c for c in closest if c is not None]
    if len(closest) < 3:
        return None
    touch = params.get("touch_factor", 0.35)
    median = float(np.median(closest))
    margins = [o - c for c, o in zip(closest, others) if c is not None and o is not None]
    notes = []
    if median > touch * 0.8:
        notes.append(f"closest touch at median closeness {median:.2f} (touch counts below {touch}): "
                     "measure the hand again, or raise touch_factor")
    if margins and np.median(margins) < params.get("dominance_margin", 0.15):
        notes.append(f"the next finger was only {np.median(margins):.2f} further away on median: "
                     "the fingers are hard to tell apart here (camera angle, hand turned?)")
    return f"{finger}: " + "; ".join(notes) if notes else None


def markdown(s):
    lines = [f"# Verbose log: {s.get('user')} {s.get('session_id')}\n",
             f"Folder: `{s['folder']}`  Started: {s.get('started')}\n"]
    ses = s["session"]
    loop = ses.get("loop_ms") or {}
    lines += ["## Session\n",
              f"- {ses['frames']} frames in {ses['duration_s']:.0f} s "
              f"({_r(ses.get('frames_per_s'), 1)} per second)",
              f"- loop time: median {_r(loop.get('median'), 1)} ms, 90% below "
              f"{_r(loop.get('p90'), 1)} ms; {ses['slow_frames_over_50ms']} frames over 50 ms",
              f"- video frames written {ses['video'].get('video_frames_written')}, dropped "
              f"{ses['video'].get('video_frames_dropped')}\n"]
    tr = s["tracking"]
    lines += ["## Tracking\n",
              "| | value |", "|---|---|"]
    for key in ("hand_present_share", "correct_hand_share", "too_far_share", "palm_turned_away_share",
                "handedness_flips"):
        lines.append(f"| {key.replace('_', ' ')} | {_r(tr[key])} |")
    for group, v in tr["hand_seen_share"].items():
        lines.append(f"| hand seen during {group} | {_r(v)} |")
    hs = tr.get("handedness_score") or {}
    lines.append(f"| handedness score (median) | {_r(hs.get('median'))} |")
    ps = tr.get("palm_size_image") or {}
    lines.append(f"| palm size in the picture (median) | {_r(ps.get('median'))} |")
    d = tr["dropouts"]
    lines.append(f"| drop-outs (longest) | {d['count']} ({_r(d['longest_s'], 2)} s) |")
    for problem, frames in tr["quality_message_frames"].items():
        lines.append(f"| quality message \"{problem}\" | {frames} frames |")
    lines.append("")
    if s["noise_floor"]:
        lines += ["## Noise floor (calibration holds)\n",
                  "Standard deviation while the hand was held still.\n",
                  "| exercise/step | landmark jitter (mm) | largest measure jitter |", "|---|---|---|"]
        for key, entry in s["noise_floor"].items():
            measures = {k: v for k, v in entry.items() if k != "world_landmark_jitter_mm"}
            worst = max(measures.items(), key=lambda kv: kv[1], default=(None, None))
            lines.append(f"| {key} | {_r(entry.get('world_landmark_jitter_mm'), 2)} | "
                         f"{worst[0]} = {_r(worst[1])} |")
        lines.append("")
    if s["detection"]:
        lines += ["## Detection\n",
                  "Signal: lift score (1 = threshold) for finger tapping, closeness "
                  "(0 = her calibrated touch) for thumb opposition.\n",
                  "| exercise | finger | prompts | found | wrong | prompted (median) | "
                  "neighbour (median) | near misses |", "|---|---|---|---|---|---|---|---|"]
        suggestions = []
        for exercise, fingers in s["detection"].items():
            for finger, e in fingers.items():
                lines.append(f"| {exercise} | {finger} | {e['prompts']} | {e['found']} | "
                             f"{e['wrong_fingers'] or ''} | {_r((e['prompted_peak'] or {}).get('median'))} | "
                             f"{_r((e['neighbour_peak'] or {}).get('median'))} | {e['near_misses']} |")
                if e.get("suggestion"):
                    suggestions.append(f"- {exercise}: {e['suggestion']}")
        lines.append("")
        if suggestions:
            lines += ["### Suggestions\n"] + suggestions + [""]
    rt = s["ratings"]
    if rt["answers"] or rt["raw_counts"]:
        lines += ["## Ratings\n"]
        for a in rt["answers"]:
            counts = rt["raw_counts"].get(a["question"], {})
            lines.append(f"- {a['question']}: answer {a['value']}; fingers seen per frame {counts}")
        lines.append(f"- frames whose count differed from the voted count: {_r(rt['flicker_share'])}\n")
    if s["calibrations"]:
        lines += ["## Calibrations\n"]
        for c in s["calibrations"]:
            lines.append(f"- {c['exercise']} (attempt {c['attempt']}): "
                         f"{json.dumps(c['steps'])[:400]}")
        lines.append("")
    return "\n".join(lines) + "\n"


def latest(root=None):
    """The newest verbose folder under data/ (her own and every guest's)."""
    root = Path(root or config.BASE_DATA_DIR)
    folders = [p.parent for p in root.glob("**/verbose/*/session.json")]
    return max(folders, key=lambda p: p.stat().st_mtime, default=None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", help="a folder written by python main.py --verbose")
    p.add_argument("--latest", action="store_true", help="the newest verbose log")
    args = p.parse_args()
    folder = Path(args.folder) if args.folder else (latest() if args.latest else None)
    if folder is None:
        p.error("give a folder, or --latest")
    s = summarize(folder)
    (folder / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    text = markdown(s)
    (folder / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
