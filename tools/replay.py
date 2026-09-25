"""
Re-run the finger tapping part of a verbose log through the current detector.

    python -m tools.replay data/verbose/eleanor_20260925-101317
    python -m tools.replay --latest
    python -m tools.replay <folder> --cal-from <other folder>   # calibrate from another log

The logged hands of every frame of the exercise stage go through the same
path as in the app: the hand is picked (features.choose_hand, palm down),
its features extracted and smoothed, and the Coach checks the quality
before the detector sees the frame. Sets, pauses and lost hands in the log
start a set or resume here too. The replay runs its own sequence of
prompts, so after its first difference from the log its prompts may differ.

Prints one timeline, in seconds from the start of the log:
  log     what the app said and detected at the time
  replay  what the current code detects (start / end of a lift, with the
          finger the log was prompting at that moment), the quality
          problems it would show, and when it measures the resting
          position again (and why)

The calibration comes from the log's own finger tapping calibration frames,
else from the stored calibration if it is in the current format, else from
--cal-from, else none (the thresholds are then min_lift).

Use it to check a change of the detector or of its settings in
rehab/config.py on a real session before trying it live.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from rehab import config, features
from rehab.Act import SilentSpeaker
from rehab.exercises.finger_tapping import FingerTapping
from rehab.filters import FeatureFilter
from rehab.Think import Coach
from tools.verbose_summary import _read_jsonl, latest

EXERCISE = FingerTapping.name


def _meta(folder):
    path = Path(folder) / "session.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def image_size(folder):
    """(width, height) of the camera image: logged, else from the video, else 1280 x 720."""
    for row in _read_jsonl(Path(folder) / "frames.jsonl.gz"):
        f = row.get("features") or {}
        if f.get("image_size"):
            return tuple(int(v) for v in f["image_size"])
        if f.get("present"):
            break                       # an older log: not logged
    video = Path(folder) / "video.mp4"
    if video.is_file():
        cap = cv2.VideoCapture(str(video))
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        if w and h:
            return int(w), int(h)
    return 1280, 720


def _observations(row):
    return [features.HandObservation(image=np.asarray(h["image"], dtype=float),
                                     world=np.asarray(h["world"], dtype=float),
                                     handedness=h.get("handedness"),
                                     handedness_score=h.get("score") or 0.0,
                                     gesture=h.get("gesture"),
                                     gesture_score=h.get("gesture_score") or 0.0)
            for h in row.get("hands") or []]


def frames(folder, hand, size):
    """(row, t, HandFeatures) of every logged frame, the hand picked and smoothed like main.py."""
    smoother = FeatureFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
                             config.ONE_EURO_D_CUTOFF)
    for row in _read_jsonl(Path(folder) / "frames.jsonl.gz"):
        t = row.get("t") or 0.0
        obs = features.choose_hand(_observations(row), hand, palm_down=True)
        yield row, t, features.smooth(features.extract(obs, t, size, hand), smoother)


def calibration_from_frames(folder, hand):
    """The finger tapping calibration measured again from the log's calibration holds."""
    steps = {s.name: s for s in FingerTapping.calibration_steps()}
    samples = {name: [] for name in steps}
    for row, _, f in frames(folder, hand, image_size(folder)):
        st = row.get("state") or {}
        cal = st.get("calibration") or {}
        if (st.get("stage") == "calibrating" and st.get("exercise") == EXERCISE
                and cal.get("stage") == "hold" and cal.get("step") in steps
                and f.quality_problem(palm_down=True) is None):
            samples[cal["step"]].append(steps[cal["step"]].extract(f))
    if not all(samples.values()):
        return None
    return {name: {k: float(np.median([s[k] for s in values])) for k in values[0]}
            for name, values in samples.items()}


def pick_calibration(folder, hand, cal_from=None):
    """(calibration steps or None, where they came from)."""
    if cal_from:
        cal = calibration_from_frames(cal_from, hand)
        if cal and FingerTapping.calibration_valid(cal):
            return cal, f"calibration frames of {cal_from}"
    cal = calibration_from_frames(folder, hand)
    if cal and FingerTapping.calibration_valid(cal):
        return cal, "this log's calibration frames"
    stored = (((_meta(folder).get("profile") or {}).get("calibration") or {})
              .get(EXERCISE) or {}).get("steps")
    if stored and FingerTapping.calibration_valid(stored):
        return stored, "the stored calibration"
    return None, "none (thresholds = min_lift)"


def _logged(folder, t0):
    """What the app said and detected during finger tapping, from events.jsonl."""
    out, active = [], False
    for ev in _read_jsonl(Path(folder) / "events.jsonl"):
        kind = ev.get("type")
        if kind == "stage":
            active = ev.get("exercise") == EXERCISE and ev.get("stage") in (
                "intro", "calibration_offer", "calibrating", "exercise", "rest")
        t = (ev.get("t") or 0.0) - t0
        if kind == "detect" and ev.get("exercise") == EXERCISE:
            if ev.get("event") == "start":
                ok = "ok" if ev.get("correct") else "WRONG"
                out.append((t, "log", f"start {ev.get('finger'):6s} prompted {ev.get('prompted')} {ok}"))
            else:
                iso = ev.get("isolation")
                out.append((t, "log", f"end   {ev.get('finger'):6s}"
                            + (f" isolation {iso:.2f}" if iso is not None else "")))
        elif kind == "speech" and active:
            out.append((t, "log", f'said "{ev.get("text")}"'))
    return out


def replay(folder, hand=None, cal_from=None, params=None):
    """
    Run the log's finger tapping frames through the current code.

    Returns (timeline, info): timeline is a list of (t, source, text) in
    seconds from the start of the log; info has the calibration source,
    the thresholds and the replay's detections.
    """
    folder = Path(folder)
    meta = _meta(folder)
    hand = hand or (meta.get("profile") or {}).get("affected_hand") or config.AFFECTED_HAND
    cal, cal_source = pick_calibration(folder, hand, cal_from)
    ex = FingerTapping(calibration=cal, params=params)
    speaker = SilentSpeaker()
    timeline, starts = [], []
    size = image_size(folder)
    t0 = None
    now = [0.0]
    logged_prompt = [None]

    def trace(kind, **data):
        t = now[0] - t0
        if kind == "detect":
            if data.get("event") == "start":
                starts.append((t, data["finger"]))
                timeline.append((t, "replay", f"start {data['finger']:6s} prompted "
                                 f"{data.get('prompted')} (log prompted {logged_prompt[0]})"))
            else:
                timeline.append((t, "replay", f"end   {data['finger']:6s} isolation "
                                 f"{data.get('isolation', float('nan')):.2f} ({data.get('reason')})"))
        elif kind == "quality":
            timeline.append((t, "replay", f"quality {data.get('problem') or 'ok'}"))
        elif kind == "baseline" and data.get("reason") not in ("set", "interrupt"):
            timeline.append((t, "replay", f"resting position measured again ({data.get('reason')})"))

    ex.trace = trace
    coach = Coach(ex, speaker, None, hand=hand, trace=trace)
    set_no, running = None, False
    for row, t, f in frames(folder, hand, size):
        t0 = t if t0 is None else t0
        now[0] = t
        st = row.get("state") or {}
        if st.get("stage") != "exercise" or st.get("exercise") != EXERCISE or st.get("paused"):
            running = False
            continue
        logged_prompt[0] = (st.get("detector") or {}).get("prompted")
        if st.get("set") != set_no:
            set_no = st.get("set")
            timeline.append((t - t0, "replay", f"set {set_no} starts"))
            coach.start_set(set_no or 1, t)
        elif not running:
            coach.resume(t)
        running = True
        coach.update(f, t)

    timeline += _logged(folder, t0 or 0.0)
    timeline.sort(key=lambda e: (e[0], e[1] != "log"))
    info = {"calibration": cal_source, "lift_threshold": ex.lift_threshold,
            "thresholds": dict(ex.thresholds), "image_size": size, "hand": hand,
            "starts": starts}
    return timeline, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", help="a folder written by python main.py --verbose")
    p.add_argument("--latest", action="store_true", help="the newest verbose log")
    p.add_argument("--cal-from", help="take the calibration from this log's calibration frames")
    p.add_argument("--hand", choices=("Left", "Right"), help="the affected hand (default: the log's)")
    p.add_argument("--quiet-log", action="store_true", help="leave out what the app said")
    args = p.parse_args()
    folder = Path(args.folder) if args.folder else (latest() if args.latest else None)
    if folder is None:
        p.error("give a folder, or --latest")
    timeline, info = replay(folder, hand=args.hand, cal_from=args.cal_from)
    print(f"Replay of {folder}")
    print(f"hand {info['hand']}, image {info['image_size'][0]} x {info['image_size'][1]}, "
          f"calibration: {info['calibration']}")
    print("thresholds (hand sizes): " + ", ".join(f"{k} {v:.3f}" for k, v in info["thresholds"].items()))
    print()
    for t, source, text in timeline:
        if args.quiet_log and source == "log" and text.startswith("said"):
            continue
        print(f"{t:8.2f}s  {source:6s}  {text}")
    print()
    order = " -> ".join(finger for _, finger in info["starts"]) or "none"
    print(f"Replay detections: {order}")


if __name__ == "__main__":
    main()
