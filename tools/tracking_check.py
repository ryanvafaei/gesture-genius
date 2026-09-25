"""
Phase 1 tracking check: how well does *this* camera see the hand?

Run it before building on an exercise, especially finger tapping (hand flat on
the table, low camera angle) and thumb opposition (fingertips overlap).

    python -m tools.tracking_check --seconds 30 --record tapping.mp4
    python -m tools.tracking_check --video tapping.mp4 --headless

For every frame it logs the features to a CSV and at the end prints:
  * detection rate, correct-hand rate, share of frames "too far"
  * jitter (median 1 s standard deviation) of the key measures, raw and
    after the One Euro filter, next to the size of the movement we need to
    detect. Jitter should be well below the movement.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from rehab import config, features
from rehab.filters import FeatureFilter
from rehab.Sense import Sense

KEYS = (
    [f"openness.{f}" for f in features.FINGERS]
    + [f"spread.{g}" for g in features.GAP_NAMES]
    + [f"thumb_tip_dist.{f}" for f in features.FINGERS]
    + [f"tip_height.{f}" for f in features.FINGERS]
    + [f"tip_rise.{f}" for f in features.FINGERS]
    + [f"mcp.{f}" for f in features.FINGERS]
    + ["thumb_flexion", "palm_facing", "palm_size_image"]
)

# Rough size of the movement each measure has to show (for the report).
NEEDED = {"tip_height": 0.08, "tip_rise": 0.12, "thumb_tip_dist": 0.3, "openness": 0.3,
          "spread": 10.0, "mcp": 15.0, "thumb_flexion": 30.0}


def row_of(f):
    row = {"t": round(f.t, 3), "present": int(f.present), "handedness": f.handedness or "",
           "handedness_score": round(f.handedness_score, 3), "correct_hand": int(f.correct_hand),
           "too_small": int(f.too_small), "gesture": f.gesture or ""}
    if not f.present:
        return row
    for key in KEYS:
        if key.startswith("mcp."):
            row[key] = float(f.joint_flexion[key[4:]][0])
        elif "." in key:
            group, name = key.split(".")
            row[key] = float(getattr(f, group)[name])
        else:
            row[key] = float(getattr(f, key))
    return row


def jitter(times, values, window=1.0):
    times, values = np.asarray(times), np.asarray(values)
    stds = []
    start = 0
    for i in range(len(times)):
        while times[i] - times[start] > window:
            start += 1
        if i - start >= 10:
            stds.append(np.std(values[start:i + 1]))
    return float(np.median(stds)) if stds else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", help="analyse a recording instead of the webcam")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--record", help="also save the raw (unmirrored) webcam video here")
    p.add_argument("--csv", default=str(config.DATA_DIR / "tracking_check.csv"))
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-mirror", action="store_true")
    args = p.parse_args()

    sense = Sense(args.video if args.video else args.camera, mirror=not args.no_mirror)
    smoother = FeatureFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA, config.ONE_EURO_D_CUTOFF)
    writer = None
    raw_rows, smooth_rows = [], []

    while True:
        frame, t = sense.read()
        if frame is None or (not args.video and t > args.seconds):
            break
        if args.record:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            writer.write(cv2.flip(frame, 1) if sense.mirror else frame)
        obs = features.choose_hand(sense.observe(frame, t))
        f = features.extract(obs, t, (frame.shape[1], frame.shape[0]))
        raw_rows.append(row_of(f))
        features.smooth(f, smoother)
        smooth_rows.append(row_of(f))

        if not args.headless:
            if f.present:
                for a, b in features.HAND_CONNECTIONS:
                    cv2.line(frame, tuple(f.image_points[a].astype(int)),
                             tuple(f.image_points[b].astype(int)), (255, 255, 255), 2)
            status = "hand OK" if f.quality_problem() is None else (f.quality_problem() or "")
            cv2.putText(frame, f"{t:5.1f}s  {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Tracking check", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    sense.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(raw_rows[0].keys()) + [k for k in KEYS if k not in raw_rows[0]] if raw_rows else []
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filtered"] + fields)
        w.writeheader()
        for r in raw_rows:
            w.writerow({"filtered": 0, **r})
        for r in smooth_rows:
            w.writerow({"filtered": 1, **r})

    n = len(raw_rows)
    if not n:
        print("No frames.")
        return
    present = [r for r in raw_rows if r["present"]]
    print(f"\nFrames: {n}   hand detected: {len(present) / n:.0%}   "
          f"correct hand: {sum(r['correct_hand'] for r in raw_rows) / n:.0%}   "
          f"too far: {sum(r['too_small'] for r in raw_rows) / n:.0%}")
    if len(present) < 20:
        print("Too few frames with a hand to judge jitter.")
        return
    print(f"\n{'measure':28s} {'raw jitter':>11s} {'filtered':>10s} {'movement':>10s}")
    s_present = [r for r in smooth_rows if r["present"]]
    for key in KEYS:
        group = key.split(".")[0]
        raw_j = jitter([r["t"] for r in present], [r[key] for r in present])
        smooth_j = jitter([r["t"] for r in s_present], [r[key] for r in s_present])
        need = NEEDED.get(group)
        flag = "  <-- noisy" if need and smooth_j > 0.3 * need else ""
        print(f"{key:28s} {raw_j:11.4f} {smooth_j:10.4f} {need if need else '':>10}{flag}")
    print(f"\nPer-frame features written to {out}")
    if len(present) / n < 0.9:
        print("Detection below 90%: try more light, a plain background, or another camera angle.")


if __name__ == "__main__":
    main()
