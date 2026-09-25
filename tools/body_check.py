"""
Body tracking check for the arm exercises (benchmarks/BENCHMARK_PLAN.md, section 13).

Step 1, jitter: sit still in the exercise's view for a while. The angles
should wobble less than the measurement tolerance of the task.

    python -m tools.body_check --seconds 20
    python -m tools.body_check --video still.mp4 --headless

Step 2, agreement: hold one arm still at an angle measured with a phone
goniometer app, and log what the camera reads next to it. Repeat for about
5 angles on 2 people and compare the mean difference and range with
Lazem 2026 (the report's reliability section).

    python -m tools.body_check --seconds 8 --joint left.shoulder_elevation \\
        --goniometer 90 --note "Ryan, sagittal, 1.5 m"

Per-frame angles go to data/body_check.csv; agreement rows are appended to
data/angle_agreement.csv. The overlay shows the angles live, the view
(side-on / facing / halfway) and whether the arm is inside the box.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from rehab import benchmarks, body, config, features
from rehab.filters import LowPassBank
from rehab.Sense import Sense
from tools.tracking_check import jitter

MEASURES = ("shoulder_elevation", "elbow_flexion", "wrist_extension")
KEYS = [f"{s}.{m}" for s in body.SIDES for m in MEASURES] + ["trunk_angle", "view_ratio"]
# tolerance for each measure (degrees): the smallest of the tasks using it
TOLERANCE = {"shoulder_elevation": 5.0, "elbow_flexion": 5.0, "wrist_extension": 5.0,
             "trunk_angle": 5.0}


def row_of(f):
    row = {"t": round(f.t, 3), "present": int(f.present), "view": f.view or ""}
    if not f.present:
        return row
    for side in body.SIDES:
        for m in MEASURES:
            row[f"{side}.{m}"] = f.arm[side].get(m, float("nan"))
    row["trunk_angle"] = f.trunk_angle
    row["view_ratio"] = f.view_ratio
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", help="analyse a recording instead of the webcam")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--csv", default=str(config.DATA_DIR / "body_check.csv"))
    p.add_argument("--joint", default="left.shoulder_elevation",
                   help="side.measure compared with --goniometer")
    p.add_argument("--goniometer", type=float, help="angle read from a goniometer (degrees)")
    p.add_argument("--note", default="", help="who, view, distance ... (agreement log)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-mirror", action="store_true")
    args = p.parse_args()

    sense = Sense(args.video if args.video else args.camera, mirror=not args.no_mirror)
    bank = LowPassBank(fs=sense.fps, fc=config.LOW_PASS_HZ)
    hip_gate = body.HipGate()
    raw_rows, smooth_rows = [], []
    while True:
        frame, t = sense.read()
        if frame is None or (not args.video and t > args.seconds):
            break
        size = (frame.shape[1], frame.shape[0])
        # the same as the app (main.body_features): hands go to the arm they are at
        f = body.extract(sense.observe_pose(frame, t), t, size, hip_gate=hip_gate)
        hands = {side: features.extract(obs, t, size, side.capitalize())
                 for side, obs in body.assign_hands(sense.observe(frame, t), f).items()}
        body.attach_hands(f, hands)
        raw_rows.append(row_of(f))
        body.smooth(f, bank)
        smooth_rows.append(row_of(f))
        if not args.headless:
            show(frame, f)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    sense.close()
    cv2.destroyAllWindows()

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filtered", "t", "present", "view"] + KEYS)
        w.writeheader()
        for flag, rows in ((0, raw_rows), (1, smooth_rows)):
            for r in rows:
                w.writerow({"filtered": flag, **r})

    present = [r for r in raw_rows if r["present"]]
    if not raw_rows:
        print("No frames.")
        return
    print(f"\nFrames: {len(raw_rows)}   body seen: {len(present) / len(raw_rows):.0%}")
    if len(present) < 20:
        print("Too few frames with a body to judge jitter.")
        return
    views = [r["view"] for r in present if r["view"]]
    if views:
        print("View: " + ", ".join(f"{v} {views.count(v) / len(views):.0%}" for v in sorted(set(views))))
    s_present = [r for r in smooth_rows if r["present"]]
    print(f"\n{'measure':28s} {'raw jitter':>11s} {'filtered':>10s} {'tolerance':>10s}")
    for key in KEYS[:-1]:
        values = [r[key] for r in present]
        if not np.isfinite(values).any():
            continue
        raw_j = jitter([r["t"] for r in present], values)
        smooth_j = jitter([r["t"] for r in s_present], [r[key] for r in s_present])
        tol = TOLERANCE[key.split(".")[-1]]
        flag = "  <-- above tolerance" if smooth_j >= tol else ""
        print(f"{key:28s} {raw_j:11.2f} {smooth_j:10.2f} {tol:10.1f}{flag}")
    print(f"\nPer-frame angles written to {out}")

    if args.goniometer is not None:
        agreement(s_present, args)


def agreement(rows, args):
    """Camera angle (median of the still period) next to the goniometer, appended to a log."""
    values = np.array([r.get(args.joint, np.nan) for r in rows], float)
    values = values[np.isfinite(values)]
    if not len(values):
        print(f"No readings for {args.joint}.")
        return
    measured = float(np.median(values))
    diff = measured - args.goniometer
    measure = args.joint.split(".")[-1]
    tol = {"shoulder_elevation": benchmarks.load().tolerance("shoulder_flexion_raise"),
           "elbow_flexion": benchmarks.load().elbow_tolerance("shoulder_flexion_raise")}.get(measure, 5.0)
    path = config.DATA_DIR / "angle_agreement.csv"
    new = not path.is_file()
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "joint", "goniometer_deg", "camera_deg", "difference_deg",
                        "camera_sd_deg", "tolerance_deg", "within_tolerance", "note"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), args.joint, args.goniometer,
                    round(measured, 2), round(diff, 2), round(float(np.std(values)), 2), tol,
                    "yes" if abs(diff) <= tol else "no", args.note])
    print(f"\n{args.joint}: camera {measured:.1f} deg, goniometer {args.goniometer:.1f} deg, "
          f"difference {diff:+.1f} deg (tolerance {tol:.1f}). Logged to {path}")


def show(frame, f):
    h, w = frame.shape[:2]
    m = config.SETUP_EDGE_MARGIN
    cv2.rectangle(frame, (int(m * w), int(m * h)), (int((1 - m) * w), int((1 - m) * h)),
                  (80, 200, 80) if f.present else (0, 220, 255), 3)
    if f.present:
        pts = f.points.astype(int)
        for s in body.SIDES:
            for a, b in (("shoulder", "elbow"), ("elbow", "wrist"), ("hip", "shoulder")):
                cv2.line(frame, tuple(pts[body.POSE[f"{s}_{a}"]]), tuple(pts[body.POSE[f"{s}_{b}"]]),
                         (230, 230, 0) if s == "left" else (200, 200, 200), 3)
        lines = [f"view {f.view} ({f.view_ratio:.2f})", f"trunk {f.trunk_angle:+.0f}"]
        for s in body.SIDES:
            a = f.arm[s]
            lines.append(f"{s}: shoulder {a['shoulder_elevation']:.0f}  elbow {a['elbow_flexion']:.0f}"
                         + (f"  wrist {a['wrist_extension']:.0f}" if np.isfinite(a["wrist_extension"]) else ""))
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imshow("Body check", frame)


if __name__ == "__main__":
    main()
