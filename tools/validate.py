"""
Phase 8 validation: compare the program's rep count with a count made by hand.

Record test videos (different lighting, distances, the left hand), count the
repetitions yourself, then:

    python -m tools.validate grip.mp4 --exercise grip_release --expected 10
    python -m tools.validate taps.mp4 --exercise finger_tapping --expected 2 --calibrate 12

The video is processed exactly as live (same features, filter and exercise
code; timestamps follow the video). Calibration comes from data/profile.json,
or with --calibrate N from the first N seconds of the video (hold the
calibration positions in the order the routine asks for them).

Results are printed and appended to data/validation.csv for the report.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

from rehab import config, features, storage
from rehab.Act import SilentSpeaker
from rehab.calibration import CalibrationRoutine
from rehab.exercises import EXERCISES, create
from rehab.filters import FeatureFilter
from rehab.Sense import Sense
from rehab.Think import Coach

OUT = config.DATA_DIR / "validation.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--exercise", required=True, choices=sorted(EXERCISES))
    p.add_argument("--expected", type=int, help="reps counted by hand")
    p.add_argument("--calibrate", type=float, default=0.0,
                   help="calibrate from the first N seconds of the video")
    p.add_argument("--profile", default=str(config.PROFILE_PATH))
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--note", default="", help="e.g. 'dim light, 60 cm'")
    p.add_argument("-v", "--verbose", action="store_true", help="print what the coach would say")
    args = p.parse_args()

    profile = storage.load_profile(args.profile)
    cals = storage.calibrations(profile)
    routine = None
    if args.calibrate > 0:
        routine = CalibrationRoutine(EXERCISES[args.exercise],
                                     settle_s=1.0, hold_s=min(config.CALIBRATION_HOLD_S, args.calibrate / 4),
                                     max_attempts=1)     # a recording can't redo it
    elif args.exercise not in cals:
        p.error(f"no calibration for {args.exercise} in {args.profile}; use --calibrate N")

    sense = Sense(args.video, mirror=not args.no_mirror)
    smoother = FeatureFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA, config.ONE_EURO_D_CUTOFF)
    speaker = SilentSpeaker(echo=args.verbose)
    hand = profile.get("affected_hand", config.AFFECTED_HAND)
    coach = None
    frames = with_hand = 0

    if routine:
        routine.start(0.0)
    while True:
        frame, t = sense.read()
        if frame is None:
            break
        frames += 1
        obs = features.choose_hand(sense.observe(frame, t), hand)
        f = features.smooth(features.extract(obs, t, (frame.shape[1], frame.shape[0]), hand), smoother)
        with_hand += f.present

        if routine and not routine.done:
            need = routine.step.need_palm_facing if routine.step else False
            speaker.say_all(routine.update(f, t, quality_ok=f.quality_problem(need) is None))
            if routine.done:
                cals[args.exercise] = routine.result
            continue
        if coach is None:
            exercise = create(args.exercise, cals, profile.get("thresholds"),
                              params={"reps": 10 ** 6, "sets": 1})
            coach = Coach(exercise, speaker, None, hand)
            coach.start_set(1, t)
        coach.update(f, t)
    sense.close()

    if routine and not routine.done:
        raise SystemExit("Calibration did not finish within the video.")
    reps = coach.exercise.reps if coach else []
    print(f"\n{args.video}: {frames} frames, hand visible in {with_hand / max(1, frames):.0%}")
    print(f"Reps counted by the program: {len(reps)}")
    for r in reps:
        print(f"  rep {r.rep_no:2d}  t={r.t_end:6.1f}s  range={r.range_high:5.2f}/{r.range_low:5.2f}  "
              f"move={r.movement_time:4.1f}s  peaks={r.smoothness_peaks}  "
              f"comp={','.join(r.compensation) or '-'}  {r.extra}")
    if args.expected is not None:
        err = len(reps) - args.expected
        print(f"Counted by hand: {args.expected}   difference: {err:+d}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.is_file()
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "video", "exercise", "expected", "counted", "difference",
                        "hand_visible", "note"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), Path(args.video).name,
                    args.exercise, args.expected if args.expected is not None else "",
                    len(reps), (len(reps) - args.expected) if args.expected is not None else "",
                    round(with_hand / max(1, frames), 3), args.note])


if __name__ == "__main__":
    main()
