"""
Hand rehabilitation coach for Eleanor.

    python main.py                          full session
    python main.py --exercise grip_release  one exercise (repeatable)
    python main.py --video test.mp4         run on a recording instead of the webcam
    python main.py --no-speech              print instead of speaking

Keys: space = start / pause / continue, q or Esc = stop.

Wiring only: Sense -> features -> Think (Coach + SessionManager) -> Act.
Nothing imports this file.
"""

import argparse

import cv2

from rehab import config, features, storage
from rehab.Act import Display, SilentSpeaker, Speaker
from rehab.exercises import EXERCISES
from rehab.filters import FeatureFilter
from rehab.Sense import Sense
from rehab.Think import SessionManager


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exercise", action="append", choices=sorted(EXERCISES),
                   help="run only this exercise (may be given more than once)")
    p.add_argument("--video", help="video file to use instead of the webcam")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--no-speech", action="store_true")
    p.add_argument("--no-mirror", action="store_true",
                   help="for recordings that are already mirrored")
    return p.parse_args()


def main():
    args = parse_args()

    sense = Sense(args.video if args.video else args.camera,
                  mirror=not args.no_mirror)
    speaker = SilentSpeaker(echo=True) if args.no_speech else Speaker()
    display = Display()
    profile = storage.load_profile()
    log = storage.SessionLog()
    session = SessionManager(speaker, profile, log, exercises=args.exercise,
                             save_profile=storage.save_profile)
    smoother = FeatureFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
                             config.ONE_EURO_D_CUTOFF)
    hand = profile.get("affected_hand", config.AFFECTED_HAND)
    t = 0.0

    try:
        while not session.done:
            frame, t = sense.read()
            if frame is None:
                break

            # Sense
            observations = sense.observe(frame, t)
            obs = features.choose_hand(observations, hand)
            f = features.extract(obs, t, (frame.shape[1], frame.shape[0]), hand)
            features.smooth(f, smoother)

            # Think
            session.update(f, t)

            # Act
            display.show(display.render(frame, session.view(), f))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                session.on_key(" ", t)
    finally:
        session.stop(t)
        speaker.close()
        sense.close()
        display.close()


if __name__ == "__main__":
    main()
