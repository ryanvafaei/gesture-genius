"""
Hand rehabilitation coach for Eleanor.

    python main.py                          full session
    python main.py --exercise grip_release  one exercise (repeatable)
    python main.py --video test.mp4         run on a recording instead of the webcam
    python main.py --no-speech              print instead of speaking
    python main.py --windowed               in a window instead of full screen

It opens with a greeting and a check-in ("How is your hand feeling today?"),
answered with thumbs up / thumbs down (or space = yes, n = no). Without
--exercise it then shows a menu: press a number (or up/down and space) to
pick one exercise, space for all of today's exercises, or "Finish for today"
to see the garden and say goodbye.

Keys: space = start / pause / continue, y / n = yes / no, m = back to the
menu, p = her profile (from the menu; d there deletes it and starts again),
f = full screen on / off, q or Esc = stop (what was done is kept).

Wiring only: Sense -> features -> Think (Coach + SessionManager) -> Act.
Nothing imports this file.
"""

import argparse

import cv2

from rehab import config, features, storage
from rehab.Act import Display, SilentSpeaker, Speaker, screen_size
from rehab.exercises import EXERCISES
from rehab.feedback import Feedback
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
    p.add_argument("--windowed", action="store_true",
                   help="open in a window instead of full screen (f switches)")
    return p.parse_args()


# Arrow key codes from cv2.waitKeyEx: Linux, Windows, macOS.
UP_KEYS = {0xFF52, 0x260000, 0xF700}
DOWN_KEYS = {0xFF54, 0x280000, 0xF701}


def key_name(code):
    """cv2.waitKeyEx code -> "up", "down", "q", " ", a digit ... or None."""
    if code < 0:
        return None
    # on Linux, modifier state (e.g. NumLock) is added in the high bits
    if code in UP_KEYS or code & 0xFFFF == 0xFF52:
        return "up"
    if code in DOWN_KEYS or code & 0xFFFF == 0xFF54:
        return "down"
    if code & 0xFF00 or not code & 0xFF:
        return None
    char = code & 0xFF
    return "q" if char == 27 else chr(char).lower()


def main():
    args = parse_args()
    sense = Sense(args.video if args.video else args.camera,
                  mirror=not args.no_mirror)
    display = Display(screen=screen_size(),
                      fullscreen=config.FULLSCREEN and not args.windowed)
    try:
        # a new session after her profile was deleted: the first questions again
        while run_session(args, sense, display):
            pass
    finally:
        sense.close()
        display.close()


def run_session(args, sense, display):
    """One session, from the greeting to goodbye. True: start again with a new profile."""
    profile = storage.load_profile()
    character = storage.load_content("character")
    feedback = Feedback(name=profile["name"], coach=profile.get("coach_name"))
    speaker = (SilentSpeaker(echo=True, feedback=feedback) if args.no_speech else
               Speaker(rate=profile.get("voice_rate", config.SPEECH_RATE),
                       voice=profile.get("voice"), feedback=feedback))
    display.character, display.feedback = character, feedback
    log = storage.SessionLog()
    session = SessionManager(speaker, profile, log, exercises=args.exercise,
                             save_profile=storage.save_profile,
                             garden=storage.load_garden(), save_garden=storage.save_garden,
                             character=character, delete_profile=storage.delete_profile)
    smoother = FeatureFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
                             config.ONE_EURO_D_CUTOFF)
    hand = profile.get("affected_hand", config.AFFECTED_HAND)
    t = 0.0

    try:
        while not session.done:
            frame, t = sense.read()
            if frame is None:
                return False

            # Sense
            observations = sense.observe(frame, t)
            obs = features.choose_hand(observations, hand)
            f = features.extract(obs, t, (frame.shape[1], frame.shape[0]), hand)
            features.smooth(f, smoother)

            # Think (thumbs up / down count from either hand)
            gestures = [(o.gesture, o.gesture_score) for o in observations if o.gesture]
            session.update(f, t, gestures)

            # Act
            display.show(display.render(frame, session.view(), f))
            key = key_name(cv2.waitKeyEx(1))
            if key == "q":
                return False
            if key == "f":
                display.toggle_fullscreen()
            elif key:
                session.on_key(key, t)
        return session.restart
    finally:
        session.stop(t)
        speaker.close()


if __name__ == "__main__":
    main()
