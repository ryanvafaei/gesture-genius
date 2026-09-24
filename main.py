"""
Hand rehabilitation coach for Eleanor.

    python main.py                          full session
    python main.py --exercise grip_release  one exercise (repeatable)
    python main.py --video test.mp4         run on a recording instead of the webcam
    python main.py --no-speech              print instead of speaking
    python main.py --windowed               in a window instead of full screen
    python main.py --guest --short          a visitor (e.g. the marketplace): own data
                                            folder, no first-time questions, short sets
    python main.py --guest --hand Right     a visitor who trains the right hand

It opens with a greeting and a check-in ("How is your hand feeling today?"),
answered with thumbs up / thumbs down (or space = yes, n = no). Without
--exercise it then shows a menu: press a number (or up/down and space) to
pick one exercise, space for all of today's exercises, or "Finish for today"
to see the garden and say goodbye.

Keys: space = start / pause / continue, y / n = yes / no, m = back to the
menu, p = her profile (from the menu; d there deletes it and starts again),
e = finish for today (menu), s = Stop / "I don't feel well", r = repeat,
1-5 = rating answers, 1-8 = cards in the memory game,
f = full screen on / off, q or Esc = stop (what was done is kept).

Wiring only: Sense -> features -> Think (Coach + SessionManager) -> Act.
Nothing imports this file.
"""

import argparse
from datetime import datetime

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
    p.add_argument("--guest", action="store_true",
                   help="a visitor: data in data/guests/<time>/, her own data is never touched")
    p.add_argument("--short", action="store_true",
                   help="one short set per exercise (see SHORT_SESSION in rehab/config.py)")
    p.add_argument("--hand", choices=("Left", "Right"),
                   help="the hand to train (default: from the profile, or AFFECTED_HAND)")
    return p.parse_args()


def start_guest(hand=None):
    """
    A visitor's own data folder with a ready profile: no first-time
    questions (the first coach name and three activities are chosen).
    Returns the folder.
    """
    folder = config.GUESTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    config.use_data_dir(folder)
    character = storage.load_content("character")
    activities = storage.load_content("activities")
    profile = storage.new_profile()
    profile["coach_name"] = (character.get("name") or (character.get("name_options") or [None])[0])
    profile["chosen_activities"] = list(activities.get("activities", {}))[:config.ACTIVITY_CHOICES]
    if hand:
        profile["affected_hand"] = hand
    storage.save_profile(profile)
    return folder


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
    guest = start_guest(args.hand) if args.guest else None
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
        if guest:
            print(f"This guest's data is in {guest}")


def run_session(args, sense, display):
    """One session, from the greeting to goodbye. True: start again with a new profile."""
    profile = storage.load_profile()
    if args.hand:
        profile["affected_hand"] = args.hand
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
                             character=character, delete_profile=storage.delete_profile,
                             short=args.short,
                             rating_questions=(config.GUEST_RATING_QUESTIONS if args.guest
                                               else config.RATING_QUESTIONS))
    smoother = FeatureFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
                             config.ONE_EURO_D_CUTOFF)
    # the other hand has its own filters (two-hand match, finger counting)
    smoother_other = FeatureFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
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
            size = (frame.shape[1], frame.shape[0])
            obs = features.choose_hand(observations, hand)
            f = features.extract(obs, t, size, hand)
            features.smooth(f, smoother)
            other = features.choose_other(observations, obs)
            if other is not None:
                f.other = features.smooth(features.extract(other, t, size, hand), smoother_other)

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
