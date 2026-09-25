"""
Hand and arm rehabilitation coach for Eleanor.

    python main.py                          full session
    python main.py --exercise grip_release  one exercise (repeatable)
    python main.py --exercise shoulder_flexion_raise   an arm exercise (body tracking)
    python main.py --video test.mp4         run on a recording instead of the webcam
    python main.py --no-speech              print instead of speaking
    python main.py --windowed               in a window instead of full screen
    python main.py --guest --short          a visitor (e.g. the marketplace): a new guest
                                            with a unique id and own data folder, no
                                            first-time questions, short sets
    python main.py --guest --hand Right     a visitor who trains the right hand
    python main.py --verbose                log every frame, event and the camera video
                                            for tuning calibration, detection and
                                            tracking (rehab/verbose.py)

It opens with a greeting and a check-in ("How is your hand feeling today?"),
answered with thumbs up / thumbs down (or space = yes, n = no). Without
--exercise it then shows a menu: press a number (or up/down and space) to
pick one exercise, space for all of today's exercises, or "Finish for today"
to see the garden and say goodbye.

Keys: space = start / pause / continue, y / n = yes / no, m = back to the
menu, p = her profile (from the menu; d there deletes it and starts again),
e = finish for today (menu), s = Stop / "I don't feel well", r = repeat,
k = skip (the exercise, or the rest), 1-5 = rating answers,
1-8 = cards in the memory game, f = full screen on / off,
q or Esc = stop (what was done is kept).

In the menu, i (or the icon at the top right) opens the toolbar: t = short
sessions on / off, g = new guest (a new session for a new guest with a
unique id, e.g. guest-007-20260925-143012, and its own folder in
data/guests/), b = back to her own profile (while a guest is active),
o = make the report (python -m tools.report) and open its folder: while a
guest is active only that guest's, in data/guests/<id>/<id>_report/.
A guest's report is also made when the guest's session ends. Buttons can
also be clicked.

Wiring only: Sense -> features -> Think (Coach + SessionManager) -> Act.
While an arm exercise runs (session.needs_body) the body is tracked too and
Think gets BodyFeatures (arm angles, 6 Hz low-pass) instead of HandFeatures.
Nothing imports this file.
"""

import argparse
import time

import cv2

from rehab import body, config, features, guests, storage, verbose
from rehab.Act import Display, SilentSpeaker, Speaker, screen_size
from rehab.exercises import EXERCISES
from rehab.feedback import Feedback
from rehab.filters import FeatureFilter, LowPassBank
from rehab.Sense import Sense
from rehab.Think import SessionManager
from tools import report_job


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
                   help="a new guest with a unique id: data in data/guests/<id>/, "
                        "her own data is never touched")
    p.add_argument("--short", action="store_true",
                   help="one short set per exercise (see SHORT_SESSION in rehab/config.py)")
    p.add_argument("--hand", choices=("Left", "Right"),
                   help="the hand to train (default: from the profile, or AFFECTED_HAND)")
    p.add_argument("--verbose", action="store_true",
                   help="log every frame (landmarks, features, detector state), every event and "
                        "the camera video to <data>/verbose/<user>_<session>/")
    p.add_argument("--no-video", action="store_true",
                   help="with --verbose: do not record the camera video")
    return p.parse_args()


def report_target(guest_id):
    """(data folder, report folder, user) of the report the toolbar makes now."""
    if guest_id:
        folder = config.DATA_DIR
        return folder, guests.report_dir(folder, guest_id), guest_id
    return config.BASE_DATA_DIR, config.BASE_DATA_DIR / "report", None


def make_report(guest_id):
    """The toolbar's report: this guest's, or everyone's while she is the user."""
    folder, out, user = report_target(guest_id)
    return report_job.start_background(folder, out, user=user)


def guest_report(guest_id, folder):
    """A guest's own report when the guest's session ends (when the guest did something)."""
    if not any((folder / name).is_file() for name in ("sessions.csv", "history.csv")):
        return None
    return report_job.start_background(folder, guests.report_dir(folder, guest_id), user=guest_id,
                                       open_when_done=False)


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
    args.guest_id = None
    seen, jobs = [], []             # guests of this run; their reports being made
    if args.guest:
        args.guest_id, folder = guests.start_guest(args.hand)
        seen.append((args.guest_id, folder))
    sense = Sense(args.video if args.video else args.camera,
                  mirror=not args.no_mirror)
    display = Display(screen=screen_size(),
                      fullscreen=config.FULLSCREEN and not args.windowed)
    try:
        while True:
            result = run_session(args, sense, display)
            if args.guest_id:
                jobs.append(guest_report(args.guest_id, config.DATA_DIR))
            if result in ("new_guest", "main_profile"):
                # toolbar: a new session for a new guest, or back in her own folder
                if result == "new_guest":
                    args.guest_id, folder = guests.start_guest(args.hand)
                    seen.append((args.guest_id, folder))
                else:
                    args.guest_id = None
                    config.use_data_dir(config.BASE_DATA_DIR)
                args.guest = args.guest_id is not None
            elif not result:
                break
            # else her profile was deleted: the first questions again
    finally:
        sense.close()
        display.close()
        for guest_id, folder in seen:
            print(f"{guest_id}: data in {folder}")
        waiting = [j for j in jobs if j is not None and j.status()[0] == "running"]
        if waiting:
            print("Making the guests' reports ...")
            for job in waiting:
                job.wait()
        for job in jobs:
            if job is not None:
                print(f"{job.user}: {job.status()[1]}")


def run_session(args, sense, display):
    """
    One session, from the greeting to goodbye. Returns True to start again
    with a new profile, "new_guest" or "main_profile" to start again as a
    new guest or as her (toolbar), or False to quit.
    """
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
    vlog = None
    if args.verbose:
        user = args.guest_id or guests.MAIN_USER
        vlog = verbose.VerboseLog(
            verbose.session_folder(user, log.session_id),
            meta={"session_id": log.session_id, "user": user, "args": vars(args),
                  "source": args.video or args.camera, "camera_fps": sense.fps,
                  "data_dir": config.DATA_DIR,
                  "profile": {k: profile.get(k) for k in ("name", "affected_hand", "calibration",
                                                         "levels", "targets", "guest_id")}},
            video=not args.no_video, fps=sense.fps, mirrored=sense.mirror)
        speaker.on_spoken = lambda text: vlog.event("speech", text=text)
        print(f"Verbose log: {vlog.folder}")
    session = SessionManager(speaker, profile, log, exercises=args.exercise,
                             save_profile=storage.save_profile,
                             garden=storage.load_garden(), save_garden=storage.save_garden,
                             character=character, delete_profile=storage.delete_profile,
                             short=args.short, guest_id=args.guest_id,
                             trace=vlog.event if vlog else None,
                             make_report=lambda: make_report(args.guest_id),
                             rating_questions=(config.GUEST_RATING_QUESTIONS if args.guest
                                               else config.RATING_QUESTIONS))
    smoother = FeatureFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
                             config.ONE_EURO_D_CUTOFF)
    # the other hand has its own filters (two-hand match, finger counting)
    smoother_other = FeatureFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
                                   config.ONE_EURO_D_CUTOFF)
    side_smoothers = {"Left": FeatureFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
                                            config.ONE_EURO_D_CUTOFF),
                      "Right": FeatureFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
                                             config.ONE_EURO_D_CUTOFF)}
    lowpass = LowPassBank(fs=sense.fps, fc=config.LOW_PASS_HZ)
    hip_gate = body.HipGate()
    hand = profile.get("affected_hand", config.AFFECTED_HAND)
    t = 0.0
    loop_ms = None

    try:
        while not session.done:
            frame, t = sense.read()
            if frame is None:
                return False
            started = time.perf_counter()

            # Sense
            size = (frame.shape[1], frame.shape[0])
            observations = sense.observe(frame, t)
            if session.needs_body:
                f = body_features(sense, frame, t, size, observations, side_smoothers, lowpass,
                                  hip_gate)
            else:
                if session.palm_down:
                    # one hand, back up: the label is unreliable, the knuckle triangle picks it
                    obs, other = features.choose_hand(observations, hand, palm_down=True), None
                else:
                    obs, other = features.pair_hands(observations, hand, sense.mirror)
                f = features.extract(obs, t, size, hand)
                features.smooth(f, smoother)
                if other is not None:
                    f.other = features.smooth(features.extract(other, t, size, hand), smoother_other)

            # Think (thumbs up / down count from either hand)
            gestures = [(o.gesture, o.gesture_score) for o in observations if o.gesture]
            session.update(f, t, gestures)
            if vlog:
                # before Act draws on the frame
                vlog.frame(t, frame, observations, f, session.debug_state(), loop_ms)

            # Act
            display.show(display.render(frame, session.view(), f))
            key = key_name(cv2.waitKeyEx(1))
            if vlog and key in ("q", "f"):
                vlog.event("key", t, key=key, stage=session.stage)
            if key == "q":
                return False
            if key == "f":
                display.toggle_fullscreen()
            elif key:
                session.on_key(key, t)
            click = display.pop_click()
            if click:
                session.on_click(click, t)
            loop_ms = (time.perf_counter() - started) * 1000
        if session.switch_user is not None:
            return session.switch_user
        return session.restart
    finally:
        session.stop(t)
        speaker.close()
        args.short = session.short      # the toolbar's choice carries over to the next session
        if vlog:
            vlog.close(stage=session.stage, summaries=[name for name, _, _ in session.summaries],
                       ratings=session.ratings, skipped=session.skipped)


def body_features(sense, frame, t, size, observations, smoothers, lowpass, hip_gate=None):
    """
    BodyFeatures of this frame: pose landmarks plus each seen hand, filtered.
    Each hand goes to the arm whose wrist it is at (body.assign_hands), not
    to MediaPipe's handedness label, which is often wrong for a hand seen
    from the side.
    """
    gesture = max(((o.gesture, o.gesture_score) for o in observations if o.gesture),
                  key=lambda g: g[1], default=(None, 0.0))
    f = body.extract(sense.observe_pose(frame, t), t, size, None, gesture, hip_gate=hip_gate)
    unique = [o for o in features.pair_hands(observations, "Left", sense.mirror) if o is not None]
    hands = {}
    for side, obs in body.assign_hands(unique, f).items():
        label = side.capitalize()
        hands[side] = features.smooth(features.extract(obs, t, size, label), smoothers[label])
    body.attach_hands(f, hands)
    return body.smooth(f, lowpass)


if __name__ == "__main__":
    main()
