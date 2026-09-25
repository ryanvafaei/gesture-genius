"""
Verbose session log (python main.py --verbose): everything needed after a
session to tune the calibration, the detection thresholds and the tracking.

One folder per session, inside the user's data folder (hers, or the
guest's own folder):

    <data>/verbose/<user>_<session id>/
      session.json     who and when, versions, camera, the whole config, the
                       stored calibrations; at the end the totals (frames,
                       frame rate, loop time, video frames dropped)
      frames.jsonl.gz  one JSON line per camera frame: timing, every detected
                       hand (21 image + 21 world landmarks, handedness,
                       gesture), the features the exercises use, the quality
                       problem, the stage and the active detector's inner
                       state (e.g. lift scores, touch closeness, finger counts)
      events.jsonl     stage changes, keys, clicks, what the coach said (when
                       it was really spoken),
                       calibration results, exercise settings, detections
                       (finger, prompted finger, right or wrong), reps,
                       ratings, skips and toolbar actions
      video.mp4        the camera image as the camera gave it (not mirrored)
                       on a real-time timeline; "video_frame" in frames.jsonl
                       is its frame number. Replay the session with
                       python main.py --video <folder>/video.mp4

The video shows the person: record only with their consent, and delete the
folder when it is no longer needed. Analyse a folder with
python -m tools.verbose_summary <folder>.
"""

import dataclasses
import gzip
import json
import math
import platform
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from rehab import config, features

# HandFeatures fields written for every frame (image_points etc. follow from the landmarks)
HAND_FIELDS = ("present", "handedness", "handedness_score", "correct_hand", "too_small",
               "palm_facing", "palm_size", "palm_width", "palm_size_image", "hand_size_image",
               "image_size", "openness", "curl",
               "spread", "thumb_tip_dist", "thumb_tip_dist_image", "tip_reach", "tip_height",
               "tip_rise", "dorsal_side", "joint_flexion", "thumb_flexion",
               "thumb_to_pinky_mcp", "thumb_to_index_mcp",
               "aperture", "tip_to_palm", "gesture", "gesture_score")
BODY_FIELDS = ("present", "view", "view_ratio", "trunk_angle", "arm", "shoulder_width",
               "torso_len", "points", "visibility", "gesture", "gesture_score")
DIGITS = 5


def jsonable(v, digits=DIGITS):
    """Numbers rounded, numpy and paths turned into plain JSON; NaN -> None."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return None if math.isnan(v) or math.isinf(v) else round(v, digits)
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist(), digits)
    if isinstance(v, dict):
        return {str(k): jsonable(x, digits) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [jsonable(x, digits) for x in v]
    if isinstance(v, Path):
        return str(v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return jsonable(vars(v), digits)
    return str(v)


def config_snapshot():
    """Every setting in rehab/config.py (UPPER_CASE names)."""
    return {k: jsonable(getattr(config, k)) for k in dir(config) if k.isupper()}


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "opencv": cv2.__version__, "numpy": np.__version__}
    try:
        import mediapipe
        out["mediapipe"] = getattr(mediapipe, "__version__", "")
    except ImportError:
        pass
    return out


def hand_row(f):
    """The logged part of HandFeatures (and the finger count of both hands)."""
    if f is None or not getattr(f, "present", False):
        return {"present": False}
    row = {k: getattr(f, k, None) for k in HAND_FIELDS}
    row["fingers_raised"] = features.count_extended(f)
    other = getattr(f, "other", None)
    if other is not None and other.present:
        row["other"] = {k: getattr(other, k, None) for k in HAND_FIELDS}
        row["other"]["fingers_raised"] = features.count_extended(other)
    return row


def body_row(f):
    row = {k: getattr(f, k, None) for k in BODY_FIELDS}
    row["hands"] = {side: hand_row(h) for side, h in (f.hands or {}).items()}
    return row


class VideoRecorder:
    """
    Writes frames in its own thread (the coach never waits for the disk).
    The timeline is real time: when frames come slower than fps, a frame is
    repeated, so a replay has the same timing as the session.
    """

    MAX_GAP_S = 2.0                 # longer gaps (e.g. a pause of the camera) are not filled

    def __init__(self, path, fps=30.0, unmirror=True, max_queue=60):
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 1 else 30.0
        self.unmirror = unmirror
        self.written = 0
        self.dropped = 0
        self._writer = None
        self._t0 = None
        self._queue = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, frame, t):
        """Queue a frame taken at time t (seconds). Returns its frame number, or None if dropped."""
        if self._t0 is None:
            self._t0 = t
        index = int(round((t - self._t0) * self.fps))
        try:
            self._queue.put_nowait((frame.copy(), index))
        except queue.Full:
            self.dropped += 1
            return None
        return index

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            frame, index = item
            if self.unmirror:
                frame = cv2.flip(frame, 1)
            if self._writer is None:
                h, w = frame.shape[:2]
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                               self.fps, (w, h))
            if index - self.written > self.MAX_GAP_S * self.fps:
                self.written = index            # do not fill a long gap
            repeat = max(1, index - self.written + 1)
            for _ in range(repeat):
                self._writer.write(frame)
            self.written += repeat

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=30)
        if self._writer is not None:
            self._writer.release()


class VerboseLog:

    def __init__(self, folder, meta=None, video=True, fps=30.0, mirrored=True):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.meta = {
            "started": datetime.now().isoformat(timespec="seconds"),
            "versions": versions(),
            "config": config_snapshot(),
            **jsonable(meta or {}),
            "mirrored_in_app": mirrored,
            "video": "video.mp4" if video else None,
        }
        self._write_meta()
        self._frames = gzip.open(self.folder / "frames.jsonl.gz", "wt", compresslevel=5)
        self._events = open(self.folder / "events.jsonl", "w")
        self.video = VideoRecorder(self.folder / "video.mp4", fps, unmirror=mirrored) if video else None
        self.n = 0
        self._t = 0.0
        self._first_t = None
        self._loop_ms = []
        self._hand_frames = 0
        self._closed = False
        self._lock = threading.Lock()   # events also come from the speaker's thread

    def _write_meta(self):
        (self.folder / "session.json").write_text(json.dumps(self.meta, indent=1))

    # --- events -------------------------------------------------------------------

    def event(self, kind, t=None, **data):
        """One line of events.jsonl, e.g. event("key", key="k")."""
        line = {"t": jsonable(self._t if t is None else t), "wall": round(time.time(), 3),
                "type": kind, **jsonable(data)}
        with self._lock:
            if self._closed:
                return
            self._events.write(json.dumps(line) + "\n")
            self._events.flush()

    # --- frames -------------------------------------------------------------------

    def frame(self, t, frame, observations, f, state=None, loop_ms=None):
        """
        One camera frame: the raw hands (observations), the features the
        coach used (f: HandFeatures or BodyFeatures) and the session's inner
        state (SessionManager.debug_state()).
        """
        if self._closed:
            return
        self._t = t
        if self._first_t is None:
            self._first_t = t
        video_frame = self.video.add(frame, t) if self.video is not None and frame is not None else None
        hands = [{"handedness": o.handedness, "score": o.handedness_score,
                  "gesture": o.gesture, "gesture_score": o.gesture_score,
                  "image": o.image, "world": o.world} for o in observations or []]
        if hands:
            self._hand_frames += 1
        if loop_ms is not None:
            self._loop_ms.append(loop_ms)
        row = {"i": self.n, "t": t, "wall": time.time(), "loop_ms": loop_ms,
               "video_frame": video_frame, "hands": hands,
               "state": state or {}}
        if f is not None:
            row["body" if hasattr(f, "arm") else "features"] = (
                body_row(f) if hasattr(f, "arm") else hand_row(f))
        self._frames.write(json.dumps(jsonable(row)) + "\n")
        self.n += 1

    # --- end ----------------------------------------------------------------------

    def close(self, **totals):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self.video is not None:
            self.video.close()
        self._frames.close()
        self._events.close()
        duration = (self._t - self._first_t) if self._first_t is not None else 0.0
        loop = np.array(self._loop_ms) if self._loop_ms else np.array([np.nan])
        self.meta["ended"] = datetime.now().isoformat(timespec="seconds")
        self.meta["totals"] = jsonable({
            "frames": self.n,
            "duration_s": duration,
            "frames_per_s": self.n / duration if duration > 0 else None,
            "frames_with_a_hand": self._hand_frames,
            "loop_ms_median": float(np.nanmedian(loop)),
            "loop_ms_p95": float(np.nanpercentile(loop, 95)) if self._loop_ms else None,
            "video_frames_written": self.video.written if self.video else 0,
            "video_frames_dropped": self.video.dropped if self.video else 0,
            **totals,
        })
        self._write_meta()


def session_folder(user, session_id):
    """<current data folder>/verbose/<user>_<session id>/"""
    return Path(config.DATA_DIR) / "verbose" / f"{user}_{session_id}"
