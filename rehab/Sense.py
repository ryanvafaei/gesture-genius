"""
Sense: camera (or video file) -> frame -> hand observations.

Uses MediaPipe's GestureRecognizer. Besides the gesture label (kept only as a
secondary check) it returns `hand_landmarks` (image coordinates) and
`hand_world_landmarks` (3D, metres, centred on the hand), which features.py
turns into measurements. No extra model is needed.
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from rehab import config
from rehab.features import HandObservation


class Sense:

    def __init__(self, source=None, model_path=config.MODEL_PATH, mirror=config.MIRROR):
        """
        source  None -> webcam config.CAMERA_INDEX; int -> that webcam;
                str/Path -> a recorded video (timestamps follow the video,
                so a recording gives the same result as live).
        """
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Gesture recognizer model not found at {model_path}")

        vision = mp.tasks.vision
        options = vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.mirror = mirror

        self.is_file = isinstance(source, (str, Path))
        if source is None:
            source = config.CAMERA_INDEX
        self.cap = cv2.VideoCapture(str(source) if self.is_file else source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {source!r}")
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_index = 0
        self._t0 = time.monotonic()
        self._last_ms = -1

    def read(self):
        """Next frame and its timestamp in seconds, or (None, None) at the end."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        if self.is_file:
            t = self._frame_index / self.fps
        else:
            t = time.monotonic() - self._t0
        self._frame_index += 1
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame, t

    def observe(self, frame, t):
        """
        Run the recognizer on a BGR frame as returned by read(); returns
        [HandObservation] in the coordinates of that (mirrored) frame.

        With RECOGNIZE_UNMIRRORED the recognizer sees the camera's own view
        and the landmarks are mirrored afterwards, so the handedness label is
        the real hand (see config).
        """
        unflip = self.mirror and config.RECOGNIZE_UNMIRRORED
        if unflip:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(t * 1000), self._last_ms + 1)     # must strictly increase
        self._last_ms = ms
        result = self.recognizer.recognize_for_video(image, ms)
        return to_observations(result, mirror_x=unflip)

    def close(self):
        self.recognizer.close()
        self.cap.release()


def to_observations(result, mirror_x=False):
    """MediaPipe result -> [HandObservation]; mirror_x flips x of both landmark sets."""
    hands = []
    for i, landmarks in enumerate(result.hand_landmarks):
        image = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=float)
        world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]], dtype=float)
        if mirror_x:
            image[:, 0] = 1.0 - image[:, 0]
            world[:, 0] = -world[:, 0]
        handed = result.handedness[i][0] if result.handedness and result.handedness[i] else None
        gesture = result.gestures[i][0] if result.gestures and result.gestures[i] else None
        hands.append(HandObservation(
            image=image,
            world=world,
            handedness=handed.category_name if handed else None,
            handedness_score=handed.score if handed else 0.0,
            gesture=gesture.category_name if gesture else None,
            gesture_score=gesture.score if gesture else 0.0,
        ))
    return hands
