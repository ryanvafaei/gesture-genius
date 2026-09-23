import cv2
import mediapipe as mp
from pathlib import Path


class Sense_recognizer:

    def __init__(self, model_path=None):

        # ---------------------------------------------------------
        # Gesture recognition setup
        # ---------------------------------------------------------

        model_path = Path(
            model_path or
            Path(__file__).parents[1] / 'models' / 'gesture_recognizer.task'
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Gesture recognizer model not found at {model_path}"
            )

        # MediaPipe classes
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        RunningMode = mp.tasks.vision.RunningMode

        # Create gesture recognizer
        options = GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=2
        )

        self.gesture_recognizer = GestureRecognizer.create_from_options(
            options
        )

        self.timestamp_ms = 0

        # ---------------------------------------------------------
        # Your existing pose-recognition setup
        # ---------------------------------------------------------

        # Put your existing PoseLandmarker initialization here.
        # For example:
        #
        # self.mp_pose = vision.PoseLandmarker.create_from_options(options)
        #
        # self.previous_angle = -1
        # self.angle_window = [-1] * 10

        self.previous_angle = -1
        self.angle_window = [-1] * 10


    # =============================================================
    # GESTURE RECOGNITION
    # =============================================================

    def detect_gesture(self, frame):
        """
        Detects hand gestures in a camera frame.

        Parameters:
            frame: OpenCV BGR image

        Returns:
            MediaPipe GestureRecognizerResult
        """

        # OpenCV -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # RGB -> MediaPipe Image
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp must increase for every frame
        self.timestamp_ms += 33

        # Run gesture recognition
        result = self.gesture_recognizer.recognize_for_video(
            image,
            self.timestamp_ms
        )

        return result


    def get_gesture_name(self, result):
        """
        Returns the name of the most confident detected gesture.

        Example:
            'Thumb_Up'
            'Open_Palm'
            'Closed_Fist'

        Returns None if no gesture is detected.
        """

        if not result.gestures:
            return None

        # First detected hand
        if not result.gestures[0]:
            return None

        gesture = result.gestures[0][0]

        return gesture.category_name


    def get_gesture_confidence(self, result):
        """
        Returns confidence of the most confident gesture.
        """

        if not result.gestures:
            return 0.0

        if not result.gestures[0]:
            return 0.0

        return result.gestures[0][0].score


    def draw_gesture(self, frame, result):
        """
        Draws the detected gesture and hand landmarks
        onto the OpenCV frame.
        """

        # ---------------------------------------------------------
        # Draw gesture name
        # ---------------------------------------------------------

        gesture_name = self.get_gesture_name(result)
        confidence = self.get_gesture_confidence(result)

        if gesture_name is not None:

            text = f"{gesture_name} ({confidence:.2f})"

            cv2.putText(
                frame,
                text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "No gesture",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # ---------------------------------------------------------
        # Draw hand landmarks
        # ---------------------------------------------------------

        height, width, _ = frame.shape

        for hand in result.hand_landmarks:

            # Draw points
            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 0, 0),
                    -1
                )

            # Connections between landmarks
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]

            for start_index, end_index in connections:

                start = hand[start_index]
                end = hand[end_index]

                x1 = int(start.x * width)
                y1 = int(start.y * height)

                x2 = int(end.x * width)
                y2 = int(end.y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


    # =============================================================
    # CLEANUP
    # =============================================================

    def close(self):
        """
        Close the MediaPipe gesture recognizer.
        """

        self.gesture_recognizer.close()