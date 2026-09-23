import cv2
import mediapipe as mp
import math
import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Sense Component: Detect joints using the camera
# Things you need to improve: Make the skeleton tracking smoother and robust to errors.
class Sense:

    def __init__(self, model_path=None):
        model_path = Path(model_path or Path(__file__).parents[1] / 'models' / 'pose_landmarker_lite.task')
        if not model_path.is_file():
            raise FileNotFoundError(
                f'MediaPipe pose model not found at {model_path}. '
                'Restore models/pose_landmarker_lite.task before running the application.'
            )

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_pose = vision.PoseLandmarker.create_from_options(options)
        self.timestamp_ms = 0

        # used later for having a moving avergage
        self.angle_window = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.previous_angle = -1

    def detect_joints(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.timestamp_ms += 1
        return self.mp_pose.detect_for_video(image, self.timestamp_ms)

    def calculate_angle(self, joint1, joint2, joint3):
        """
        Calculates the angle between three joints.

        Parameters:
        - joint1: Tuple of (x, y) for the first joint (e.g., shoulder)
        - joint2: Tuple of (x, y) for the middle joint (e.g., elbow)
        - joint3: Tuple of (x, y) for the last joint (e.g., wrist)

        Returns:
        - Angle in degrees between the three joints
        """
        # Calculate vectors
        vector1 = [joint1[0] - joint2[0], joint1[1] - joint2[1]]
        vector2 = [joint3[0] - joint2[0], joint3[1] - joint2[1]]

        # Calculate the dot product and magnitude of the vectors
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        # Calculate the angle in radians and convert to degrees
        angle = math.acos(dot_product / (magnitude1 * magnitude2))

        # get a moving average
        self.angle_window.pop(0)
        self.angle_window.append(angle)
        # Use np.convolve to calculate the moving average
        window_size = 10
        angle_mvg = np.convolve(np.asarray(self.angle_window), np.ones(window_size) / window_size, mode='valid')

        return math.degrees(float(angle_mvg[0]))

    def extract_joint_coordinates(self, landmarks, joint):
        """
        Extracts the (x, y) coordinates of a specific joint.

        Parameters:
        - landmarks: The list of pose landmarks from MediaPipe
        - joint: The name of the joint (e.g., 'left_elbow')

        Returns:
        - A tuple of (x, y) coordinates of the specified joint
        """
        joint_index_map = {
            'left_shoulder': vision.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': vision.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': vision.PoseLandmark.LEFT_ELBOW,
            'right_elbow': vision.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': vision.PoseLandmark.LEFT_WRIST,
            'right_wrist': vision.PoseLandmark.RIGHT_WRIST,
            'left_hip': vision.PoseLandmark.LEFT_HIP,
            'right_hip': vision.PoseLandmark.RIGHT_HIP,
            'left_knee': vision.PoseLandmark.LEFT_KNEE,
            'right_knee': vision.PoseLandmark.RIGHT_KNEE,
            'left_ankle': vision.PoseLandmark.LEFT_ANKLE,
            'right_ankle': vision.PoseLandmark.RIGHT_ANKLE
        }

        landmark = landmarks[joint_index_map[joint]]

        return landmark.x, landmark.y

    ### Example for defining a function that extracts an angle
    def extract_hip_angle(self, landmarks):
        """
                Extracts the hip angle.

                Parameters:
                - landmarks: The list of pose landmarks from MediaPipe

                Returns:
                - An angle in degrees for the hip
                """
        # extract the x and y coordinates
        left_hip = [landmarks[vision.PoseLandmark.LEFT_HIP].x, landmarks[vision.PoseLandmark.LEFT_HIP].y]
        left_shoulder = [landmarks[vision.PoseLandmark.LEFT_SHOULDER].x, landmarks[vision.PoseLandmark.LEFT_SHOULDER].y]
        left_knee = [landmarks[vision.PoseLandmark.LEFT_KNEE].x, landmarks[vision.PoseLandmark.LEFT_KNEE].y]
        return self.calculate_angle(left_shoulder, left_hip, left_knee)

    # Extracts the angle of the knee by measuring the angle between the left hip, left knee, and left ankle
    def extract_knee_angle(self, landmarks):
        # extract the x and y coordinates
        left_hip = [landmarks[vision.PoseLandmark.LEFT_HIP].x, landmarks[vision.PoseLandmark.LEFT_HIP].y]
        left_knee = [landmarks[vision.PoseLandmark.LEFT_KNEE].x, landmarks[vision.PoseLandmark.LEFT_KNEE].y]
        left_ankle = [landmarks[vision.PoseLandmark.LEFT_ANKLE].x, landmarks[vision.PoseLandmark.LEFT_ANKLE].y]
        return self.calculate_angle(left_hip, left_knee, left_ankle)
