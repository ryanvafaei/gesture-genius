import cv2
import mediapipe as mp

# Old Sense class - commented out
# from coach import Sense

from coach.Sense_recognizer import Sense_recognizer
from coach import Think
from coach import Act
from coach import Sense

import numpy as np


# Main Program Loop
def main():
    """
    Main function to initialize the exercise tracking application.

    This version uses Sense_recognizer for gesture recognition.
    The old Sense class is currently commented out.
    """

    # =========================================================
    # INITIALIZE COMPONENTS
    # =========================================================

    # Old Sense class - commented out
    # sense = Sense.Sense()

    # New gesture recognizer
    sense = Sense_recognizer()

    act = Act.Act()
    think = Think.Think(act)

    # =========================================================
    # INITIALIZE WEBCAM
    # =========================================================

    cap = cv2.VideoCapture(0)

    # =========================================================
    # MAIN LOOP
    # =========================================================

    while cap.isOpened():

        # Capture frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Flip camera image like a mirror
        frame = cv2.flip(frame, 1)

        # =====================================================
        # SENSE: GESTURE RECOGNITION
        # =====================================================

        gesture_result = sense.detect_gesture(frame)

        gesture_name = sense.get_gesture_name(
            gesture_result
        )

        gesture_confidence = sense.get_gesture_confidence(
            gesture_result
        )

        print(
            f"Gesture: {gesture_name}, "
            f"confidence: {gesture_confidence:.2f}"
        )

        # Draw gesture and hand skeleton
        sense.draw_gesture(
            frame,
            gesture_result
        )

        # =====================================================
        # OLD POSE RECOGNITION
        # =====================================================

        # The old Sense class used to do this:
        #
        # joints = sense.detect_joints(frame)
        #
        # landmarks = (
        #     joints.pose_landmarks[0]
        #     if joints.pose_landmarks
        #     else None
        # )
        #
        # if landmarks:
        #
        #     shoulder = sense.extract_joint_coordinates(
        #         landmarks,
        #         'left_shoulder'
        #     )
        #
        #     elbow = sense.extract_joint_coordinates(
        #         landmarks,
        #         'left_elbow'
        #     )
        #
        #     wrist = sense.extract_joint_coordinates(
        #         landmarks,
        #         'left_wrist'
        #     )
        #
        #     elbow_angle_mvg = sense.calculate_angle(
        #         shoulder,
        #         elbow,
        #         wrist
        #     )
        #
        #     think.update_state(
        #         elbow_angle_mvg,
        #         sense.previous_angle
        #     )
        #
        #     sense.previous_angle = elbow_angle_mvg
        #
        #     decision = think.state
        #
        #     act.provide_feedback(
        #         decision,
        #         frame=frame,
        #         joints=joints,
        #         elbow_angle_mvg=elbow_angle_mvg
        #     )
        #
        #     act.visualize_balloon()

        # =====================================================
        # DISPLAY
        # =====================================================

        cv2.imshow(
            "Gesture Recognition",
            frame
        )

        # =====================================================
        # EXIT
        # =====================================================

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # =========================================================
    # CLEAN UP
    # =========================================================

    cap.release()
    cv2.destroyAllWindows()

    sense.close()


# =============================================================
# RUN PROGRAM
# =============================================================

if __name__ == "__main__":
    main()