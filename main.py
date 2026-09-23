import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands = vision.HandLandmarksConnections
mp_drawing = vision.drawing_utils
mp_drawing_styles = vision.drawing_styles

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        print(f"hand_landmarks update loop {idx}", hand_landmarks)
        handedness = handedness_list[idx]
        print(f"handedness update loop {idx}", handedness)

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        height, width = annotated_image.shape
        print(f"annotated_image height {height} width {width} update loop {idx}", annotated_image.shape)
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        print(f"x_coordinates {x_coordinates} update loop {idx}", x_coordinates)
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        print(f"y_coordinates {y_coordinates} update loop {idx}", y_coordinates)
        text_x = int(min(x_coordinates) * width)
        print(f"text_x {text_x} update loop {idx}", text_x)
        text_y = int(min(y_coordinates) * height) - MARGIN
        print(f"text_y {text_y} update loop {idx}", text_y)

        cv.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv.LINE_AA
        )

        return annotated_image

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)

detector = vision.HandLandmarker.create_from_options(options=options)

while True:
    ret, frame = cap.read()
    print("ret init", ret)
    print("frame init", frame)
    if ret:
        rbg_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rbg_image)
        detection_result = detector.detect(mp_image)
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                print(draw_landmarks_on_image(rbg_image, detection_result))

        cv.imshow("capture image", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cv.destroyAllWindows()