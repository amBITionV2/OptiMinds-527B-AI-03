import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True)


def detect_gaze(image):
    """
    Detects gaze direction from an OpenCV image.
    Returns final JSON with gaze status.
    """
    if image is None or not isinstance(image, np.ndarray):
        return {"error": "Invalid image format"}

    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get Eye Landmarks
            left_eye_inner = face_landmarks.landmark[133]
            left_eye_outer = face_landmarks.landmark[33]
            left_iris = face_landmarks.landmark[468]

            right_eye_inner = face_landmarks.landmark[362]
            right_eye_outer = face_landmarks.landmark[263]
            right_iris = face_landmarks.landmark[473]

            # Convert to Pixel Coordinates
            h, w, _ = image.shape

            def to_px(landmark):
                return int(landmark.x * w), int(landmark.y * h)

            left_inner_px, left_outer_px, left_iris_px = map(
                to_px, [left_eye_inner, left_eye_outer, left_iris])
            right_inner_px, right_outer_px, right_iris_px = map(
                to_px, [right_eye_inner, right_eye_outer, right_iris])

            # Compute iris position relative to eye width
            left_ratio = (left_iris_px[0] - left_outer_px[0]) / \
                (left_inner_px[0] - left_outer_px[0])
            right_ratio = (
                right_iris_px[0] - right_outer_px[0]) / (right_inner_px[0] - right_outer_px[0])

            # **Adjust Thresholds for Better Accuracy**
            LOOK_LEFT_THRESHOLD = 0.35
            LOOK_RIGHT_THRESHOLD = 0.65
            LOOK_UP_THRESHOLD = 5  # Pixel difference
            LOOK_DOWN_THRESHOLD = 5

            # Determine Gaze Direction
            if left_ratio < LOOK_LEFT_THRESHOLD and right_ratio < LOOK_LEFT_THRESHOLD:
                return {"gaze_status": "Looking Left"}
            elif left_ratio > LOOK_RIGHT_THRESHOLD and right_ratio > LOOK_RIGHT_THRESHOLD:
                return {"gaze_status": "Looking Right"}
            elif left_iris_px[1] < left_inner_px[1] - LOOK_UP_THRESHOLD and right_iris_px[1] < right_inner_px[1] - LOOK_UP_THRESHOLD:
                return {"gaze_status": "Looking Up"}
            elif left_iris_px[1] > left_inner_px[1] + LOOK_DOWN_THRESHOLD and right_iris_px[1] > right_inner_px[1] + LOOK_DOWN_THRESHOLD:
                return {"gaze_status": "Looking Down"}
            else:
                return {"gaze_status": "Looking at Screen"}

    return {"gaze_status": "No Face Detected"}
