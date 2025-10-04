import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


def detect_multiple_faces(frame):
    """
    Detects faces and checks if multiple faces are present.

    :param frame: Input image frame (numpy array)
    :return: Dictionary with face detection result.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return {"error": "Invalid image format"}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        face_count = len(results.detections)
        if face_count > 1:
            return {"multiple_faces": "Two Faces Found"}
        return {"multiple_faces": "One Face Detected"}

    return {"multiple_faces": "No Face Detected"}


def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        return "Face detected"
    else:
        return "No face detected"
