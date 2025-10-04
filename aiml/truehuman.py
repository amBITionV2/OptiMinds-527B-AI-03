# import cv2
# import dlib
# import numpy as np
# from scipy.spatial import distance

# # Load face detector and landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# THRESHOLD_MOVEMENT = 10  # Head movement threshold
# THRESHOLD_EAR = 0.25  # Eye Aspect Ratio threshold for partially closed eyes

# # Store previous face landmarks for movement check
# previous_landmarks = None


# def eye_aspect_ratio(eye):
#     """Calculate Eye Aspect Ratio (EAR) to check eye openness"""
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)


# def detect_liveness(frame):
#     """
#     Detects liveness based on head movement and eye openness.
#     Returns a dictionary with the liveness status.
#     """
#     global previous_landmarks

#     if frame is None or not isinstance(frame, np.ndarray):
#         return {"error": "Invalid image format"}

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     if len(faces) == 0:
#         return {"liveness_status": "No Face Detected"}

#     for face in faces:
#         landmarks = predictor(gray, face)

#         # Extract eye landmarks
#         left_eye = [(landmarks.part(n).x, landmarks.part(n).y)
#                     for n in range(36, 42)]
#         right_eye = [(landmarks.part(n).x, landmarks.part(n).y)
#                      for n in range(42, 48)]

#         # Compute Eye Aspect Ratio (EAR)
#         ear_left = eye_aspect_ratio(left_eye)
#         ear_right = eye_aspect_ratio(right_eye)
#         ear = (ear_left + ear_right) / 2.0

#         # **HEAD MOVEMENT CHECK**
#         if previous_landmarks is not None:
#             movement = sum(abs(landmarks.part(i).x -
#                            previous_landmarks.part(i).x) for i in range(68))
#             if movement < THRESHOLD_MOVEMENT:
#                 return {"liveness_status": "Possible Spoofing (Static Image Detected)"}

#         # **EYE OPENNESS CHECK**
#         if ear < THRESHOLD_EAR:
#             return {"liveness_status": "Real Face Detected (Partially Closed Eyes)"}

#         previous_landmarks = landmarks  # Store landmarks for next frame

#     return {"liveness_status": "Real Face Detected"}

# import cv2
# import numpy as np
# import dlib
# import json
# from scipy.spatial import distance

# # Load OpenCV's face detector & Dlib's facial landmarks predictor
# face_detector = cv2.dnn.readNetFromCaffe(
#     "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
# predictor = dlib.shape_predictor(
#     "shape_predictor_68_face_landmarks.dat")

# EYE_AR_THRESH = 0.25  # Eye aspect ratio threshold for blinking
# BLINK_FRAMES = 3  # Frames required to confirm a blink
# TOTAL_FRAMES = 10  # Reduced frames for faster spoofing detection
# blink_counter = 0  # Tracks consecutive frames without blinking


# def eye_aspect_ratio(eye):
#     """Computes the eye aspect ratio to detect blinks."""
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)


# def detect_face(image):
#     """Detects face in an image using OpenCV's DNN model."""
#     h, w = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
#     face_detector.setInput(blob)
#     detections = face_detector.forward()

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             return image[startY:endY, startX:endX], (startX, startY, endX, endY)
#     return None, None


# def detect_liveness(image):
#     """Detects if a face is real or spoofed from an image."""
#     global blink_counter

#     face, bbox = detect_face(image)
#     if face is None:
#         return "No Face Detected"

#     # Convert to grayscale for texture analysis
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
#     landmarks = predictor(gray, rect)

#     # Get eye landmarks
#     left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y)
#                         for n in range(36, 42)])
#     right_eye = np.array(
#         [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

#     # Compute blink detection
#     left_ear = eye_aspect_ratio(left_eye)
#     right_ear = eye_aspect_ratio(right_eye)
#     ear = (left_ear + right_ear) / 2.0

#     if ear < EYE_AR_THRESH:
#         # Reset blink count when detected
#         blink_counter = max(blink_counter - 3, 0)
#     else:
#         blink_counter += 1  # Increase if no blink detected

#     if blink_counter >= TOTAL_FRAMES:
#         return "Spoofing Detected"

#     # Texture analysis (edges & color distortion check)
#     edges = cv2.Laplacian(gray, cv2.CV_64F).var()
#     if edges < 50:  # Low edge variation means likely a printed photo
#         return "Spoofing Detected"

#     return "Real Face Detected"


# def process_image(image):
#     """Runs face detection and liveness detection, returning a JSON response."""
#     face_detected, _ = detect_face(image)
#     liveness_result = detect_liveness(image)

#     result = {
#         "face_detected": face_detected is not None and face_detected.size > 0,
#         "liveness_status": liveness_result
#     }

#     return json.dumps(result)


import cv2
import numpy as np
import dlib
import json
from scipy.spatial import distance
from collections import deque
import os
_THIS_DIR = os.path.dirname(__file__)  # add this once near the imports

face_detector = cv2.dnn.readNetFromCaffe(
    os.path.join(_THIS_DIR, "deploy.prototxt"),
    os.path.join(_THIS_DIR, "res10_300x300_ssd_iter_140000.caffemodel"),
)
predictor = dlib.shape_predictor(os.path.join(_THIS_DIR, "shape_predictor_68_face_landmarks.dat"))

EYE_AR_THRESH = 0.25  # Eye aspect ratio threshold for blinking
TOTAL_FRAMES = 10  # Number of frames to analyze
MOTION_THRESH = 2.0  # Motion threshold for real face

# Store recent images for analysis
image_queue = deque(maxlen=TOTAL_FRAMES)
blink_counter = 0  # Tracks consecutive frames without blinking
motion_counter = 0  # Tracks motion consistency
last_bbox = None  # Stores the previous face position


def eye_aspect_ratio(eye):
    """Computes the eye aspect ratio to detect blinks."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def detect_face(image):
    """Detects face in an image using OpenCV's DNN model."""
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            return image[startY:endY, startX:endX], (startX, startY, endX, endY)
    return None, None


def detect_motion(bbox):
    """Detects if there is significant motion between frames."""
    global last_bbox, motion_counter
    if last_bbox is None:
        last_bbox = bbox
        return False
    motion = sum(abs(np.array(bbox) - np.array(last_bbox)))
    last_bbox = bbox
    if motion > MOTION_THRESH:
        motion_counter += 1
    return motion > MOTION_THRESH


def analyze_images():
    """Analyzes stored images for blinking and motion detection."""
    global blink_counter, motion_counter
    blink_detected = False
    motion_detected = False
    texture_spoof = False

    for image in image_queue:
        face, bbox = detect_face(image)
        if face is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
        landmarks = predictor(gray, rect)

        left_eye = np.array(
            [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye = np.array(
            [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_AR_THRESH:
            blink_detected = True

        if detect_motion(bbox):
            motion_detected = True

        edges = cv2.Laplacian(gray, cv2.CV_64F).var()
        if edges < 50:
            texture_spoof = True

    if not blink_detected and texture_spoof:
        return "Spoofing Detected"
    if motion_detected:
        return "Real Face Detected"
    return "Uncertain"


def process_image(image):
    """Stores image and runs liveness detection on stored frames."""
    image_queue.append(image)
    liveness_result = analyze_images()
    face_detected, _ = detect_face(image)

    result = {
        "face_detected": face_detected is not None and face_detected.size > 0,
        "liveness_status": liveness_result
    }
    return json.dumps(result)

