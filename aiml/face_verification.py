# # import cv2
# # import numpy as np
# # from deepface import DeepFace
# # from sklearn.metrics.pairwise import cosine_similarity

# # # Store multiple face embeddings for a more reliable identity check
# # stored_face_embeddings = []
# # THRESHOLD = 0.3  # Lower threshold for stricter matching
# # NUM_SAMPLES = 5  # Number of face samples to capture


# # def get_face_embedding(frame):
# #     """
# #     Extracts the face embedding (numerical feature representation) from the frame.
# #     """
# #     try:
# #         # Convert BGR (OpenCV) to RGB (DeepFace expects RGB)
# #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #         # result = DeepFace.represent(
# #         #     rgb_frame, model_name="Facenet", enforce_detection=False, detector_backend="opencv"
# #         # )
# #         result = DeepFace.represent(
# #             rgb_frame, model_name="Facenet512", enforce_detection=False, detector_backend="retinaface"
# #         )

# #         if result:
# #             # Reshape for cosine similarity
# #             return np.array(result[0]['embedding']).reshape(1, -1)

# #     except Exception as e:
# #         print(f"Error extracting face embedding: {e}")

# #     return None  # No face detected or error occurred


# # def average_embedding(embeddings):
# #     """
# #     Computes the average embedding from multiple stored face embeddings.
# #     Ensures the output is of shape (1, N) where N is the feature dimension.
# #     """
# #     if len(embeddings) == 0:
# #         return None
# #     return np.mean(np.vstack(embeddings), axis=0, keepdims=True)


# # def verify_identity(frame):
# #     """
# #     Verifies if the detected face matches the stored face profile.
# #     """
# #     global stored_face_embeddings

# #     if frame is None or not isinstance(frame, np.ndarray):
# #         return {"error": "Invalid image format"}

# #     face_embedding = get_face_embedding(frame)

# #     if face_embedding is not None:
# #         if len(stored_face_embeddings) < NUM_SAMPLES:
# #             stored_face_embeddings.append(face_embedding)
# #             return {"identity_status": f"Face Stored {len(stored_face_embeddings)}/{NUM_SAMPLES}"}

# #         # Compute average embedding from stored samples
# #         avg_embedding = average_embedding(stored_face_embeddings)

# #         # Compute Cosine Similarity
# #         similarity = cosine_similarity(avg_embedding, face_embedding)[0][0]

# #         if similarity > 1 - THRESHOLD:  # Higher similarity means same face
# #             return {"identity_status": "Same Face"}
# #         else:
# #             return {"identity_status": "Different Face Detected"}

# #     return {"identity_status": "No Face Detected"}


# # import cv2
# # import numpy as np
# # from deepface import DeepFace
# # from sklearn.metrics.pairwise import cosine_similarity

# # # Store the first detected face embedding
# # initial_face_embedding = None
# # THRESHOLD = 0.3  # Lower threshold for stricter matching


# # def get_face_embedding(frame):
# #     """
# #     Extracts the face embedding (numerical feature representation) from the frame.
# #     """
# #     try:
# #         # Convert BGR (OpenCV) to RGB (DeepFace expects RGB)
# #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #         # result = DeepFace.represent(
# #         #     rgb_frame, model_name="Facenet", enforce_detection=False, detector_backend="opencv"
# #         # )
# #         result = DeepFace.represent(
# #             rgb_frame, model_name="Facenet512", enforce_detection=False, detector_backend="retinaface"
# #         )

# #         if result:
# #             # Reshape for cosine similarity
# #             return np.array(result[0]['embedding']).reshape(1, -1)

# #     except Exception as e:
# #         print(f"Error extracting face embedding: {e}")

# #     return None  # No face detected or error occurred


# # def verify_identity(frame):
# #     """
# #     Verifies if the detected face matches the initially stored face.
# #     """
# #     global initial_face_embedding

# #     if frame is None or not isinstance(frame, np.ndarray):
# #         return {"error": "Invalid image format"}

# #     face_embedding = get_face_embedding(frame)

# #     if face_embedding is not None:
# #         if initial_face_embedding is None:
# #             initial_face_embedding = face_embedding
# #             return {"identity_status": "Face Stored"}

# #         # Compute Cosine Similarity
# #         similarity = cosine_similarity(
# #             initial_face_embedding, face_embedding)[0][0]

# #         if similarity > 1 - THRESHOLD:  # Higher similarity means same face
# #             return {"identity_status": "Same Face"}
# #         else:
# #             return {"identity_status": "Different Face Detected"}

# #     return {"identity_status": "No Face Detected"}


# # import cv2
# # import numpy as np
# # from deepface import DeepFace
# # from sklearn.metrics.pairwise import cosine_similarity

# # # Load OpenCV's DNN face detector
# # face_net = cv2.dnn.readNetFromCaffe(
# #     "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# # # Store the first detected face embedding
# # initial_face_embedding = None
# # THRESHOLD = 0.3  # Lower threshold for stricter matching


# # def detect_face(frame):
# #     """Detects faces in an image using OpenCV's DNN model."""
# #     h, w = frame.shape[:2]
# #     blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(
# #         300, 300), mean=(104.0, 177.0, 123.0))
# #     face_net.setInput(blob)
# #     detections = face_net.forward()

# #     for i in range(detections.shape[2]):
# #         confidence = detections[0, 0, i, 2]
# #         if confidence > 0.5:  # Minimum confidence threshold
# #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# #             (startX, startY, endX, endY) = box.astype("int")
# #             face = frame[startY:endY, startX:endX]
# #             return face if face.size > 0 else None
# #     return None


# # def get_face_embedding(frame):
# #     """Extracts the face embedding from the detected face."""
# #     try:
# #         face = detect_face(frame)
# #         if face is None:
# #             return None

# #         # Convert BGR (OpenCV) to RGB (DeepFace expects RGB)
# #         rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

# #         # Get face embedding using DeepFace
# #         result = DeepFace.represent(
# #             rgb_face, model_name="Facenet512", enforce_detection=False, detector_backend="opencv"
# #         )

# #         if result:
# #             return np.array(result[0]['embedding']).reshape(1, -1)
# #     except Exception as e:
# #         print(f"Error extracting face embedding: {e}")
# #     return None


# # def verify_identity(frame):
# #     """Verifies if the detected face matches the initially stored face."""
# #     global initial_face_embedding

# #     if frame is None or not isinstance(frame, np.ndarray):
# #         return {"error": "Invalid image format"}

# #     face_embedding = get_face_embedding(frame)
# #     if face_embedding is not None:
# #         if initial_face_embedding is None:
# #             initial_face_embedding = face_embedding
# #             return {"identity_status": "Face Stored"}

# #         # Compute Cosine Similarity
# #         similarity = cosine_similarity(
# #             initial_face_embedding, face_embedding)[0][0]

# #         if similarity > 1 - THRESHOLD:
# #             return {"identity_status": "Same Face"}
# #         else:
# #             return {"identity_status": "Different Face Detected"}

# #     return {"identity_status": "No Face Detected"}


# # REALLY GOOD WORKING CODE IF NOTHING WORKS THIS IS BEST DONT FORGET THIS BELOW CODE ***********
# # import cv2
# # import numpy as np
# # from deepface import DeepFace
# # from sklearn.metrics.pairwise import cosine_similarity

# # # Load OpenCV's DNN face detector
# # face_net = cv2.dnn.readNetFromCaffe(
# #     "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"
# # )

# # # Store multiple detected face embeddings
# # initial_face_embeddings = []
# # MAX_INITIAL_IMAGES = 5
# # THRESHOLD = 0.3  # Lower threshold for stricter matching


# # def detect_face(frame):
# #     """Detects faces in an image using OpenCV's DNN model."""
# #     h, w = frame.shape[:2]
# #     blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(
# #         300, 300), mean=(104.0, 177.0, 123.0))
# #     face_net.setInput(blob)
# #     detections = face_net.forward()

# #     for i in range(detections.shape[2]):
# #         confidence = detections[0, 0, i, 2]
# #         if confidence > 0.5:  # Minimum confidence threshold
# #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# #             (startX, startY, endX, endY) = box.astype("int")
# #             face = frame[startY:endY, startX:endX]
# #             return face if face.size > 0 else None
# #     return None


# # def get_face_embedding(frame):
# #     """Extracts the face embedding from the detected face."""
# #     try:
# #         face = detect_face(frame)
# #         if face is None:
# #             return None

# #         # Convert BGR (OpenCV) to RGB (DeepFace expects RGB)
# #         rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

# #         # Get face embedding using DeepFace
# #         result = DeepFace.represent(
# #             rgb_face, model_name="Facenet512", enforce_detection=False, detector_backend="opencv"
# #         )

# #         if result:
# #             return np.array(result[0]['embedding']).reshape(1, -1)
# #     except Exception as e:
# #         print(f"Error extracting face embedding: {e}")
# #     return None


# # def verify_identity(frame):
# #     """Verifies if the detected face matches the initially stored face."""
# #     global initial_face_embeddings

# #     if frame is None or not isinstance(frame, np.ndarray):
# #         return {"error": "Invalid image format"}

# #     face_embedding = get_face_embedding(frame)
# #     if face_embedding is not None:
# #         if len(initial_face_embeddings) < MAX_INITIAL_IMAGES:
# #             initial_face_embeddings.append(face_embedding)
# #             return {"identity_status": "Face Stored", "stored_faces": len(initial_face_embeddings)}

# #         # Compute average embedding
# #         avg_embedding = np.mean(initial_face_embeddings, axis=0)

# #         # Compute Cosine Similarity
# #         similarity = cosine_similarity(avg_embedding, face_embedding)[0][0]

# #         if similarity > 1 - THRESHOLD:
# #             return {"identity_status": "Same Face"}
# #         else:
# #             return {"identity_status": "Different Face Detected"}

# #     return {"identity_status": "No Face Detected"}


# import cv2
# import numpy as np
# from deepface import DeepFace
# from sklearn.metrics.pairwise import cosine_similarity
# import firebase_admin
# from firebase_admin import credentials, firestore
# import base64
# import os
# MODEL_DIR = os.path.dirname(__file__)
# # Load OpenCV's DNN face detector
# face_net = cv2.dnn.readNetFromCaffe(    # <- change to these 2 lines
#     os.path.join(MODEL_DIR, "deploy.prototxt"),
#     os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel"),
# )

# MODEL_DIR = os.path.dirname(__file__)
# CRED_PATH = os.path.join(MODEL_DIR, "serviceAccountKey.json")

# cred = credentials.Certificate(CRED_PATH)
# firebase_admin.initialize_app(cred)
# db = firestore.client()

# # Store multiple detected face embeddings
# initial_face_embeddings = []
# MAX_INITIAL_IMAGES = 5
# THRESHOLD = 0.3  # Lower threshold for stricter matching


# def detect_face(frame):
#     """Detects faces in an image using OpenCV's DNN model."""
#     h, w = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(
#         300, 300), mean=(104.0, 177.0, 123.0))
#     face_net.setInput(blob)
#     detections = face_net.forward()

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:  # Minimum confidence threshold
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             face = frame[startY:endY, startX:endX]
#             return face if face.size > 0 else None
#     return None


# def get_face_embedding(frame):
#     """Extracts the face embedding from the detected face."""
#     try:
#         face = detect_face(frame)
#         if face is None:
#             return None

#         # Convert BGR (OpenCV) to RGB (DeepFace expects RGB)
#         rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

#         # Get face embedding using DeepFace
#         result = DeepFace.represent(
#             rgb_face, model_name="Facenet512", enforce_detection=False, detector_backend="opencv"
#         )

#         if result:
#             return np.array(result[0]['embedding']).reshape(1, -1)
#     except Exception as e:
#         print(f"Error extracting face embedding: {e}")
#     return None


# def fetch_initial_images(unique_id):
#     """Fetch the first 5 images for a specific user from Firestore and store their embeddings."""
#     global initial_face_embeddings
#     initial_face_embeddings.clear()

#     print(f"[INFO] Fetching images for user: {unique_id}")

#     # Reference Firestore document
#     doc_ref = db.collection("images").document(unique_id)
#     doc = doc_ref.get()

#     if not doc.exists:
#         print(f"[ERROR] No images found for user: {unique_id}")
#         return

#     data = doc.to_dict()
#     image_list = data.get("images", [])[:MAX_INITIAL_IMAGES]

#     print(
#         f"[INFO] Retrieved {len(image_list)} images from Firestore for user {unique_id}")

#     if not image_list:
#         print("[ERROR] No images available in Firestore document.")
#         return

#     for idx, image_data in enumerate(image_list):
#         try:
#             # Decode base64 image
#             image_bytes = base64.b64decode(image_data)
#             image_array = np.frombuffer(image_bytes, np.uint8)
#             frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

#             if frame is None:
#                 print(f"[ERROR] Image {idx+1} could not be decoded.")
#                 continue

#             # Get face embedding
#             embedding = get_face_embedding(frame)
#             if embedding is not None:
#                 initial_face_embeddings.append(embedding)
#             else:
#                 print(f"[WARNING] No face detected in image {idx+1}.")

#         except Exception as e:
#             print(f"[ERROR] Exception processing image {idx+1}: {e}")

#     print(
#         f"[INFO] Loaded {len(initial_face_embeddings)} face embeddings for user: {unique_id}")

#  # Load initial images when script starts


# def verify_identity(frame, userId):
#     fetch_initial_images(userId)
#     """Verifies if the detected face matches the initially stored face."""
#     global initial_face_embeddings

#     if frame is None or not isinstance(frame, np.ndarray):
#         return {"error": "Invalid image format"}

#     face_embedding = get_face_embedding(frame)
#     if face_embedding is not None:
#         if len(initial_face_embeddings) < MAX_INITIAL_IMAGES:
#             return {"identity_status": "Not enough initial faces stored", "stored_faces": len(initial_face_embeddings)}

#         # Compute average embedding
#         avg_embedding = np.mean(initial_face_embeddings, axis=0)

#         # Compute Cosine Similarity
#         similarity = cosine_similarity(avg_embedding, face_embedding)[0][0]

#         if similarity > 1 - THRESHOLD:
#             return {"identity_status": "Same Face"}
#         else:
#             return {"identity_status": "Different Face Detected"}

#     return {"identity_status": "No Face Detected"}


import os
import time
import base64
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

import firebase_admin
from firebase_admin import credentials, firestore

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = os.path.dirname(__file__)
CRED_PATH = os.path.join(MODEL_DIR, "serviceAccountKey.json")
CAFFE_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

MAX_INITIAL_IMAGES = 5
THRESHOLD = 0.3          # stricter is smaller
MIN_FACE_CONF = 0.5
CACHE_TTL_SEC = 3600     # set to None to disable TTL
MAX_CACHE_USERS = 100    # simple cap to avoid unbounded memory

# -----------------------------
# One-time init
# -----------------------------
face_net = cv2.dnn.readNetFromCaffe(CAFFE_PROTO, CAFFE_MODEL)

if not firebase_admin._apps:
    cred = credentials.Certificate(CRED_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# -----------------------------
# Multi-user in-memory cache
# cache[user_id] = {"embs": np.ndarray (K,D), "ts": float}
# -----------------------------
_USER_CACHE: dict[str, dict] = {}

def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)

def _cache_valid(user_id: str) -> bool:
    item = _USER_CACHE.get(user_id)
    if not item:
        return False
    if item["embs"] is None or len(item["embs"]) < MAX_INITIAL_IMAGES:
        return False
    if CACHE_TTL_SEC is not None and (time.time() - item["ts"]) > CACHE_TTL_SEC:
        return False
    return True

def _touch_cache(user_id: str):
    """Simple LRU-ish control: trim if over capacity."""
    if len(_USER_CACHE) > MAX_CACHE_USERS:
        # evict oldest by timestamp
        oldest = min(_USER_CACHE.items(), key=lambda kv: kv[1]["ts"])[0]
        _USER_CACHE.pop(oldest, None)

# -----------------------------
# Face utilities
# -----------------------------
def _detect_face(frame: np.ndarray) -> np.ndarray | None:
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    best = None
    best_conf = 0.0
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= MIN_FACE_CONF and conf > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype(int)
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = max(0, endX), max(0, endY)
            face = frame[startY:endY, startX:endX]
            if face.size > 0:
                best = face
                best_conf = conf
    return best

def _get_face_embedding(frame: np.ndarray) -> np.ndarray | None:
    try:
        face = _detect_face(frame)
        if face is None:
            return None
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        result = DeepFace.represent(
            rgb_face,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="opencv"
        )
        if result:
            emb = np.array(result[0]["embedding"], dtype=np.float32).reshape(1, -1)
            return _l2_normalize(emb)
    except Exception as e:
        print(f"[ERROR] _get_face_embedding: {e}")
    return None

# -----------------------------
# Firestore fetch (only when needed)
# -----------------------------
def _extract_b64(any_item) -> str | None:
    """
    Accept base64 in multiple shapes:
    - "iVBOR..." (raw string)
    - {"data": "..."} or {"image": "..."} or {"b64": "..."} etc.
    """
    if isinstance(any_item, str):
        return any_item
    if isinstance(any_item, dict):
        for k in ("data", "image", "b64", "base64", "content"):
            val = any_item.get(k)
            if isinstance(val, str):
                return val
    return None

def _load_user_embeddings_from_firestore(user_id: str) -> np.ndarray | None:
    print(f"[INFO] Fetching images for user: {user_id}")
    doc = db.collection("images").document(user_id).get()
    if not doc.exists:
        print(f"[ERROR] No images found for user: {user_id}")
        return None

    data = doc.to_dict() or {}
    image_list = (data.get("images") or [])[:MAX_INITIAL_IMAGES]
    if not image_list:
        print("[ERROR] No images in document.")
        return None

    embs = []
    for idx, item in enumerate(image_list, start=1):
        try:
            b64 = _extract_b64(item)
            if not b64:
                print(f"[WARN] Image {idx}: unsupported format ({type(item)}).")
                continue
            img_bytes = base64.b64decode(b64)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Image {idx} decode failed.")
                continue
            emb = _get_face_embedding(img)
            if emb is not None:
                embs.append(emb)
            else:
                print(f"[WARN] No face detected in enrollment image {idx}.")
        except Exception as e:
            print(f"[ERROR] Enrollment image {idx}: {e}")

    if not embs:
        print("[ERROR] No usable embeddings from Firestore.")
        return None

    stacked = np.vstack(embs).astype(np.float32)  # (K, D)
    stacked = _l2_normalize(stacked)              # row-wise normalize
    return stacked

def _ensure_user_cached(user_id: str):
    if _cache_valid(user_id):
        return
    embs = _load_user_embeddings_from_firestore("77")
    if embs is not None:
        _USER_CACHE[user_id] = {"embs": embs, "ts": time.time()}
        _touch_cache(user_id)
    else:
        # keep a negative cache to avoid re-fetching in the same call
        _USER_CACHE[user_id] = {"embs": None, "ts": time.time()}

# -----------------------------
# Public API
# -----------------------------
def verify_identity(frame: np.ndarray, user_id: str) -> dict:
    """
    Call this for each frame.
    Firestore is consulted only when we have no valid cache for `user_id`.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return {"error": "Invalid image format"}

    _ensure_user_cached(user_id)
    slot = _USER_CACHE.get(user_id, {})
    embs = slot.get("embs")

    if embs is None or len(embs) < MAX_INITIAL_IMAGES:
        return {
            "identity_status": "Not enough initial faces stored",
            "stored_faces": 0 if embs is None else int(len(embs))
        }

    # Average template -> re-normalize (cosine-friendly)
    avg_template = _l2_normalize(np.mean(embs, axis=0, keepdims=True).astype(np.float32))  # (1, D)

    probe = _get_face_embedding(frame)
    if probe is None:
        return {"identity_status": "No Face Detected"}

    sim = float(cosine_similarity(avg_template, probe)[0][0])
    return {
        "identity_status": "Same Face" if sim > (1.0 - THRESHOLD) else "Different Face Detected",
        "similarity": round(sim, 4)
    }
