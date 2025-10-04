# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import mysql.connector
from datetime import datetime

from face_verification import verify_identity
from face_detection import detect_multiple_faces, detect_faces
from device_detection import detect_device
from gaze_tracking import detect_gaze
from truehuman import process_image

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
# MySQL Connection
def get_db_conn():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root123",  # your MySQL password
        database="exam_monitoring"
    )

# -------------------------
# Upload 5 reference images
# -------------------------
@app.route("/upload_faces", methods=["POST"])
def upload_faces():
    if "images" not in request.files or "userId" not in request.form:
        return jsonify({"error": "Missing files or userId"}), 400

    user_id = request.form["userId"]
    files = request.files.getlist("images")
    if len(files) != 5:
        return jsonify({"error": "Exactly 5 images required"}), 400

    user_folder = os.path.join(UPLOAD_FOLDER, user_id)
    os.makedirs(user_folder, exist_ok=True)

    conn = get_db_conn()
    cursor = conn.cursor()
    saved_paths = []

    for idx, file in enumerate(files):
        filename = f"{idx+1}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        path = os.path.join(user_folder, filename)
        file.save(path)
        saved_paths.append(path)
        cursor.execute(
            "INSERT INTO reference_images (user_id, image_path) VALUES (%s, %s)",
            (user_id, path)
        )

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Reference images uploaded", "paths": saved_paths}), 200


# -------------------------
# Monitor student
# -------------------------
@app.route("/monitor", methods=["POST"])
def monitor_student():
    if "frame" not in request.files or "userId" not in request.form:
        return jsonify({"error": "No frame or userId received"}), 400

    file = request.files["frame"]
    user_id = request.form["userId"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Run your existing detection logic
    face_result = detect_faces(frame)
    gaze_result = detect_gaze(frame)
    device_result = detect_device(frame)
    multiple_faces_result = detect_multiple_faces(frame)
    identity_result = verify_identity(frame, user_id)
    liveness_result = process_image(frame)

    # Save frame locally
    user_folder = os.path.join(UPLOAD_FOLDER, user_id, "monitoring")
    os.makedirs(user_folder, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    path = os.path.join(user_folder, filename)
    cv2.imwrite(path, frame)

    # Save result in DB
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO monitoring_results 
           (user_id, face_detected, gaze, device_detected, multiple_faces, identity_verified, liveness, image_path) 
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            user_id,
            bool(face_result),
            str(gaze_result),
            str(device_result),
            bool(multiple_faces_result),
            bool(identity_result),
            bool(liveness_result),
            path
        )
    )
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({
        "face_detection": face_result,
        "gaze_detection": gaze_result,
        "device_detection": device_result,
        "multiple_faces": multiple_faces_result,
        "identity_verification": identity_result,
        "liveness_detection": liveness_result
    }), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
