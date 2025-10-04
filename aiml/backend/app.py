from face_verification import verify_identity
from face_detection import detect_multiple_faces, detect_faces
from device_detection import detect_device
from gaze_tracking import detect_gaze
from truehuman import process_image
from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)

# Allow CORS for all origins
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/process', methods=['POST'])
def process_frame():
    if 'frame' not in request.files or 'userId' not in request.form:
        return jsonify({"error": "No frame or userId received"}), 400

    file = request.files['frame']
    userId = request.form['userId']  # Assign userId AFTER checking it exists

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Perform detection tasks
    face_result = detect_faces(frame)
    gaze_result = detect_gaze(frame)
    device_result = detect_device(frame)
    multiple_faces_result = detect_multiple_faces(frame)
    identity_result = verify_identity(frame, userId)
    liveness_result = process_image(frame)  # Call liveness detection

    # Ensure gaze is correctly classified
    # if gaze_result not in ['Looking at screen', 'No face detected']:
    #     gaze_result = "Not looking at screen"

    # Construct response
    response = {
        "face_detection": face_result,
        "gaze_detection": gaze_result,
        "device_detection": device_result,
        "multiple_faces": multiple_faces_result,
        "identity_verification": identity_result,
        "liveness_detection": liveness_result  # Add liveness result
    }

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
