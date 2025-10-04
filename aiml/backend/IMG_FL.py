from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import os
from face_detection import detect_multiple_faces
from flask_cors import CORS
import cv2
import numpy as np


# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Firebase Initialization
cred = credentials.Certificate("serviceAccountKey.json")  # Ensure this exists
firebase_admin.initialize_app(cred)
db = firestore.client()

# Upload Route


@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        data = request.json
        image_data = data.get("image")  # Base64-encoded image
        unique_id = str(data.get("unique_id"))  # Unique integer ID

        if not image_data or not unique_id:
            return jsonify({"error": "Missing image or unique_id"}), 400

        # Ensure the base64 string is correctly formatted
        if "," in image_data:
            # Remove "data:image/jpeg;base64,"
            image_data = image_data.split(",")[1]

        # Decode Base64 string into OpenCV image format
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Face Detection
        face_result = detect_multiple_faces(frame)["multiple_faces"]

        print(f"Face Detection Result: {face_result}")

        if face_result != "One Face Detected":
            return jsonify({"error": "Invalid face count", "result": face_result}), 400

        # Retrieve existing images from Firestore
        doc_ref = db.collection("images").document(unique_id)
        doc = doc_ref.get()

        if doc.exists:
            existing_data = doc.to_dict()
            image_list = existing_data.get(
                "images", [])  # Get existing images list
        else:
            image_list = []  # No existing images

        # Limit to 5 images
        if len(image_list) >= 5:
            return jsonify({"error": "Maximum of 5 images allowed"}), 400

        # Append the new image
        image_list.append(image_data)

        # Store updated images list in Firestore
        doc_ref.set({"unique_id": unique_id, "images": image_list})

        return jsonify({"message": "Image stored successfully!", "images_count": len(image_list)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Retrieve Image by Unique ID


@app.route("/retrieve/<unique_id>", methods=["GET"])
def retrieve_images(unique_id):
    try:
        doc = db.collection("images").document(unique_id).get()

        if not doc.exists:
            return jsonify({"error": "No image found for this ID"}), 404

        return jsonify(doc.to_dict()), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010, debug=True)
