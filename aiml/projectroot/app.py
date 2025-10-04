import threading
import queue
import time
import uuid  # For unique task IDs
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from detection.person_detection import detect_person
from detection.electronic_device_detection import detect_electronic_devices
from detection.notebook_detection import detect_notebook
from detection.activity_recognition import recognize_activities

app = Flask(__name__)
CORS(app)

# Create a queue to store requests
request_queue = queue.Queue()
results = {}  # Store results

# Worker function to process queue


def process_queue():
    while True:
        task_id, img = request_queue.get()
        if task_id is None:
            break  # Stop worker thread if None is received

        results[task_id] = process_image(img)
        request_queue.task_done()


worker_thread = threading.Thread(target=process_queue, daemon=True)
worker_thread.start()


def process_image(img):
    """Processes the image and returns analysis results."""
    return {
        "person": detect_person(img),
        "electronic_devices": detect_electronic_devices(img),
        "notebook": detect_notebook(img),
        "activity": recognize_activities(img)
    }


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Receive image as blob
        blob = request.files['image'].read()
        npimg = np.frombuffer(blob, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Generate a unique task ID
        task_id = str(uuid.uuid4())  # Unique ID for each request

        # Add request to queue
        request_queue.put((task_id, img))

        return jsonify({"status": "queued", "task_id": task_id})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    if task_id in results:
        return jsonify(results.pop(task_id))  # Return and remove result
    else:
        return jsonify({"status": "processing"}), 202


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


# the code using before changing yesterday
# import threading
# import queue
# import time
# from flask import Flask, request, jsonify
# from flask_cors import CORS  # Import CORS
# import cv2
# import numpy as np
# from detection.person_detection import detect_person
# from detection.electronic_device_detection import detect_electronic_devices
# from detection.notebook_detection import detect_notebook
# from detection.activity_recognition import recognize_activities

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Create a request queue
# request_queue = queue.Queue()

# # Worker function to process requests


# def process_queue():
#     while True:
#         task = request_queue.get()
#         if task is None:
#             break  # Stop worker thread if None is received

#         task_id, img = task
#         response = process_image(img)
#         results[task_id] = response
#         request_queue.task_done()


# # Dictionary to store results
# results = {}

# # Start a background thread for processing
# worker_thread = threading.Thread(target=process_queue, daemon=True)
# worker_thread.start()


# def process_image(img):
#     """Processes the image and returns analysis results."""
#     person_result = detect_person(img)
#     device_result = detect_electronic_devices(img)
#     notebook_result = detect_notebook(img)
#     activity_result = recognize_activities(img)

#     return {
#         "person": person_result,
#         "electronic_devices": device_result,
#         "notebook": notebook_result,
#         "activity": activity_result
#     }


# @app.route('/analyze', methods=['POST'])
# def analyze():
#     try:
#         # Receive image as blob
#         blob = request.files['image'].read()
#         npimg = np.frombuffer(blob, np.uint8)
#         img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#         # Generate unique request ID
#         task_id = str(time.time())

#         # Add request to queue
#         request_queue.put((task_id, img))

#         return jsonify({"status": "queued", "task_id": task_id})
#     except Exception as e:
#         return jsonify({"error": str(e)})


# @app.route('/result/<task_id>', methods=['GET'])
# def get_result(task_id):
#     if task_id in results:
#         return jsonify(results.pop(task_id))  # Return and remove result
#     else:
#         return jsonify({"status": "processing"}), 202


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)


# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# from detection.person_detection import detect_person
# from detection.electronic_device_detection import detect_electronic_devices
# from detection.notebook_detection import detect_notebook
# from detection.activity_recognition import recognize_activity

# app = Flask(__name__)


# @app.route('/analyze', methods=['POST'])
# def analyze():
#     try:
#         # Receive image as blob
#         blob = request.files['image'].read()
#         npimg = np.frombuffer(blob, np.uint8)
#         img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#         # Run detections
#         person_result = detect_person(img)
#         device_result = detect_electronic_devices(img)
#         notebook_result = detect_notebook(img)
#         activity_result = recognize_activity(img)

#         # Consolidate results
#         result = {
#             "person": person_result,
#             "electronic_devices": device_result,
#             "notebook": notebook_result,
#             "activity": activity_result
#         }

#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)})


# if __name__ == '__main__':
#     app.run(debug=True)
