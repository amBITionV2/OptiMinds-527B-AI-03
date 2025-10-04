import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Define class IDs for electronic devices
DEVICE_CLASSES = {
    63: "Laptop",
    67: "Mobile Phone",
    77: "Tablet"  # Closely related to TV/Monitor in COCO dataset
}


def detect_device(frame):
    """
    Detects mobile phones, laptops, and tablets in a frame.
    :param frame: Image frame as a numpy array.
    :return: JSON object with detected devices.
    """
    results = model(frame)  # Run YOLOv8 inference
    detected_devices = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get detected object class ID
            if class_id in DEVICE_CLASSES:
                # Store detected device name
                detected_devices.append(DEVICE_CLASSES[class_id])

    return {"device_detection": detected_devices if detected_devices else "No Device Detected"}
