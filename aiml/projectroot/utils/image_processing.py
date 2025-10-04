import cv2
import numpy as np


def preprocess_image(blob):
    """Decodes a blob image and resizes it for model processing."""
    try:
        npimg = np.frombuffer(blob, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (640, 480))  # Standardized size
        return img_resized
    except Exception as e:
        return None, str(e)
