# import cv2
# import numpy as np
# import torch
# from torchvision import models, transforms

# # Load pre-trained Faster R-CNN model
# model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()

# # Define transformation
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ToTensor()
# ])


# def detect_person(image):
#     """Detects if exactly one person is present in the image."""
#     try:
#         # Convert image to tensor
#         img_tensor = transform(image).unsqueeze(0)

#         # Perform inference
#         with torch.no_grad():
#             predictions = model(img_tensor)[0]

#         # Count detected persons
#         person_count = sum(1 for i, label in enumerate(
#             predictions['labels']) if label == 1 and predictions['scores'][i] > 0.8)

#         # Determine if valid
#         if person_count == 1:
#             return {"status": "valid", "message": "One person detected."}
#         elif person_count == 0:
#             return {"status": "violation", "message": "No person detected."}
#         else:
#             return {"status": "violation", "message": "Multiple people detected."}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}



import cv2
import numpy as np
import torch
from torchvision import models, transforms

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load pre-trained Faster R-CNN model and move it to device
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


def detect_person(image):
    """Detects if exactly one person is present in the image."""
    try:
        # Convert image to tensor and move to device
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            predictions = model(img_tensor)[0]

        # Count persons with high confidence
        person_count = sum(
            1 for i, label in enumerate(predictions['labels'])
            if label.item() == 1 and predictions['scores'][i].item() > 0.8
        )

        print(f"[DEBUG] Person detections: {person_count} (Device: {device})")

        # Decision logic
        if person_count == 1:
            return {"status": "valid", "message": "One person detected."}
        elif person_count == 0:
            return {"status": "violation", "message": "No person detected."}
        else:
            return {"status": "violation", "message": "Multiple people detected."}
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        return {"status": "error", "message": str(e)}
