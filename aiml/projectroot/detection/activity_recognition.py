import cv2
import torch
from torchvision import models, transforms
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load pre-trained pose estimation model (Keypoint R-CNN)
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


def convert_numpy_types(data):
    """Convert NumPy data types to native Python types for JSON serialization."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    else:
        return data


def recognize_activities(image):
    """Determines various activities with percentage confidence."""
    try:
        img_tensor = transform(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            predictions = model(img_tensor)[0]

        if len(predictions['keypoints']) == 0:
            return {"status": "violation", "message": "No person detected."}

        keypoints = predictions['keypoints'][0].cpu().numpy()
        nose = keypoints[0][:2]  # Nose
        left_hand = keypoints[9][:2]  # Left wrist
        right_hand = keypoints[10][:2]  # Right wrist
        left_foot = keypoints[15][:2]  # Left ankle
        right_foot = keypoints[16][:2]  # Right ankle

        # --- Activity Recognition ---
        looking_at_screen = False
        writing_typing = False
        sitting = False
        standing = False
        other_action = False

        confidence_scores = {
            "looking_at_screen": 0,
            "writing_typing": 0,
            "sitting": 0,
            "standing": 0,
            "other_action": 0
        }

        # --- Looking at Screen ---
        if nose[0] > left_hand[0] and nose[0] > right_hand[0]:  # Face direction logic
            looking_at_screen = True
            confidence_scores["looking_at_screen"] = 85  # Adjust as needed

        # --- Writing/Typing ---
        hand_distance = np.linalg.norm(left_hand - right_hand)
        if hand_distance < 50:
            writing_typing = True
            confidence_scores["writing_typing"] = 80
        # Hand below nose (writing posture)
        elif left_hand[1] < nose[1] or right_hand[1] < nose[1]:
            writing_typing = True
            confidence_scores["writing_typing"] = 70

        # --- Sitting/Standing ---
        height_ratio = abs(left_foot[1] - right_foot[1])
        if height_ratio < 50:  # Small height difference = Sitting
            sitting = True
            confidence_scores["sitting"] = 90
        else:
            standing = True
            confidence_scores["standing"] = 90

        # --- Other Action ---
        if not looking_at_screen and not writing_typing and not sitting and not standing:
            other_action = True
            confidence_scores["other_action"] = 95

        # Construct response JSON
        result = {
            "looking_at_screen": looking_at_screen,
            "looking_at_screen_confidence": confidence_scores["looking_at_screen"],
            "writing_typing": writing_typing,
            "writing_typing_confidence": confidence_scores["writing_typing"],
            "sitting": sitting,
            "sitting_confidence": confidence_scores["sitting"],
            "standing": standing,
            "standing_confidence": confidence_scores["standing"],
            "other_action": other_action,
            "other_action_confidence": confidence_scores["other_action"]
        }

        return convert_numpy_types(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.route('/detect_activity', methods=['POST'])
def detect_activity():
    """API Endpoint to receive an image and return detected activity."""
    try:
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(
            file.read(), np.uint8), cv2.IMREAD_COLOR)
        result = recognize_activities(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
