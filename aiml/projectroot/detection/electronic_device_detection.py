import cv2
import torch
from torchvision import models, transforms, ops

# =========================
# Model & device setup
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the default COCO-pretrained Faster R-CNN
model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()
model.to(DEVICE)

# =========================
# Labels (COCO 91 classes)
# =========================
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
IDX = {name: i for i, name in enumerate(COCO_INSTANCE_CATEGORY_NAMES)}

# =========================
# What counts as "electronics" here
# NOTE: COCO has NO "tablet" class.
# =========================
LAPTOP = IDX['laptop']           # 64
CELL_PHONE = IDX['cell phone']   # 68

# Optional & rough tablet approximation: treat "tv" as a "tablet" proxy (OFF by default)
USE_TABLET_APPROX = False
TABLET_PROXY = IDX['tv'] if 'tv' in IDX else None

if USE_TABLET_APPROX and TABLET_PROXY is not None:
    ELECTRONIC_CLASS_IDS = {LAPTOP, CELL_PHONE, TABLET_PROXY}
else:
    ELECTRONIC_CLASS_IDS = {LAPTOP, CELL_PHONE}

# =========================
# Preprocessing
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),  # [0,1], CHW
])

# =========================
# NMS per class
# =========================
def _apply_nms_per_class(boxes, scores, labels, iou_thresh=0.5):
    """
    Apply class-wise NMS and return kept indices (sorted by descending score).
    """
    keep_indices = []
    labels_list = labels.tolist()
    for cl in sorted(set(labels_list)):
        cls_idx = [i for i, L in enumerate(labels_list) if L == cl]
        if not cls_idx:
            continue
        b = boxes[cls_idx]
        s = scores[cls_idx]
        keep = ops.nms(b, s, iou_thresh)
        keep_indices.extend([cls_idx[i.item()] for i in keep])
    keep_indices = sorted(keep_indices, key=lambda i: scores[i].item(), reverse=True)
    return keep_indices

# =========================
# Detector
# =========================
def detect_electronic_devices(image_bgr, conf_thresh=0.5, nms_iou=0.5, max_laptops_allowed=1):
    """
    Detect ONLY the allowed electronic devices (laptop, cell phone, and optionally a crude tablet proxy)
    and enforce a policy:
      - Valid if <= max_laptops_allowed and no OTHER allowed devices present
      - Otherwise violation

    Returns:
        {
          "status": "valid" | "violation" | "error",
          "message": str,
          "counts": {"laptop": int, "other_electronics": int, "total_electronics": int},
          "detections": [{"label": str, "score": float, "bbox": [x1,y1,x2,y2]}]
        }
    """
    try:
        if image_bgr is None:
            return {"status": "error", "message": "Input image is None."}

        # BGR -> RGB -> tensor
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = transform(image_rgb).to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)[0]

        # Raw outputs
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]

        # Confidence filter
        conf_mask = scores >= conf_thresh
        if conf_mask.sum().item() == 0:
            return {
                "status": "valid",
                "message": "No electronic devices detected.",
                "counts": {"laptop": 0, "other_electronics": 0, "total_electronics": 0},
                "detections": []
            }
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]
        labels = labels[conf_mask]

        # Keep ONLY our (allowed) electronic devices
        elec_mask = torch.tensor(
            [int(l.item()) in ELECTRONIC_CLASS_IDS for l in labels],
            device=labels.device,
            dtype=torch.bool
        )

        if elec_mask.sum().item() == 0:
            return {
                "status": "valid",
                "message": "No electronic devices detected.",
                "counts": {"laptop": 0, "other_electronics": 0, "total_electronics": 0},
                "detections": []
            }

        boxes = boxes[elec_mask]
        scores = scores[elec_mask]
        labels = labels[elec_mask]

        # NMS per class
        keep_idx = _apply_nms_per_class(boxes, scores, labels, iou_thresh=nms_iou)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

        # Summarize
        detections = []
        laptop_count = 0
        other_count = 0

        for i in range(len(labels)):
            lid = int(labels[i].item())
            score = float(scores[i].item())
            x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
            name = COCO_INSTANCE_CATEGORY_NAMES[lid] if 0 <= lid < len(COCO_INSTANCE_CATEGORY_NAMES) else f"id_{lid}"

            # Normalize label names in output (map tv->tablet if using proxy)
            label_name = name
            if USE_TABLET_APPROX and TABLET_PROXY is not None and lid == TABLET_PROXY:
                label_name = "tablet (approx)"

            detections.append({"label": label_name, "score": score, "bbox": [x1, y1, x2, y2]})

            if lid == LAPTOP:
                laptop_count += 1
            else:
                # any other allowed electronic (cell phone, or tablet proxy if enabled)
                other_count += 1

        total_elec = laptop_count + other_count

        # Decision: valid only if <= max_laptops_allowed and no other electronics
        if laptop_count <= max_laptops_allowed and other_count == 0:
            status = "valid"
            message = "Only one laptop detected." if laptop_count == 1 else "No electronic devices detected."
        else:
            status = "violation"
            reasons = []
            if laptop_count > max_laptops_allowed:
                reasons.append(f"{laptop_count} laptops detected")
            if other_count > 0:
                # list distinct other labels for clarity
                other_names = sorted({d["label"] for d in detections if d["label"] != "laptop"})
                reasons.append("Other devices detected: " + ", ".join(other_names))
            message = "; ".join(reasons) if reasons else "Violation detected."

        return {
            "status": status,
            "message": message,
            "counts": {"laptop": laptop_count, "other_electronics": other_count, "total_electronics": total_elec},
            "detections": detections
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Example: read an image and run detection
    # img = cv2.imread("path_to_image.jpg")
    # result = detect_electronic_devices(img, conf_thresh=0.6, nms_iou=0.5, max_laptops_allowed=1)
    # print(result)
    pass
