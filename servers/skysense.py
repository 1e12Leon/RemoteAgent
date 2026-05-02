import base64
import logging
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mmdet.apis import inference_detector, init_detector

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SkySenseDetAPI")

CONFIG_PATH = "./detection/configs/swin_transformer_v2/faster_rcnn_swinv2_huge_patch4_window8_fpn-1x_dior.py"
CHECKPOINT_PATH = "./detection/work_dirs/faster_rcnn_swinv2_huge_patch4_window8_fpn-1x_dior/latest.pth"

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
HOST = "0.0.0.0"
PORT = 6654 

det_model = None

# --- FastAPI Setup ---
app = FastAPI(
    title="SkySense Detection API Server",
    description="Object detection using SkySense (SwinV2 Huge) + MMDetection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class DetectRequest(BaseModel):
    task: str = "detection"
    image: str
    score_thr: float = 0.3
    classes: Optional[List[str]] = None
    return_image: bool = True

# --- Utility Functions ---
def base64_to_cv2(b64_str: str) -> Optional[np.ndarray]:
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        data = base64.b64decode(b64_str)
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        return None

def cv2_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

def draw_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: List[str],
    color: tuple = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    canvas = image.copy()
    h, w = image.shape[:2]
    font_scale = max(0.5, min(w, h) / 1000.0)
    font_thickness = max(1, int(font_scale * 2))

    for box, label in zip(boxes, labels):
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x1 >= x2 or y1 >= y2:
            continue

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        y_text = max(y1, th + 5)
        
        cv2.rectangle(
            canvas,
            (x1, y_text - th - 5),
            (x1 + tw + 10, y_text + baseline),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 5, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return canvas

# --- Startup Event ---
@app.on_event("startup")
async def load_model():
    global det_model
    logger.info(f"Loading SkySense detection model on {DEVICE}...")
    if CHECKPOINT_PATH is not None and not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    det_model = init_detector(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
    logger.info("SkySense detection model loaded successfully.")

# --- Main API Endpoint ---
@app.post("/predict")
async def predict(req: DetectRequest):
    if det_model is None:
        raise HTTPException(status_code=503, detail="Detection model is not loaded")

    if req.task != "detection":
        raise HTTPException(status_code=400, detail=f"Unsupported task: {req.task}")

    img_bgr = base64_to_cv2(req.image)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    with torch.no_grad():
        result = inference_detector(det_model, img_bgr)

    class_names = list(getattr(det_model, "CLASSES", [])) or [
        f"class_{i}" for i in range(len(result))
    ]

    score_thr = float(req.score_thr)
    all_dets: Dict[str, List[List[float]]] = {}
    vis_boxes: List[List[float]] = []
    vis_labels: List[str] = []
    total = 0

    classes_filter = set(req.classes) if req.classes else None

    for cls_idx, cls_name in enumerate(class_names):
        boxes = result[cls_idx]
        if boxes is None or len(boxes) == 0:
            continue

        cls_dets = []
        for b in boxes:
            x1, y1, x2, y2, score = map(float, b[:5])
            if score < score_thr:
                continue
            
            det = [x1, y1, x2, y2, score]
            cls_dets.append(det)

            if classes_filter is None or cls_name in classes_filter:
                vis_boxes.append([x1, y1, x2, y2])
                vis_labels.append(f"{cls_name} {score:.2f}")

        if cls_dets and (classes_filter is None or cls_name in classes_filter):
            all_dets[cls_name] = cls_dets
            total += len(cls_dets)

    result_image_b64 = None
    if req.return_image and vis_boxes:
        vis_img = draw_boxes(img_bgr, vis_boxes, vis_labels)
        result_image_b64 = cv2_to_base64(vis_img)

    return {
        "status": "success",
        "task": req.task,
        "num_detections": total,
        "detections": all_dets,
        "classes": class_names,
        "score_thr": score_thr,
        "result_image": result_image_b64,
    }

if __name__ == "__main__":
    print(f"🚀 SkySense Detection API running at: http://{HOST}:{PORT}")
    print(f"📄 API Docs available at: http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT)