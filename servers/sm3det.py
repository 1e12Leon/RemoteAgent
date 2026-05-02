import base64
import logging
import os
import sys
import traceback
from typing import List, Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from mmcv.parallel import collate, scatter
    from mmdet.apis import init_detector
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose
    import mmrotate  # noqa: F401
except ModuleNotFoundError as e:
    raise ImportError(f"Failed to import openmmlab dependencies: {e}. Please ensure you are in the correct conda environment.")

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

CONFIG_PATH = "./configs/SM3Det/SM3Det_convnext_b.py"
CHECKPOINT_PATH = "./checkpoints/SM3Det_convnext_b.pth"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PORT = 6655
HOST = "0.0.0.0"

detector_model = None

# --- FastAPI Setup ---
app = FastAPI(
    title="SM3Det API Server",
    description="Oriented Object Detection (OBB) API supporting multi-modality (RGB, SAR, IFR).",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class PredictRequest(BaseModel):
    image: str
    classes: Optional[List[str]] = []
    box_threshold: float = 0.3
    modality: Optional[str] = "rgb"

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

def cv2_to_base64(img_arr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img_arr)
    return base64.b64encode(buffer).decode('utf-8')

def draw_rotated_boxes(
    image: np.ndarray, 
    bboxes: np.ndarray, 
    labels: np.ndarray,
    class_names: List[str], 
    score_thr: float
) -> np.ndarray:
    """Draw Oriented Bounding Boxes (OBB) [xc, yc, w, h, angle, score]"""
    if bboxes is None or len(bboxes) == 0:
        return image
    
    canvas = image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, bbox in enumerate(bboxes):
        try:
            if len(bbox) >= 6 and bbox[5] < score_thr:
                continue
            
            xc, yc, w, h, angle = map(float, bbox[:5])
            if not all(np.isfinite([xc, yc, w, h, angle])):
                continue
            
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            wx, wy = w / 2 * cos_a, w / 2 * sin_a
            hx, hy = -h / 2 * sin_a, h / 2 * cos_a
            
            pts = np.array([
                [xc - wx - hx, yc - wy - hy], [xc + wx - hx, yc + wy - hy],
                [xc + wx + hx, yc + wy + hy], [xc - wx + hx, yc - wy + hy]
            ], dtype=np.int32)
            
            label_idx = int(labels[i]) if i < len(labels) else 0
            color = colors[label_idx % len(colors)]
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
            
            score = float(bbox[5]) if len(bbox) >= 6 else 1.0
            class_name = class_names[label_idx] if label_idx < len(class_names) else f"Class_{label_idx}"
            cv2.putText(canvas, f"{class_name}: {score:.2f}", (int(xc), int(yc)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except Exception as e:
            logger.warning(f"Failed to draw rotated box: {e}")
            continue
            
    return canvas

def inference_sm3det(model, img, subdataset_mode="rgb"):
    """Run TriSourceDetector inference."""
    cfg = model.cfg.copy()
    if isinstance(img, np.ndarray):
        cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
    
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    data = dict(img=img) if isinstance(img, np.ndarray) else dict(img_info=dict(filename=img), img_prefix=None)
    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)
    data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
    data["img"] = [img.data[0] for img in data["img"]]
    data["subdataset"] = [[subdataset_mode]]

    device = next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]

    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    return results[0]

# --- Startup Event ---
@app.on_event("startup")
async def load_model():
    global detector_model
    logger.info(f"Loading SM3Det model on {DEVICE}...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    config_abs = os.path.abspath(CONFIG_PATH)
    checkpoint_abs = os.path.abspath(CHECKPOINT_PATH)
    
    if not os.path.exists(checkpoint_abs):
        raise RuntimeError(f"Checkpoint not found: {checkpoint_abs}")
        
    detector_model = init_detector(config_abs, checkpoint_abs, device=DEVICE)
    detector_model.eval()
    logger.info("SM3Det model loaded successfully.")

# --- API Endpoints ---
@app.post("/predict")
async def predict(request: PredictRequest):
    if detector_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    img_bgr = base64_to_cv2(request.image)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    try:
        if not request.classes:
            raise ValueError("The 'classes' parameter is required and cannot be empty")

        req_mode = (request.modality or "rgb").lower().strip()
        if req_mode not in ("rgb", "sar", "ifr"):
            req_mode = "rgb"

        result = inference_sm3det(detector_model, img_bgr, subdataset_mode=req_mode)
        class_names = list(getattr(detector_model, 'CLASSES', []))
        
        all_bboxes, all_labels = [], []
        score_thr = request.box_threshold

        for class_idx, class_name in enumerate(class_names):
            if class_name not in request.classes:
                continue
                
            bboxes = result[class_idx]
            for bbox in bboxes:
                score = float(bbox[-1]) if len(bbox) in (5, 6) else 1.0
                if score >= score_thr:
                    all_bboxes.append(bbox)
                    all_labels.append(class_idx)

        result_data = {
            "status": "success",
            "task": "Oriented Object Detection",
            "modality": req_mode,
        }

        if not all_bboxes:
            result_data.update({
                "message": "No objects detected",
                "num_detections": 0,
                "detections": {c: [] for c in request.classes},
                "result_image": cv2_to_base64(img_bgr)
            })
            return result_data

        all_bboxes_np = np.array(all_bboxes, dtype=np.float32)
        all_labels_np = np.array(all_labels, dtype=np.int32)

        vis_img = draw_rotated_boxes(img_bgr, all_bboxes_np, all_labels_np, class_names, score_thr)
        
        detections = {c: [] for c in request.classes}
        for bbox, label_idx in zip(all_bboxes, all_labels):
            cn = class_names[label_idx]
            arr = bbox.tolist()
            formatted_box = [float(x) for x in arr[:5]]
            formatted_box.append(float(arr[5]) if len(arr) >= 6 else 1.0)
            detections[cn].append(formatted_box)

        result_data.update({
            "message": f"Detected {len(all_bboxes)} objects",
            "num_detections": len(all_bboxes),
            "detections": detections,
            "result_image": cv2_to_base64(vis_img)
        })
        return result_data

    except ValueError as e:
        logger.warning(f"Parameter error: {e}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Error processing request: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {
        "name": "SM3Det API Server",
        "version": "1.3.0",
        "supported_task": "Oriented Object Detection (OBB)",
        "supported_modalities": ["rgb", "sar", "ifr"]
    }

@app.get("/classes")
async def get_classes():
    class_names = list(getattr(detector_model, 'CLASSES', []))
    return {"status": "success", "classes": class_names, "num_classes": len(class_names)}

if __name__ == "__main__":
    print(f"🚀 Server running at: http://{HOST}:{PORT}")
    print(f"📄 API Docs available at: http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT)