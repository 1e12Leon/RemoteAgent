import base64
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

try:
    from tasks.code.model import RemoteSAM, init_demo_model
except ImportError as e:
    raise ImportError(f"Failed to import RemoteSAM model: {e}")

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "./pretrained_weights/checkpoint.pth"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PORT = 6657
HOST = "0.0.0.0"

sam_model = None

# --- FastAPI Setup ---
app = FastAPI(
    title="RemoteSAM API Server",
    description="Based on RemoteSAM for Referring Segmentation Tasks",
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
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Ignore unexpected fields from remote agents
    task: str
    image: str
    text: Optional[str] = ""
    classes: Optional[List[str]] = []
    box_threshold: Optional[float] = None
    text_threshold: Optional[float] = None
    mask_size: Optional[int] = None  # If specified, skip upsampling and output masks at this target resolution

# --- Startup Event ---
@app.on_event("startup")
async def load_model():
    global sam_model
    logger.info(f"Loading RemoteSAM model on {DEVICE}...")
    base_model = init_demo_model(CHECKPOINT_PATH, DEVICE)
    sam_model = RemoteSAM(base_model, DEVICE, use_EPOC=True)
    logger.info("Model loaded successfully.")

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

def binary_mask_to_base64(mask: np.ndarray) -> str:
    """
    Convert a (H, W) binary mask into base64 PNG bytes.
    Encoded as uint8 0/255 to maintain evaluation consistency.
    """
    m = np.asarray(mask)
    m = np.squeeze(m)
    m = (m > 0).astype(np.uint8) * 255
    _, buf = cv2.imencode(".png", m)
    return base64.b64encode(buf).decode("utf-8")

def align_mask_to_image(mask: np.ndarray, image_hw: Tuple[int, int]) -> np.ndarray:
    """
    Ensure mask spatial size matches the input image.
    Handles cases where RemoteSAM outputs square masks for non-square input images.
    """
    if mask is None:
        return mask

    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    
    if mask.ndim < 2:
        return mask

    h, w = image_hw
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask

def apply_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    """Overlay binary mask on the image."""
    vis = image.copy()
    if mask is None:
        return vis
        
    h, w = image.shape[:2]
    mask = align_mask_to_image(mask, (h, w))
    mask_bool = mask > 0
    
    if not mask_bool.any():
        return vis
        
    overlay = np.zeros_like(image)
    overlay[mask_bool] = color
    vis[mask_bool] = cv2.addWeighted(
        image[mask_bool], 1 - alpha, overlay[mask_bool], alpha, 0
    )
    return vis

# --- Main API Endpoint ---
@app.post("/predict")
async def predict(request: PredictRequest):
    if sam_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    img_bgr = base64_to_cv2(request.image)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result_data = {}

    try:
        if request.task in ("referring_seg", "referring_segmentation"):
            if not (request.text or "").strip():
                raise ValueError("Text prompt is required for referring segmentation")
            
            m_hw = (
                (request.mask_size, request.mask_size)
                if request.mask_size is not None
                else None
            )
            mask = sam_model.referring_seg(
                image=img_rgb, sentence=request.text.strip(), mask_output_hw=m_hw
            )
            
            if mask is not None:
                mask = np.asarray(mask).squeeze()
                if request.mask_size is None:
                    mask = align_mask_to_image(mask, img_bgr.shape[:2])
            
            vis = apply_mask(img_bgr, mask)
            result_data["result_image"] = cv2_to_base64(vis)
            
            # Generate binary mask for downstream evaluation
            if mask is not None:
                mask_uint8 = (mask > 0).astype(np.uint8) * 255
                _, buf = cv2.imencode(".png", mask_uint8)
                result_data["mask"] = base64.b64encode(buf).decode("utf-8")
                
        elif request.task in ("semantic_seg", "semantic_segmentation"):
            if not request.classes:
                raise ValueError("Classes are required for semantic segmentation")

            m_hw = (
                (request.mask_size, request.mask_size)
                if request.mask_size is not None
                else None
            )
            res = sam_model.semantic_seg(
                image=img_rgb, classnames=request.classes, mask_output_hw=m_hw
            )

            vis = img_bgr.copy()
            combined_mask = None
            per_class_masks: dict = {}

            for cls in request.classes:
                if not isinstance(res, dict) or cls not in res:
                    continue
                m = res.get(cls, None)
                if m is None:
                    continue

                m = np.asarray(m).squeeze()

                # Reduce multiple masks for a single class (N, H, W) into a single union mask
                if m.ndim == 3:
                    m = (m > 0).any(axis=0).astype(np.uint8)

                if m.ndim < 2:
                    continue

                if request.mask_size is None:
                    m = align_mask_to_image(m, img_bgr.shape[:2])
                
                vis = apply_mask(vis, m)
                per_class_masks[cls] = binary_mask_to_base64(m)

                if combined_mask is None:
                    combined_mask = m.copy()
                else:
                    combined_mask = np.maximum(combined_mask, m)

            result_data["result_image"] = cv2_to_base64(vis)

            if combined_mask is not None:
                result_data["mask"] = binary_mask_to_base64(combined_mask)

            if per_class_masks:
                result_data["masks"] = per_class_masks

        else:
            raise ValueError(f"Unsupported task: {request.task}")

        return {
            "status": "success",
            "task": request.task,
            **result_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "task": request.task}

if __name__ == "__main__":
    print(f"🚀 Server running at: http://{HOST}:{PORT}")
    print(f"📄 API Docs available at: http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT)