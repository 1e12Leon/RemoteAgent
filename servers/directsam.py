import base64
import logging
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# --- DirectSAM Path Setup ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DIRECTSAM_DIR = os.path.join(_SCRIPT_DIR, "DirectSAM")
if _DIRECTSAM_DIR not in sys.path:
    sys.path.insert(0, _DIRECTSAM_DIR)

try:
    from utils import inference_single_image, probs_to_masks
except ImportError as e:
    raise ImportError(f"Failed to import DirectSAM utils: {e}. Please ensure DirectSAM is correctly placed.")

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHECKPOINT_ENTITY_SEG = "chendelong/DirectSAM-EntitySeg-1024px-0501"
CHECKPOINT_1800PX = "chendelong/DirectSAM-1800px-0424"
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
PORT = 6659
HOST = "0.0.0.0"

# Global Models
image_processor_entity_seg = None
directsam_model_entity_seg = None
image_processor_1800px = None
directsam_model_1800px = None

# --- FastAPI Setup ---
app = FastAPI(
    title="DirectSAM Contour Extraction API",
    description="Contour, Region, and Subobject contour extraction using DirectSAM.",
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
    task: str
    image: str
    box: Optional[List[float]] = None
    threshold: float = 0.25
    resolution: Optional[int] = None
    pyramid_layers: int = 0
    max_masks: int = 256


# --- Startup Event ---
@app.on_event("startup")
async def load_models():
    global image_processor_entity_seg, directsam_model_entity_seg
    global image_processor_1800px, directsam_model_1800px

    logger.info(f"Loading EntitySeg model on {DEVICE}...")
    image_processor_entity_seg = AutoImageProcessor.from_pretrained(CHECKPOINT_ENTITY_SEG, reduce_labels=True)
    directsam_model_entity_seg = AutoModelForSemanticSegmentation.from_pretrained(CHECKPOINT_ENTITY_SEG).to(DEVICE).eval()
    logger.info("EntitySeg model loaded successfully.")

    logger.info(f"Loading 1800px model on {DEVICE}...")
    image_processor_1800px = AutoImageProcessor.from_pretrained(CHECKPOINT_1800PX, reduce_labels=True)
    directsam_model_1800px = AutoModelForSemanticSegmentation.from_pretrained(CHECKPOINT_1800PX).to(DEVICE).eval()
    logger.info("1800px model loaded successfully.")


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


def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_base64(img_arr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img_arr)
    return base64.b64encode(buffer).decode('utf-8')


def _crop_image_pil(pil_img: Image.Image, box: List[float]) -> Image.Image:
    x1, y1, x2, y2 = [int(round(x)) for x in box[:4]]
    w, h = pil_img.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid box dimensions")
    return pil_img.crop((x1, y1, x2, y2))


def _paste_probs_to_full(probs_crop: np.ndarray, full_h: int, full_w: int, box: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    out = np.zeros((full_h, full_w), dtype=probs_crop.dtype)
    hc, wc = probs_crop.shape
    out[y1:y1 + hc, x1:x1 + wc] = probs_crop
    return out


def _masks_to_contour_vis(image_bgr: np.ndarray, masks: List[np.ndarray], color=(0, 255, 0), thickness=2) -> np.ndarray:
    canvas = image_bgr.copy()
    for mask in masks:
        mask_u8 = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, thickness)
    return canvas


def _contour_map_vis(image_bgr: np.ndarray, contour_binary: np.ndarray, color=(0, 255, 0), alpha=0.6) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay[contour_binary > 0] = color
    return cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)


# --- Core Task Inference Functions ---
def run_contour_extraction(pil_image: Image.Image, threshold: float, resolution: Optional[int], pyramid_layers: int) -> Tuple[np.ndarray, np.ndarray]:
    probs = inference_single_image(pil_image, image_processor_entity_seg, directsam_model_entity_seg, resolution=resolution, pyramid_layers=pyramid_layers)
    contour_binary = (probs >= threshold).astype(np.uint8) * 255
    return probs, contour_binary


def run_region_contour_extraction(pil_image: Image.Image, box: List[float], threshold: float, resolution: Optional[int], pyramid_layers: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    crop = _crop_image_pil(pil_image, box)
    probs_crop = inference_single_image(crop, image_processor_entity_seg, directsam_model_entity_seg, resolution=resolution, pyramid_layers=pyramid_layers)
    contour_crop = (probs_crop >= threshold).astype(np.uint8) * 255

    w, h = pil_image.size
    box_int = [int(round(x)) for x in box[:4]]
    probs_full = _paste_probs_to_full(probs_crop, h, w, box_int)
    contour_full = _paste_probs_to_full(contour_crop, h, w, box_int)

    return probs_full, contour_full, probs_crop, contour_crop


def run_subobject_contour_extraction(pil_image: Image.Image, threshold: float, resolution: Optional[int], pyramid_layers: int, max_masks: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    probs = inference_single_image(pil_image, image_processor_1800px, directsam_model_1800px, resolution=resolution, pyramid_layers=pyramid_layers)
    masks = probs_to_masks(probs, threshold=threshold)[:max_masks]
    return probs, masks


def run_region_subobject_contour_extraction(pil_image: Image.Image, box: List[float], threshold: float, resolution: Optional[int], pyramid_layers: int, max_masks: int) -> Tuple[np.ndarray, List[np.ndarray], Tuple[int, int]]:
    crop = _crop_image_pil(pil_image, box)
    probs_crop = inference_single_image(crop, image_processor_1800px, directsam_model_1800px, resolution=resolution, pyramid_layers=pyramid_layers)
    masks_crop = probs_to_masks(probs_crop, threshold=threshold)[:max_masks]
    return probs_crop, masks_crop, crop.size


# --- API Endpoint ---
@app.post("/predict")
async def predict(request: PredictRequest):
    if directsam_model_entity_seg is None or directsam_model_1800px is None:
        raise HTTPException(status_code=503, detail="Models are not fully loaded")

    img_bgr = base64_to_cv2(request.image)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    pil_image = cv2_to_pil(img_bgr)
    result_data = {}
    msg = ""

    try:
        if request.task == "contour_extraction":
            probs, contour_binary = run_contour_extraction(pil_image, request.threshold, request.resolution, request.pyramid_layers)
            vis = _contour_map_vis(img_bgr, contour_binary)

            result_data["result_image"] = cv2_to_base64(vis)
            result_data["contour_mask_image"] = cv2_to_base64(cv2.cvtColor(contour_binary, cv2.COLOR_GRAY2BGR))
            msg = "Global contour extraction completed"

        elif request.task == "region_contour_extraction":
            if not request.box or len(request.box) < 4:
                raise ValueError("The 'box' parameter [x1, y1, x2, y2] is required")

            probs_full, contour_full, probs_crop, contour_crop = run_region_contour_extraction(
                pil_image, request.box, request.threshold, request.resolution, request.pyramid_layers
            )
            vis = _contour_map_vis(img_bgr, contour_full)

            result_data["result_image"] = cv2_to_base64(vis)
            result_data["contour_mask_image"] = cv2_to_base64(cv2.cvtColor(contour_full, cv2.COLOR_GRAY2BGR))
            result_data["region_contour_mask_image"] = cv2_to_base64(cv2.cvtColor(contour_crop, cv2.COLOR_GRAY2BGR))
            msg = "Region contour extraction completed"

        elif request.task == "subobject_contour_extraction":
            probs, masks = run_subobject_contour_extraction(
                pil_image, request.threshold, request.resolution, request.pyramid_layers, request.max_masks
            )
            vis = _masks_to_contour_vis(img_bgr, masks)

            result_data["result_image"] = cv2_to_base64(vis)
            result_data["num_subobjects"] = len(masks)
            msg = f"DirectSAM subobject contour extraction completed. Found {len(masks)} regions."

        elif request.task == "region_subobject_contour_extraction":
            if not request.box or len(request.box) < 4:
                raise ValueError("The 'box' parameter [x1, y1, x2, y2] is required")

            probs_crop, masks_crop, _ = run_region_subobject_contour_extraction(
                pil_image, request.box, request.threshold, request.resolution, request.pyramid_layers, request.max_masks
            )

            crop_bgr = pil_to_cv2(_crop_image_pil(pil_image, request.box))
            vis_crop = _masks_to_contour_vis(crop_bgr, masks_crop)
            x1, y1, x2, y2 = [int(round(x)) for x in request.box[:4]]

            full_vis = img_bgr.copy()
            full_vis[y1:y2, x1:x2] = vis_crop

            result_data["result_image"] = cv2_to_base64(full_vis)
            result_data["region_result_image"] = cv2_to_base64(vis_crop)
            result_data["num_subobjects"] = len(masks_crop)
            msg = f"DirectSAM region subobject contour extraction completed. Found {len(masks_crop)} regions."

        else:
            raise ValueError(f"Unsupported task: {request.task}")

        return {
            "status": "success",
            "task": request.task,
            "message": msg,
            **result_data,
        }

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "task": request.task}


if __name__ == "__main__":
    print(f"🚀 DirectSAM Contour API running at: http://{HOST}:{PORT}")
    print(f"📄 API Docs available at: http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT)
