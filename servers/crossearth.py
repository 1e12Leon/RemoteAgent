import base64
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Project Root Setup ---
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

try:
    import CrossEarth  # noqa: E402
    import rs_dataset  # noqa: E402
    from mmengine.config import Config
    from mmseg.apis import inference_model, init_model, show_result_pyplot
except ImportError as e:
    raise ImportError(f"Failed to import required modules: {e}. Please ensure CrossEarth and mmseg are installed.")

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = os.environ.get("CROSSEARTH_CONFIG", "configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_512x512_bs1x4.py")
CHECKPOINT_PATH = os.environ.get("CROSSEARTH_CHECKPOINT", "./checkpoints/Potsdam(i)-source.pth")
DEVICE = os.environ.get("CROSSEARTH_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
HOST = os.environ.get("CROSSEARTH_HOST", "0.0.0.0")
PORT = int(os.environ.get("CROSSEARTH_PORT", "6656"))

_cfg_options_env = os.environ.get("CROSSEARTH_CFG_OPTIONS", "").strip()
CFG_OPTIONS: Optional[Dict[str, Any]] = None
if _cfg_options_env:
    try:
        CFG_OPTIONS = json.loads(_cfg_options_env)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in CROSSEARTH_CFG_OPTIONS, ignoring: {e}")

seg_model = None

# --- FastAPI Setup ---
app = FastAPI(
    title="CrossEarth Semantic Segmentation API",
    description="Remote HTTP API for CrossEarth (mmseg) semantic segmentation",
    version="1.0.0",
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
    task: str = "semantic_seg"
    image: str
    text: Optional[str] = ""
    classes: Optional[List[str]] = []

# --- Utility Functions ---
def base64_to_cv2(b64_str: str) -> Optional[np.ndarray]:
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        data = base64.b64decode(b64_str)
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        return None

def cv2_to_base64(img_arr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", img_arr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buffer).decode("utf-8")

def _ensure_test_pipeline(cfg: Config) -> None:
    """Ensure test_pipeline exists in config for mmseg inference."""
    if cfg.get("test_pipeline") is not None:
        return
        
    td = cfg.get("test_dataloader")
    if not td:
        raise RuntimeError("Missing test_pipeline and test_dataloader in config.")
        
    ds = td.get("dataset")
    if not ds:
        raise RuntimeError("test_dataloader is missing the dataset field.")
        
    if ds.get("type") == "ConcatDataset":
        datasets = ds.get("datasets") or []
        if not datasets:
            raise RuntimeError("ConcatDataset 'datasets' list is empty.")
        if isinstance(datasets[0], dict) and "pipeline" in datasets[0]:
            cfg.test_pipeline = datasets[0]["pipeline"]
            return
        raise RuntimeError("Failed to extract pipeline from ConcatDataset.")
        
    if "pipeline" in ds:
        cfg.test_pipeline = ds["pipeline"]
        return
        
    raise RuntimeError("Failed to infer test_pipeline from test_dataloader.dataset.")

# --- Startup Event ---
@app.on_event("startup")
async def load_model() -> None:
    global seg_model
    cfg_path = os.path.join(ROOT, CONFIG_PATH) if not os.path.isabs(CONFIG_PATH) else CONFIG_PATH
    ckpt_path = os.path.join(ROOT, CHECKPOINT_PATH) if not os.path.isabs(CHECKPOINT_PATH) else CHECKPOINT_PATH

    if not os.path.isfile(cfg_path):
        logger.error(f"Config file not found: {cfg_path}")
        return
    if not os.path.isfile(ckpt_path):
        logger.error(f"Checkpoint file not found: {ckpt_path}")
        return

    logger.info(f"Loading CrossEarth model on {DEVICE}...")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Checkpoint: {ckpt_path}")

    cfg = Config.fromfile(cfg_path)
    if CFG_OPTIONS:
        cfg.merge_from_dict(CFG_OPTIONS)
    
    _ensure_test_pipeline(cfg)

    seg_model = init_model(cfg, ckpt_path, device=DEVICE)
    logger.info("CrossEarth model loaded successfully.")

# --- API Endpoints ---
@app.get("/health")
async def health():
    ok = seg_model is not None
    return {"status": "ok" if ok else "degraded", "model_loaded": ok}

@app.post("/predict")
async def predict(request: PredictRequest):
    if seg_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please check config/checkpoint paths.")

    if request.task != "semantic_seg":
        raise HTTPException(status_code=400, detail="This service only supports task='semantic_seg'.")

    img_bgr = base64_to_cv2(request.image)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    try:
        result = inference_model(seg_model, img_bgr)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    result_data: Dict[str, Any] = {}

    # 1. Colorized Overlay Visualization
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        vis_rgb = show_result_pyplot(
            seg_model,
            img_rgb,
            result,
            opacity=0.5,
            show=False,
            draw_gt=False,
        )
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        result_data["result_image"] = cv2_to_base64(vis_bgr)
    except Exception as e:
        logger.warning(f"Visualization failed, returning raw mask only: {e}")
        result_data["result_image"] = cv2_to_base64(img_bgr)

    # 2. Class Index Map (16-bit PNG)
    try:
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy()
        h, w = img_bgr.shape[:2]
        
        if pred.shape != (h, w):
            pred_u8 = cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        else:
            pred_u8 = np.asarray(pred, dtype=np.int32)
            
        mask_png = pred_u8.astype(np.uint16)
        ok, buf = cv2.imencode(".png", mask_png)
        
        if ok:
            result_data["mask"] = base64.b64encode(buf).decode("utf-8")
            result_data["mask_encoding"] = "png_uint16_class_index"
        else:
            result_data["mask"] = ""
    except Exception as e:
        logger.warning(f"Failed to export mask: {e}")
        result_data["mask"] = ""

    # 3. Class Names (from metadata)
    meta = getattr(seg_model, "dataset_meta", None) or {}
    clss = meta.get("classes")
    if clss:
        result_data["classes"] = list(clss)

    return {
        "status": "success",
        "task": request.task,
        "message": "Semantic segmentation completed",
        **result_data,
    }

if __name__ == "__main__":
    print(f"🚀 CrossEarth API Server running at: http://{HOST}:{PORT}")
    print(f"📄 API Docs available at: http://{HOST}:{PORT}/docs")
    print(f"🔧 Default CONFIG={CONFIG_PATH} | CHECKPOINT={CHECKPOINT_PATH}")
    uvicorn.run(app, host=HOST, port=PORT)