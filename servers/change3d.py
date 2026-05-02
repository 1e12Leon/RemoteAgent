import asyncio
import base64
import logging
import os
import sys
import time
import traceback
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from model.trainer import Trainer
except ImportError as e:
    raise ImportError(f"Failed to import Change3D model dependencies: {e}. Please ensure 'model.trainer' is accessible.")

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATHS = {
    'bcd': os.path.join(ROOT_DIR, "exp", "LEVIR-CD_iter_80000_lr_0.0002", "best_model.pth"),
    'scd': os.path.join(ROOT_DIR, "exp", "HRSCD_iter_80000_lr_0.0002", "best_model.pth"),
    'bda': os.path.join(ROOT_DIR, "exp", "xBD_iter_200000_lr_0.0002", "best_model.pth"),
}
ALLOWED_TASKS = ('bcd', 'scd', 'bda')

DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
PORT = 6658
HOST = "0.0.0.0"

TASK_CONFIGS = {
    'bcd': {'num_perception_frame': 1, 'num_class': 2, 'dataset': 'BCD'},
    'scd': {'num_perception_frame': 3, 'num_class': 6, 'dataset': 'HRSCD'},
    'bda': {'num_perception_frame': 2, 'num_class': 5, 'dataset': 'BDA'},
}

COMMON_CONFIG = {
    'in_height': 256,
    'in_width': 256,
    'pretrained': os.path.join(ROOT_DIR, 'X3D_L.pyth'),
}

NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5
DEFAULT_THRESHOLD = 0.5

change3d_models = {}
model_configs = {}
request_locks = {}
model_loading_locks = {}

# --- FastAPI Setup ---
app = FastAPI(
    title="Change3D API Server",
    description="Change Detection API based on Change3D model",
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
    pre_image: str 
    post_image: str 
    num_class: Optional[int] = 2
    threshold: Optional[float] = 0.5

# --- Model Management ---
def get_or_load_model(task: str):
    global change3d_models, model_configs
    task_lower = task.lower()
    
    if task_lower in change3d_models:
        return change3d_models[task_lower], model_configs[task_lower]
    
    class Args:
        def __init__(self, config):
            for key, value in config.items():
                setattr(self, key, value)
    
    task_config = {**COMMON_CONFIG, **TASK_CONFIGS.get(task_lower, TASK_CONFIGS['bcd'])}
    model_args = Args(task_config)
    model_configs[task_lower] = model_args
    
    logger.info(f"Loading Change3D model ({task_lower}) on {DEVICE}...")
    
    try:
        model = Trainer(model_args).to(DEVICE)
        model.eval()
        checkpoint_path = CHECKPOINT_PATHS[task_lower]
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
            
        change3d_models[task_lower] = model
        logger.info(f"Change3D model ({task_lower}) loaded successfully.")
        
        return model, model_args
        
    except Exception as e:
        logger.error(f"Failed to load model ({task_lower}): {e}", exc_info=True)
        raise

@app.on_event("startup")
async def load_all_models():
    logger.info("Initializing Change3D API service...")
    for task in TASK_CONFIGS.keys():
        request_locks[task] = asyncio.Lock()
        model_loading_locks[task] = asyncio.Lock()

    loop = asyncio.get_event_loop()
    for task in ALLOWED_TASKS:
        try:
            await loop.run_in_executor(None, get_or_load_model, task)
        except Exception as e:
            logger.error(f"Startup loading failed for {task}: {e}")

    logger.info(f"Startup complete. Loaded tasks: {list(change3d_models.keys())}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"Internal server error: {str(exc)}",
            "error_type": type(exc).__name__
        }
    )

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

def preprocess_image(img_bgr: np.ndarray, target_size: tuple = (256, 256), use_bgr: bool = False) -> torch.Tensor:
    img = img_bgr if use_bgr else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img.astype(np.float32) / 255.0
    img_normalized = (img_float - NORMALIZE_MEAN) / NORMALIZE_STD
    img_resized = cv2.resize(img_normalized, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0)

def postprocess_binary_mask(mask: torch.Tensor, threshold: Optional[float] = None) -> np.ndarray:
    thresh = threshold if threshold is not None else DEFAULT_THRESHOLD
    mask_np = mask.squeeze().cpu().detach().numpy()
    return (mask_np > thresh).astype(np.uint8) * 255

def postprocess_multiclass_mask(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.squeeze(0).cpu().detach().numpy()
    if len(mask_np.shape) == 3:
        mask_np = np.argmax(mask_np, axis=0)
    return mask_np.astype(np.uint8)

def apply_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    if mask is None:
        return image.copy()
    
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 or image.shape[2] == 1 else image.copy()
    
    if len(mask.shape) > 2:
        mask = mask.squeeze()
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
            
    if mask.shape[:2] != vis.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    mask_bool = mask > 127
    if not mask_bool.any():
        return vis
        
    mask_3d = np.stack([mask_bool] * vis.shape[2], axis=2)
    color_layer = np.full_like(vis, np.array(color, dtype=np.uint8))
    
    return np.where(mask_3d,
                    (vis.astype(np.float32) * (1 - alpha) + color_layer.astype(np.float32) * alpha).astype(np.uint8),
                    vis)

def visualize_multi_class_mask(image: np.ndarray, mask: np.ndarray, num_classes: int) -> np.ndarray:
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 or image.shape[2] == 1 else image.copy()
    
    if len(mask.shape) > 2:
        mask = mask.squeeze()
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
            
    if mask.shape[:2] != vis.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    colors = [
        (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
    ]
    
    for cls_id in range(1, min(num_classes + 1, len(colors))):
        mask_cls = (mask == cls_id)
        if not mask_cls.any():
            continue
            
        mask_3d = np.stack([mask_cls] * vis.shape[2], axis=2)
        color_layer = np.full_like(vis, np.array(colors[cls_id], dtype=np.uint8))
        vis = np.where(mask_3d,
                       (vis.astype(np.float32) * 0.5 + color_layer.astype(np.float32) * 0.5).astype(np.uint8),
                       vis)
    return vis

# --- Core API Endpoint ---
@app.post("/predict")
async def predict(request: PredictRequest):
    task_lower = request.task.lower().strip()
    if task_lower not in ALLOWED_TASKS:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Unsupported task: {request.task}"})
        
    start_time = time.time()
    if task_lower not in request_locks:
        request_locks[task_lower] = asyncio.Lock()
        
    async with request_locks[task_lower]:
        try:
            loop = asyncio.get_event_loop()
            change3d_model, model_args = await loop.run_in_executor(None, get_or_load_model, request.task)
        except Exception as e:
            return JSONResponse(status_code=503, content={"status": "error", "message": f"Model load failed: {e}"})

        pre_img_bgr = base64_to_cv2(request.pre_image)
        post_img_bgr = base64_to_cv2(request.post_image)
        
        if pre_img_bgr is None or post_img_bgr is None:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid image data"})
            
        try:
            target_size = (model_args.in_height, model_args.in_width)
            use_bgr = (task_lower == "bda")
            pre_img_tensor = preprocess_image(pre_img_bgr, target_size=target_size, use_bgr=use_bgr).to(DEVICE)
            post_img_tensor = preprocess_image(post_img_bgr, target_size=target_size, use_bgr=use_bgr).to(DEVICE)
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": f"Preprocessing failed: {e}"})

        result_data = {}
        
        with torch.no_grad():
            if task_lower == "bcd":
                change_mask = change3d_model.update_bcd(pre_img_tensor, post_img_tensor)
                change_mask_np = postprocess_binary_mask(change_mask, request.threshold)
                
                vis_img = apply_mask(pre_img_bgr, change_mask_np, color=(0, 255, 0), alpha=0.5)
                change_pixels = np.sum(change_mask_np > 127)
                change_ratio = float(change_pixels / change_mask_np.size)
                
                result_data.update({
                    "result_image": cv2_to_base64(vis_img),
                    "change_mask": cv2_to_base64(change_mask_np),
                    "pre_image_original": cv2_to_base64(pre_img_bgr),
                    "post_image_original": cv2_to_base64(post_img_bgr),
                    "change_ratio": change_ratio,
                    "change_pixels": int(change_pixels)
                })
                msg = f"Binary Change Detection completed. Change ratio: {change_ratio*100:.2f}%"

            elif task_lower == "scd":
                pre_mask, post_mask, change_mask = change3d_model.update_scd(pre_img_tensor, post_img_tensor)
                
                pre_mask_np = postprocess_multiclass_mask(pre_mask)
                post_mask_np = postprocess_multiclass_mask(post_mask)
                change_mask_np = postprocess_binary_mask(change_mask, request.threshold)
                
                change_mask_bool = change_mask_np > 127
                pre_mask_np = pre_mask_np * change_mask_bool.astype(np.uint8)
                post_mask_np = post_mask_np * change_mask_bool.astype(np.uint8)
                
                vis_pre = visualize_multi_class_mask(pre_img_bgr, pre_mask_np, model_args.num_class)
                vis_post = visualize_multi_class_mask(post_img_bgr, post_mask_np, model_args.num_class)
                vis_change = apply_mask(pre_img_bgr, change_mask_np, color=(255, 0, 0))
                
                result_data.update({
                    "pre_mask_image": cv2_to_base64(vis_pre),
                    "post_mask_image": cv2_to_base64(vis_post),
                    "change_mask_image": cv2_to_base64(vis_change),
                    "pre_mask": cv2_to_base64(pre_mask_np),
                    "post_mask": cv2_to_base64(post_mask_np),
                    "change_mask": cv2_to_base64(change_mask_np)
                })
                msg = f"Semantic Change Detection completed."

            elif task_lower == "bda":
                pred_cls, pred_loc = change3d_model.update_bda(pre_img_tensor, post_img_tensor)
                
                loc_np = postprocess_binary_mask(pred_loc, request.threshold)
                cls_np = postprocess_multiclass_mask(pred_cls)
                loc_mask_bool = loc_np > 127
                
                damage_class = 0
                if loc_mask_bool.any():
                    cls_in_damage = cls_np[loc_mask_bool]
                    unique_classes, counts = np.unique(cls_in_damage, return_counts=True)
                    damage_class = int(unique_classes[np.argmax(counts)])
                    
                vis_img = apply_mask(pre_img_bgr, loc_np, color=(255, 0, 0))
                
                result_data.update({
                    "damage_location": cv2_to_base64(loc_np),
                    "damage_class_map": cv2_to_base64(cls_np),
                    "result_image": cv2_to_base64(vis_img),
                    "damage_class": damage_class,
                    "has_damage": bool(loc_mask_bool.any())
                })
                msg = f"Building Damage Assessment completed. Damage class: {damage_class}"

        result_data["processing_time"] = f"{time.time() - start_time:.2f}s"
        
        return {
            "status": "success",
            "message": msg,
            "task": request.task,
            **result_data
        }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(change3d_models.keys()),
        "device": DEVICE,
        "available_tasks": list(ALLOWED_TASKS)
    }

if __name__ == "__main__":
    print(f"🚀 Change3D API running at: http://{HOST}:{PORT}")
    print(f"📄 API Docs available at: http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT)