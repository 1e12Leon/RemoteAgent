import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
SCRIPT_DIR = EVAL_DIR

from remoteagent.eval_common import (
    get_vllm_model_id,
    http_post_predict,
    load_eval_system_prompt,
    run_vllm_chat,
)
from remoteagent.parsing import parse_tool_call_with_bcd_fallback
from remoteagent.services import execute_tool
from remoteagent.utils import encode_image, extract_answer_tag

# --- Image Processing Helpers ---
def img_to_base64(img_bgr: np.ndarray) -> str:
    """Encode BGR image to base64 PNG string."""
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")

def base64_to_mask(b64_str: str) -> np.ndarray:
    """Decode base64 string to a grayscale numpy array mask [H, W]."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if mask is None and nparr.size == 256 * 256:
        mask = nparr.reshape(256, 256)
    return mask if mask is not None else np.zeros((256, 256), dtype=np.uint8)

def encode_image_bgr(path: str) -> str:
    """Read image and encode to BGR base64 (maintaining consistency with Change3D preprocessing)."""
    try:
        from skimage import io as skio
        rgb = skio.imread(path)
    except (ImportError, ValueError, OSError):
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return img_to_base64(bgr)
        
    if len(rgb.shape) == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return img_to_base64(bgr)

# --- Data Loading Helpers ---
def load_scd_test_data(file_root: str, split: str = "test") -> List[tuple]:
    """
    Load SCD test dataset paths.
    Returns: List of (pre_path, post_path, label1_path, label2_path, change_path, name)
    """
    label1_dir = os.path.join(file_root, split, "label1")
    if not os.path.exists(label1_dir):
        raise FileNotFoundError(f"Label directory not found: {label1_dir}")
        
    file_list = os.listdir(label1_dir)
    samples = []
    for name in sorted(file_list):
        pre_path = os.path.join(file_root, split, "t1", name)
        post_path = os.path.join(file_root, split, "t2", name)
        label1_path = os.path.join(file_root, split, "label1", name)
        label2_path = os.path.join(file_root, split, "label2", name)
        change_path = os.path.join(file_root, split, "change", name)
        
        if all(os.path.exists(p) for p in [pre_path, post_path, label1_path, label2_path, change_path]):
            samples.append((pre_path, post_path, label1_path, label2_path, change_path, name))
    return samples

def build_user_prompt(pre_path: Path, post_path: Path) -> str:
    return (
        f"Pre image path: {pre_path}\n"
        f"Post image path: {post_path}\n\n"
        f"User request: Given this pair of images, output the change category mask for each pixel.\n\n"
    )

def run_scd(
    pre_path: Path,
    post_path: Path,
    api_url: str,
    threshold: float = 0.5,
    num_class: int = 6,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Call Change3D SCD API, returns (pre_mask, post_mask, change_mask)."""
    if not api_url:
        return None
    try:
        payload = {
            "task": "scd",
            "pre_image": encode_image_bgr(str(pre_path)),
            "post_image": encode_image_bgr(str(post_path)),
            "num_class": num_class,
            "threshold": threshold,
        }
        data = http_post_predict(api_url, payload, timeout=180.0)
        if data.get("status") != "success":
            return None
            
        pre_b64 = data.get("pre_mask")
        post_b64 = data.get("post_mask")
        change_b64 = data.get("change_mask")
        
        if not pre_b64 or not post_b64:
            return None
            
        pre_mask = base64_to_mask(pre_b64)
        post_mask = base64_to_mask(post_b64)
        change_mask = base64_to_mask(change_b64) if change_b64 else np.ones_like(pre_mask)
        return (pre_mask, post_mask, change_mask)
    except Exception as e:
        print(f"  [ERROR] SCD API call failed: {e}")
    return None

# --- Metrics Dependency ---
_change3d_root = REPO_ROOT / "change3d"
if not _change3d_root.exists():
    sys.exit(
        f"[ERROR] change3d directory not found: {_change3d_root}\n"
        "Clone or symlink the Change3D codebase at the repository root as ./change3d, or set PYTHONPATH."
    )
sys.path.insert(0, str(_change3d_root))

try:
    from model.utils import SCDD_eval_all as scdd_eval_all
except ImportError as e:
    sys.exit(f"[ERROR] Failed to import SCDD_eval_all from change3d: {e}")

# --- Main Evaluation Pipeline ---
def main():
    parser = argparse.ArgumentParser(description="HRSCD Semantic Change Detection Evaluation (RemoteAgent Tool Selection)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--file_root",
        "--dataset_root",
        dest="file_root",
        type=str,
        default=str(
            (REPO_ROOT / "change3d" / "datasets" / "HRSCD")
            if (REPO_ROOT / "change3d" / "datasets" / "HRSCD").exists()
            else (SCRIPT_DIR / "HRSCD")
        ),
        help="Root directory of the HRSCD dataset",
    )
    parser.add_argument("--change3d_url", type=str, default="http://localhost:6658")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000")
    parser.add_argument("--vllm_model_name", type=str, default=None)
    parser.add_argument("--system_prompt_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(SCRIPT_DIR / "results_hrscd_scd_remoteagent_choose.json"))
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold for change detection")
    parser.add_argument("--num_classes", "--num_class", dest="num_classes", type=int, default=6, help="Number of semantic classes")
    args = parser.parse_args()

    samples = load_scd_test_data(args.file_root, args.split)
    if not samples:
        sys.exit(f"[ERROR] No test samples found in: {args.file_root}/{args.split}")
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    def _url(v: str, key: str) -> Optional[str]:
        return (v or "").strip() or os.environ.get(key) or None

    change3d_url = _url(args.change3d_url, "CHANGE3D_API_URL") or "http://localhost:6658"
    api_urls = {"change3d": change3d_url}

    model_name = (
        get_vllm_model_id(args.vllm_url) or args.vllm_model_name or "RemoteAgent-7B-merged-6000+9000"
    )
    system_prompt = load_eval_system_prompt(
        SCRIPT_DIR, Path(args.system_prompt_path) if args.system_prompt_path else None
    )

    print("=" * 56)
    print("HRSCD SCD Evaluation (RemoteAgent)")
    print("=" * 56)
    print(f"  vLLM URL: {args.vllm_url} | Model: {model_name}")
    print(f"  Change3D API (6658): {change3d_url}")
    print(f"  Split: {args.split} | Samples: {len(samples)} | Num Classes: {args.num_classes}")
    print("-" * 56)

    tool_choice_counts: Dict[str, int] = {}
    scd_ok_count = 0
    preds_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    sample_results: List[Dict] = []
    sample_count = 0

    try:
        from skimage import io as skio
    except ImportError:
        skio = None

    for i, (pre_path_str, post_path_str, label1_path, label2_path, change_path, name) in enumerate(samples):
        pre_path = Path(pre_path_str)
        post_path = Path(post_path_str)

        user_prompt = build_user_prompt(pre_path, post_path)
        raw = run_vllm_chat(
            system_prompt,
            user_prompt,
            args.vllm_url,
            model_name,
            image_paths=[pre_path, post_path],
        )

        answer_content = extract_answer_tag(raw)
        parsed = None
        pred_pre = None
        pred_post = None
        pred_change = None
        scd_success = False

        if answer_content is not None:
            tool_choice_counts["(T_in)"] = tool_choice_counts.get("(T_in)", 0) + 1
        else:
            parsed = parse_tool_call_with_bcd_fallback(raw, pre_path, post_path)
            if parsed:
                tool_name, tool_args = parsed
                tool_choice_counts[tool_name] = tool_choice_counts.get(tool_name, 0) + 1
                
                if tool_name == "change3d_scd":
                    scd_ok_count += 1
                    pre_p = Path((tool_args.get("pre_image_path") or str(pre_path)).strip().strip("'").strip('"'))
                    post_p = Path((tool_args.get("post_image_path") or str(post_path)).strip().strip('"').strip("'"))
                    
                    if not pre_p.exists(): pre_p = pre_path
                    if not post_p.exists(): post_p = post_path
                        
                    result = run_scd(pre_p, post_p, change3d_url, threshold=args.threshold, num_class=args.num_classes)
                    if result:
                        pred_pre, pred_post, pred_change = result
                        scd_success = True
                else:
                    execute_tool(tool_name, tool_args, api_urls)
            else:
                tool_choice_counts["(parse_fail)"] = tool_choice_counts.get("(parse_fail)", 0) + 1

        # Ground Truth Loading
        def _load_gray(p) -> np.ndarray:
            if skio is not None:
                try:
                    return skio.imread(str(p), as_gray=True).astype(np.uint8)
                except (ValueError, OSError):
                    pass
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            return img if img is not None else np.zeros((256, 256), dtype=np.uint8)

        pre_label = _load_gray(label1_path)
        post_label = _load_gray(label2_path)
        label_change = _load_gray(change_path)

        if pre_label.max() > 5 or post_label.max() > 5:
            pre_label = np.clip(np.round(pre_label.astype(np.float32) / 255.0 * 5).astype(np.int32), 0, 5).astype(np.uint8)
            post_label = np.clip(np.round(post_label.astype(np.float32) / 255.0 * 5).astype(np.int32), 0, 5).astype(np.uint8)
        if label_change.max() > 1:
            label_change = (label_change > 0).astype(np.uint8)

        # Skip evaluation iteration if prediction failed
        if pred_pre is None or pred_post is None or pred_change is None:
            sample_results.append({
                "file_name": name,
                "parsed_tool": parsed[0] if parsed else None,
                "is_t_in": answer_content is not None,
                "scd_ok": scd_success,
            })
            
            if (i + 1) % 20 == 0 or (i + 1) == len(samples):
                if preds_all:
                    F1, mIoU, Sek, OA = scdd_eval_all(preds_all, labels_all, args.num_classes)
                    print(f"🔄 Progress: {i + 1}/{len(samples)} | SCD_OK: {scd_ok_count} | F1: {F1*100:.2f}% | mIoU: {mIoU*100:.2f}% | OA: {OA*100:.2f}% | Sek: {Sek*100:.2f}% | Tools: {dict(sorted(tool_choice_counts.items()))}")
                else:
                    print(f"🔄 Progress: {i + 1}/{len(samples)} | SCD_OK: {scd_ok_count} | Tools: {dict(sorted(tool_choice_counts.items()))}")
            continue

        # Resize and align GT masks with predictions
        h_pred, w_pred = pred_pre.shape[:2]
        if pre_label.shape[:2] != (h_pred, w_pred):
            pre_label = cv2.resize(pre_label.astype(np.uint8), (w_pred, h_pred), interpolation=cv2.INTER_NEAREST)
            post_label = cv2.resize(post_label.astype(np.uint8), (w_pred, h_pred), interpolation=cv2.INTER_NEAREST)
            label_change = cv2.resize(label_change.astype(np.uint8), (w_pred, h_pred), interpolation=cv2.INTER_NEAREST)

        change_mask_bool = (pred_change > 0).astype(np.uint8)
        label_change_bool = (label_change > 0).astype(np.uint8)

        pred_pre_masked = np.clip(pred_pre.astype(np.int64), 0, args.num_classes - 1) * change_mask_bool
        pred_post_masked = np.clip(pred_post.astype(np.int64), 0, args.num_classes - 1) * change_mask_bool
        gt_pre = pre_label.astype(np.int64) * label_change_bool
        gt_post = post_label.astype(np.int64) * label_change_bool

        preds_all.extend([pred_pre_masked, pred_post_masked])
        labels_all.extend([gt_pre, gt_post])
        sample_count += 1

        sample_results.append({
            "file_name": name,
            "parsed_tool": parsed[0] if parsed else None,
            "is_t_in": answer_content is not None,
            "scd_ok": scd_success,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(samples):
            if preds_all:
                F1, mIoU, Sek, OA = scdd_eval_all(preds_all, labels_all, args.num_classes)
                print(f"🔄 Progress: {i + 1}/{len(samples)} | SCD_OK: {scd_ok_count} | F1: {F1*100:.2f}% | mIoU: {mIoU*100:.2f}% | OA: {OA*100:.2f}% | Sek: {Sek*100:.2f}% | Tools: {dict(sorted(tool_choice_counts.items()))}")
            else:
                print(f"🔄 Progress: {i + 1}/{len(samples)} | SCD_OK: {scd_ok_count} | Tools: {dict(sorted(tool_choice_counts.items()))}")

    n = len(samples)
    scd_acc = scd_ok_count / n * 100 if n > 0 else 0

    F1, mIoU, Sek, OA = 0.0, 0.0, 0.0, 0.0
    if preds_all and labels_all:
        F1, mIoU, Sek, OA = scdd_eval_all(preds_all, labels_all, args.num_classes)

    print("\n" + "=" * 56)
    print("HRSCD SCD Evaluation Report (RemoteAgent)")
    print("=" * 56)
    print("Tool Selection Distribution:", dict(sorted(tool_choice_counts.items())))
    print(f"SCD Tool Accuracy: {scd_ok_count}/{n} ({scd_acc:.2f}%)")
    print(f"Evaluated Samples: {sample_count} (successful predictions only)")
    print(f"  F1   = {F1*100:.2f}%")
    print(f"  mIoU = {mIoU*100:.2f}%")
    print(f"  OA   = {OA*100:.2f}%")
    print(f"  Sek  = {Sek*100:.2f}%")
    print("=" * 56)

    metrics = {
        "scd_tool_accuracy_pct": round(scd_acc, 2),
        "F1": round(F1, 4),
        "mIoU": round(mIoU, 4),
        "OA": round(OA, 4),
        "Sek": round(Sek, 4),
        "tool_choice_counts": dict(sorted(tool_choice_counts.items())),
    }
    
    out = {"metrics": metrics, "samples": sample_results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        
    print(f"Results successfully saved to: {args.output}")

if __name__ == "__main__":
    main()