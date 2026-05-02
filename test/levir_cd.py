import argparse
import base64
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def build_user_prompt(pre_path: Path, post_path: Path) -> str:
    return (
        f"Pre image path: {pre_path}\n"
        f"Post image path: {post_path}\n\n"
        f"User request: These two images show the same area at different times. Find where the scene has changed and give me pixel-level localization.\n\n"
    )

def run_bcd(pre_path: Path, post_path: Path, api_url: str, threshold: float = 0.5) -> Optional[np.ndarray]:
    """Call Change3D BCD API and return the binary mask [H,W] (0/255)."""
    if not api_url:
        return None
    try:
        payload = {
            "task": "bcd",
            "pre_image": encode_image(str(pre_path)),
            "post_image": encode_image(str(post_path)),
            "threshold": threshold,
        }
        data = http_post_predict(api_url, payload, timeout=180.0)
        if data.get("status") == "success" and "change_mask" in data:
            b64 = data["change_mask"]
            if "," in b64:
                b64 = b64.split(",")[1]
            img_data = base64.b64decode(b64)
            nparr = np.frombuffer(img_data, np.uint8)
            mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if mask is None and nparr.size == 256 * 256:
                mask = nparr.reshape(256, 256)
            return mask
    except Exception as e:
        print(f"  [ERROR] BCD API call failed: {e}")
    return None

# --- Image Processing & Metrics Helpers ---
def load_gt_mask(path: Path, target_shape: Optional[tuple] = None) -> np.ndarray:
    """Load ground truth binary mask and optionally resize to target shape."""
    try:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((256, 256), dtype=np.uint8)
        mask = (mask > 127).astype(np.uint8) * 255
        if target_shape and mask.shape != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask
    except Exception:
        return np.zeros((256, 256), dtype=np.uint8)

def raw_output_to_mask(raw: str, shape: tuple) -> Optional[np.ndarray]:
    """Parse model raw output string to a binary mask using coordinate boxes."""
    content = raw.strip()
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL | re.IGNORECASE)
    if m:
        content = m.group(1).strip()
    box_m = re.findall(r"\[\s*([\d.\s,]+)\s*\]", content)
    if not box_m:
        return None
        
    try:
        h, w = shape[:2] if len(shape) >= 2 else (256, 256)
        mask = np.zeros((h, w), dtype=np.uint8)
        for part in box_m:
            nums = re.findall(r"[\d.]+", part)
            if len(nums) >= 4:
                coords = [int(float(x)) for x in nums[:4]]
                x1 = max(0, min(coords[0], coords[2], w - 1))
                x2 = max(0, min(max(coords[0], coords[2]), w))
                y1 = max(0, min(coords[1], coords[3], h - 1))
                y2 = max(0, min(max(coords[1], coords[3]), h))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 255
        return mask
    except Exception:
        return None

def get_confuse_matrix(num_classes: int, label_gts: List[np.ndarray], label_preds: List[np.ndarray]) -> np.ndarray:
    """Calculate the confusion matrix for semantic segmentation / change detection."""
    def _fast_hist(lt: np.ndarray, lp: np.ndarray) -> np.ndarray:
        mask = (lt >= 0) & (lt < num_classes)
        hist = np.bincount(
            num_classes * lt[mask].astype(int) + lp[mask],
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes)
        return hist

    cm = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        cm += _fast_hist(lt.flatten(), lp.flatten())
    return cm

def cm2score(cm: np.ndarray) -> Dict[str, float]:
    """Calculate F1, IoU, OA, Kappa, Recall, and Precision from a confusion matrix."""
    tp, fn = cm[1, 1], cm[1, 0]
    fp, tn = cm[0, 1], cm[0, 0]
    eps = np.finfo(np.float32).eps
    
    oa = (tp + tn) / (tp + fn + fp + tn + eps)
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * recall * precision / (recall + precision + eps)
    iou = tp / (tp + fp + fn + eps)
    pre = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn) ** 2
    kappa = (oa - pre) / (1 - pre + eps)
    
    return {"F1": f1, "IoU": iou, "OA": oa, "Kappa": kappa, "recall": recall, "precision": precision}

# --- Main Evaluation Pipeline ---
def main():
    parser = argparse.ArgumentParser(description="LEVIR-CD Binary Change Detection Evaluation (RemoteAgent Tool Selection)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset_root", type=str, default=str(SCRIPT_DIR / "LEVIR-CD"))
    parser.add_argument("--change3d_url", type=str, default="http://localhost:6658")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000")
    parser.add_argument("--vllm_model_name", type=str, default=None)
    parser.add_argument("--system_prompt_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(SCRIPT_DIR / "results_levir_cd_remoteagent_choose.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    t1_dir = dataset_root / args.split / "t1"
    t2_dir = dataset_root / args.split / "t2"
    label_dir = dataset_root / args.split / "label"

    if not t1_dir.exists() or not t2_dir.exists():
        sys.exit(f"[ERROR] LEVIR-CD {args.split} directories not found: {t1_dir}, {t2_dir}")
    if not label_dir.exists():
        sys.exit(f"[ERROR] Label directory not found: {label_dir}")

    images = sorted([f.name for f in t1_dir.glob("*.png") if (t2_dir / f.name).exists() and (label_dir / f.name).exists()])
    if args.max_samples is not None:
        images = images[: args.max_samples]

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
    print("LEVIR-CD BCD Evaluation (RemoteAgent)")
    print("=" * 56)
    print(f"  vLLM URL: {args.vllm_url} | Model: {model_name}")
    print(f"  Change3D API (6658): {change3d_url}")
    print(f"  Split: {args.split} | Samples: {len(images)}")
    print("-" * 56)

    tool_choice_counts: Dict[str, int] = {}
    bcd_ok_count = 0
    preds_binary: List[np.ndarray] = []
    gts_binary: List[np.ndarray] = []
    sample_results: List[Dict] = []

    for i, name in enumerate(images):
        pre_path = t1_dir / name
        post_path = t2_dir / name
        gt_path = label_dir / name
        
        if not pre_path.exists() or not post_path.exists() or not gt_path.exists():
            continue

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
        pred_mask = None

        if answer_content is not None:
            tool_choice_counts["(T_in)"] = tool_choice_counts.get("(T_in)", 0) + 1
        else:
            parsed = parse_tool_call_with_bcd_fallback(raw, pre_path, post_path)
            if parsed:
                tool_name, tool_args = parsed
                tool_choice_counts[tool_name] = tool_choice_counts.get(tool_name, 0) + 1
                
                if tool_name == "change3d_bcd":
                    bcd_ok_count += 1
                    pre_p = Path((tool_args.get("pre_image_path") or str(pre_path)).strip().strip("'").strip('"'))
                    post_p = Path((tool_args.get("post_image_path") or str(post_path)).strip().strip('"').strip("'"))
                    
                    if not pre_p.exists(): pre_p = pre_path
                    if not post_p.exists(): post_p = post_path
                        
                    pred_mask = run_bcd(pre_p, post_p, change3d_url, threshold=args.threshold)
                else:
                    execute_tool(tool_name, tool_args, api_urls)
            else:
                tool_choice_counts["(parse_fail)"] = tool_choice_counts.get("(parse_fail)", 0) + 1

        gt_mask = load_gt_mask(gt_path)
        if pred_mask is None:
            pred_mask = raw_output_to_mask(raw, gt_mask.shape)

        if pred_mask is not None and gt_mask.size > 0:
            h_pred, w_pred = pred_mask.shape[:2]
            gt_resized = cv2.resize(gt_mask.astype(np.uint8), (w_pred, h_pred), interpolation=cv2.INTER_NEAREST)
            pred_binary = (pred_mask > 127).astype(np.uint8)
            gt_binary = (gt_resized > 127).astype(np.uint8)
            
            preds_binary.append(pred_binary)
            gts_binary.append(gt_binary)

        sample_results.append({
            "file_name": name,
            "parsed_tool": parsed[0] if parsed else None,
            "is_t_in": answer_content is not None,
            "bcd_ok": pred_mask is not None,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(images):
            if preds_binary:
                cm = get_confuse_matrix(2, gts_binary, preds_binary)
                s = cm2score(cm)
                print(f"🔄 Progress: {i + 1}/{len(images)} | BCD_OK: {bcd_ok_count} | mIoU: {s['IoU']*100:.2f}% | mF1: {s['F1']*100:.2f}% | OA: {s['OA']*100:.2f}% | Tools: {dict(sorted(tool_choice_counts.items()))}")
            else:
                print(f"🔄 Progress: {i + 1}/{len(images)} | BCD_OK: {bcd_ok_count} | Tools: {dict(sorted(tool_choice_counts.items()))}")

    n = len(images)
    bcd_acc = bcd_ok_count / n * 100 if n > 0 else 0

    scores = {"F1": 0.0, "IoU": 0.0, "OA": 0.0}
    if preds_binary:
        cm = get_confuse_matrix(2, gts_binary, preds_binary)
        scores = cm2score(cm)

    print("\n" + "=" * 56)
    print("LEVIR-CD BCD Evaluation Report (RemoteAgent)")
    print("=" * 56)
    print("Tool Selection Distribution:", dict(sorted(tool_choice_counts.items())))
    print(f"BCD Tool Accuracy: {bcd_ok_count}/{n} ({bcd_acc:.2f}%)")
    print(f"  F1  = {scores['F1']:.4f}")
    print(f"  IoU = {scores['IoU']:.4f}")
    print(f"  OA  = {scores['OA']:.4f}")
    print("=" * 56)

    metrics = {
        "bcd_tool_accuracy_pct": round(bcd_acc, 2),
        "F1": round(scores["F1"], 4),
        "IoU": round(scores["IoU"], 4),
        "OA": round(scores["OA"], 4),
        "tool_choice_counts": dict(sorted(tool_choice_counts.items())),
    }
    
    out = {"metrics": metrics, "samples": sample_results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        
    print(f"Results successfully saved to: {args.output}")

if __name__ == "__main__":
    main()