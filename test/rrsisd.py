import argparse
import base64
import json
import os
import pickle
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
SCRIPT_DIR = EVAL_DIR
RRSISD_ROOT = SCRIPT_DIR / "RRSISD_Test"
IMAGE_DIR = RRSISD_ROOT / "origin"
REFS_PICKLE = RRSISD_ROOT / "refs(unc).p"
INSTANCES_JSON = RRSISD_ROOT / "instances.json"

from remoteagent.eval_common import (
    get_vllm_model_id,
    http_post_predict,
    load_eval_system_prompt,
    run_vllm_chat,
)
from remoteagent.parsing import parse_tool_call
from remoteagent.services import execute_tool
from remoteagent.utils import encode_image, extract_answer_tag

# --- Data Loading Helpers ---
def load_referring_data(split: str = "test") -> List[Dict]:
    if not REFS_PICKLE.exists():
        raise FileNotFoundError(f"Reference annotations not found: {REFS_PICKLE}")
        
    with open(REFS_PICKLE, "rb") as f:
        all_refs = pickle.load(f)
        
    refs = [r for r in all_refs if r.get("split") == split]
    out = []
    for r in refs:
        expr = ""
        if r.get("sentences"):
            expr = r["sentences"][0].get("raw") or r["sentences"][0].get("sent", "")
        out.append({
            "file_name": r.get("file_name", ""),
            "expression": expr,
            "ann_id": r.get("ann_id"),
            "ref_id": r.get("ref_id"),
            "image_id": r.get("image_id"),
        })
    return out

def load_instances() -> Tuple[Dict, Dict[int, Dict], Dict[int, Tuple[int, int]]]:
    if not INSTANCES_JSON.exists():
        raise FileNotFoundError(f"Instances file not found: {INSTANCES_JSON}")
        
    with open(INSTANCES_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)
        
    ann_by_id = {a["id"]: a for a in coco.get("annotations", [])}
    image_size_by_id = {img["id"]: (img.get("width", 0), img.get("height", 0)) for img in coco.get("images", [])}
    
    return coco, ann_by_id, image_size_by_id

def get_gt_mask_from_annotation(ann: Dict) -> Optional[np.ndarray]:
    seg = ann.get("segmentation")
    if not seg:
        return None
    try:
        from pycocotools import mask as mask_util
        rle = seg[0] if isinstance(seg, list) and len(seg) > 0 else seg
        if isinstance(rle, dict) and "counts" in rle:
            mask = mask_util.decode(rle)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            return (mask > 0).astype(np.uint8)
    except Exception as e:
        print(f"  [ERROR] Failed to decode GT mask: {e}")
    return None

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    
    if pred_bin.shape != gt_bin.shape:
        h, w = gt_bin.shape[:2]
        from PIL import Image
        pred_pil = Image.fromarray(pred_bin)
        pred_pil = pred_pil.resize((w, h), Image.NEAREST)
        pred_bin = np.array(pred_pil)
        
    inter = np.logical_and(pred_bin > 0, gt_bin > 0).sum()
    union = np.logical_or(pred_bin > 0, gt_bin > 0).sum()
    return float(inter), float(union)

def run_remote_sam_ref_seg(
    image_path: Path, text: str, api_url: str
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if not api_url or not text.strip():
        return None, None
    try:
        payload = {
            "task": "referring_seg",
            "image": encode_image(str(image_path)),
            "text": text.strip(),
            "classes": [],
        }
        t0 = time.perf_counter()
        data = http_post_predict(api_url, payload)
        elapsed_sec = time.perf_counter() - t0
        if data.get("status") != "success":
            return None, elapsed_sec
            
        mask_b64 = data.get("mask")
        if not mask_b64:
            return None, elapsed_sec
            
        raw = base64.b64decode(mask_b64)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, elapsed_sec
            
        return (img > 0).astype(np.uint8) * 255, elapsed_sec
    except Exception as e:
        print(f"  [ERROR] RemoteSAM referring_seg failed: {e}")
        return None, None

def build_user_prompt(image_path: Path, expression: str) -> str:
    return (
        f"Image absolute path: {image_path}\n\n"
        f"User request: Segment the region described by the following referring expression : {expression}\n\n"
    )

# --- Main Evaluation Pipeline ---
def main():
    parser = argparse.ArgumentParser(description="RRSISD Referring Expression Segmentation Evaluation (prompt.txt)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--remote_sam_url", type=str, default="http://localhost:6657/predict")
    parser.add_argument("--change3d_url", type=str, default="http://localhost:6658")
    parser.add_argument("--sm3det_url", type=str, default="http://localhost:6655")
    parser.add_argument("--crossearth_url", type=str, default="http://localhost:6656")
    parser.add_argument("--skysense_det_url", type=str, default="http://localhost:6654")
    parser.add_argument("--directsam_url", type=str, default="http://localhost:6659")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000")
    parser.add_argument("--vllm_model_name", type=str, default=None)
    parser.add_argument("--system_prompt_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(SCRIPT_DIR / "results_rrsisd_prompt.json"))
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--debug_raw", action="store_true", help="Print raw output to stdout on parse failure")
    args = parser.parse_args()

    if not IMAGE_DIR.exists():
        sys.exit(f"[ERROR] Image directory not found: {IMAGE_DIR}")
    if not REFS_PICKLE.exists():
        sys.exit(f"[ERROR] References file not found: {REFS_PICKLE}")

    refs = load_referring_data(split=args.split)
    coco, ann_by_id, image_size_by_id = load_instances()
    if args.max_samples is not None:
        refs = refs[: args.max_samples]

    model_name = (
        args.vllm_model_name or get_vllm_model_id(args.vllm_url) or "RemoteAgent-7B-merged-6000+9000"
    )
    system_prompt = load_eval_system_prompt(
        SCRIPT_DIR, Path(args.system_prompt_path) if args.system_prompt_path else None
    )

    def _url(arg_val: Optional[str], env_key: str) -> Optional[str]:
        v = (arg_val or "").strip() or os.environ.get(env_key)
        return (v or "").strip() or None

    api_urls = {
        "remotesam": _url(args.remote_sam_url, "REMOTE_API_URL"),
        "change3d": _url(args.change3d_url, "CHANGE3D_API_URL"),
        "sm3det": _url(args.sm3det_url, "SM3DET_API_URL"),
        "crossearth": _url(args.crossearth_url, "CROSSEARTH_API_URL"),
        "skysense_det": _url(args.skysense_det_url, "SKYSENSE_DET_API_URL"),
        "directsam": _url(args.directsam_url, "DIRECTSAM_API_URL"),
    }

    print("=" * 56)
    print("RRSISD Referring Expression Segmentation Evaluation")
    print("=" * 56)
    print(f"  vLLM URL: {args.vllm_url} | Model: {model_name}")
    print(f"  RemoteSAM (6657): {api_urls['remotesam'] or 'Not Configured'}")
    print(f"  Split: {args.split} | Samples: {len(refs)}")
    
    if not api_urls.get("remotesam"):
        print("  [WARNING] RemoteSAM is not configured. IoU will default to 0 for 'referring_expression_segmentation'.")
    if not all(api_urls.get(k) for k in ("change3d", "sm3det", "crossearth", "skysense_det", "directsam")):
        print("  [WARNING] Some backend APIs are not configured. Execution may fail if the model selects them.")
    print("-" * 56)

    total_intersection = 0.0
    total_union = 0.0
    ious: List[float] = []
    tool_ok_count = 0
    tool_choice_counts: Dict[str, int] = {}
    remoteagent_latencies_sec: List[float] = []
    remotesam_latencies_sec: List[float] = []
    sample_results: List[Dict] = []
    debug_failures: List[Dict] = []

    for i, ref in enumerate(refs):
        file_name = ref.get("file_name") or ""
        expression = ref.get("expression") or ""
        ann_id = ref.get("ann_id")

        if not file_name or not expression or ann_id is None:
            print(f"  [WARNING] Skipping (missing fields): ref_id={ref.get('ref_id')}")
            continue

        ann = ann_by_id.get(ann_id)
        if not ann:
            print(f"  [WARNING] Skipping (ann_id not in instances): {ann_id}")
            continue

        gt_mask = get_gt_mask_from_annotation(ann)
        if gt_mask is None:
            print(f"  [WARNING] Skipping (failed to decode GT mask): ann_id={ann_id}")
            continue

        image_path = IMAGE_DIR / file_name
        if not image_path.exists():
            print(f"  [WARNING] Skipping (image not found): {file_name}")
            continue

        user_prompt = build_user_prompt(image_path, expression)
        t_agent_start = time.perf_counter()
        raw = run_vllm_chat(
            system_prompt, user_prompt, args.vllm_url, model_name, image_path=image_path
        )
        remoteagent_elapsed_sec = time.perf_counter() - t_agent_start
        remoteagent_latencies_sec.append(remoteagent_elapsed_sec)
        
        tool_ok = False
        use_path, use_text = image_path, expression
        execution_result: Optional[str] = None
        remotesam_elapsed_sec: Optional[float] = None
        answer_content = extract_answer_tag(raw)
        
        if answer_content is not None:
            tool_choice_counts["(T_in)"] = tool_choice_counts.get("(T_in)", 0) + 1
            parsed = None
        else:
            parsed = parse_tool_call(raw)
            if parsed:
                tool_name, tool_args = parsed
                tool_choice_counts[tool_name] = tool_choice_counts.get(tool_name, 0) + 1
                
                if "image_path" in tool_args and tool_args.get("image_path"):
                    p = Path((tool_args["image_path"] or "").strip().strip("'").strip('"'))
                    if not p.exists():
                        tool_args["image_path"] = str(image_path)
                        
                if tool_name == "referring_expression_segmentation":
                    p_path = (tool_args.get("image_path") or "").strip().strip("'").strip('"')
                    p_prompt = (tool_args.get("prompt") or "").strip()
                    if p_path and p_prompt:
                        tool_ok = True
                        if Path(p_path).exists():
                            use_path = Path(p_path)
                        use_text = p_prompt
                    else:
                        tool_ok = True
                        use_path = image_path
                        use_text = expression
                else:
                    execution_result = execute_tool(tool_name, tool_args, api_urls)
            else:
                tool_choice_counts["(parse_fail)"] = tool_choice_counts.get("(parse_fail)", 0) + 1
                fail_count = tool_choice_counts["(parse_fail)"]
                
                if fail_count <= 5:
                    debug_failures.append({
                        "index": fail_count,
                        "file_name": file_name,
                        "expression": expression,
                        "raw_output": raw,
                        "raw_len": len(raw),
                    })
                if args.debug_raw and fail_count <= 3:
                    print(f"\n[DEBUG] Parse failure #{fail_count}, raw output:\n---\n{raw[:800]}{'...' if len(raw) > 800 else ''}\n---\n")

        if tool_ok:
            tool_ok_count += 1

        pred_mask = None
        if tool_ok and api_urls.get("remotesam"):
            pred_mask, remotesam_elapsed_sec = run_remote_sam_ref_seg(
                use_path, use_text, api_urls["remotesam"]
            )
            if remotesam_elapsed_sec is not None:
                remotesam_latencies_sec.append(remotesam_elapsed_sec)
            
        if pred_mask is None:
            pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)

        inter, uni = compute_iou(pred_mask, gt_mask)
        iou = inter / uni if uni > 0 else 0.0
        ious.append(iou)
        total_intersection += inter
        total_union += uni

        sample_results.append({
            "file_name": file_name,
            "expression": expression,
            "ann_id": ann_id,
            "tool_ok": tool_ok,
            "parsed_tool": parsed[0] if parsed else None,
            "is_t_in": answer_content is not None,
            "iou": iou,
            "remoteagent_inference_time_ms": round(remoteagent_elapsed_sec * 1000.0, 2),
            "remotesam_inference_time_ms": (
                round(remotesam_elapsed_sec * 1000.0, 2) if remotesam_elapsed_sec is not None else None
            ),
        })
        
        if execution_result is not None:
            sample_results[-1]["execution_result"] = execution_result[:500]
        if args.save_raw:
            sample_results[-1]["raw"] = (raw[:500] + "…") if len(raw) > 500 else raw

        if (i + 1) % 20 == 0 or (i + 1) == len(refs):
            print(f"🔄 Progress: {i + 1}/{len(refs)} | Target Tool Used: {tool_ok_count}/{i+1} | Distribution: {dict(sorted(tool_choice_counts.items()))}")

    if len(ious) == 0:
        print("\n[ERROR] No valid evaluation results generated.")
        return

    n = len(ious)
    mIoU = float(np.mean(ious) * 100)
    oIoU = float((total_intersection / total_union) * 100) if total_union > 0 else 0.0
    tool_acc = tool_ok_count / n * 100

    def _pr_at(thr: float) -> float:
        return float(sum(1 for v in ious if v >= thr) / n * 100)

    pr_50, pr_60, pr_70, pr_80, pr_90 = _pr_at(0.5), _pr_at(0.6), _pr_at(0.7), _pr_at(0.8), _pr_at(0.9)
    ra_avg_ms = float(np.mean(remoteagent_latencies_sec) * 1000.0) if remoteagent_latencies_sec else 0.0
    rs_avg_ms = float(np.mean(remotesam_latencies_sec) * 1000.0) if remotesam_latencies_sec else 0.0

    print("\n" + "=" * 56)
    print("RRSISD Referring Expression Segmentation Report")
    print("=" * 56)
    print("Tool Selection Distribution:", dict(sorted(tool_choice_counts.items())))
    print(f"Target Tool Selection Accuracy: {tool_ok_count}/{n} ({tool_acc:.2f}%)")
    print("-" * 56)
    print(f"  oIoU (%):  {oIoU:.2f}")
    print(f"  mIoU (%):  {mIoU:.2f}")
    print("-" * 56)
    print(f"  RemoteAgent Inference Time avg (ms): {ra_avg_ms:.2f}")
    print(f"  RemoteSAM Response Time avg (ms): {rs_avg_ms:.2f}")
    print("-" * 56)
    print(f"  Pr@0.5: {pr_50:.2f} | Pr@0.6: {pr_60:.2f} | Pr@0.7: {pr_70:.2f} | Pr@0.8: {pr_80:.2f} | Pr@0.9: {pr_90:.2f}")
    print("=" * 56)

    out = {
        "metrics": {
            "tool_selection_accuracy_pct": round(tool_acc, 2),
            "tool_choice_counts": dict(sorted(tool_choice_counts.items())),
            "oIoU": oIoU,
            "mIoU": mIoU,
            "Pr@0.5": pr_50,
            "Pr@0.6": pr_60,
            "Pr@0.7": pr_70,
            "Pr@0.8": pr_80,
            "Pr@0.9": pr_90,
            "remoteagent_inference_time_avg_ms": round(ra_avg_ms, 2),
            "remotesam_response_time_avg_ms": round(rs_avg_ms, 2),
        },
        "samples": sample_results,
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults successfully saved to: {args.output}")

    if debug_failures:
        debug_path = SCRIPT_DIR / "debug_parse_failures.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_failures, f, indent=2, ensure_ascii=False)
        print(f"Parse failure logs (top {len(debug_failures)}) saved to: {debug_path}")

if __name__ == "__main__":
    main()