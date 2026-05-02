import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
SCRIPT_DIR = EVAL_DIR

from remoteagent.parsing import parse_tool_call_with_image_fallback
from remoteagent.services import execute_tool
from remoteagent.utils import encode_image, extract_answer_tag

from remoteagent.eval_common import (
    get_vllm_model_id,
    http_post_predict,
    load_eval_system_prompt,
    run_vllm_chat,
)


def run_skysense_detection_raw(
    image_path: Path, classes: List[str], api_url: str, score_thr: float = 0.05
) -> Optional[Dict[str, List]]:
    if not api_url:
        return None
    try:
        payload = {
            "task": "detection",
            "image": encode_image(str(image_path)),
            "score_thr": score_thr,
            "classes": classes if classes else None,
            "return_image": False,
        }
        data = http_post_predict(api_url, payload)
        if data.get("status") == "success" and "detections" in data:
            return data["detections"]
    except Exception as e:
        print(f"  [ERROR] SkySense Detection API call failed: {e}")
    return None

def _detections_to_outputs(
    detections: Optional[Dict], class_names: List[str], num_cats: Optional[int] = None
) -> List[np.ndarray]:
    if not detections:
        n = num_cats if num_cats is not None else len(class_names)
        return [np.zeros((0, 5), dtype=np.float32) for _ in range(n)]
        
    per_img = []
    for cls_name in class_names:
        boxes = detections.get(cls_name, [])
        if boxes:
            arr = np.array([[b[0], b[1], b[2], b[3], b[4]] for b in boxes], dtype=np.float32)
        else:
            arr = np.zeros((0, 5), dtype=np.float32)
        per_img.append(arr)
        
    if num_cats is not None:
        if len(per_img) > num_cats:
            per_img = per_img[:num_cats]
        elif len(per_img) < num_cats:
            per_img = per_img + [np.zeros((0, 5), dtype=np.float32)] * (num_cats - len(per_img))
            
    return per_img

def build_user_prompt(image_path: Path, class_names: List[str]) -> str:
    return (
        f"Image path: {image_path}\n\n"
        f"User request: Please detect objects of the following classes and their bounding boxes: {', '.join(class_names)}\n\n"
    )

# --- Main Evaluation Pipeline ---
def main():
    parser = argparse.ArgumentParser(description="DIOR Object Detection Evaluation (RemoteAgent Tool Selection)")
    parser.add_argument("--max_samples", "--max", type=int, default=None, dest="max_samples")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset_root", "--data-root", dest="data_root", type=str, default=str(SCRIPT_DIR / "DIOR"))
    parser.add_argument("--remote_sam_url", type=str, default="http://localhost:6657/predict")
    parser.add_argument("--change3d_url", type=str, default="http://localhost:6658")
    parser.add_argument("--sm3det_url", type=str, default="http://localhost:6655")
    parser.add_argument("--crossearth_url", type=str, default="http://localhost:6656")
    parser.add_argument("--skysense_det_url", type=str, default="http://localhost:6654")
    parser.add_argument("--directsam_url", type=str, default="http://localhost:6659")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000")
    parser.add_argument("--vllm_model_name", type=str, default=None)
    parser.add_argument("--system_prompt_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(SCRIPT_DIR / "results_dior_remoteagent_choose.json"))
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--score_thr", type=float, default=0.001)
    parser.add_argument(
        "--mmdet_config",
        type=str,
        default=str(REPO_ROOT / "SkySense" / "detection" / "configs" / "swin_transformer_v2" / "faster_rcnn_swinv2_huge_patch4_window8_fpn-1x_dior.py"),
    )
    args = parser.parse_args()

    dataset_root = Path(args.data_root).resolve()
    coco_labels = dataset_root / "COCO_labels"
    ann_file = str(dataset_root / "COCO_labels" / "test.json")
    
    img_prefix = str(dataset_root / "JPEGImages-test") + "/"
    if not Path(img_prefix.rstrip("/")).exists():
        img_prefix = str(dataset_root / args.split / "images") + "/"
    if not Path(img_prefix.rstrip("/")).exists():
        sys.exit(f"[ERROR] Image directory not found: {img_prefix}")

    if not coco_labels.exists():
        sys.exit(f"[ERROR] COCO_labels directory not found: {coco_labels}")

    with open(ann_file) as f:
        ann = json.load(f)
        
    images = ann["images"]
    if args.max_samples is not None:
        images = images[: args.max_samples]
        img_ids = {i["id"] for i in images}
        subset = {
            "images": images,
            "annotations": [a for a in ann["annotations"] if a["image_id"] in img_ids],
            "categories": ann["categories"],
        }
        subset_path = SCRIPT_DIR / "api_eval_gt_subset_dior.json"
        with open(subset_path, "w") as f:
            json.dump(subset, f)
        ann_file = str(subset_path)
        print(f"Evaluating subset: First {args.max_samples} images")
    else:
        print(f"Evaluating all {len(images)} images")

    sys.path.insert(0, str(SCRIPT_DIR / "detection"))
    from mmcv import Config
    from mmdet.datasets import build_dataset, replace_ImageToTensor
    from mmdet.utils import compat_cfg, replace_cfg_vals, setup_multi_processes, update_data_root

    cfg_paths = [
        SCRIPT_DIR / "detection/configs/swin_transformer_v2/faster_rcnn_swinv2_huge_patch4_window8_fpn-1x_dior.py",
        Path(args.mmdet_config) if getattr(args, "mmdet_config", None) else None,
    ]
    
    cfg_path = next((str(p) for p in cfg_paths if p and Path(p).exists()), 
                    str(SCRIPT_DIR / "detection/configs/swin_transformer_v2/faster_rcnn_swinv2_huge_patch4_window8_fpn-1x_dior.py"))

    cfg = Config.fromfile(cfg_path)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)

    cfg.merge_from_dict({
        "data.test.ann_file": ann_file,
        "data.test.img_prefix": img_prefix,
    })
    cfg.data.test.test_mode = True
    
    if "pretrained" in cfg.model:
        cfg.model.pretrained = None
    elif hasattr(cfg.model, "backbone") and hasattr(cfg.model.backbone, "init_cfg") and cfg.model.backbone.init_cfg:
        cfg.model.backbone.init_cfg = None
    cfg.model.train_cfg = None

    test_dataloader = getattr(cfg.data, "test_dataloader", None) or {}
    if test_dataloader.get("samples_per_gpu", 1) > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test)
    class_names = list(dataset.CLASSES)
    num_cats = len(dataset.cat_ids)
    
    img_prefix_path = Path(cfg.data.test.img_prefix.rstrip("/"))
    img_id_to_path = {img["id"]: img_prefix_path / img["file_name"] for img in images}

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

    model_name = (
        get_vllm_model_id(args.vllm_url) or args.vllm_model_name or "RemoteAgent-7B-merged-6000+9000"
    )
    system_prompt = load_eval_system_prompt(
        SCRIPT_DIR, Path(args.system_prompt_path) if args.system_prompt_path else None
    )

    print("=" * 56)
    print("DIOR Object Detection Evaluation (RemoteAgent)")
    print("=" * 56)
    if num_cats != len(class_names):
        print(f"  [WARNING] num_cats({num_cats}) != len(CLASSES)({len(class_names)}). Outputs aligned.")
    print(f"  vLLM URL: {args.vllm_url} | Model: {model_name}")
    print(f"  SkySense Det API: {api_urls['skysense_det'] or 'Not Configured'}")
    print(f"  Split: {args.split} | Samples: {len(images)} | Score Thr: {args.score_thr}")
    print("-" * 56)

    outputs: List[List[np.ndarray]] = []
    tool_choice_counts: Dict[str, int] = {}
    detection_ok_count = 0
    sample_results: List[Dict] = []

    for i, img in enumerate(images):
        img_id = img["id"]
        file_name = img["file_name"]
        image_path = img_id_to_path.get(img_id)
        
        if image_path is None or not image_path.exists():
            outputs.append([np.zeros((0, 5), dtype=np.float32) for _ in range(num_cats)])
            sample_results.append({"file_name": file_name, "image_id": img_id, "tool_ok": False, "skip": "image_not_found"})
            continue

        detections = None
        tool_ok = False
        parsed = None
        parse_fail = False
        
        user_prompt = build_user_prompt(image_path, class_names)
        raw = run_vllm_chat(
            system_prompt, user_prompt, args.vllm_url, model_name, image_path=image_path
        )
        answer_content = extract_answer_tag(raw)
        
        if answer_content is not None:
            tool_choice_counts["(T_in)"] = tool_choice_counts.get("(T_in)", 0) + 1
        else:
            parsed = parse_tool_call_with_image_fallback(raw, image_path)
            if parsed:
                tool_name, tool_args = parsed
                tool_choice_counts[tool_name] = tool_choice_counts.get(tool_name, 0) + 1
                
                if "image_path" in tool_args and tool_args.get("image_path"):
                    p = Path((tool_args["image_path"] or "").strip().strip("'").strip('"'))
                    if not p.exists():
                        tool_args["image_path"] = str(image_path)
                        
                use_path = Path((tool_args.get("image_path") or str(image_path)).strip().strip("'").strip('"'))
                if not use_path.exists():
                    use_path = image_path
                    
                api_classes = tool_args.get("classes") or class_names
                
                if tool_name == "skysense_detection" and api_urls.get("skysense_det"):
                    tool_ok = True
                    detections = run_skysense_detection_raw(
                        use_path, api_classes, api_urls["skysense_det"], score_thr=args.score_thr
                    )
                else:
                    execute_tool(tool_name, tool_args, api_urls)
            else:
                tool_choice_counts["(parse_fail)"] = tool_choice_counts.get("(parse_fail)", 0) + 1
                parse_fail = True

        if tool_ok:
            detection_ok_count += 1
            
        per_img = _detections_to_outputs(detections, class_names, num_cats)
        outputs.append(per_img)

        sample_results.append({
            "file_name": file_name,
            "image_id": img_id,
            "parsed_tool": parsed[0] if parsed else None,
            "is_t_in": answer_content is not None,
            "parse_fail": parse_fail,
            "tool_ok": tool_ok,
            "num_preds": sum(arr.shape[0] for arr in per_img),
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(images):
            print(f"🔄 Progress: {i + 1}/{len(images)} | Detection Tools: {detection_ok_count}/{i+1} | Distribution: {dict(sorted(tool_choice_counts.items()))}")

    n = len(images)
    tool_acc = detection_ok_count / n * 100 if n > 0 else 0

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(metric="bbox")
    
    metrics_result = dataset.evaluate(outputs, **eval_kwargs)

    print("\n" + "=" * 56)
    print("DIOR Object Detection (RemoteAgent) Evaluation Report")
    print("=" * 56)
    print("Tool Selection Distribution:", dict(sorted(tool_choice_counts.items())))
    print(f"Detection Tool Accuracy: {detection_ok_count}/{n} ({tool_acc:.2f}%)")
    
    if isinstance(metrics_result, dict):
        if "bbox_mAP_50" in metrics_result:
            print(f"  mAP50 (IoU=0.50):      {metrics_result['bbox_mAP_50'] * 100:.2f}%")
        if "bbox_mAP" in metrics_result:
            print(f"  mAP (IoU=0.50:0.95):   {metrics_result['bbox_mAP'] * 100:.2f}%")
    else:
        print(metrics_result)
    print("=" * 56)

    metrics: Dict[str, Any] = {
        "tool_selection_accuracy_pct": round(tool_acc, 2),
        "tool_choice_counts": dict(sorted(tool_choice_counts.items())),
    }
    if isinstance(metrics_result, dict):
        metrics["mAP50"] = round(metrics_result.get("bbox_mAP_50", 0) * 100, 2)
        metrics["mAP50_95"] = round(metrics_result.get("bbox_mAP", 0) * 100, 2)

    out = {"metrics": metrics, "samples": sample_results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        
    print(f"Results successfully saved to: {args.output}")

if __name__ == "__main__":
    main()