import argparse
import base64
import json
import os
import sys
import urllib.request
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


def build_user_prompt(pre_path: Path, post_path: Path) -> str:
    return (
        f"Pre image path: {pre_path}\n"
        f"Post image path: {post_path}\n\n"
        f"User request: These two images show the same area before and after a disaster. "
        f"Assess building damage: locate damaged buildings and classify damage levels (non-damaged, minor, major, destroyed). "
        f"Give me pixel-level damage location and classification.\n\n"
    )

def run_bda(
    pre_path: Path, post_path: Path, api_url: str, threshold: float = 0.5, num_class: int = 5
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Call Change3D BDA API and return (pred_loc [H,W], pred_cls [H,W])."""
    if not api_url:
        return None
    try:
        payload = {
            "task": "bda",
            "pre_image": encode_image(str(pre_path)),
            "post_image": encode_image(str(post_path)),
            "threshold": threshold,
            "num_class": num_class,
        }
        data = http_post_predict(api_url, payload, timeout=180.0)
        if data.get("status") != "success":
            return None
            
        loc_b64 = data.get("damage_location")
        cls_b64 = data.get("damage_class_map")
        if not loc_b64 or not cls_b64:
            return None

        def _decode(b64: str) -> np.ndarray:
            if "," in b64:
                b64 = b64.split(",")[1]
            img_data = base64.b64decode(b64)
            nparr = np.frombuffer(img_data, np.uint8)
            mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if mask is None and nparr.size == 256 * 256:
                mask = nparr.reshape(256, 256)
            return mask if mask is not None else np.zeros((256, 256), dtype=np.uint8)

        pred_loc = (_decode(loc_b64) > 127).astype(np.uint8)
        pred_cls = np.clip(_decode(cls_b64).astype(np.uint8), 0, num_class - 1)
        return (pred_loc, pred_cls)
    except Exception as e:
        print(f"  [ERROR] BDA API call failed: {e}")
    return None

# --- Metrics Evaluator ---
class BDAEvaluator:
    """Evaluator for Building Damage Assessment computing Pixel F1 and Damage F1 metrics."""

    def __init__(self, num_class: int):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((num_class, num_class), dtype=np.int64)

    def _generate_matrix(self, gt_image: np.ndarray, pre_image: np.ndarray) -> np.ndarray:
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype(np.int64) + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        return count.reshape(self.num_class, self.self.num_class)

    def add_batch(self, gt_image: np.ndarray, pre_image: np.ndarray):
        gt_image = np.asarray(gt_image).flatten()
        pre_image = np.asarray(pre_image).flatten()
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def pixel_f1_score(self) -> float:
        """Calculate binary F1 score (assuming num_class=2 setup for location)."""
        assert self.confusion_matrix.shape[0] == 2
        cm = self.confusion_matrix
        prec = cm[1, 1] / (cm[0, 1] + cm[1, 1] + 1e-10)
        rec = cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-10)
        return float(2 * rec * prec / (rec + prec + 1e-10))

    def damage_f1_per_class(self) -> np.ndarray:
        """Calculate F1 score for each specific damage class (ignoring background)."""
        TPs = np.diag(self.confusion_matrix)[1:]
        FNs = np.sum(self.confusion_matrix, axis=1)[1:] - TPs
        FPs = np.sum(self.confusion_matrix, axis=0)[1:] - TPs
        precisions = TPs / (TPs + FPs + 1e-7)
        recalls = TPs / (TPs + FNs + 1e-7)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        return f1_scores

# --- Data Loading ---
def load_bda_samples(dataset_root: Path, split: str) -> List[Tuple[Path, Path, Path, Path, str]]:
    """Load xBD dataset samples with their corresponding masks."""
    t1_dir = dataset_root / split / "t1"
    t2_dir = dataset_root / split / "t2"
    label1_dir = dataset_root / split / "label1"
    label2_dir = dataset_root / split / "label2"
    
    if not t1_dir.exists() or not t2_dir.exists() or not label1_dir.exists() or not label2_dir.exists():
        return []

    samples = []
    for f in sorted(t1_dir.iterdir()):
        if not f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            continue
        name = f.name
        pre_path = t1_dir / name
        post_path = t2_dir / name
        label_loc_name = name.replace("disaster", "disaster_target")
        
        label_loc_path = label1_dir / label_loc_name
        label_cls_path = label2_dir / label_loc_name
        
        if not label_loc_path.exists():
            label_loc_path = label1_dir / name
            label_cls_path = label2_dir / name
            
        if pre_path.exists() and post_path.exists() and label_loc_path.exists() and label_cls_path.exists():
            samples.append((pre_path, post_path, label_loc_path, label_cls_path, name))
    return samples

# --- Main Evaluation Pipeline ---
def main():
    parser = argparse.ArgumentParser(description="xBD Building Damage Assessment Evaluation (RemoteAgent Tool Selection)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "hold", "test"])
    parser.add_argument("--dataset_root", type=str, default=str(SCRIPT_DIR / "xBD"))
    parser.add_argument("--change3d_url", type=str, default="http://localhost:6658")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000")
    parser.add_argument("--vllm_model_name", type=str, default=None)
    parser.add_argument("--system_prompt_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(SCRIPT_DIR / "results_xbd_bda_remoteagent_choose.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_class", type=int, default=5)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    samples = load_bda_samples(dataset_root, args.split)
    if not samples:
        sys.exit(f"[ERROR] No xBD samples found in: {dataset_root}/{args.split}")

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

    print("=" * 60)
    print("xBD Building Damage Assessment Evaluation (RemoteAgent)")
    print("=" * 60)
    print(f"  vLLM URL: {args.vllm_url} | Model: {model_name}")
    print(f"  Change3D API (6658): {change3d_url}")
    print(f"  Split: {args.split} | Samples: {len(samples)}")
    print("-" * 60)

    tool_choice_counts: Dict[str, int] = {}
    bda_ok_count = 0
    evaluator_loc = BDAEvaluator(num_class=2)
    evaluator_cls = BDAEvaluator(num_class=args.num_class)
    sample_results: List[Dict] = []

    for i, (pre_path, post_path, label_loc_path, label_cls_path, name) in enumerate(samples):
        pre_img = cv2.imread(str(pre_path))
        post_img = cv2.imread(str(post_path))
        label_loc = cv2.imread(str(label_loc_path), cv2.IMREAD_GRAYSCALE)
        label_cls = cv2.imread(str(label_cls_path), cv2.IMREAD_GRAYSCALE)

        if pre_img is None or post_img is None or label_loc is None or label_cls is None:
            sample_results.append({"file_name": name, "skip": "load_fail"})
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
        pred_loc = None
        pred_cls = None

        if answer_content is not None:
            tool_choice_counts["(T_in)"] = tool_choice_counts.get("(T_in)", 0) + 1
        else:
            parsed = parse_tool_call_with_bcd_fallback(raw, pre_path, post_path)
            if parsed:
                tool_name, tool_args = parsed
                tool_choice_counts[tool_name] = tool_choice_counts.get(tool_name, 0) + 1
                
                if tool_name == "change3d_bda":
                    bda_ok_count += 1
                    pre_p = Path((tool_args.get("pre_image_path") or str(pre_path)).strip().strip("'").strip('"'))
                    post_p = Path((tool_args.get("post_image_path") or str(post_path)).strip().strip('"').strip("'"))
                    
                    if not pre_p.exists(): pre_p = pre_path
                    if not post_p.exists(): post_p = post_path
                        
                    result = run_bda(pre_p, post_p, change3d_url, threshold=args.threshold, num_class=args.num_class)
                    if result:
                        pred_loc, pred_cls = result
                else:
                    execute_tool(tool_name, tool_args, api_urls)
            else:
                tool_choice_counts["(parse_fail)"] = tool_choice_counts.get("(parse_fail)", 0) + 1

        if pred_loc is not None and pred_cls is not None:
            h_pred, w_pred = pred_loc.shape[:2]
            label_loc_resized = cv2.resize(label_loc.astype(np.uint8), (w_pred, h_pred), interpolation=cv2.INTER_NEAREST)
            label_cls_resized = cv2.resize(label_cls.astype(np.uint8), (w_pred, h_pred), interpolation=cv2.INTER_NEAREST)
            
            label_loc_binary = (label_loc_resized > 0).astype(np.uint8)
            label_cls_resized = np.clip(label_cls_resized, 0, args.num_class - 1)

            evaluator_loc.add_batch(label_loc_binary.flatten(), pred_loc.flatten())
            mask = label_loc_binary > 0
            if mask.any():
                evaluator_cls.add_batch(label_cls_resized[mask].flatten(), pred_cls[mask].flatten())

        sample_results.append({
            "file_name": name,
            "parsed_tool": parsed[0] if parsed else None,
            "is_t_in": answer_content is not None,
            "bda_ok": pred_loc is not None and pred_cls is not None,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(samples):
            f1_loc = evaluator_loc.pixel_f1_score()
            damage_f1 = evaluator_cls.damage_f1_per_class()
            inv_sum = np.sum(1.0 / damage_f1) if len(damage_f1) > 0 else np.inf
            f1_cls = float(len(damage_f1) / inv_sum) if len(damage_f1) > 0 and np.isfinite(inv_sum) else 0.0
            f1_overall = 0.3 * f1_loc + 0.7 * f1_cls
            print(f"🔄 Progress: {i + 1}/{len(samples)} | BDA_OK: {bda_ok_count} | F1_loc: {f1_loc:.4f} | F1_cls: {f1_cls:.4f} | F1_overall: {f1_overall:.4f} | Tools: {dict(sorted(tool_choice_counts.items()))}")

    n = len(samples)
    bda_acc = bda_ok_count / n * 100 if n > 0 else 0

    f1_loc = evaluator_loc.pixel_f1_score()
    damage_f1_scores = evaluator_cls.damage_f1_per_class()
    if len(damage_f1_scores) > 0:
        inv_sum = np.sum(1.0 / damage_f1_scores)
        harmonic_mean_f1 = float(len(damage_f1_scores) / inv_sum) if np.isfinite(inv_sum) else 0.0
    else:
        harmonic_mean_f1 = 0.0
        
    f1_cls = harmonic_mean_f1
    f1_overall = 0.3 * f1_loc + 0.7 * f1_cls

    class_names = ["Non", "Minor", "Major", "Destroy"]
    per_class = {}
    for k, cname in enumerate(class_names):
        if k < len(damage_f1_scores):
            per_class[cname] = float(damage_f1_scores[k])
        else:
            per_class[cname] = 0.0

    print("\n" + "=" * 60)
    print("xBD Building Damage Assessment Evaluation Report")
    print("=" * 60)
    print("Tool Selection Distribution:", dict(sorted(tool_choice_counts.items())))
    print(f"Target Tool Selection Accuracy: {bda_ok_count}/{n} ({bda_acc:.2f}%)")
    print(f"  F1_loc     = {f1_loc:.4f}")
    print(f"  F1_cls     = {f1_cls:.4f}")
    print(f"  F1_overall = {f1_overall:.4f}")
    print("  Damage F1 Per-class:")
    for cname, score in per_class.items():
        print(f"    {cname:8s} = {score:.4f}")
    print("=" * 60)

    metrics = {
        "bda_tool_accuracy_pct": round(bda_acc, 2),
        "F1_loc": round(f1_loc, 4),
        "F1_cls": round(f1_cls, 4),
        "F1_overall": round(f1_overall, 4),
        "Damage_F1_per_class": per_class,
        "tool_choice_counts": dict(sorted(tool_choice_counts.items())),
    }
    
    out = {"metrics": metrics, "samples": sample_results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults successfully saved to: {args.output}")

if __name__ == "__main__":
    main()