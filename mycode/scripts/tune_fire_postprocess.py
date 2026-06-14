# -*- coding: utf-8 -*-
"""
Tune fire post-processing with guardrails against overfitting.

The script can either:
1. load a raw prediction JSON that contains scores, or
2. run inference through the submission predict.py pipeline and export raw scored boxes.

Example:
  python mycode/scripts/tune_fire_postprocess.py \
    --pred-json raw_preds_with_score.json \
    --gt A_train/coco/annotations/instance_val.json \
    --out fire_postprocess_tuning.json

To generate raw predictions from the real submission script:
  python mycode/scripts/tune_fire_postprocess.py \
    --predict-py "D:/work/Marchine Dog/predict.py" \
    --infer-txt val.txt \
    --gt A_train/coco/annotations/instance_val.json \
    --raw-out raw_preds_with_score.json \
    --out fire_postprocess_tuning.json
"""
import argparse
import copy
import importlib.util
import itertools
import json
import os
from collections import defaultdict


CATEGORY_IDS = (1, 2, 3)
CATEGORY_NAMES = {1: "battery", 2: "board", 3: "fire"}
FIRE_ID = 3


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_thresholds(text):
    values = parse_float_list(text)
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values: battery,board,fire")
    return {1: values[0], 2: values[1], 3: values[2]}


def load_json_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "result" in data:
        return data["result"]
    if isinstance(data, list):
        return data
    raise ValueError("prediction JSON must be a list or a dict containing key 'result'")


def prediction_bbox(pred):
    if "bbox" in pred:
        bbox = pred["bbox"]
    else:
        bbox = [pred["x"], pred["y"], pred["width"], pred["height"]]
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


def prediction_category(pred):
    cat = pred.get("type", pred.get("category_id"))
    if cat is None:
        raise ValueError("prediction is missing 'type' or 'category_id'")
    return int(cat)


def normalize_prediction(pred, require_score=True):
    if "image_id" not in pred:
        raise ValueError("prediction is missing 'image_id'")
    if require_score and "score" not in pred:
        raise ValueError(
            "prediction JSON is missing 'score'. "
            "Use --predict-py to export raw scored boxes, or use a raw JSON generated before score filtering."
        )
    return {
        "image_id": os.path.splitext(str(pred["image_id"]))[0],
        "type": prediction_category(pred),
        "bbox": prediction_bbox(pred),
        "score": float(pred.get("score", 1.0)),
        "_order": int(pred.get("_order", 0)),
    }


def load_predictions(path, require_score=True):
    normalized = []
    for order, pred in enumerate(load_json_predictions(path)):
        item = normalize_prediction(pred, require_score=require_score)
        item["_order"] = order
        normalized.append(item)
    return normalized


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename_to_id = {
        os.path.splitext(img["file_name"])[0]: img["id"]
        for img in data["images"]
    }
    gt_by_image = defaultdict(list)
    for ann in data["annotations"]:
        gt_by_image[ann["image_id"]].append({
            "bbox": [float(v) for v in ann["bbox"]],
            "category_id": int(ann["category_id"]),
        })
    return gt_by_image, filename_to_id


def bbox_area_xywh(bbox):
    return max(0.0, bbox[2]) * max(0.0, bbox[3])


def iou_xywh(a, b):
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = bbox_area_xywh(a) + bbox_area_xywh(b) - inter
    return inter / union if union > 0 else 0.0


def nms(preds, threshold):
    if threshold >= 1.0:
        return sorted(preds, key=lambda p: p["score"], reverse=True)
    ordered = sorted(preds, key=lambda p: p["score"], reverse=True)
    kept = []
    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        ordered = [
            pred for pred in ordered
            if iou_xywh(best["bbox"], pred["bbox"]) < threshold
        ]
    return kept


def apply_postprocess(predictions, thresholds, nms_thresholds, fire_min_area):
    by_image_class = defaultdict(list)
    for order, pred in enumerate(predictions):
        cat_id = prediction_category(pred)
        if cat_id not in CATEGORY_IDS:
            continue
        norm = normalize_prediction(pred, require_score=True)
        norm["_order"] = int(pred.get("_order", order))
        if norm["score"] < thresholds.get(cat_id, 0.0):
            continue
        if cat_id == FIRE_ID and bbox_area_xywh(norm["bbox"]) < fire_min_area:
            continue
        by_image_class[(norm["image_id"], cat_id)].append(norm)

    filtered = []
    for (_image_id, cat_id), preds in sorted(by_image_class.items()):
        filtered.extend(nms(preds, nms_thresholds.get(cat_id, 1.0)))
    return sorted(filtered, key=lambda p: p["_order"])


def to_submission_predictions(predictions):
    result = []
    for pred in predictions:
        x, y, w, h = pred["bbox"]
        item = {
            "image_id": pred["image_id"],
            "type": int(pred["type"]),
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "segmentation": [],
        }
        if "score" in pred:
            item["score"] = float(pred["score"])
        result.append(item)
    return result


def evaluate(predictions, gt_by_image, filename_to_id, iou_threshold=0.5):
    stats = {cat_id: {"tp": 0, "fp": 0, "fn": 0} for cat_id in CATEGORY_IDS}
    preds_by_image = defaultdict(list)
    for pred in predictions:
        img_id = filename_to_id.get(os.path.splitext(str(pred["image_id"]))[0])
        if img_id is None:
            continue
        preds_by_image[img_id].append(pred)

    all_image_ids = set(gt_by_image.keys()) | set(preds_by_image.keys())
    for img_id in all_image_ids:
        gts = gt_by_image.get(img_id, [])
        preds = sorted(preds_by_image.get(img_id, []), key=lambda p: p["score"], reverse=True)
        matched_gt = set()

        for pred in preds:
            cat_id = int(pred["type"])
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gts):
                if idx in matched_gt or gt["category_id"] != cat_id:
                    continue
                score = iou_xywh(pred["bbox"], gt["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                matched_gt.add(best_idx)
                stats[cat_id]["tp"] += 1
            else:
                stats[cat_id]["fp"] += 1

        for idx, gt in enumerate(gts):
            if idx not in matched_gt:
                stats[gt["category_id"]]["fn"] += 1

    per_class = {}
    f1_values = []
    for cat_id in CATEGORY_IDS:
        tp = stats[cat_id]["tp"]
        fp = stats[cat_id]["fp"]
        fn = stats[cat_id]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[CATEGORY_NAMES[cat_id]] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
    return sum(f1_values) / len(f1_values), per_class


def load_predict_module(path):
    spec = importlib.util.spec_from_file_location("submission_predict", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_image_list(predict_module, infer_txt):
    if hasattr(predict_module, "get_test_images"):
        return predict_module.get_test_images(infer_txt)
    infer_dir = os.path.dirname(os.path.abspath(infer_txt))
    with open(infer_txt, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [line if os.path.isabs(line) else os.path.join(infer_dir, line) for line in lines]


def export_raw_predictions_from_predict_py(predict_py, infer_txt, model_dir, raw_out):
    module = load_predict_module(predict_py)
    import paddle

    paddle.enable_static()
    pred_config = module.PredictConfig(model_dir)
    detector = module.Detector(pred_config, model_dir)
    image_list = get_image_list(module, infer_txt)

    results = {"result": []}
    for im_path in image_list:
        input_im, im_info = module.preprocess(im_path, detector.preprocess_ops)
        inputs = module.create_inputs([input_im], [im_info])
        det_results = detector.predict(inputs)
        num = int(det_results["boxes_num"][0])
        image_id = os.path.splitext(os.path.basename(im_path))[0]
        if num <= 0:
            continue
        boxes = det_results["boxes"][:num, 2:]
        ids = det_results["boxes"][:num, 0]
        scores = det_results["boxes"][:num, 1]
        for idx in range(num):
            x1, y1, x2, y2 = [float(v) for v in boxes[idx]]
            results["result"].append({
                "image_id": image_id,
                "type": int(ids[idx]) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(scores[idx]),
                "_order": len(results["result"]),
            })

    with open(raw_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results["result"]


def candidate_record(params, overall_f1, per_class, baseline):
    fire = per_class["fire"]
    baseline_fire = baseline["per_class"]["fire"]
    raw_gain = overall_f1 - baseline["overall_f1"]
    recall_drop = max(0.0, baseline_fire["recall"] - fire["recall"])

    # Penalize aggressive choices so the recommendation is less val-specific.
    threshold_penalty = max(0.0, params["fire_threshold"] - 0.50) * 0.020
    nms_penalty = max(0.0, 0.45 - params["fire_nms"]) * 0.020
    area_penalty = max(0.0, params["fire_min_area"] - 120.0) / 40000.0
    recall_penalty = max(0.0, recall_drop - 0.02) * 0.50
    stable_score = overall_f1 - threshold_penalty - nms_penalty - area_penalty - recall_penalty

    return {
        **params,
        "overall_f1": overall_f1,
        "raw_gain": raw_gain,
        "stable_score": stable_score,
        "per_class": per_class,
    }


def choose_stable_candidate(candidates, best_raw, tolerance):
    near_best = [c for c in candidates if best_raw["overall_f1"] - c["overall_f1"] <= tolerance]
    return max(
        near_best,
        key=lambda c: (
            c["stable_score"],
            -abs(c["fire_threshold"] - 0.45),
            c["fire_nms"],
            -c["fire_min_area"],
        ),
    )


def run_search(predictions, gt_path, args):
    gt_by_image, filename_to_id = load_ground_truth(gt_path)
    base_thresholds = copy.deepcopy(args.base_thresholds)
    base_nms = {1: args.base_nms, 2: args.base_nms, 3: args.base_fire_nms}

    baseline_preds = apply_postprocess(
        predictions,
        thresholds=base_thresholds,
        nms_thresholds=base_nms,
        fire_min_area=args.base_fire_min_area,
    )
    baseline_f1, baseline_per_class = evaluate(baseline_preds, gt_by_image, filename_to_id)
    baseline = {
        "params": {
            "fire_threshold": base_thresholds[3],
            "fire_nms": base_nms[3],
            "fire_min_area": args.base_fire_min_area,
        },
        "overall_f1": baseline_f1,
        "per_class": baseline_per_class,
    }

    candidates = []
    for fire_thr, fire_nms, min_area in itertools.product(
        args.fire_thresholds, args.fire_nms_values, args.fire_min_areas
    ):
        thresholds = copy.deepcopy(base_thresholds)
        thresholds[3] = fire_thr
        nms_thresholds = copy.deepcopy(base_nms)
        nms_thresholds[3] = fire_nms
        processed = apply_postprocess(predictions, thresholds, nms_thresholds, min_area)
        overall_f1, per_class = evaluate(processed, gt_by_image, filename_to_id)
        params = {
            "fire_threshold": fire_thr,
            "fire_nms": fire_nms,
            "fire_min_area": min_area,
        }
        candidates.append(candidate_record(params, overall_f1, per_class, baseline))

    best_raw = max(candidates, key=lambda c: (c["overall_f1"], c["per_class"]["fire"]["precision"]))
    recommended = choose_stable_candidate(candidates, best_raw, args.stable_tolerance)
    candidates_sorted = sorted(
        candidates,
        key=lambda c: (c["overall_f1"], c["stable_score"]),
        reverse=True,
    )

    return {
        "baseline": baseline,
        "best_raw": best_raw,
        "recommended_stable": recommended,
        "top_candidates": candidates_sorted[:args.top_k],
        "search_space": {
            "base_thresholds": {CATEGORY_NAMES[k]: v for k, v in base_thresholds.items()},
            "base_nms": args.base_nms,
            "base_fire_nms": args.base_fire_nms,
            "fire_thresholds": args.fire_thresholds,
            "fire_nms_values": args.fire_nms_values,
            "fire_min_areas": args.fire_min_areas,
            "stable_tolerance": args.stable_tolerance,
        },
    }


def print_summary(result):
    def line(label, item):
        fire = item["per_class"]["fire"]
        params = item.get("params", item)
        print(
            f"{label:20s} "
            f"thr={params['fire_threshold']:.3f} "
            f"nms={params['fire_nms']:.3f} "
            f"min_area={params['fire_min_area']:.0f} "
            f"overall={item['overall_f1']:.4f} "
            f"fire P/R/F1={fire['precision']:.4f}/{fire['recall']:.4f}/{fire['f1']:.4f} "
            f"FP={fire['fp']} FN={fire['fn']}"
        )

    print("\nFire postprocess tuning summary")
    print("=" * 90)
    line("baseline", result["baseline"])
    line("best_raw", result["best_raw"])
    line("recommended_stable", result["recommended_stable"])
    print("\nTop candidates:")
    for idx, item in enumerate(result["top_candidates"][:10], start=1):
        fire = item["per_class"]["fire"]
        print(
            f"{idx:2d}. thr={item['fire_threshold']:.3f} nms={item['fire_nms']:.3f} "
            f"min_area={item['fire_min_area']:.0f} overall={item['overall_f1']:.4f} "
            f"stable={item['stable_score']:.4f} fire_fp={fire['fp']} fire_recall={fire['recall']:.4f}"
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Tune fire threshold, NMS, and min-area filtering.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pred-json", help="Raw prediction JSON with score fields.")
    source.add_argument("--predict-py", help="Path to the real submission predict.py.")
    parser.add_argument("--infer-txt", help="Input txt used when --predict-py is set.")
    parser.add_argument("--model-dir", default=None, help="Model directory used when --predict-py is set.")
    parser.add_argument("--raw-out", default="raw_preds_with_score.json", help="Where to save raw scored predictions.")
    parser.add_argument("--gt", required=True, help="COCO validation annotation JSON.")
    parser.add_argument("--out", default="fire_postprocess_tuning.json", help="Output JSON report.")

    parser.add_argument("--base-thresholds", type=parse_thresholds, default={1: 0.3, 2: 0.3, 3: 0.3})
    parser.add_argument("--base-nms", type=float, default=0.60)
    parser.add_argument("--base-fire-nms", type=float, default=0.60)
    parser.add_argument("--base-fire-min-area", type=float, default=0.0)
    parser.add_argument("--fire-thresholds", type=parse_float_list, default=parse_float_list("0.30,0.35,0.40,0.45,0.50,0.55"))
    parser.add_argument("--fire-nms-values", type=parse_float_list, default=parse_float_list("0.45,0.50,0.55,0.60"))
    parser.add_argument("--fire-min-areas", type=parse_float_list, default=parse_float_list("0,50,80,120,160,200"))
    parser.add_argument("--stable-tolerance", type=float, default=0.003)
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.predict_py:
            if not args.infer_txt:
                parser.error("--infer-txt is required with --predict-py")
            predict_dir = os.path.dirname(os.path.abspath(args.predict_py))
            model_dir = args.model_dir or os.path.join(predict_dir, "model")
            predictions = export_raw_predictions_from_predict_py(
                args.predict_py,
                args.infer_txt,
                model_dir,
                args.raw_out,
            )
        else:
            predictions = load_predictions(args.pred_json, require_score=True)
    except ValueError as exc:
        parser.error(str(exc))

    result = run_search(predictions, args.gt, args)
    print_summary(result)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved report: {args.out}")


if __name__ == "__main__":
    main()
