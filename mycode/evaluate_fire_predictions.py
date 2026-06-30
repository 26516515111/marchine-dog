"""Evaluate one-box firebig predictions and render annotated false positives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


ABSOLUTE_SCORE_FLOOR = 0.02
RELATIVE_SCORE_RATIO = 0.5
FIRE_LABELS = {"fire", "firebig"}


def bbox_iou_xywh(first: list[float], second: list[float]) -> float:
    first_x1, first_y1, first_width, first_height = [float(value) for value in first]
    second_x1, second_y1, second_width, second_height = [
        float(value) for value in second
    ]
    first_x2 = first_x1 + max(0.0, first_width)
    first_y2 = first_y1 + max(0.0, first_height)
    second_x2 = second_x1 + max(0.0, second_width)
    second_y2 = second_y1 + max(0.0, second_height)
    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(
        0.0, min(first_y2, second_y2) - max(first_y1, second_y1)
    )
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_width) * max(0.0, first_height)
    second_area = max(0.0, second_width) * max(0.0, second_height)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _prediction_bbox(prediction: dict[str, Any]) -> list[float]:
    if "bbox" in prediction:
        return [float(value) for value in prediction["bbox"]]
    return [
        float(prediction["x"]),
        float(prediction["y"]),
        float(prediction["width"]),
        float(prediction["height"]),
    ]


def _key(value: Any) -> str:
    return Path(str(value)).stem


def _ground_truth_by_image(
    coco_ground_truth: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    images_by_id = {
        image["id"]: image for image in coco_ground_truth.get("images", [])
    }
    annotations_by_image: dict[Any, list[dict[str, Any]]] = {}
    for annotation in coco_ground_truth.get("annotations", []):
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    ground_truth: dict[str, dict[str, Any]] = {}
    for image_id, image in images_by_id.items():
        annotations = annotations_by_image.get(image_id, [])
        if not annotations:
            continue
        largest = max(
            annotations,
            key=lambda annotation: float(annotation["bbox"][2])
            * float(annotation["bbox"][3]),
        )
        key = _key(image["file_name"])
        ground_truth[key] = {
            "image_id": key,
            "file_name": str(image["file_name"]),
            "bbox": [float(value) for value in largest["bbox"]],
        }
    return ground_truth


def evaluate(
    coco_ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    """Evaluate predictions; IoU must be strictly greater than the threshold."""
    ground_truth = _ground_truth_by_image(coco_ground_truth)
    predictions_by_image: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        prediction_key = _key(
            prediction.get("image_id", prediction.get("file_name", ""))
        )
        if prediction_key:
            predictions_by_image.setdefault(prediction_key, []).append(prediction)

    true_positives = false_positives = false_negatives = 0
    rows: list[dict[str, Any]] = []
    for image_key, truth in ground_truth.items():
        image_predictions = predictions_by_image.pop(image_key, [])
        prediction = image_predictions[0] if image_predictions else None
        if prediction is None:
            false_negatives += 1
            rows.append(
                {
                    "image_id": image_key,
                    "file_name": truth["file_name"],
                    "status": "FN",
                    "gt_bbox": truth["bbox"],
                    "prediction_bbox": None,
                    "score": None,
                    "iou": 0.0,
                }
            )
        else:
            prediction_bbox = _prediction_bbox(prediction)
            iou = bbox_iou_xywh(truth["bbox"], prediction_bbox)
            if iou > iou_threshold:
                true_positives += 1
                status = "TP"
            else:
                false_positives += 1
                false_negatives += 1
                status = "FP"
            rows.append(
                {
                    "image_id": image_key,
                    "file_name": truth["file_name"],
                    "status": status,
                    "gt_bbox": truth["bbox"],
                    "prediction_bbox": prediction_bbox,
                    "score": float(prediction.get("score", 0.0)),
                    "iou": iou,
                }
            )

        for duplicate in image_predictions[1:]:
            false_positives += 1
            rows.append(
                {
                    "image_id": image_key,
                    "file_name": truth["file_name"],
                    "status": "FP",
                    "gt_bbox": truth["bbox"],
                    "prediction_bbox": _prediction_bbox(duplicate),
                    "score": float(duplicate.get("score", 0.0)),
                    "iou": bbox_iou_xywh(
                        truth["bbox"], _prediction_bbox(duplicate)
                    ),
                }
            )

    for image_key, unmatched in predictions_by_image.items():
        for prediction in unmatched:
            false_positives += 1
            rows.append(
                {
                    "image_id": image_key,
                    "file_name": str(
                        prediction.get("file_name", f"{image_key}.jpg")
                    ),
                    "status": "FP",
                    "gt_bbox": None,
                    "prediction_bbox": _prediction_bbox(prediction),
                    "score": float(prediction.get("score", 0.0)),
                    "iou": 0.0,
                }
            )

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    metrics: dict[str, float | int] = {
        "images": len(ground_truth),
        "iou_threshold": float(iou_threshold),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics, rows


def _is_fire_candidate(candidate: dict[str, Any]) -> bool:
    bbox = candidate.get("bbox")
    return (
        str(candidate.get("category", "")) in FIRE_LABELS
        and isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and float(bbox[2]) > 0.0
        and float(bbox[3]) > 0.0
    )


def predictions_for_strategy(
    raw_candidates: dict[str, list[dict[str, Any]]],
    strategy: str,
) -> list[dict[str, Any]]:
    """Select one candidate per image using one fixed, non-tuned strategy."""
    predictions: list[dict[str, Any]] = []
    for image_id, raw in raw_candidates.items():
        candidates = [candidate for candidate in raw if _is_fire_candidate(candidate)]
        if not candidates:
            continue
        if strategy == "highest_confidence":
            selected = max(candidates, key=lambda item: float(item.get("score", 0.0)))
        elif strategy == "largest_area":
            selected = max(
                candidates,
                key=lambda item: float(item["bbox"][2]) * float(item["bbox"][3]),
            )
        elif strategy == "relative_gate":
            best_score = max(float(item.get("score", 0.0)) for item in candidates)
            threshold = max(
                ABSOLUTE_SCORE_FLOOR, RELATIVE_SCORE_RATIO * best_score
            )
            credible = [
                item
                for item in candidates
                if float(item.get("score", 0.0)) >= threshold
            ]
            selected = max(
                credible,
                key=lambda item: (
                    float(item["bbox"][2]) * float(item["bbox"][3]),
                    float(item.get("score", 0.0)),
                ),
            )
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        predictions.append(
            {
                "image_id": _key(image_id),
                "category": "firebig",
                "bbox": [float(value) for value in selected["bbox"]],
                "score": float(selected.get("score", 0.0)),
            }
        )
    return predictions


def _box_points(bbox: list[float]) -> tuple[tuple[int, int], tuple[int, int]]:
    x, y, width, height = [float(value) for value in bbox]
    return (round(x), round(y)), (round(x + width), round(y + height))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_false_positives(
    rows: list[dict[str, Any]],
    image_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    fp_image_dir = output_dir / "fp_images"
    fp_image_dir.mkdir(parents=True, exist_ok=True)
    for stale_image in fp_image_dir.glob("*.jpg"):
        stale_image.unlink()

    false_positives = [row for row in rows if row["status"] == "FP"]
    false_negatives = [
        row for row in rows if row["status"] in {"FP", "FN"}
    ]
    for row in false_positives:
        source_path = image_dir / row["file_name"]
        image = cv2.imread(str(source_path))
        if image is None:
            row["render_error"] = f"unable to read {source_path}"
            continue
        if row.get("gt_bbox") is not None:
            cv2.rectangle(
                image,
                *_box_points(row["gt_bbox"]),
                (0, 255, 0),
                2,
            )
        if row.get("prediction_bbox") is not None:
            cv2.rectangle(
                image,
                *_box_points(row["prediction_bbox"]),
                (0, 0, 255),
                2,
            )
        score = float(row.get("score") or 0.0)
        iou = float(row.get("iou") or 0.0)
        cv2.putText(
            image,
            f"FP score={score:.3f} IoU={iou:.3f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "GT=green  prediction=red",
            (8, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        destination = fp_image_dir / row["file_name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image):
            row["render_error"] = f"unable to write {destination}"

    summary = {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
    _write_json(output_dir / "fp_summary.json", summary)
    return summary


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def run_evaluation(
    ground_truth_path: str | Path,
    predictions_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    ground_truth = _load_json(ground_truth_path)
    prediction_document = _load_json(predictions_path)
    if isinstance(prediction_document, dict):
        predictions = prediction_document.get("result", [])
        raw_candidates = prediction_document.get("raw_candidates", {})
    elif isinstance(prediction_document, list):
        predictions = prediction_document
        raw_candidates = {}
    else:
        raise ValueError("predictions must be a list or a JSON object")

    metrics, rows = evaluate(ground_truth, predictions, iou_threshold)
    baselines: dict[str, dict[str, float | int]] = {}
    if raw_candidates:
        for strategy in ("highest_confidence", "largest_area", "relative_gate"):
            strategy_predictions = predictions_for_strategy(raw_candidates, strategy)
            strategy_metrics, _ = evaluate(
                ground_truth, strategy_predictions, iou_threshold
            )
            baselines[strategy] = strategy_metrics

    complete_metrics: dict[str, Any] = dict(metrics)
    complete_metrics["baselines"] = baselines
    output_dir = Path(output_dir)
    _write_json(output_dir / "metrics.json", complete_metrics)
    _write_json(output_dir / "per_image.json", rows)
    render_false_positives(rows, image_dir, output_dir)
    return complete_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate firebig F1 and render false positives"
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    metrics = run_evaluation(
        args.ground_truth,
        args.predictions,
        args.image_dir,
        args.output_dir,
        args.iou_threshold,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
