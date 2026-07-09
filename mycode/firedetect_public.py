# -*- coding: utf-8 -*-
"""Utilities for the firedetect_public firebig-only workflow.

The training set contains both ``fire`` and ``firebig`` labels, while the
competition metric only evaluates one final ``firebig`` box per image.  This
module keeps data conversion, post-processing, and metric calculation in one
place so training scripts and submission scripts can share the same behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree


CATEGORIES = [{"id": 1, "name": "fire"}, {"id": 2, "name": "firebig"}]
CATEGORY_NAME_TO_ID = {item["name"]: item["id"] for item in CATEGORIES}
CATEGORY_ID_TO_NAME = {item["id"]: item["name"] for item in CATEGORIES}


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _dataset_subdir(dataset_dir: Path, name: str) -> Path:
    subdir = dataset_dir / name
    return subdir if subdir.exists() else dataset_dir


def _categories_for(class_names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    if class_names is None:
        return CATEGORIES.copy()

    names = [name.strip() for name in class_names if name.strip()]
    if not names:
        raise ValueError("class_names must contain at least one class")

    known = set(CATEGORY_NAME_TO_ID)
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"unknown class names: {', '.join(unknown)}")

    return [{"id": idx, "name": name} for idx, name in enumerate(names, start=1)]


def _read_int(node: ElementTree.Element, name: str) -> int:
    value = node.findtext(name)
    if value is None:
        raise ValueError(f"missing XML field: {name}")
    return int(float(value))


def _clip_voc_box(box: list[int], width: int, height: int) -> list[int] | None:
    xmin, ymin, xmax, ymax = box
    xmin = max(0, min(width - 1, xmin))
    ymin = max(0, min(height - 1, ymin))
    xmax = max(0, min(width - 1, xmax))
    ymax = max(0, min(height - 1, ymax))
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax, ymax]


def _voc_box_to_coco(box: list[int]) -> list[int]:
    xmin, ymin, xmax, ymax = box
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def convert_voc_dataset(
    dataset_dir: str | Path,
    image_names: Iterable[str] | None = None,
    class_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Convert ``firedetect_public`` VOC XML annotations to COCO dict data.

    ``dataset_dir`` may contain ``images`` and ``annotations`` folders, or keep
    JPG/XML files in one flat directory.  When ``image_names`` is provided, only
    those files are converted.  ``class_names`` can be used to export a subset
    with contiguous COCO category ids, for example ``["firebig"]``.
    """
    dataset_dir = _as_path(dataset_dir)
    images_dir = _dataset_subdir(dataset_dir, "images")
    annotations_dir = _dataset_subdir(dataset_dir, "annotations")
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"annotations directory not found: {annotations_dir}")

    if image_names is None:
        image_paths = sorted(images_dir.glob("*.jpg"))
    else:
        image_paths = [images_dir / name for name in image_names]

    categories = _categories_for(class_names)
    category_name_to_id = {item["name"]: item["id"] for item in categories}
    coco: dict[str, Any] = {"images": [], "annotations": [], "categories": categories}
    ann_id = 1
    for image_id, image_path in enumerate(image_paths, start=1):
        xml_path = annotations_dir / f"{image_path.stem}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"annotation not found for {image_path.name}: {xml_path}")

        tree = ElementTree.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            raise ValueError(f"missing <size> in {xml_path}")
        width = _read_int(size, "width")
        height = _read_int(size, "height")

        coco["images"].append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )

        firebig_count = 0
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name == "firebig":
                firebig_count += 1
            if name not in category_name_to_id:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            voc_box = [
                _read_int(bndbox, "xmin"),
                _read_int(bndbox, "ymin"),
                _read_int(bndbox, "xmax"),
                _read_int(bndbox, "ymax"),
            ]
            clipped = _clip_voc_box(voc_box, width, height)
            if clipped is None:
                continue
            bbox = _voc_box_to_coco(clipped)
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_name_to_id[name],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        if firebig_count != 1:
            raise ValueError(f"{xml_path} must contain exactly one firebig, got {firebig_count}")

    return coco


def split_image_names(dataset_dir: str | Path, val_ratio: float = 0.2, seed: int = 2026) -> tuple[list[str], list[str]]:
    """Return deterministic train/val image file names."""
    import random

    images_dir = _dataset_subdir(_as_path(dataset_dir), "images")
    names = sorted(path.name for path in images_dir.glob("*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(names)
    val_count = max(1, int(round(len(names) * val_ratio))) if names else 0
    val_names = sorted(names[:val_count])
    train_names = sorted(names[val_count:])
    return train_names, val_names


def copy_images(dataset_dir: str | Path, output_dir: str | Path, image_names: Iterable[str] | None = None) -> int:
    """Copy dataset JPG files to ``output_dir/images`` for COCO training."""
    dataset_dir = _as_path(dataset_dir)
    output_dir = _as_path(output_dir)
    images_dir = _dataset_subdir(dataset_dir, "images")
    target_dir = output_dir / "images"
    target_dir.mkdir(parents=True, exist_ok=True)

    if image_names is None:
        image_paths = sorted(images_dir.glob("*.jpg"))
    else:
        image_paths = [images_dir / name for name in image_names]

    copied = 0
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        shutil.copy2(image_path, target_dir / image_path.name)
        copied += 1
    return copied


def write_coco_json(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = _as_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """Calculate IoU for COCO-format ``[x, y, w, h]`` boxes."""
    x11, y11, x12, y12 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    x21, y21, x22, y22 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    ix1, iy1 = max(x11, x21), max(y11, y21)
    ix2, iy2 = min(x12, x22), min(y12, y22)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = bbox_area(box1) + bbox_area(box2) - inter
    return inter / union if union > 0 else 0.0


def _category_name(detection: dict[str, Any]) -> str:
    if "category" in detection:
        return str(detection["category"])
    category_id = detection.get("category_id", detection.get("type"))
    if isinstance(category_id, str):
        return category_id
    return CATEGORY_ID_TO_NAME.get(int(category_id), str(category_id)) if category_id is not None else ""


def _detection_score(detection: dict[str, Any]) -> float:
    return float(detection.get("score", 1.0))


def _weighted_area_score(detection: dict[str, Any], image_area: float, area_alpha: float) -> float:
    score = _detection_score(detection)
    if area_alpha <= 0:
        return score
    normalized_area = bbox_area(detection["bbox"]) / max(1.0, image_area)
    return score * math.pow(max(0.0, normalized_area), area_alpha)


def select_firebig(
    detections: list[dict[str, Any]],
    firebig_threshold: float = 0.5,
    fire_threshold: float = 0.25,
    area_alpha: float = 0.3,
    image_width: int = 640,
    image_height: int = 480,
) -> dict[str, Any] | None:
    """Select one final firebig box from model detections.

    Reliable ``firebig`` predictions are preferred.  If none exists, all
    fire-like boxes above ``fire_threshold`` compete with a confidence-area
    weighted score.
    """
    if not detections:
        return None

    candidates = [det for det in detections if "bbox" in det and _category_name(det) in {"fire", "firebig"}]
    if not candidates:
        return None

    reliable_big = [
        det for det in candidates if _category_name(det) == "firebig" and _detection_score(det) >= firebig_threshold
    ]
    if reliable_big:
        return max(reliable_big, key=_detection_score).copy()

    fallback = [det for det in candidates if _detection_score(det) >= fire_threshold]
    if not fallback:
        fallback = candidates
    image_area = float(image_width * image_height)
    return max(fallback, key=lambda det: _weighted_area_score(det, image_area, area_alpha)).copy()


def postprocess_predictions(
    predictions: list[dict[str, Any]],
    firebig_threshold: float = 0.5,
    fire_threshold: float = 0.25,
    area_alpha: float = 0.3,
) -> list[dict[str, Any]]:
    """Reduce multi-box predictions to one ``firebig`` prediction per image."""
    by_image: dict[str, list[dict[str, Any]]] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    for pred in predictions:
        image_id = str(pred.get("image_id", pred.get("file_name", "")))
        if not image_id:
            continue
        by_image.setdefault(image_id, []).append(pred)
        width = int(pred.get("width", 640))
        height = int(pred.get("height", 480))
        image_sizes[image_id] = (width, height)

    results = []
    for image_id in sorted(by_image):
        width, height = image_sizes.get(image_id, (640, 480))
        selected = select_firebig(
            by_image[image_id],
            firebig_threshold=firebig_threshold,
            fire_threshold=fire_threshold,
            area_alpha=area_alpha,
            image_width=width,
            image_height=height,
        )
        if selected is None:
            continue
        results.append(
            {
                "image_id": image_id,
                "category": "firebig",
                "category_id": CATEGORY_NAME_TO_ID["firebig"],
                "bbox": [int(round(v)) for v in selected["bbox"]],
                "score": _detection_score(selected),
            }
        )
    return results


def _firebig_gt_by_filename(coco_gt: dict[str, Any]) -> dict[str, list[float]]:
    image_id_to_name = {image["id"]: image["file_name"] for image in coco_gt.get("images", [])}
    firebig_category_ids = {
        category["id"] for category in coco_gt.get("categories", []) if category.get("name") == "firebig"
    }
    result = {}
    for ann in coco_gt.get("annotations", []):
        if ann.get("category_id") in firebig_category_ids:
            filename = image_id_to_name.get(ann.get("image_id"))
            if filename:
                result[filename] = ann["bbox"]
    return result


def calculate_firebig_f1(
    coco_gt: dict[str, Any], predictions: list[dict[str, Any]], iou_threshold: float = 0.5
) -> dict[str, float | int]:
    """Calculate F1 for the firebig-only competition rule."""
    gt_by_name = _firebig_gt_by_filename(coco_gt)
    preds_by_name = {str(pred.get("image_id", pred.get("file_name", ""))): pred for pred in predictions}

    tp = fp = fn = 0
    for filename, gt_box in gt_by_name.items():
        pred = preds_by_name.get(filename) or preds_by_name.get(Path(filename).stem)
        if pred is None:
            fn += 1
            continue
        iou = calculate_iou([float(v) for v in pred["bbox"]], [float(v) for v in gt_box])
        if iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
            fn += 1

    for filename in preds_by_name:
        if filename and filename not in gt_by_name and f"{filename}.jpg" not in gt_by_name:
            fp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _load_json(path: str | Path) -> Any:
    return json.loads(_as_path(path).read_text(encoding="utf-8"))


def _cmd_convert(args: argparse.Namespace) -> None:
    dataset_dir = _as_path(args.dataset_dir)
    output_dir = _as_path(args.output_dir)
    class_names = [name.strip() for name in args.classes.split(",")] if args.classes else None
    train_names, val_names = split_image_names(dataset_dir, args.val_ratio, args.seed)
    write_coco_json(
        convert_voc_dataset(dataset_dir, train_names, class_names),
        output_dir / "annotations" / "instance_train.json",
    )
    write_coco_json(
        convert_voc_dataset(dataset_dir, val_names, class_names),
        output_dir / "annotations" / "instance_val.json",
    )
    write_coco_json(
        convert_voc_dataset(dataset_dir, class_names=class_names),
        output_dir / "annotations" / "instance_train_full.json",
    )
    if args.copy_images:
        copied = copy_images(dataset_dir, output_dir)
        print(f"copied {copied} images to {output_dir / 'images'}")
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)
    (output_dir / "splits" / "train.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
    (output_dir / "splits" / "val.txt").write_text("\n".join(val_names) + "\n", encoding="utf-8")
    print(f"converted {len(train_names)} train and {len(val_names)} val images to {output_dir}")


def _cmd_postprocess(args: argparse.Namespace) -> None:
    predictions = _load_json(args.predictions)
    if isinstance(predictions, dict) and "result" in predictions:
        predictions = predictions["result"]
    result = postprocess_predictions(
        predictions,
        firebig_threshold=args.firebig_threshold,
        fire_threshold=args.fire_threshold,
        area_alpha=args.area_alpha,
    )
    write_coco_json({"result": result}, args.output)
    print(f"wrote {len(result)} firebig predictions to {args.output}")


def _cmd_eval(args: argparse.Namespace) -> None:
    coco_gt = _load_json(args.ground_truth)
    predictions = _load_json(args.predictions)
    if isinstance(predictions, dict) and "result" in predictions:
        predictions = predictions["result"]
    metrics = calculate_firebig_f1(coco_gt, predictions, args.iou_threshold)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="firedetect_public utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert-voc", help="convert VOC XML annotations to COCO JSON")
    convert.add_argument("--dataset-dir", required=True)
    convert.add_argument("--output-dir", required=True)
    convert.add_argument("--val-ratio", type=float, default=0.2)
    convert.add_argument("--seed", type=int, default=2026)
    convert.add_argument("--classes", default=None, help="comma-separated classes to export, e.g. firebig")
    convert.add_argument("--copy-images", action="store_true", help="copy JPG files to output-dir/images")
    convert.set_defaults(func=_cmd_convert)

    postprocess = subparsers.add_parser("postprocess", help="select one firebig prediction per image")
    postprocess.add_argument("--predictions", required=True)
    postprocess.add_argument("--output", required=True)
    postprocess.add_argument("--firebig-threshold", type=float, default=0.5)
    postprocess.add_argument("--fire-threshold", type=float, default=0.25)
    postprocess.add_argument("--area-alpha", type=float, default=0.3)
    postprocess.set_defaults(func=_cmd_postprocess)

    evaluate = subparsers.add_parser("eval-firebig", help="evaluate firebig-only F1")
    evaluate.add_argument("--ground-truth", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
