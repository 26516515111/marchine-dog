"""Merge two firebig COCO datasets and create a deterministic stratified split."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path
    filename_prefix: str


@dataclass(frozen=True)
class ImageRecord:
    source: str
    source_path: Path
    output_filename: str
    width: int
    height: int
    annotations: tuple[dict[str, Any], ...]

    @property
    def is_positive(self) -> bool:
        return bool(self.annotations)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_source_records(source: SourceSpec) -> list[ImageRecord]:
    annotation_path = (
        source.root / "annotations" / "instance_train_full.json"
    )
    coco = _load_json(annotation_path)
    annotations_by_image: dict[Any, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        annotations_by_image.setdefault(annotation["image_id"], []).append(
            annotation
        )

    records: list[ImageRecord] = []
    for image in sorted(coco.get("images", []), key=lambda item: item["file_name"]):
        source_path = source.root / "images" / image["file_name"]
        if not source_path.is_file():
            raise FileNotFoundError(f"source image not found: {source_path}")
        output_filename = f"{source.filename_prefix}{image['file_name']}"
        records.append(
            ImageRecord(
                source=source.name,
                source_path=source_path,
                output_filename=output_filename,
                width=int(image["width"]),
                height=int(image["height"]),
                annotations=tuple(
                    annotations_by_image.get(image["id"], [])
                ),
            )
        )
    return records


def stratified_split(
    records: list[ImageRecord],
    val_ratio: float = 0.2,
    seed: int = 2026,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    strata: dict[tuple[str, bool], list[ImageRecord]] = {}
    for record in records:
        strata.setdefault((record.source, record.is_positive), []).append(record)

    train: list[ImageRecord] = []
    val: list[ImageRecord] = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda item: item.output_filename)
        random.Random(f"{seed}:{key[0]}:{int(key[1])}").shuffle(group)
        if len(group) <= 1:
            val_count = 0
        else:
            val_count = max(1, min(len(group) - 1, int(round(len(group) * val_ratio))))
        val.extend(group[:val_count])
        train.extend(group[val_count:])
    return (
        sorted(train, key=lambda item: item.output_filename),
        sorted(val, key=lambda item: item.output_filename),
    )


def build_coco(records: list[ImageRecord], description: str) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, record in enumerate(
        sorted(records, key=lambda item: item.output_filename),
        start=1,
    ):
        images.append(
            {
                "id": image_id,
                "file_name": record.output_filename,
                "width": record.width,
                "height": record.height,
            }
        )
        for original in record.annotations:
            annotation = {
                key: value
                for key, value in original.items()
                if key not in {"id", "image_id", "category_id"}
            }
            annotation.update(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                }
            )
            if "area" not in annotation:
                bbox = annotation["bbox"]
                annotation["area"] = float(bbox[2]) * float(bbox[3])
            annotation.setdefault("iscrowd", 0)
            annotation.setdefault("segmentation", [])
            annotations.append(annotation)
            annotation_id += 1
    return {
        "info": {"description": description},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "firebig", "supercategory": "fire"}
        ],
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _split_summary(records: list[ImageRecord], coco: dict[str, Any]) -> dict[str, int]:
    positive_images = sum(record.is_positive for record in records)
    return {
        "images": len(records),
        "positive_images": positive_images,
        "negative_images": len(records) - positive_images,
        "annotations": len(coco["annotations"]),
    }


def merge_datasets(
    sources: list[SourceSpec],
    output_dir: str | Path,
    val_ratio: float = 0.2,
    seed: int = 2026,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    records = [
        record
        for source in sources
        for record in load_source_records(source)
    ]
    filenames = [record.output_filename for record in records]
    if len(filenames) != len(set(filenames)):
        raise ValueError("output filename collision after source prefixing")

    train_records, val_records = stratified_split(records, val_ratio, seed)
    full_records = sorted(records, key=lambda item: item.output_filename)
    full_coco = build_coco(full_records, "merged firebig full")
    train_coco = build_coco(train_records, "merged firebig train")
    val_coco = build_coco(val_records, "merged firebig val")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    expected_filenames = set(filenames)
    for stale in images_dir.iterdir():
        if stale.is_file() and stale.name not in expected_filenames:
            stale.unlink()
    for record in full_records:
        shutil.copy2(record.source_path, images_dir / record.output_filename)

    annotations_dir = output_dir / "annotations"
    _write_json(annotations_dir / "instance_train_full.json", full_coco)
    _write_json(annotations_dir / "instance_train.json", train_coco)
    _write_json(annotations_dir / "instance_val.json", val_coco)

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.txt").write_text(
        "\n".join(record.output_filename for record in train_records) + "\n",
        encoding="utf-8",
    )
    (splits_dir / "val.txt").write_text(
        "\n".join(record.output_filename for record in val_records) + "\n",
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "seed": seed,
        "val_ratio": val_ratio,
        "sources": {
            source.name: str(source.root.resolve()) for source in sources
        },
        "full": _split_summary(full_records, full_coco),
        "train": _split_summary(train_records, train_coco),
        "val": _split_summary(val_records, val_coco),
    }
    _write_json(output_dir / "merge_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Merge new_label_coco and coco_firebig with a fresh split"
    )
    parser.add_argument(
        "--new-label",
        type=Path,
        default=base_dir / "new_label_coco",
    )
    parser.add_argument(
        "--coco-firebig",
        type=Path,
        default=base_dir / "coco_firebig",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "new_label_coco_firebig_merged",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = merge_datasets(
        [
            SourceSpec("new_label", args.new_label, "new_label__"),
            SourceSpec("coco_firebig", args.coco_firebig, "coco_firebig__"),
        ],
        args.output,
        args.val_ratio,
        args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
