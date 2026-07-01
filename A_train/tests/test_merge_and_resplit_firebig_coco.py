import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "merge_and_resplit_firebig_coco.py"
)


def load_builder():
    name = "merge_and_resplit_firebig_coco_under_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_source(root: Path, prefix: str) -> None:
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    images = []
    annotations = []
    annotation_id = 1
    for index in range(10):
        filename = f"shared_{index:02d}.jpg"
        (images_dir / filename).write_bytes(f"{prefix}-{index}".encode())
        images.append(
            {
                "id": index + 1,
                "file_name": filename,
                "width": 640,
                "height": 480,
            }
        )
        if index < 5:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": index + 1,
                    "category_id": 7,
                    "bbox": [10 + index, 20, 30, 40],
                    "area": 1200,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 7, "name": "firebig"}],
    }
    (annotations_dir / "instance_train_full.json").write_text(
        json.dumps(coco),
        encoding="utf-8",
    )


class MergeAndResplitTests(unittest.TestCase):
    def test_merge_preserves_empty_images_and_stratifies_each_source(self):
        builder = load_builder()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            output = root / "merged"
            make_source(first, "first")
            make_source(second, "second")

            summary = builder.merge_datasets(
                [
                    builder.SourceSpec("first", first, "first__"),
                    builder.SourceSpec("second", second, "second__"),
                ],
                output,
                val_ratio=0.2,
                seed=2026,
            )

            full = json.loads(
                (output / "annotations" / "instance_train_full.json").read_text(
                    encoding="utf-8"
                )
            )
            train = json.loads(
                (output / "annotations" / "instance_train.json").read_text(
                    encoding="utf-8"
                )
            )
            val = json.loads(
                (output / "annotations" / "instance_val.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(len(full["images"]), 20)
        self.assertEqual(len(full["annotations"]), 10)
        self.assertEqual(len(train["images"]), 16)
        self.assertEqual(len(val["images"]), 4)
        self.assertEqual(summary["full"]["negative_images"], 10)
        self.assertEqual(summary["train"]["negative_images"], 8)
        self.assertEqual(summary["val"]["negative_images"], 2)
        self.assertEqual(
            {image["id"] for image in full["images"]},
            set(range(1, 21)),
        )
        self.assertEqual(
            {annotation["id"] for annotation in full["annotations"]},
            set(range(1, 11)),
        )
        self.assertEqual(
            {annotation["category_id"] for annotation in full["annotations"]},
            {1},
        )
        filenames = {image["file_name"] for image in full["images"]}
        self.assertEqual(len(filenames), 20)
        self.assertTrue(any(name.startswith("first__") for name in filenames))
        self.assertTrue(any(name.startswith("second__") for name in filenames))
        self.assertTrue(
            {annotation["image_id"] for annotation in full["annotations"]}
            <= {image["id"] for image in full["images"]}
        )
        train_names = {image["file_name"] for image in train["images"]}
        val_names = {image["file_name"] for image in val["images"]}
        self.assertFalse(train_names & val_names)
        self.assertEqual(train_names | val_names, filenames)

    def test_same_seed_produces_identical_annotations(self):
        builder = load_builder()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            make_source(first, "first")
            make_source(second, "second")
            sources = [
                builder.SourceSpec("first", first, "first__"),
                builder.SourceSpec("second", second, "second__"),
            ]

            builder.merge_datasets(sources, root / "out_a", seed=2026)
            builder.merge_datasets(sources, root / "out_b", seed=2026)

            for filename in (
                "instance_train.json",
                "instance_val.json",
                "instance_train_full.json",
            ):
                self.assertEqual(
                    (root / "out_a" / "annotations" / filename).read_bytes(),
                    (root / "out_b" / "annotations" / filename).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
