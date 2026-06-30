import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))


def coco_with_box(bbox):
    return {
        "images": [
            {
                "id": 1,
                "file_name": "sample_001.jpg",
                "width": 100,
                "height": 100,
            }
        ],
        "categories": [{"id": 1, "name": "firebig"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": bbox,
            }
        ],
    }


def task_prediction(bbox, score=0.8):
    return {
        "image_id": "sample_001",
        "type": 1,
        "x": bbox[0],
        "y": bbox[1],
        "width": bbox[2],
        "height": bbox[3],
        "segmentation": [],
        "score": score,
    }


class EvaluateFirePredictionsTests(unittest.TestCase):
    def test_iou_equal_to_half_is_fp_and_fn(self):
        from mycode.evaluate_fire_predictions import evaluate

        metrics, rows = evaluate(
            coco_with_box([0, 0, 3, 1]),
            [task_prediction([1, 0, 3, 1])],
            iou_threshold=0.5,
        )

        self.assertEqual(metrics["true_positives"], 0)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(rows[0]["status"], "FP")
        self.assertEqual(rows[0]["iou"], 0.5)

    def test_perfect_prediction_has_f1_one(self):
        from mycode.evaluate_fire_predictions import evaluate

        metrics, rows = evaluate(
            coco_with_box([10, 20, 30, 40]),
            [task_prediction([10, 20, 30, 40], score=0.9)],
        )

        self.assertEqual(metrics["images"], 1)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(rows[0]["status"], "TP")

    def test_missing_prediction_is_fn(self):
        from mycode.evaluate_fire_predictions import evaluate

        metrics, rows = evaluate(coco_with_box([10, 20, 30, 40]), [])

        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(rows[0]["status"], "FN")
        self.assertIsNone(rows[0]["prediction_bbox"])

    def test_fixed_strategies_use_raw_candidates_without_tuning(self):
        from mycode.evaluate_fire_predictions import predictions_for_strategy

        raw_candidates = {
            "sample_001": [
                {"category": "firebig", "score": 0.9, "bbox": [0, 0, 20, 20]},
                {"category": "firebig", "score": 0.4, "bbox": [0, 0, 100, 100]},
                {"category": "firebig", "score": 0.6, "bbox": [0, 0, 40, 40]},
            ]
        }

        highest = predictions_for_strategy(raw_candidates, "highest_confidence")
        largest = predictions_for_strategy(raw_candidates, "largest_area")
        relative = predictions_for_strategy(raw_candidates, "relative_gate")

        self.assertEqual(highest[0]["score"], 0.9)
        self.assertEqual(largest[0]["score"], 0.4)
        self.assertEqual(relative[0]["score"], 0.6)

    def test_render_false_positives_writes_annotated_image_and_summary(self):
        from mycode.evaluate_fire_predictions import render_false_positives

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            output_dir = root / "evaluation"
            image_dir.mkdir()
            cv2.imwrite(
                str(image_dir / "sample_001.jpg"),
                np.zeros((100, 100, 3), dtype=np.uint8),
            )
            rows = [
                {
                    "image_id": "sample_001",
                    "file_name": "sample_001.jpg",
                    "status": "FP",
                    "gt_bbox": [10, 10, 20, 20],
                    "prediction_bbox": [50, 50, 30, 30],
                    "score": 0.7,
                    "iou": 0.0,
                }
            ]

            summary = render_false_positives(rows, image_dir, output_dir)

            rendered = output_dir / "fp_images" / "sample_001.jpg"
            self.assertTrue(rendered.is_file())
            self.assertGreater(rendered.stat().st_size, 0)
            saved = json.loads(
                (output_dir / "fp_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved, summary)
            self.assertEqual(saved["false_positives"][0]["score"], 0.7)
            self.assertEqual(saved["false_positives"][0]["iou"], 0.0)


if __name__ == "__main__":
    unittest.main()
