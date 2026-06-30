import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


WORKSPACE_ROOT = Path(r"D:\work\Marchine_Dog")
PREDICT_PATH = WORKSPACE_ROOT / "fire" / "predict.py"


def load_predict():
    module_name = "fire_predict_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, PREDICT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FirePredictRawTests(unittest.TestCase):
    def test_relative_gate_rejects_low_confidence_large_box(self):
        predict = load_predict()
        detections = [
            {"category": "fire", "score": 0.90, "bbox": [0, 0, 20, 20]},
            {"category": "fire", "score": 0.40, "bbox": [0, 0, 100, 100]},
            {"category": "fire", "score": 0.60, "bbox": [0, 0, 40, 40]},
        ]

        selected = predict.select_largest_credible_fire(detections)

        self.assertEqual(selected["score"], 0.60)

    def test_relative_gate_accepts_firebig_export_label(self):
        predict = load_predict()

        selected = predict.select_largest_credible_fire(
            [{"category": "firebig", "score": 0.8, "bbox": [1, 2, 30, 40]}]
        )

        self.assertEqual(selected["bbox"], [1, 2, 30, 40])

    def test_relative_gate_equal_area_prefers_higher_score(self):
        predict = load_predict()
        detections = [
            {"category": "fire", "score": 0.7, "bbox": [0, 0, 20, 20]},
            {"category": "fire", "score": 0.8, "bbox": [5, 5, 10, 40]},
        ]

        selected = predict.select_largest_credible_fire(detections)

        self.assertEqual(selected["score"], 0.8)

    def test_relative_gate_ignores_non_fire_and_degenerate_boxes(self):
        predict = load_predict()
        detections = [
            {"category": "smoke", "score": 0.99, "bbox": [0, 0, 100, 100]},
            {"category": "fire", "score": 0.95, "bbox": [0, 0, 0, 100]},
            {"category": "fire", "score": 0.70, "bbox": [1, 2, 10, 20]},
        ]

        selected = predict.select_largest_credible_fire(detections)

        self.assertEqual(selected["bbox"], [1, 2, 10, 20])

    def test_build_preprocess_ops_uses_only_declared_operators(self):
        predict = load_predict()
        infos = [
            {"type": "Resize", "target_size": [640, 640], "keep_ratio": True, "interp": 2},
            {
                "type": "NormalizeImage",
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
                "norm_type": "none",
            },
            {"type": "Permute"},
            {"type": "PadStride", "stride": 32},
        ]

        ops = predict.build_preprocess_ops(infos)

        self.assertEqual(
            [type(op).__name__ for op in ops],
            ["Resize", "NormalizeImage", "Permute", "PadStride"],
        )
        self.assertFalse(hasattr(predict, "_hsv_enhance_atmosphere"))

    def test_build_preprocess_ops_rejects_unknown_operator(self):
        predict = load_predict()

        with self.assertRaisesRegex(ValueError, "Unsupported preprocess operator"):
            predict.build_preprocess_ops([{"type": "MadeUpEnhancement"}])

    def test_predict_config_reads_exported_labels_and_preprocess(self):
        predict = load_predict()
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            config = {
                "Preprocess": [{"type": "Permute"}],
                "label_list": ["firebig"],
            }
            (model_dir / "infer_cfg.yml").write_text(
                "Preprocess:\n- type: Permute\nlabel_list:\n- firebig\n",
                encoding="utf-8",
            )

            parsed = predict.PredictConfig(model_dir)

        self.assertEqual(parsed.preprocess_infos, config["Preprocess"])
        self.assertEqual(parsed.id_to_category, {0: "firebig"})

    def test_format_firebig_result_has_exact_submission_fields(self):
        predict = load_predict()

        result = predict.format_firebig_result(
            "sample_001.jpg",
            {"bbox": [1.5, 2.5, 30.0, 40.0], "score": 0.75},
        )

        self.assertEqual(
            result,
            {
                "image_id": "sample_001",
                "type": 1,
                "x": 1.5,
                "y": 2.5,
                "width": 30.0,
                "height": 40.0,
                "segmentation": [],
            },
        )

    def test_format_submission_excludes_diagnostic_top_level_fields(self):
        predict = load_predict()
        internal_predictions = {
            "result": [
                {
                    "image_id": "sample_001",
                    "type": 1,
                    "x": 1.5,
                    "y": 2.5,
                    "width": 30.0,
                    "height": 40.0,
                    "segmentation": [],
                }
            ],
            "raw_candidates": {"sample_001": [{"score": 0.75}]},
            "unreadable_images": [],
        }

        submission = predict.format_submission(internal_predictions)

        self.assertEqual(list(submission), ["result"])
        self.assertEqual(
            set(submission["result"][0]),
            {"image_id", "type", "x", "y", "width", "height", "segmentation"},
        )

    def test_xyxy_conversion_clips_and_rejects_degenerate_box(self):
        predict = load_predict()

        self.assertEqual(
            predict._xyxy_to_xywh([-5, -4, 120, 80], 100, 60),
            [0.0, 0.0, 100.0, 60.0],
        )
        self.assertIsNone(predict._xyxy_to_xywh([10, 10, 10, 20], 100, 60))

    def test_source_has_no_atmosphere_or_merge_postprocessing(self):
        source = PREDICT_PATH.read_text(encoding="utf-8")

        self.assertNotIn("ATMOS_", source)
        self.assertNotIn("merge_close_fire_boxes", source)
        self.assertNotIn("_saturation_score_boost", source)


if __name__ == "__main__":
    unittest.main()
