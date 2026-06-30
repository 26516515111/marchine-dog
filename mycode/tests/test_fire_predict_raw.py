import ast
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


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
    def test_configure_gpu_rejects_cpu_paddle(self):
        predict = load_predict()

        class FakeCuda:
            @staticmethod
            def device_count():
                return 0

        class FakePaddle:
            device = type("Device", (), {"cuda": FakeCuda})

            @staticmethod
            def is_compiled_with_cuda():
                return False

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            predict.configure_gpu(object(), FakePaddle)

    def test_configure_gpu_rejects_small_memory_pool(self):
        predict = load_predict()

        class FakeCuda:
            @staticmethod
            def device_count():
                return 1

        class FakePaddle:
            device = type("Device", (), {"cuda": FakeCuda})

            @staticmethod
            def is_compiled_with_cuda():
                return True

        class FakeConfig:
            def enable_use_gpu(self, pool_mb, device_id):
                raise AssertionError("must reject before enabling GPU")

        with patch.dict(os.environ, {"PREDICT_GPU_POOL_MB": "999"}):
            with self.assertRaisesRegex(ValueError, "at least 1000"):
                predict.configure_gpu(FakeConfig(), FakePaddle)

    def test_configure_gpu_enables_device_zero(self):
        predict = load_predict()

        class FakeCuda:
            @staticmethod
            def device_count():
                return 1

        class FakePaddle:
            device = type("Device", (), {"cuda": FakeCuda})

            @staticmethod
            def is_compiled_with_cuda():
                return True

        class FakeConfig:
            enabled = None

            def enable_use_gpu(self, pool_mb, device_id):
                self.enabled = (pool_mb, device_id)

        config = FakeConfig()
        with patch.dict(os.environ, {"PREDICT_GPU_POOL_MB": "2000"}):
            predict.configure_gpu(config, FakePaddle)

        self.assertEqual(config.enabled, (2000, 0))

    def test_paddle_imports_are_at_module_scope(self):
        tree = ast.parse(PREDICT_PATH.read_text(encoding="utf-8"))
        module_imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertTrue(
            any(
                isinstance(node, ast.Import)
                and any(alias.name == "paddle" for alias in node.names)
                for node in module_imports
            )
        )
        for function in [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]:
            nested_imports = [
                node
                for node in ast.walk(function)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            self.assertFalse(
                any(
                    (
                        isinstance(node, ast.Import)
                        and any(alias.name == "paddle" for alias in node.names)
                    )
                    or (
                        isinstance(node, ast.ImportFrom)
                        and node.module
                        and node.module.startswith("paddle")
                    )
                    for node in nested_imports
                ),
                f"Paddle import found inside {function.name}",
            )

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

    def test_predict_images_returns_only_final_result(self):
        predict = load_predict()

        class FakeDetector:
            preprocess_ops = []
            pred_config = type("Config", (), {"id_to_category": {0: "firebig"}})()

            @staticmethod
            def predict(_inputs):
                return {
                    "boxes": np.asarray(
                        [[0.0, 0.9, 10.0, 20.0, 40.0, 60.0]],
                        dtype=np.float32,
                    ),
                    "boxes_num": np.asarray([1], dtype=np.int32),
                }

        image = np.zeros((3, 64, 64), dtype=np.float32)
        info = {
            "origin_shape": np.asarray([100, 100], dtype=np.float32),
            "im_shape": np.asarray([64, 64], dtype=np.float32),
            "scale_factor": np.asarray([0.64, 0.64], dtype=np.float32),
        }
        with (
            patch.object(predict, "preprocess", return_value=(image, info)),
            patch.object(predict, "create_inputs", return_value={}),
        ):
            result = predict.predict_images(FakeDetector(), ["sample_001.jpg"])

        self.assertEqual(list(result), ["result"])
        self.assertEqual(len(result["result"]), 1)

    def test_write_submission_uses_compact_json(self):
        predict = load_predict()
        item = {
            "image_id": "sample_001",
            "type": 1,
            "x": 1.5,
            "y": 2.5,
            "width": 30.0,
            "height": 40.0,
            "segmentation": [],
        }
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"

            predict.write_submission(output, {"result": [item]})

            text = output.read_text(encoding="utf-8")
        self.assertNotIn("\n", text)
        self.assertEqual(json.loads(text), {"result": [item]})

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
