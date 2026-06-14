# -*- coding: utf-8 -*-
import importlib.util
import pathlib
import unittest


SCRIPT_PATH = pathlib.Path(__file__).with_name("tune_fire_postprocess.py")


def load_module():
    spec = importlib.util.spec_from_file_location("tune_fire_postprocess", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FirePostprocessTest(unittest.TestCase):
    def test_fire_min_area_filters_only_fire(self):
        mod = load_module()
        preds = [
            {"image_id": "img001", "type": 3, "bbox": [0, 0, 5, 5], "score": 0.9},
            {"image_id": "img001", "type": 3, "bbox": [20, 20, 20, 20], "score": 0.8},
            {"image_id": "img001", "type": 1, "bbox": [40, 40, 5, 5], "score": 0.7},
        ]

        filtered = mod.apply_postprocess(
            preds,
            thresholds={1: 0.0, 2: 0.0, 3: 0.0},
            nms_thresholds={1: 1.0, 2: 1.0, 3: 1.0},
            fire_min_area=100.0,
        )

        self.assertEqual([(p["type"], p["bbox"]) for p in filtered], [
            (3, [20, 20, 20, 20]),
            (1, [40, 40, 5, 5]),
        ])

    def test_fire_nms_suppresses_same_image_fire_only(self):
        mod = load_module()
        preds = [
            {"image_id": "img001", "type": 3, "bbox": [0, 0, 100, 100], "score": 0.9},
            {"image_id": "img001", "type": 3, "bbox": [10, 10, 100, 100], "score": 0.8},
            {"image_id": "img001", "type": 2, "bbox": [10, 10, 100, 100], "score": 0.7},
        ]

        filtered = mod.apply_postprocess(
            preds,
            thresholds={1: 0.0, 2: 0.0, 3: 0.0},
            nms_thresholds={1: 1.0, 2: 1.0, 3: 0.5},
            fire_min_area=0.0,
        )

        self.assertEqual([(p["type"], p["score"]) for p in filtered], [
            (3, 0.9),
            (2, 0.7),
        ])


if __name__ == "__main__":
    unittest.main()
