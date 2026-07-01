import re
import unittest
from pathlib import Path


CONFIG_PATH = Path(r"D:\work\Marchine_Dog\firebig\B_firebig.yml")


class FirebigMergedConfigTests(unittest.TestCase):
    def test_uses_merged_dataset_and_keeps_empty_training_images(self):
        text = CONFIG_PATH.read_text(encoding="utf-8")

        self.assertEqual(
            text.count(
                "dataset_dir: ../A_train/new_label_coco_firebig_merged"
            ),
            2,
        )
        self.assertIn(
            "anno_path: ../A_train/new_label_coco_firebig_merged/"
            "annotations/instance_val.json",
            text,
        )
        train_dataset = re.search(
            r"TrainDataset:(.*?)(?=\nEvalDataset:)",
            text,
            re.DOTALL,
        ).group(1)
        self.assertIn("allow_empty: true", train_dataset)

    def test_training_and_inference_resolutions_are_aligned(self):
        text = CONFIG_PATH.read_text(encoding="utf-8")
        train_reader = re.search(
            r"TrainReader:(.*?)(?=\nEvalReader:)",
            text,
            re.DOTALL,
        ).group(1)
        eval_reader = re.search(
            r"EvalReader:(.*?)(?=\nTestReader:)",
            text,
            re.DOTALL,
        ).group(1)
        test_reader = text.split("\nTestReader:", 1)[1]

        self.assertIn("target_size: [576, 640, 704]", train_reader)
        self.assertIn("target_size: [640, 640]", eval_reader)
        self.assertIn("target_size: [640, 640]", test_reader)


if __name__ == "__main__":
    unittest.main()
