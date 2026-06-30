import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDICT_PATH = PROJECT_ROOT / "firebig" / "predict.py"


def load_predict():
    module_name = "firebig_predict_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, PREDICT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FirebigPredictGpuTests(unittest.TestCase):
    @staticmethod
    def fake_paddle(cuda_build, gpu_count):
        class FakeCuda:
            @staticmethod
            def device_count():
                return gpu_count

        class FakePaddle:
            device = type("Device", (), {"cuda": FakeCuda})

            @staticmethod
            def is_compiled_with_cuda():
                return cuda_build

        return FakePaddle

    def test_configure_gpu_rejects_cpu_paddle(self):
        predict = load_predict()

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            predict.configure_gpu(object(), self.fake_paddle(False, 0))

    def test_configure_gpu_rejects_missing_gpu(self):
        predict = load_predict()

        with self.assertRaisesRegex(RuntimeError, "GPU"):
            predict.configure_gpu(object(), self.fake_paddle(True, 0))

    def test_configure_gpu_rejects_small_pool(self):
        predict = load_predict()

        with patch.dict(os.environ, {"PREDICT_GPU_POOL_MB": "999"}):
            with self.assertRaisesRegex(ValueError, "at least 1000"):
                predict.configure_gpu(object(), self.fake_paddle(True, 1))

    def test_configure_gpu_enables_gpu_zero(self):
        predict = load_predict()

        class FakeConfig:
            enabled = None

            def enable_use_gpu(self, pool_mb, device_id):
                self.enabled = (pool_mb, device_id)

        config = FakeConfig()
        with patch.dict(os.environ, {"PREDICT_GPU_POOL_MB": "2000"}):
            predict.configure_gpu(config, self.fake_paddle(True, 1))

        self.assertEqual(config.enabled, (2000, 0))

    def test_paddle_imports_are_outside_timed_functions(self):
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


if __name__ == "__main__":
    unittest.main()
