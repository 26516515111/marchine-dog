"""Run the exported fire detector and output one ``firebig`` box per image.

Usage:
    python predict.py <image_list.txt> <result.json> [model_dir]

Preprocessing is built only from ``model_dir/infer_cfg.yml``. The exported
model already applies NMS. From those unmodified candidates, this script keeps
boxes whose score is at least ``max(output_score, 0.6 * best_score)`` and
outputs the largest remaining fire box.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import paddle
from paddle.inference import Config, create_predictor


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "model"
ABSOLUTE_SCORE_FLOOR = 0.02
RELATIVE_SCORE_RATIO = 0.5
FIRE_LABELS = {"fire", "firebig"}
MIN_OUTPUT_SCORE = float(os.getenv("FIRE_MIN_OUTPUT_SCORE", "0.1"))


def configure_gpu(config: Config, paddle_module=paddle) -> None:
    """Require CUDA and configure a stable GPU memory pool."""
    if not paddle_module.is_compiled_with_cuda():
        raise RuntimeError(
            "CUDA-enabled Paddle is required; CPU fallback is disabled"
        )
    if paddle_module.device.cuda.device_count() < 1:
        raise RuntimeError("CUDA GPU is not available; CPU fallback is disabled")
    gpu_pool_mb = int(os.getenv("PREDICT_GPU_POOL_MB", "2000"))
    if gpu_pool_mb < 1000:
        raise ValueError("PREDICT_GPU_POOL_MB must be at least 1000")
    config.enable_use_gpu(gpu_pool_mb, 0)


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


def select_largest_credible_fire(
    detections: list[dict[str, Any]],
    absolute_score_floor: float = ABSOLUTE_SCORE_FLOOR,
    relative_score_ratio: float = RELATIVE_SCORE_RATIO,
    minimum_output_score: float = MIN_OUTPUT_SCORE,
) -> dict[str, Any] | None:
    """Select the largest box whose confidence is credible for this image."""
    if not 0.0 <= relative_score_ratio <= 1.0:
        raise ValueError("relative_score_ratio must be between 0 and 1")
    if absolute_score_floor < 0.0:
        raise ValueError("absolute_score_floor must be non-negative")
    if minimum_output_score < 0.0:
        raise ValueError("minimum_output_score must be non-negative")

    candidates: list[tuple[int, dict[str, Any]]] = []
    for original_index, detection in enumerate(detections):
        bbox = detection.get("bbox")
        if str(detection.get("category", "")) not in FIRE_LABELS:
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if _bbox_area([float(value) for value in bbox]) <= 0.0:
            continue
        candidates.append((original_index, detection))

    if not candidates:
        return None

    max_score = max(float(detection.get("score", 0.0)) for _, detection in candidates)
    threshold = max(
        float(absolute_score_floor),
        float(minimum_output_score),
        float(relative_score_ratio) * max_score,
    )
    credible = [
        (index, detection)
        for index, detection in candidates
        if float(detection.get("score", 0.0)) >= threshold
    ]
    if not credible:
        return None

    _, selected = max(
        credible,
        key=lambda item: (
            _bbox_area(item[1]["bbox"]),
            float(item[1].get("score", 0.0)),
            -item[0],
        ),
    )
    return selected.copy()


def format_firebig_result(image_id: str, selected: dict[str, Any]) -> dict[str, Any]:
    x, y, width, height = [float(value) for value in selected["bbox"]]
    return {
        "image_id": Path(image_id).stem,
        "type": 1,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "segmentation": [],
    }


def format_submission(predictions: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return the exact JSON contract accepted by the evaluator."""
    fields = ("image_id", "type", "x", "y", "width", "height", "segmentation")
    return {
        "result": [
            {field: prediction[field] for field in fields}
            for prediction in predictions["result"]
        ]
    }


def write_submission(
    output_path: str | Path,
    submission: dict[str, list[dict[str, Any]]],
) -> None:
    """Write the exact evaluator schema without formatting overhead."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(submission, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def get_test_images(infer_file: str | Path) -> list[str]:
    infer_path = Path(infer_file)
    candidates = (infer_path, Path.cwd() / infer_path, BASE_DIR / infer_path)
    for candidate in candidates:
        if candidate.is_file():
            infer_path = candidate.resolve()
            break
    else:
        raise FileNotFoundError(f"image list not found: {infer_file}")

    image_paths: list[str] = []
    for raw_line in infer_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        path = Path(line)
        path_candidates = (
            path,
            infer_path.parent / path,
            Path.cwd() / path,
            BASE_DIR / path,
        )
        resolved = next((item.resolve() for item in path_candidates if item.is_file()), None)
        image_paths.append(str(resolved) if resolved is not None else line)
    return image_paths


class Resize:
    def __init__(
        self,
        target_size: int | list[int],
        keep_ratio: bool = True,
        interp: int | None = None,
    ):
        import cv2

        self.target_size = (
            [target_size, target_size] if isinstance(target_size, int) else list(target_size)
        )
        self.keep_ratio = bool(keep_ratio)
        self.interp = cv2.INTER_LINEAR if interp is None else int(interp)

    def __call__(self, image, image_info):
        import cv2
        import numpy as np

        target_height, target_width = self.target_size
        source_height, source_width = image.shape[:2]
        if self.keep_ratio:
            scale = min(
                float(target_height) / float(source_height),
                float(target_width) / float(source_width),
            )
            resize_height = max(1, int(round(source_height * scale)))
            resize_width = max(1, int(round(source_width * scale)))
            scale_y = scale_x = scale
        else:
            resize_height = int(target_height)
            resize_width = int(target_width)
            scale_y = resize_height / float(source_height)
            scale_x = resize_width / float(source_width)

        resized = cv2.resize(
            image,
            (resize_width, resize_height),
            interpolation=self.interp,
        )
        image_info["im_shape"] = np.asarray(
            [resize_height, resize_width], dtype=np.float32
        )
        image_info["scale_factor"] = np.asarray([scale_y, scale_x], dtype=np.float32)
        return resized, image_info


class NormalizeImage:
    def __init__(
        self,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        is_scale: bool = True,
        norm_type: str | None = None,
        **_kwargs,
    ):
        import numpy as np

        self.mean = np.asarray(mean or [0.485, 0.456, 0.406], dtype=np.float32).reshape(
            1, 1, 3
        )
        self.std = np.asarray(std or [0.229, 0.224, 0.225], dtype=np.float32).reshape(
            1, 1, 3
        )
        self.is_scale = bool(is_scale)
        self.norm_type = norm_type

    def __call__(self, image, image_info):
        image = image.astype("float32")
        if self.is_scale or self.norm_type == "none":
            image = image / 255.0
        image = (image - self.mean) / self.std
        return image, image_info


class Permute:
    def __init__(self, to_bgr: bool = False):
        self.to_bgr = bool(to_bgr)

    def __call__(self, image, image_info):
        image = image.transpose((2, 0, 1))
        if self.to_bgr:
            image = image[[2, 1, 0], :, :]
        return image, image_info


class PadStride:
    def __init__(self, stride: int = 32):
        self.stride = int(stride)

    def __call__(self, image, image_info):
        import numpy as np

        if self.stride <= 0:
            return image, image_info
        channels, height, width = image.shape
        padded_height = (height + self.stride - 1) // self.stride * self.stride
        padded_width = (width + self.stride - 1) // self.stride * self.stride
        if (padded_height, padded_width) == (height, width):
            return image, image_info
        padded = np.zeros(
            (channels, padded_height, padded_width),
            dtype=image.dtype,
        )
        padded[:, :height, :width] = image
        return padded, image_info


class PredictConfig:
    def __init__(self, model_dir: str | Path):
        import yaml

        deploy_file = Path(model_dir) / "infer_cfg.yml"
        if not deploy_file.is_file():
            raise FileNotFoundError(f"inference config not found: {deploy_file}")
        config = yaml.safe_load(deploy_file.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict) or "Preprocess" not in config:
            raise ValueError(f"invalid inference config: {deploy_file}")
        self.preprocess_infos = config["Preprocess"]
        self.labels = list(config.get("label_list") or ["fire"])
        self.id_to_category = dict(enumerate(self.labels))


def build_preprocess_ops(preprocess_infos: list[dict[str, Any]]) -> list[Any]:
    operator_types = {
        "Resize": Resize,
        "NormalizeImage": NormalizeImage,
        "Permute": Permute,
        "PadStride": PadStride,
    }
    operators: list[Any] = []
    for info in preprocess_infos:
        if not isinstance(info, dict) or "type" not in info:
            raise ValueError(f"invalid preprocess entry: {info!r}")
        operator_name = str(info["type"])
        operator_type = operator_types.get(operator_name)
        if operator_type is None:
            raise ValueError(f"Unsupported preprocess operator: {operator_name}")
        parameters = {key: value for key, value in info.items() if key != "type"}
        operators.append(operator_type(**parameters))
    return operators


def preprocess(image_path: str, preprocess_ops: list[Any]):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None, None

    original_shape = np.asarray(image.shape[:2], dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_info = {
        "im_shape": original_shape.copy(),
        "scale_factor": np.asarray([1.0, 1.0], dtype=np.float32),
        "origin_shape": original_shape,
    }
    for operator in preprocess_ops:
        image, image_info = operator(image, image_info)
    return image, image_info


def create_inputs(image_list, info_list):
    import numpy as np

    if not image_list:
        raise ValueError("image_list must not be empty")
    max_height = max(image.shape[1] for image in image_list)
    max_width = max(image.shape[2] for image in image_list)
    padded_images = []
    for image in image_list:
        channels, height, width = image.shape
        padded = np.zeros((channels, max_height, max_width), dtype=np.float32)
        padded[:, :height, :width] = image
        padded_images.append(padded)
    return {
        "image": np.stack(padded_images).astype("float32"),
        "im_shape": np.stack([info["im_shape"] for info in info_list]).astype("float32"),
        "scale_factor": np.stack(
            [info["scale_factor"] for info in info_list]
        ).astype("float32"),
    }


class Detector:
    def __init__(self, pred_config: PredictConfig, model_dir: str | Path):
        model_dir = Path(model_dir)
        model_file = model_dir / "model.pdmodel"
        params_file = model_dir / "model.pdiparams"
        for required_file in (model_file, params_file):
            if not required_file.is_file():
                raise FileNotFoundError(f"model file not found: {required_file}")

        config = Config(str(model_file), str(params_file))
        configure_gpu(config)
        config.enable_memory_optim()
        config.switch_ir_optim(False)
        config.switch_use_feed_fetch_ops(False)
        config.disable_glog_info()

        self.predictor = create_predictor(config)
        self.pred_config = pred_config
        self.preprocess_ops = build_preprocess_ops(pred_config.preprocess_infos)

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        for name in self.predictor.get_input_names():
            if name not in inputs:
                raise KeyError(f"model input {name!r} was not prepared")
            self.predictor.get_input_handle(name).copy_from_cpu(inputs[name])
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        if len(output_names) < 2:
            raise RuntimeError(f"expected boxes and boxes_num outputs, got {output_names}")
        boxes = self.predictor.get_output_handle(output_names[0]).copy_to_cpu()
        boxes_num = self.predictor.get_output_handle(output_names[len(output_names) // 2]).copy_to_cpu()
        return {"boxes": boxes, "boxes_num": boxes_num}


def _xyxy_to_xywh(
    box: list[float],
    image_width: int,
    image_height: int,
) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = min(max(0.0, x1), float(image_width))
    y1 = min(max(0.0, y1), float(image_height))
    x2 = min(max(0.0, x2), float(image_width))
    y2 = min(max(0.0, y2), float(image_height))
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        return None
    return [x1, y1, width, height]


def predict_images(
    detector: Detector,
    image_list: list[str],
) -> dict[str, Any]:
    batch_size = max(1, int(os.getenv("PREDICT_BATCH_SIZE", "48")))
    final_results: list[dict[str, Any]] = []

    for start in range(0, len(image_list), batch_size):
        batch_paths = image_list[start : start + batch_size]
        input_images = []
        input_infos = []
        valid_paths = []
        for image_path in batch_paths:
            image, info = preprocess(image_path, detector.preprocess_ops)
            if image is None:
                continue
            input_images.append(image)
            input_infos.append(info)
            valid_paths.append(image_path)
        if not valid_paths:
            continue

        inference = detector.predict(create_inputs(input_images, input_infos))
        box_offset = 0
        for batch_index, image_path in enumerate(valid_paths):
            image_height, image_width = [
                int(value) for value in input_infos[batch_index]["origin_shape"]
            ]
            box_count = int(inference["boxes_num"][batch_index])
            model_rows = inference["boxes"][box_offset : box_offset + box_count]
            box_offset += box_count

            detections: list[dict[str, Any]] = []
            for row in model_rows:
                class_index = int(row[0])
                bbox = _xyxy_to_xywh(
                    [float(value) for value in row[2:6]],
                    image_width,
                    image_height,
                )
                if bbox is None:
                    continue
                category = detector.pred_config.id_to_category.get(
                    class_index, str(class_index)
                )
                detections.append(
                    {
                        "image_id": Path(image_path).stem,
                        "category": category,
                        "category_id": class_index + 1,
                        "score": float(row[1]),
                        "bbox": bbox,
                    }
                )

            image_key = Path(image_path).stem
            selected = select_largest_credible_fire(detections)
            if selected is not None:
                final_results.append(format_firebig_result(image_key, selected))

    return {"result": final_results}


def main(
    infer_txt: str,
    result_path: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> None:
    paddle.enable_static()
    pred_config = PredictConfig(model_dir)
    detector = Detector(pred_config, model_dir)
    image_list = get_test_images(infer_txt)
    predictions = predict_images(detector, image_list)
    submission = format_submission(predictions)
    output_path = Path(result_path)
    write_submission(output_path, submission)
    print(
        f"Processed {len(image_list)} images; "
        f"wrote {len(submission['result'])} predictions to {output_path}"
    )


if __name__ == "__main__":
    started = time.time()
    if len(sys.argv) < 3:
        print("Usage: python predict.py <image_list.txt> <result.json> [model_dir]")
        raise SystemExit(2)
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3] if len(sys.argv) >= 4 else str(DEFAULT_MODEL_DIR),
    )
    print(f"Total time: {time.time() - started:.3f}s")
