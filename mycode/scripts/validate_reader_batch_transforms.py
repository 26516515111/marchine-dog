import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "PaddleDetection"))

from ppdet.core.workspace import load_config  # noqa: E402


CONFIGS = [
    REPO_ROOT / "PaddleDetection" / "configs" / "custom" / "ppyoloe_fire.yml",
    REPO_ROOT / "PaddleDetection" / "configs" / "custom" / "ppyoloe_fire_a1.yml",
    REPO_ROOT / "PaddleDetection" / "configs" / "custom" / "ppyoloe_fire_hn.yml",
    REPO_ROOT / "PaddleDetection" / "configs" / "custom" / "ppyoloe_plus_fire_c1.yml",
]


def batch_transform_names(config_path: Path):
    cfg = load_config(str(config_path))
    return cfg


def main():
    failed = False
    for config_path in CONFIGS:
        cfg = batch_transform_names(config_path)
        eval_size = cfg.get("eval_size")
        train_names = [next(iter(op.keys())) for op in cfg["TrainReader"]["batch_transforms"]]
        print(f"{config_path.name} TrainReader: {train_names}")
        if "PadGT" not in train_names:
            print(f"ERROR: {config_path.name} missing PadGT in TrainReader.batch_transforms")
            failed = True
        for reader_name in ["EvalReader", "TestReader"]:
            reader = cfg.get(reader_name, {})
            sample_names = [next(iter(op.keys())) for op in reader.get("sample_transforms", [])]
            batch_names = [next(iter(op.keys())) for op in reader.get("batch_transforms", [])]
            inputs_def = reader.get("inputs_def", {})
            image_shape = inputs_def.get("image_shape")
            keep_ratio_true = any(
                "Resize" in op and op["Resize"].get("keep_ratio") is True
                for op in reader.get("sample_transforms", [])
            )
            print(f"{config_path.name} {reader_name}: sample={sample_names} batch={batch_names}")
            if keep_ratio_true and "PadBatch" not in batch_names:
                print(
                    f"ERROR: {config_path.name} {reader_name} uses keep_ratio=True but missing PadBatch"
                )
                failed = True
            if keep_ratio_true and eval_size is not None:
                print(
                    f"ERROR: {config_path.name} {reader_name} uses keep_ratio=True but eval_size is fixed to {eval_size}"
                )
                failed = True
            if reader_name == "TestReader" and keep_ratio_true and image_shape not in (None, [3, -1, -1]):
                print(
                    f"ERROR: {config_path.name} TestReader uses keep_ratio=True but inputs_def.image_shape is fixed to {image_shape}"
                )
                failed = True
    if failed:
        raise SystemExit(1)
    print("Reader batch_transforms validation passed.")


if __name__ == "__main__":
    main()
