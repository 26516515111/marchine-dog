# predict.py keep_ratio retune - 2026-06-15

## Scope

- Official submission script: `D:/work/Marchine Dog/predict.py`.
- Model config: `D:/work/Marchine Dog/model/infer_cfg.yml`.
- The exported model declares `Resize target_size: [768, 768]`, `keep_ratio: true`, and `PadStride stride: 32`.

## Root Cause

Before this change, `predict.py` ignored `keep_ratio=True` and always stretched images to `768x768`.
For a 1080x1920 image, the correct exported-model preprocessing is:

1. resize with ratio kept: `1080x1920 -> 432x768`;
2. `PadStride(32)`: tensor becomes `3x448x768`;
3. `im_shape` must stay `[432, 768]`;
4. `scale_factor` must be `[0.4, 0.4]`.

The batch `create_inputs()` path also previously replaced `im_shape` with the padded batch shape.
That diverges from `PaddleDetection/deploy/python/infer.py`; only `image` should be batch-padded.

## Code Changes

`D:/work/Marchine Dog/predict.py` was updated as follows:

- `Resize.__call__()` now honors `keep_ratio=True`.
- `create_inputs()` now preserves preprocessing `im_shape` and `scale_factor`.
- `image` tensors are still padded to the largest batch shape for batched inference.
- Battery threshold was retuned after the geometry fix.

## Selected Submission Parameters

```python
class_thresholds = {1: 0.50, 2: 0.30, 3: 0.40}
nms_threshold = {
    1: 0.40,
    2: 0.50,
    3: 0.50,
}
bbox_scales = {
    1: (1.00, 1.00),
    2: (1.00, 1.00),
    3: (1.00, 1.00),
}
```

Battery threshold `0.50` removes one low-score FP (`score=0.4529`) while the lowest retained battery TP is still about `0.6926`.

Raw val search also found `fire NMS=0.60` with overall F1 `0.8935`, but the gain over `fire NMS=0.50` was only `+0.0003` and it added fire FP. It was not applied to reduce validation-set overfitting risk.

## Verification

Preprocess regression check:

```bash
D:/Anaconda/envs/dog/python.exe -m py_compile "D:/work/Marchine Dog/predict.py"
```

Official-style validation run:

```bash
D:/Anaconda/envs/dog/python.exe "D:/work/Marchine Dog/predict.py" "D:/work/Marchine Dog/val.txt" "D:/work/Marchine Dog/tmp_keep_ratio_battery_tuned.json"
D:/Anaconda/envs/dog/python.exe "D:/work/Marchine Dog/dog/mycode/calculate_f1.py" "D:/work/Marchine Dog/tmp_keep_ratio_battery_tuned.json" "D:/work/Marchine Dog/dog/A_train/coco/annotations/instance_val.json"
```

Result:

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| battery | 7 | 1 | 1 | 0.8750 | 0.8750 | 0.8750 |
| board | 16 | 1 | 3 | 0.9412 | 0.8421 | 0.8889 |
| fire | 174 | 12 | 20 | 0.9355 | 0.8969 | 0.9158 |

- Overall F1: `0.8932`.
- Runtime: `2.5629s` for 81 images.
- FPS: about `31.6`, above the official `20 FPS` constraint.

## Follow-up Notes

- If retuning again, evaluate both raw scored predictions and no-score submission replay. The local `calculate_f1.py` treats missing scores as `1.0`, so output order can affect matching.
- Do not retune only against the raw scored evaluator and then copy the raw-best parameters blindly.
- Avoid changing `fire NMS` unless the gain is meaningful on a less val-specific split or on cross-validation.
