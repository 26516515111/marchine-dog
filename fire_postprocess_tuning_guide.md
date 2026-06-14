# Fire Postprocess Tuning Guide

This note documents the script for tuning fire post-processing without blindly overfitting the validation set.

## Goal

Use `mycode/scripts/tune_fire_postprocess.py` to search:

- fire score threshold
- fire NMS IoU threshold
- fire small-box minimum area

The script reports both:

- `best_raw`: the highest validation F1 in the search space
- `recommended_stable`: a conservative choice near the best score

Prefer `recommended_stable` first. It penalizes overly high thresholds, overly strict NMS, large min-area filtering, and clear fire recall drops.

## Generate Raw Scored Predictions

The official submission JSON usually does not contain `score`, so it cannot be used for threshold tuning directly.

Use the real submission script to export raw scored predictions:

```bash
python mycode/scripts/tune_fire_postprocess.py --predict-py "D:/work/Marchine Dog/predict.py" --infer-txt val.txt --gt A_train/coco/annotations/instance_val.json --raw-out raw_preds_with_score.json --out fire_postprocess_tuning.json
```

By default, the model directory is resolved as `D:/work/Marchine Dog/model`. Override it when needed:

```bash
python mycode/scripts/tune_fire_postprocess.py --predict-py "D:/work/Marchine Dog/predict.py" --model-dir "D:/work/Marchine Dog/model" --infer-txt val.txt --gt A_train/coco/annotations/instance_val.json
```

## Reuse Existing Raw Predictions

If `raw_preds_with_score.json` already exists:

```bash
python mycode/scripts/tune_fire_postprocess.py --pred-json raw_preds_with_score.json --gt A_train/coco/annotations/instance_val.json --out fire_postprocess_tuning.json
```

## Practical Search Range

The defaults are intentionally small and conservative:

- `fire_thresholds`: `0.30,0.35,0.40,0.45,0.50,0.55`
- `fire_nms_values`: `0.45,0.50,0.55,0.60`
- `fire_min_areas`: `0,50,80,120,160,200`

For quick score improvement when fire FP is high, inspect candidates around:

- fire threshold: `0.40` to `0.50`
- fire NMS: `0.45` to `0.55`
- fire min area: `50` to `120`

Avoid jumping directly to extreme values unless the hidden test result confirms the trend.

## Verification

Run:

```bash
python mycode/scripts/test_tune_fire_postprocess.py
python -m py_compile mycode/scripts/tune_fire_postprocess.py mycode/scripts/test_tune_fire_postprocess.py
```
