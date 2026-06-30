# Fire Prediction F1 and FP Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `fire/predict.py` with configuration-faithful raw inference, select the largest credible fire through fixed relative-confidence gating, evaluate all 100 sample images, and render annotated false positives.

**Architecture:** `fire/predict.py` owns model configuration, preprocessing, Paddle inference, one-box `firebig` selection, and optional retention of unmodified model candidates for diagnostics. A separate `dog/mycode/evaluate_fire_predictions.py` module owns strict-IoU evaluation, fixed-strategy comparisons, and annotated FP rendering so evaluation concerns do not leak into deployment inference.

**Tech Stack:** Python 3.12, Paddle Inference, OpenCV, NumPy, PyYAML, `unittest`.

---

## File map

- Modify `D:\work\Marchine_Dog\fire\predict.py`: configuration-driven preprocessing, raw inference, relative-confidence gating, and task JSON output.
- Create `D:\work\Marchine_Dog\dog\mycode\evaluate_fire_predictions.py`: F1 calculation, strategy comparison, and FP visualization.
- Create `D:\work\Marchine_Dog\dog\mycode\tests\test_fire_predict_raw.py`: selection, preprocessing, format, and coordinate tests.
- Create `D:\work\Marchine_Dog\dog\mycode\tests\test_evaluate_fire_predictions.py`: strict IoU, metric, and FP-rendering tests.
- Create runtime artifacts under `D:\work\Marchine_Dog\dog\A_train\sample100_coco\fire_model_evaluation`: predictions, metrics, FP summary, and annotated images.

### Task 1: Relative-confidence largest-fire selection

**Files:**
- Modify: `D:\work\Marchine_Dog\fire\predict.py`
- Create: `D:\work\Marchine_Dog\dog\mycode\tests\test_fire_predict_raw.py`

- [ ] **Step 1: Write failing selection tests**

Load `fire/predict.py` with `importlib.util` and add tests equivalent to:

```python
def test_relative_gate_rejects_low_confidence_large_box(self):
    detections = [
        {"category": "fire", "score": 0.90, "bbox": [0, 0, 20, 20]},
        {"category": "fire", "score": 0.40, "bbox": [0, 0, 100, 100]},
        {"category": "fire", "score": 0.60, "bbox": [0, 0, 40, 40]},
    ]
    selected = predict.select_largest_credible_fire(detections)
    self.assertEqual(selected["score"], 0.60)

def test_relative_gate_accepts_firebig_export_label(self):
    selected = predict.select_largest_credible_fire(
        [{"category": "firebig", "score": 0.8, "bbox": [1, 2, 30, 40]}]
    )
    self.assertEqual(selected["bbox"], [1, 2, 30, 40])

def test_equal_area_prefers_higher_score(self):
    detections = [
        {"category": "fire", "score": 0.7, "bbox": [0, 0, 20, 20]},
        {"category": "fire", "score": 0.8, "bbox": [5, 5, 10, 40]},
    ]
    self.assertEqual(predict.select_largest_credible_fire(detections)["score"], 0.8)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest dog.mycode.tests.test_fire_predict_raw -v
```

Expected: FAIL because `select_largest_credible_fire` does not exist.

- [ ] **Step 3: Replace heuristic postprocessing with the minimal selector**

Delete atmosphere HSV enhancement, saturation scoring, weighted-area scoring, close-box merging, and force-selection code. Implement:

```python
ABSOLUTE_SCORE_FLOOR = 0.02
RELATIVE_SCORE_RATIO = 0.5

def select_largest_credible_fire(
    detections,
    absolute_score_floor=ABSOLUTE_SCORE_FLOOR,
    relative_score_ratio=RELATIVE_SCORE_RATIO,
):
    candidates = [
        det for det in detections
        if det.get("category") in {"fire", "firebig"}
        and len(det.get("bbox", [])) == 4
        and float(det["bbox"][2]) > 0
        and float(det["bbox"][3]) > 0
    ]
    if not candidates:
        return None
    max_score = max(float(det["score"]) for det in candidates)
    threshold = max(float(absolute_score_floor), float(relative_score_ratio) * max_score)
    credible = [det for det in candidates if float(det["score"]) >= threshold]
    return max(
        enumerate(credible),
        key=lambda item: (
            float(item[1]["bbox"][2]) * float(item[1]["bbox"][3]),
            float(item[1]["score"]),
            -item[0],
        ),
    )[1].copy()
```

- [ ] **Step 4: Run selection tests and verify GREEN**

Run the Task 1 unittest command. Expected: all Task 1 tests PASS.

### Task 2: Configuration-faithful preprocessing and output

**Files:**
- Modify: `D:\work\Marchine_Dog\fire\predict.py`
- Modify: `D:\work\Marchine_Dog\dog\mycode\tests\test_fire_predict_raw.py`

- [ ] **Step 1: Write failing configuration and formatting tests**

Add tests that create a temporary `infer_cfg.yml`, instantiate `PredictConfig`, and assert:

```python
self.assertEqual(
    [type(op).__name__ for op in predict.build_preprocess_ops(config.preprocess_infos)],
    ["Resize", "NormalizeImage", "Permute", "PadStride"],
)
self.assertFalse(hasattr(predict, "_hsv_enhance_atmosphere"))

result = predict.format_firebig_result(
    "sample_001.jpg",
    {"bbox": [1.5, 2.5, 30.0, 40.0], "score": 0.75},
)
self.assertEqual(result["image_id"], "sample_001")
self.assertEqual(result["type"], 1)
self.assertEqual(result["score"], 0.75)
```

Also test that `_xyxy_to_xywh` clips coordinates and returns `None` for a degenerate box, and
that `predict_images()` returns both final `result` and unchanged per-image `raw_candidates`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest dog.mycode.tests.test_fire_predict_raw -v
```

Expected: FAIL because the old atmosphere symbol exists and output/degenerate behavior differs.

- [ ] **Step 3: Implement the configuration-faithful pipeline**

Keep only operators declared in `infer_cfg.yml`. `preprocess()` must read BGR, convert once to RGB, then apply configured operators without image enhancement. `predict_images()` must:

```python
selected = select_largest_credible_fire(image_detections)
if selected is not None:
    results.append(format_firebig_result(Path(image_path).name, selected))
```

The formatter must output:

```python
{
    "image_id": Path(image_id).stem,
    "type": 1,
    "x": float(x),
    "y": float(y),
    "width": float(w),
    "height": float(h),
    "segmentation": [],
    "score": float(selected["score"]),
}
```

Keep image-list resolution, dynamic batch padding, GPU/CPU model loading, and `infer_cfg.yml` label mapping. Add CPU fallback when Paddle reports no CUDA device. Reject unknown preprocess operators with `ValueError` instead of silently ignoring them.

Return:

```python
{
    "result": final_results,
    "raw_candidates": {
        Path(image_path).stem: image_detections,
    },
}
```

`raw_candidates` is diagnostic metadata only. Candidate coordinates and scores must be copied
directly from the model output after the required xyxy-to-xywh conversion; the selector must not
mutate them.

- [ ] **Step 4: Run all predictor tests and verify GREEN**

Run the Task 2 unittest command. Expected: all tests PASS with no warnings or errors.

### Task 3: Strict F1 evaluation and fixed baselines

**Files:**
- Create: `D:\work\Marchine_Dog\dog\mycode\evaluate_fire_predictions.py`
- Create: `D:\work\Marchine_Dog\dog\mycode\tests\test_evaluate_fire_predictions.py`

- [ ] **Step 1: Write failing metric tests**

Cover strict boundary and unmatched prediction behavior:

```python
def test_iou_equal_to_half_is_false_positive_and_false_negative(self):
    metrics, rows = evaluate(coco_with_box([0, 0, 10, 10]), [prediction_with_iou_half()])
    self.assertEqual(metrics["true_positives"], 0)
    self.assertEqual(metrics["false_positives"], 1)
    self.assertEqual(metrics["false_negatives"], 1)

def test_perfect_prediction_has_f1_one(self):
    metrics, rows = evaluate(coco_with_box([0, 0, 10, 10]), [prediction([0, 0, 10, 10])])
    self.assertEqual(metrics["f1"], 1.0)
    self.assertEqual(rows[0]["status"], "TP")
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest dog.mycode.tests.test_evaluate_fire_predictions -v
```

Expected: import failure because `evaluate_fire_predictions.py` does not exist.

- [ ] **Step 3: Implement strict evaluation**

Implement `bbox_iou_xywh`, COCO image/annotation indexing, prediction parsing for the task JSON, and:

```python
is_tp = iou > iou_threshold
if is_tp:
    tp += 1
else:
    fp += int(prediction is not None)
    fn += 1
```

Return a metrics dictionary and one per-image row containing filename, GT bbox, prediction bbox,
score, IoU, and `TP`/`FP`/`FN` status.

Implement fixed baseline selectors operating on saved raw candidates:

- highest confidence;
- largest area;
- relative gate then largest area.

These are reported as diagnostics only; no parameter search is allowed.

- [ ] **Step 4: Run metric tests and verify GREEN**

Run Task 3 unittest command. Expected: all metric tests PASS.

### Task 4: Annotated FP output

**Files:**
- Modify: `D:\work\Marchine_Dog\dog\mycode\evaluate_fire_predictions.py`
- Modify: `D:\work\Marchine_Dog\dog\mycode\tests\test_evaluate_fire_predictions.py`

- [ ] **Step 1: Write failing rendering test**

Create a temporary 100×100 black JPEG and one FP row. Assert that `render_false_positives`
creates one non-empty JPEG and `fp_summary.json` whose item contains filename, score, IoU,
GT bbox, and prediction bbox.

- [ ] **Step 2: Run rendering test and verify RED**

Run Task 3 unittest command. Expected: FAIL because the renderer is missing.

- [ ] **Step 3: Implement FP rendering**

Use OpenCV with:

```python
cv2.rectangle(image, gt_p1, gt_p2, (0, 255, 0), 2)
cv2.rectangle(image, pred_p1, pred_p2, (0, 0, 255), 2)
cv2.putText(
    image,
    f"FP score={score:.3f} IoU={iou:.3f}",
    (8, 24),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 0, 255),
    2,
    cv2.LINE_AA,
)
```

Write only rows with status `FP` into the FP image folder. Write FN-only rows to the summary
JSON but do not create an FP image for them.

- [ ] **Step 4: Run evaluation tests and verify GREEN**

Run Task 3 unittest command. Expected: all tests PASS.

### Task 5: Real inference and artifact generation

**Files:**
- Runtime output: `D:\work\Marchine_Dog\dog\A_train\sample100_coco\fire_model_evaluation\*`

- [ ] **Step 1: Verify runtime dependencies**

Run:

```powershell
python -c "import paddle, cv2, numpy, yaml; print(paddle.__version__, cv2.__version__)"
```

Expected: exit code 0 and version strings.

- [ ] **Step 2: Run all unit tests**

Run:

```powershell
python -m unittest dog.mycode.tests.test_fire_predict_raw dog.mycode.tests.test_evaluate_fire_predictions -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run prediction on all 100 images**

Run:

```powershell
python fire\predict.py `
  dog\A_train\sample100_coco\sample100_images.txt `
  dog\A_train\sample100_coco\fire_model_evaluation\predictions.json `
  fire\model
```

Expected: exit code 0, 100 input images processed, and predictions JSON written.

- [ ] **Step 4: Evaluate and render FP images**

Run:

```powershell
python -m dog.mycode.evaluate_fire_predictions `
  --ground-truth dog\A_train\sample100_coco\annotations\instance_train_full.json `
  --predictions dog\A_train\sample100_coco\fire_model_evaluation\predictions.json `
  --image-dir dog\A_train\sample100_coco\images `
  --output-dir dog\A_train\sample100_coco\fire_model_evaluation
```

Expected: `metrics.json`, `per_image.json`, `fp_summary.json`, and annotated `fp_images/*.jpg`.

- [ ] **Step 5: Validate artifact consistency**

Run a Python check that asserts:

```python
assert metrics["images"] == 100
assert metrics["true_positives"] + metrics["false_negatives"] == 100
assert len(list(fp_dir.glob("*.jpg"))) == metrics["false_positives"]
assert len(fp_summary["false_positives"]) == metrics["false_positives"]
```

Expected: exit code 0.

### Task 6: Final verification

**Files:**
- Verify all files above.

- [ ] **Step 1: Compile changed Python files**

Run:

```powershell
python -m py_compile fire\predict.py dog\mycode\evaluate_fire_predictions.py
```

Expected: exit code 0.

- [ ] **Step 2: Re-run the full focused test suite**

Run:

```powershell
python -m unittest dog.mycode.tests.test_fire_predict_raw dog.mycode.tests.test_evaluate_fire_predictions -v
```

Expected: all tests PASS.

- [ ] **Step 3: Inspect final metrics and representative FP images**

Read `metrics.json`, count annotated FP images, and visually inspect at least three FP images
or all images when fewer than three exist. Confirm green GT and red prediction boxes match the
stored coordinates and the displayed score/IoU.

- [ ] **Step 4: Report exact results**

Report TP, FP, FN, precision, recall, F1, inference output path, FP folder path, and any runtime
limitations. Do not claim completion unless Tasks 6.1–6.3 have fresh passing evidence.
