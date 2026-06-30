# Fire Predict GPU FPS Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `fire/predict.py` fail fast without CUDA, remove unused candidate caching, align timing with the verified predictor, and preserve strict output, F1, and GPU FPS requirements.

**Architecture:** Keep the existing preprocessing, Paddle predictor, and largest-credible-fire selector. Isolate GPU validation in a small function, return only final predictions from `predict_images`, and write the strict submission with compact JSON.

**Tech Stack:** Python 3.12, Paddle Inference 3.0, CUDA, OpenCV, NumPy, `unittest`.

---

### Task 1: Lock GPU initialization behavior

**Files:**
- Modify: `D:\work\Marchine_Dog\fire\predict.py`
- Modify: `D:\work\Marchine_Dog\dog\mycode\tests\test_fire_predict_raw.py`

- [ ] **Step 1: Write failing tests**

Add fake Paddle/config objects and verify:

```python
with self.assertRaisesRegex(RuntimeError, "CUDA"):
    predict.configure_gpu(fake_config, FakePaddle(cuda_build=False, gpu_count=0))

with patch.dict(os.environ, {"PREDICT_GPU_POOL_MB": "999"}):
    with self.assertRaisesRegex(ValueError, "at least 1000"):
        predict.configure_gpu(fake_config, FakePaddle(cuda_build=True, gpu_count=1))
```

Also parse the source AST and assert Paddle imports exist only at module scope, not inside
`main` or `Detector`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest dog.mycode.tests.test_fire_predict_raw -v
```

Expected: FAIL because `configure_gpu` does not exist and Paddle is imported inside functions.

- [ ] **Step 3: Implement minimal GPU configuration**

Move these imports above the timer:

```python
import paddle
from paddle.inference import Config, create_predictor
```

Implement:

```python
def configure_gpu(config, paddle_module=paddle):
    if not paddle_module.is_compiled_with_cuda():
        raise RuntimeError("CUDA-enabled Paddle is required; CPU fallback is disabled")
    if paddle_module.device.cuda.device_count() < 1:
        raise RuntimeError("CUDA GPU is not available; CPU fallback is disabled")
    pool_mb = int(os.getenv("PREDICT_GPU_POOL_MB", "2000"))
    if pool_mb < 1000:
        raise ValueError("PREDICT_GPU_POOL_MB must be at least 1000")
    config.enable_use_gpu(pool_mb, 0)
```

Call it from `Detector.__init__` and remove `disable_gpu()` and CPU thread configuration.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 1 test command. Expected: all tests PASS.

### Task 2: Remove unused candidate diagnostics

**Files:**
- Modify: `D:\work\Marchine_Dog\fire\predict.py`
- Modify: `D:\work\Marchine_Dog\dog\mycode\tests\test_fire_predict_raw.py`

- [ ] **Step 1: Write failing behavior test**

Use a fake detector and patched preprocessing:

```python
result = predict.predict_images(fake_detector, ["sample.jpg"])
self.assertEqual(list(result), ["result"])
self.assertNotIn("raw_candidates", predict.PREDICT_SOURCE)
```

The fake detector returns one `[class_id, score, x1, y1, x2, y2]` row and one
`boxes_num` value.

- [ ] **Step 2: Run test and verify RED**

Expected: FAIL because the current result also contains `raw_candidates` and
`unreadable_images`.

- [ ] **Step 3: Remove diagnostic retention**

Delete:

```python
raw_candidates = {}
unreadable_images = []
raw_candidates[image_key] = [detection.copy() for detection in detections]
```

Unreadable images continue to be skipped, but are not accumulated. Return only:

```python
return {"result": final_results}
```

- [ ] **Step 4: Run test and verify GREEN**

Run the focused predictor tests. Expected: all PASS.

### Task 3: Write strict compact JSON

**Files:**
- Modify: `D:\work\Marchine_Dog\fire\predict.py`
- Modify: `D:\work\Marchine_Dog\dog\mycode\tests\test_fire_predict_raw.py`

- [ ] **Step 1: Write failing serialization test**

Create a temporary output and call:

```python
predict.write_submission(path, {"result": [exact_item]})
text = path.read_text(encoding="utf-8")
self.assertNotIn("\n", text)
self.assertEqual(json.loads(text), {"result": [exact_item]})
```

- [ ] **Step 2: Run test and verify RED**

Expected: FAIL because `write_submission` does not exist.

- [ ] **Step 3: Implement compact serialization**

```python
def write_submission(path, submission):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(submission, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
```

Use it from `main`.

- [ ] **Step 4: Run test and verify GREEN**

Expected: all predictor tests PASS.

### Task 4: Real GPU regression verification

**Files:**
- Verify: `D:\work\Marchine_Dog\fire\predict.py`
- Generate: `D:\work\Marchine_Dog\dog\A_train\sample100_coco\fire_model_evaluation\result.json`

- [ ] **Step 1: Run syntax and unit tests**

```powershell
python -m py_compile fire\predict.py
python -m unittest dog.mycode.tests.test_fire_predict_raw dog.mycode.tests.test_evaluate_fire_predictions -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run three GPU benchmarks**

Use `D:\Anaconda\envs\dog\python.exe`, batch 32, and the 100-image list. Each run starts a
fresh process and writes a separate strict JSON.

Expected: every run reports more than 20 FPS.

- [ ] **Step 3: Validate strict schema**

Assert the top-level key set equals `{"result"}`, every item has exactly seven required keys,
and exactly 100 results exist.

- [ ] **Step 4: Recalculate F1**

Evaluate `result.json` against `instance_train_full.json` with strict `IoU > 0.5`.

Expected: TP=96, FP=4, FN=4, F1=0.96.

- [ ] **Step 5: Report timing breakdown**

Report three FPS values, average, minimum, schema status, and F1. Distinguish predictor timing
from `conda run` launcher overhead.
