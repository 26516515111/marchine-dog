# Fire V2.2 Conservative Postprocess Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove HSV score boosting and the hidden 0.20 threshold from `fire/v2.2/predict.py` while retaining deterministic weighted largest-fire selection and guaranteed single-box fallback.

**Architecture:** Keep the exported model's NMS and the existing `score * normalized_area ** 0.3` ranker. Candidate filtering uses only the public `fire_threshold`; because every input image contains fire, `force_select=True` falls back to all legal model candidates when none pass. The V2.2 path no longer reads source images for HSV postprocessing.

**Tech Stack:** Python 3.12, NumPy, OpenCV, PaddleDetection inference wrapper

---

## File Structure

- `dog/mycode/test_fire_v2_2_predict.py`: behavioral regression tests for thresholding, HSV independence, and forced weighted selection.
- `fire/v2.2/predict.py`: V2.2 preprocessing, inference, and conservative largest-fire postprocessing.
- `fire/v2.2/氛围火焰v2.2更新记录.md`: records the deployed postprocessing behavior and tuning defaults.

## Version-Control Safety

`fire/v2.2` is outside the `dog` Git repository, and the V2.2 test files are
pre-existing untracked workspace work. Do not create partial implementation
commits that could accidentally claim or capture unrelated user changes. Record
the implementation through focused diffs and fresh verification output.

### Task 1: Specify Conservative Selection Behavior

**Files:**
- Modify: `dog/mycode/test_fire_v2_2_predict.py`
- Test: `dog/mycode/test_fire_v2_2_predict.py`

- [ ] **Step 1: Add failing tests for the hidden threshold and HSV boost**

Append these tests before the `if __name__ == "__main__":` block:

```python
def test_select_firebig_uses_only_requested_threshold() -> None:
    predictor = _load_predict_module()
    detections = [
        {
            "category": "fire",
            "score": 0.22,
            "bbox": [10.0, 10.0, 20.0, 20.0],
        }
    ]

    selected = predictor.select_firebig(
        detections,
        fire_threshold=0.25,
        image_width=100,
        image_height=100,
        force_select=False,
    )

    assert selected is None


def test_select_firebig_does_not_boost_orange_candidates() -> None:
    predictor = _load_predict_module()
    orange_bgr = np.full((20, 20, 3), [0, 120, 255], dtype=np.uint8)
    detections = [
        {
            "category": "fire",
            "score": 0.5,
            "bbox": [0.0, 0.0, 20.0, 20.0],
        }
    ]

    selected = predictor.select_firebig(
        detections,
        fire_threshold=0.25,
        image_width=20,
        image_height=20,
        img_bgr=orange_bgr,
    )

    assert selected is not None
    assert selected["score"] == 0.5
    assert "_sat_boosted" not in selected
```

Add both calls to the script entry point:

```python
if __name__ == "__main__":
    test_inference_clahe_matches_training_operator_pixel_for_pixel()
    test_preprocess_builder_requires_one_clahe_operator()
    test_preprocess_builder_rejects_unknown_or_duplicate_operators()
    test_select_firebig_uses_only_requested_threshold()
    test_select_firebig_does_not_boost_orange_candidates()
    print("fire V2.2 prediction preprocessing and postprocessing tests passed")
```

- [ ] **Step 2: Run the tests and verify both new behaviors fail for the intended reasons**

Run:

```powershell
python D:/work/Marchine_Dog/dog/mycode/test_fire_v2_2_predict.py
```

Expected first failure:

```text
AssertionError
```

at `assert selected is None`, because the current implementation lowers `0.25` to `0.20`.
After temporarily running the HSV test alone, expect `selected["score"]` to be `0.6`, proving the current C1 boost is active.

### Task 2: Remove HSV Boost and Hidden Threshold

**Files:**
- Modify: `fire/v2.2/predict.py`
- Test: `dog/mycode/test_fire_v2_2_predict.py`

- [ ] **Step 1: Remove V2.2 HSV postprocessing constants and helpers**

Delete these constants:

```python
ATMOS_SCORE_BOOST = True
ATMOS_S_BOOST_THRESH = 75
ATMOS_SCORE_BOOST_FACTOR = 1.2
ATMOS_ORANGE_S_MIN = 60
ATMOS_ORANGE_RATIO_THRESH = 0.03
ATMOS_ORANGE_S_PERCENTILE = 90
ATMOS_FIRE_THRESHOLD_SCORE = 0.20
```

Delete these functions:

```python
_legacy_saturation_score_boost
_bbox_crop
_atmosphere_color_stats
_saturation_score_boost
```

Retain:

```python
ATMOS_FORCE_SELECT = True
MERGE_CLOSE_BOXES = False
```

- [ ] **Step 2: Simplify `select_firebig` to one threshold source**

Remove the `img_bgr` and `firebig_threshold` arguments, all C1 score-copying
logic, and the special `reliable_big` branch. Replace the threshold block with:

```python
fallback = [
    det for det in candidates
    if _detection_score(det) >= fire_threshold
]
if not fallback:
    if not force_select:
        return None
    fallback = candidates

image_area = float(image_width * image_height)
return max(
    fallback,
    key=lambda det: _weighted_area_score(det, image_area, area_alpha),
).copy()
```

The complete signature becomes:

```python
def select_firebig(
    detections: list[dict[str, Any]],
    fire_threshold: float = 0.25,
    area_alpha: float = 0.3,
    image_width: int = 640,
    image_height: int = 480,
    force_select: bool = ATMOS_FORCE_SELECT,
) -> dict[str, Any] | None:
```

- [ ] **Step 3: Remove image rereading from `predict_images`**

Delete:

```python
import cv2 as _cv2
_img_bgr_orig = _cv2.imread(image_path) if ATMOS_SCORE_BOOST else None
```

Remove this keyword from the `select_firebig` call:

```python
img_bgr=_img_bgr_orig,
```

Remove `firebig_threshold` from the `predict_images` signature and remove this
keyword from the `select_firebig` call:

```python
firebig_threshold=firebig_threshold,
```

- [ ] **Step 4: Update the HSV-independence test to the final API**

Remove `orange_bgr` and `img_bgr=orange_bgr` from
`test_select_firebig_does_not_boost_orange_candidates`, leaving:

```python
def test_select_firebig_preserves_model_scores() -> None:
    predictor = _load_predict_module()
    detections = [
        {
            "category": "fire",
            "score": 0.5,
            "bbox": [0.0, 0.0, 20.0, 20.0],
        }
    ]

    selected = predictor.select_firebig(
        detections,
        fire_threshold=0.25,
        image_width=20,
        image_height=20,
    )

    assert selected is not None
    assert selected["score"] == 0.5
    assert "_sat_boosted" not in selected
```

Rename the entry-point call to `test_select_firebig_preserves_model_scores()`.

- [ ] **Step 5: Add and run the forced-selection regression test**

Add:

```python
def test_force_select_uses_weighted_area_when_all_candidates_are_below_threshold() -> None:
    predictor = _load_predict_module()
    detections = [
        {
            "category": "fire",
            "score": 0.24,
            "bbox": [0.0, 0.0, 10.0, 10.0],
        },
        {
            "category": "fire",
            "score": 0.20,
            "bbox": [0.0, 0.0, 30.0, 30.0],
        },
    ]

    selected = predictor.select_firebig(
        detections,
        fire_threshold=0.25,
        area_alpha=0.3,
        image_width=100,
        image_height=100,
    )

    assert selected is not None
    assert selected["bbox"] == [0.0, 0.0, 30.0, 30.0]
```

Add its call to the script entry point, then run:

```powershell
python D:/work/Marchine_Dog/dog/mycode/test_fire_v2_2_predict.py
```

Expected:

```text
fire V2.2 prediction preprocessing and postprocessing tests passed
```

### Task 3: Document and Verify the V2.2 Contract

**Files:**
- Modify: `fire/v2.2/氛围火焰v2.2更新记录.md`
- Test: `dog/mycode/test_fire_v2_2_predict.py`
- Test: `dog/mycode/test_fire_v2_2_config.py`
- Test: `dog/mycode/test_clahe_dehaze_operator.py`

- [ ] **Step 1: Replace the old C1 statement in the update record**

Replace:

```markdown
预测端保留 C1 的 HSV 检测框置信度修正；该逻辑发生在模型输出之后，
不修改模型输入。
```

with:

```markdown
预测端不再使用 HSV 检测框置信度修正，也不读取原图进行颜色加分。
候选框只使用模型原始置信度和 `score × normalized_area^0.3` 排序。
默认阈值为 `0.25`；由于预测数据保证每张图存在火焰，低于阈值时仍从
合法模型候选中按同一规则强制选择一个框。近框合并默认关闭，避免密集
火颗粒被聚合成异常大框。
```

- [ ] **Step 2: Run syntax and focused behavior verification**

Run:

```powershell
python -m py_compile D:/work/Marchine_Dog/fire/v2.2/predict.py
python D:/work/Marchine_Dog/dog/mycode/test_fire_v2_2_predict.py
python D:/work/Marchine_Dog/dog/mycode/test_fire_v2_2_config.py
python D:/work/Marchine_Dog/dog/mycode/test_clahe_dehaze_operator.py
```

Expected: all four commands exit with code `0`; the three test scripts print their success messages.

- [ ] **Step 3: Inspect the final diff**

Run:

```powershell
git -C D:/work/Marchine_Dog/dog diff --check -- mycode/test_fire_v2_2_predict.py
git -C D:/work/Marchine_Dog/dog diff -- mycode/test_fire_v2_2_predict.py
```

Expected: the focused test diff contains only the conservative V2.2
postprocessing cases. The `fire` directory is outside the `dog` Git repository,
so inspect `fire/v2.2/predict.py` and its update record directly rather than
creating an unrelated repository.
