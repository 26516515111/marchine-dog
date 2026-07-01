# Merged Firebig Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `new_label_coco` and `coco_firebig`, create a deterministic stratified 80/20 split with empty images retained, and point `B_firebig.yml` at the merged dataset with aligned resolutions.

**Architecture:** A standalone dataset builder reads each source's full COCO annotation, prefixes filenames, remaps IDs, stratifies records by source and positive/negative state, copies images, and writes train/val/full artifacts. Focused tests exercise the builder on temporary miniature datasets before it touches the real 4027-image output.

**Tech Stack:** Python 3.12, standard library (`json`, `random`, `shutil`, `pathlib`), PyYAML, `unittest`.

---

### Task 1: Dataset merge and stratified split

**Files:**
- Create: `dog/A_train/merge_and_resplit_firebig_coco.py`
- Create: `dog/A_train/tests/test_merge_and_resplit_firebig_coco.py`

- [ ] Write failing tests for source-prefixed filenames, continuous IDs, empty-image retention,
  annotation remapping, train/val disjointness, 80/20 per-stratum split, and seed determinism.
- [ ] Run:

```powershell
python -m unittest dog.A_train.tests.test_merge_and_resplit_firebig_coco -v
```

  Expected: import failure because the builder does not exist.
- [ ] Implement `load_source_records`, `stratified_split`, `build_coco`, and `merge_datasets`.
  Use category `{"id": 1, "name": "firebig"}` and prefixes `new_label__` and
  `coco_firebig__`.
- [ ] Write images with IDs starting at 1 and annotations with IDs starting at 1. Keep all
  source annotations associated with their remapped image IDs.
- [ ] Copy images, write three annotation JSONs, split text files, and a summary.
- [ ] Re-run the focused tests; expected all PASS.

### Task 2: Training configuration

**Files:**
- Modify: `firebig/B_firebig.yml`
- Create: `dog/mycode/tests/test_firebig_merged_config.py`

- [ ] Write a failing test that loads the YAML as text and asserts:
  - merged dataset path appears in Train/Eval/Test datasets;
  - `allow_empty: true`;
  - training sizes are `[576, 640, 704]`;
  - Eval/Test sizes are `[640, 640]`.
- [ ] Run:

```powershell
python -m unittest dog.mycode.tests.test_firebig_merged_config -v
```

  Expected: FAIL against the current config.
- [ ] Modify only the requested dataset and resolution fields.
- [ ] Re-run the test; expected PASS.

### Task 3: Generate and verify real merged data

**Files:**
- Generate: `dog/A_train/new_label_coco_firebig_merged/**`

- [ ] Run:

```powershell
python dog\A_train\merge_and_resplit_firebig_coco.py
```

- [ ] Verify:
  - 4027 copied images;
  - 3823 full annotations;
  - 207 empty full images;
  - train + val = 4027;
  - train/val filenames are disjoint;
  - every annotation image ID exists;
  - repeated execution produces identical annotation JSON hashes.
- [ ] Run both focused test modules and `py_compile`.
- [ ] Report exact train/val positive and negative counts plus output paths.
