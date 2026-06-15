# PP-YOLOE+ Objects365 C1 - 2026-06-15

## Purpose

Hidden test F1 dropped to about `0.84` while visible val F1 was much higher.
The likely cause is weak generalization from a small validation set plus val-specific post-processing.

The C1 direction uses PP-YOLOE+ s with Objects365 pretraining to improve feature transfer while keeping model size far below the `200MB` limit.

## Active Config

- Authoritative config: `mycode/configs/ppyoloe_plus_fire_c1.yml`
- Training copy: `PaddleDetection/configs/custom/ppyoloe_plus_fire_c1.yml`
- Pretrain weights: `ppyoloe_crn_s_obj365_pretrained.pdparams`
- Scale: `depth_mult: 0.33`, `width_mult: 0.50`
- Save dir: `output/ppyoloe_plus_fire_c1`

Both config files are synchronized.

## Geometry Policy

C1 was updated to match the corrected official prediction geometry:

- `TrainReader.BatchRandomResize.keep_ratio: True`
- `TrainReader.batch_transforms` includes `PadBatch: {pad_to_stride: 32}` and `PadGT: {}`
- `EvalReader.Resize target_size: [768, 768], keep_ratio: True`
- `TestReader.Resize target_size: [768, 768], keep_ratio: True`
- Eval/Test include `PadBatch: {pad_to_stride: 32}`
- `eval_size: ~` disables fixed `[640, 640]` anchors inherited from the PP-YOLOE+ reader base.
- `TestReader.inputs_def.image_shape: [3, -1, -1]` avoids fixed static `640x640` export shape.

For a 1080x1920 image, the expected prediction preprocessing is still:

1. resize to `432x768`;
2. pad to `3x448x768`;
3. keep `im_shape=[432,768]` and `scale_factor=[0.4,0.4]`.

## Verification

Reader validation:

```bash
D:/Anaconda/envs/dog/python.exe D:/work/Marchine\ Dog/dog/mycode/scripts/validate_reader_batch_transforms.py
```

Result:

- `ppyoloe_plus_fire_c1.yml TrainReader`: `BatchRandomResize`, `NormalizeImage`, `Permute`, `PadBatch`, `PadGT`
- `EvalReader`: `Resize`, `NormalizeImage`, `Permute` + `PadBatch`
- `TestReader`: `Resize`, `NormalizeImage`, `Permute` + `PadBatch`
- validation passed.

One-epoch smoke test:

```bash
cd D:/work/Marchine\ Dog/dog/PaddleDetection
D:/Anaconda/envs/dog/python.exe tools/train.py -c configs/custom/ppyoloe_plus_fire_c1.yml --eval -o epoch=1 snapshot_epoch=1
```

Result:

- Objects365 pretrained weights downloaded and loaded.
- Classification head weights were skipped because the pretrained head has 365 classes and this task has 3 classes. This is expected.
- Train loop, evaluation loop, keep-ratio padding, and checkpoint save all completed.
- Checkpoints were saved under `output/ppyoloe_plus_fire_c1`.
- One-epoch eval AP was not meaningful for model selection, but confirmed the pipeline works.

## Full Training Command

Run this in normal or high-performance power mode:

```bash
cd D:/work/Marchine\ Dog/dog/PaddleDetection
D:/Anaconda/envs/dog/python.exe tools/train.py -c configs/custom/ppyoloe_plus_fire_c1.yml --eval
```

After training, export the best model:

```bash
cd D:/work/Marchine\ Dog/dog/PaddleDetection
D:/Anaconda/envs/dog/python.exe tools/export_model.py -c configs/custom/ppyoloe_plus_fire_c1.yml --output_dir=../../model_ppyoloe_plus_c1 -o weights=output/ppyoloe_plus_fire_c1/best_model.pdparams
```

Then copy or point the submission `predict.py` to the exported model directory and retune class thresholds on raw scored predictions.

## Retuning Rule

Do not reuse the old PP-YOLOE-s thresholds blindly.
After exporting C1:

1. generate raw scored val predictions;
2. search battery threshold and class-wise NMS;
3. compare raw-score evaluation and no-score submission replay;
4. prefer stable parameters over tiny val-only gains.
