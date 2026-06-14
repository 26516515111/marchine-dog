# AGENTS.md - 火焰检测项目

## 项目结构
- `mycode/` 是 PP-YOLOE 配置和脚本的权威来源，`PaddleDetection/configs/custom/` 是训练用副本
- `RT-DETR/` 是 RT-DETR 方案的权威来源，包含配置、文档、训练脚本
- 修改配置必须同步到两处：权威来源 → `PaddleDetection/configs/custom/`
- `Agent.md` 记录开发规范和配置变更历史
- `README.md` 是面向用户的项目文档
- `训练结果改进调研报告.md` 记录优化方案调研与落地方案
- 训练输出在 `PaddleDetection/output/`，导出模型复制到项目根 `model/`

## 训练路径规则
- 训练命令默认从 `PaddleDetection/` 目录执行，例如 `python tools/train.py -c configs/custom/ppyoloe_fire.yml --eval`
- 训练配置中的数据集路径必须使用相对路径，不要写死 `D:/work/...`
- 常规 COCO 配置使用 `dataset_dir: ../A_train/coco`、`image_dir: train|val`、`anno_path: annotations/*.json`
- Hard Negative 配置使用 `dataset_dir: ../A_train`、`image_dir: Image`、`anno_path: coco/annotations/instance_train_with_hn.json`
- 2026-06-12 已在提交 `a59f428` 将权威训练配置改为相对路径；`PaddleDetection/configs/custom/` 中的训练副本也已同步，但该目录按规范不作为顶层提交内容

### 当前活跃配置文件

| 方案 | 权威配置 | PaddleDetection 副本 |
|------|----------|---------------------|
| PP-YOLOE-s 原始版 | `mycode/configs/ppyoloe_fire.yml` | `configs/custom/ppyoloe_fire.yml` |
| PP-YOLOE-s A1 调参版 | `mycode/configs/ppyoloe_fire_a1.yml` | `configs/custom/ppyoloe_fire_a1.yml` |
| RT-DETR-R18 | `RT-DETR/configs/rtdetr_r18vd_fire.yml` | `configs/custom/rtdetr_r18vd_fire.yml` |

## PaddleDetection 配置陷阱
- `_BASE_` 继承的 `ppyoloe_crn.yml` 默认 `width_mult=1.0`（full size），必须显式设置 `width_mult`/`depth_mult` 才能缩放模型
- PP-YOLOE-s/m/l/x 的缩放因子：s=(0.33, 0.50), m=(0.67, 0.75), l=(1.0, 1.0), x=(1.33, 1.25)
- `PPYOLOEHead.loss_weight` 仅接受 `{class, iou, dfl}` dict，`class_weight` 列表会被静默忽略
- VarifocalLoss 不支持 per-class 权重，类别不平衡只能通过推理后处理解决
- 如果 `TrainReader.batch_transforms` 使用 `PadBatch` 或其他不会补齐 GT 的流程，必须显式保留 `PadGT: {}`
- 否则当 batch 内不同图片的目标数不一致时，`default_collate_fn` 会在 `np.stack` 处报 `ValueError: all input arrays must have the same shape`
- 2026-06-13 已修复 `mycode/configs/ppyoloe_fire.yml` 及 `PaddleDetection/configs/custom/ppyoloe_fire.yml` 缺失 `PadGT` 的问题，并新增 `mycode/scripts/validate_reader_batch_transforms.py` 做校验
- 如果 `EvalReader/TestReader` 使用 `Resize(..., keep_ratio=True)`，必须同步配置 `batch_transforms: [PadBatch: {pad_to_stride: 32}]`
- 原因：`Resize(keep_ratio=True)` 只做等比缩放，不会自动 padding；若输入尺寸不是稳定的 32 倍数，PP-YOLOE 的 FPN/PAN 在上采样拼接时可能出现 27/28 这类错一格
- 典型报错为 `custom_pan.py` 中 `paddle.concat` 失败，提示 `The 2-th dimension of input[0] and input[1] is expected to be equal`
- 2026-06-13 已修复 `mycode/configs/ppyoloe_fire.yml` 及 `PaddleDetection/configs/custom/ppyoloe_fire.yml` 的 Eval/Test Reader 缺失 `PadBatch` 问题，校验脚本也已覆盖该场景

## 类别不平衡注意事项
- 图片级过采样无效：battery/board 图片大多同时包含 fire，复制时 fire 也被一起复制
- Hard Negative Mining 中 `check_iou_with_gt(iou_threshold=0.0)` 过于严格，改为 0.3
- **正确方案**：通过推理后处理（NMS + 扩框）解决类别不平衡

## 推理后处理配置（2026-06-12 新增）

`predict.py` 实现了以下后处理流水线：
1. **阈值过滤**：class_thresholds = {1: 0.4, 2: 0.4, 3: 0.5}
2. **class-wise NMS**：nms_threshold = 0.55
3. **扩框**：battery (1.55x, 1.15y), fire (1.18x, 1.18y)
4. **clip 到图片边界**
5. **输出**

board 不扩框，只靠 NMS 去重。battery/fire 在 NMS 后扩框，不影响 board 去重。

## Fire 后处理调参脚本（2026-06-13 新增）

- `mycode/scripts/tune_fire_postprocess.py` 用于搜索 fire 阈值、fire NMS、fire 小框过滤面积。
- 该脚本不是单纯选择 val 最高分，会同时输出 `best_raw` 和 `recommended_stable`。
- `recommended_stable` 会对过高 fire 阈值、过低 fire NMS、过大的 fire 小框过滤面积、明显 recall 下降做惩罚，避免过拟合当前 val。
- 输入预测 JSON 必须包含 `score` 字段；官方提交格式结果通常不包含 score，不能直接用于阈值搜索。
- 推荐从真实提交脚本导出 raw scored predictions：

```bash
python mycode/scripts/tune_fire_postprocess.py --predict-py "D:/work/Marchine Dog/predict.py" --infer-txt val.txt --gt A_train/coco/annotations/instance_val.json --raw-out raw_preds_with_score.json --out fire_postprocess_tuning.json
```

- 如果已有带 score 的 raw 预测文件，可直接离线搜索：

```bash
python mycode/scripts/tune_fire_postprocess.py --pred-json raw_preds_with_score.json --gt A_train/coco/annotations/instance_val.json --out fire_postprocess_tuning.json
```

- 快速压 fire FP 时优先看 `recommended_stable`，不要只照抄 `best_raw`。

## 性能约束
- 模型大小 ≤ 200MB
- 推理速度 ≥ 20 FPS
- 目标 F1 ≥ 0.85

## 最终成绩（2026-06-12）

| 数据集 | 综合 F1 | battery F1 | board F1 | fire F1 |
|--------|---------|------------|----------|---------|
| 训练集 | **0.9408** | 0.906 | 0.973 | 0.943 |
| 验证集 | **0.9189** | 0.875 | 0.919 | 0.963 |

**综合 F1 超过目标 0.85，所有类别均达标！**

### 优化历程

| 阶段 | F1 | 关键改进 |
|------|-----|----------|
| 初始模型 | 0.80614 | baseline |
| 学习率提升 | 0.82368 | base_lr: 0.0005 → 0.01 |
| 阈值优化 | 0.8455 | battery=0.4, board=0.4, fire=0.5 |
| NMS 后处理 | 0.8701 | predict.py 添加 NMS (threshold=0.55) |
| 扩框优化 | **0.9189** | battery 扩框 1.55x/1.15y, fire 扩框 1.18x/1.18y |

## RT-DETR 特殊注意事项
- RT-DETR 不用 Mosaic/Mixup/GridMask 等 YOLO 系增强（DETR 架构不兼容）
- RT-DETR 的预训练权重是 ImageNet backbone 权重，不是 COCO 检测权重（与 PP-YOLOE 不同）
- RT-DETR 使用 AdamW 优化器 + lr=0.0001（远低于 PP-YOLOE 的 Momentum + lr=0.01）
- RT-DETR 的 Normalize 用 `mean=[0,0,0], std=[1,1,1], norm_type=none`（不在预处理中做归一化，模型内部处理）
- RT-DETR 输入需要 `NormalizeBox` + `BboxXYXY2XYWH` 变换（DETR 格式要求）
- RT-DETR 无需 NMS 后处理，导出时 `post_process: True`

## 环境
- PaddlePaddle GPU 3.0.0, CUDA 12.x, conda env `dog`
- RTX 4060 Laptop 8GB VRAM, batch_size 最大 4
- Windows 路径使用 `D:/work/Marchine Dog/dog/` 格式（正斜杠）
