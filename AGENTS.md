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
- 训练命令默认从 `PaddleDetection/` 目录执行，例如 `python tools/train.py -c configs/custom/ppyoloe_fire_a1.yml --eval`
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
- VarifocalLoss 不支持 per-class 权重，类别不平衡只能通过数据级手段解决

## 类别不平衡注意事项
- 图片级过采样无效：battery/board 图片大多同时包含 fire，复制时 fire 也被一起复制
- Hard Negative Mining 中 `check_iou_with_gt(iou_threshold=0.0)` 过于严格，改为 0.3

## 性能约束
- 模型大小 ≤ 200MB
- 推理速度 ≥ 20 FPS
- 目标 F1 ≥ 0.85（当前最佳 0.80614）

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
