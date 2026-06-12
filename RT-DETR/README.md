# RT-DETR 火焰检测训练

基于 PaddleDetection RT-DETR-R18 的火焰/电池/指示牌目标检测方案。

## 为什么选 RT-DETR-R18？

| 特性 | 说明 |
|------|------|
| 架构 | DETR-based（Transformer 端到端检测），无需 NMS |
| 对小数据集友好 | DETR 的 Hungarian 匹配天然对 minority 类别更友好 |
| 内存友好 | R18 backbone 约 78MB，RTX 4060 8GB 轻松跑 |
| 推理速度 | >= 25 FPS (RTX 4060) |
| 类别不平衡处理 | DINOHead + VFL loss 对类别不平衡有天然抑制 |

## 与 PP-YOLOE-s 对比

| 维度 | PP-YOLOE-s | RT-DETR-R18 |
|------|-----------|-------------|
| 参数量 | ~31MB | ~78MB |
| 推理速度 | ~30 FPS | ~25 FPS |
| 小数据集表现 | 需要强增强 | 天然更好 |
| 类别不平衡 | 需数据级解决 | VFL loss 天然抑制 |
| NMS 后处理 | 需要 | 不需要（端到端） |
| 预训练 | COCO 检测 | ImageNet 分类（backbone-only） |

## 目录结构

```
RT-DETR/
├── configs/
│   └── rtdetr_r18vd_fire.yml    # 训练配置（权威来源）
├── train.bat                     # 一键训练脚本
└── README.md                     # 本文件
```

## 训练步骤

训练配置使用相对数据路径，默认从项目根下的 `PaddleDetection/` 目录执行。项目迁移到其他电脑时，只要保持 `A_train/` 和 `PaddleDetection/` 同级，无需修改 RT-DETR 配置中的数据路径。

### 1. 配置同步

```bash
# 将配置复制到 PaddleDetection 训练目录
copy RT-DETR\configs\rtdetr_r18vd_fire.yml PaddleDetection\configs\custom\
```

### 2. 启动训练

```bash
cd PaddleDetection
python tools/train.py -c configs/custom/rtdetr_r18vd_fire.yml --eval
# 或直接运行
RT-DETR\train.bat
```

### 3. 导出模型

```bash
cd PaddleDetection
python tools/export_model.py -c configs/custom/rtdetr_r18vd_fire.yml --output_dir=../output_inference -o weights=output/rtdetr_r18_fire/best_model.pdparams
```

### 4. 评估

```bash
cd PaddleDetection
python tools/eval.py -c configs/custom/rtdetr_r18vd_fire.yml -o weights=output/rtdetr_r18_fire/best_model
```

## 配置要点

### 数据集
- 训练集：324 张（`../A_train/coco/train`，从 `PaddleDetection/` 目录解析）
- 验证集：81 张（`../A_train/coco/val`，从 `PaddleDetection/` 目录解析）
- 3 个类别：battery(1), board(2), fire(3)

### 训练参数
| 参数 | 值 | 说明 |
|------|-----|------|
| epoch | 300 | 小数据集需要更多迭代 |
| base_lr | 0.0001 | Transformer 推荐低学习率 |
| batch_size | 4 | RTX 4060 8GB 上限 |
| 优化器 | AdamW | DETR 标配 |
| weight_decay | 0.0001 | 防止过拟合 |
| 多尺度训练 | 480~800 | 提高泛化能力 |

### 与 PP-YOLOE 配置的差异

| 差异点 | PP-YOLOE | RT-DETR | 原因 |
|--------|----------|---------|------|
| 优化器 | Momentum | AdamW | DETR 架构要求 |
| 学习率 | 0.01 | 0.0001 | Transformer vs CNN |
| 数据增强 | Mosaic+Mixup+GridMask | RandomDistort+Crop+Flip | DETR 不需要 Mosaic |
| 归一化 | ImageNet stats | [0,1] 无归一化 | DETR 不同预处理 |
| NMS | MultiClassNMS | 不需要 | 端到端检测 |
| close_mosaic | 有 | 无 | 不使用 Mosaic |

## 预期效果

基于 324 张训练数据 + 3 类检测任务：
- 预期 F1：0.84~0.88
- 训练时间：RTX 4060 约 4~6 小时

## 注意事项

1. RT-DETR 训练初期 loss 可能很高（Transformer 收敛慢），前 50 epochs 不需要担心
2. 如果出现 OOM，将 batch_size 从 4 降到 2
3. pretain_weights 是 ImageNet 分类权重，不是 COCO 检测权重（不同于 PP-YOLOE）
