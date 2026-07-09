# 火焰检测最终结果与思路过程

## 最终结果

本次最终提交结果存储在以下目录：

```
dog/最后提交结果/B/
├── B_fire_atmosphere.yml    ← 训练配置文件
├── predict.py               ← 推理提交脚本
└── model/
    ├── infer_cfg.yml
    ├── model.pdmodel
    ├── model.pdiparams
    └── model.pdiparams.info
```

当前使用的最终方案要点如下：

- **模型路线**：PP-YOLOE+ CRN M 级别模型，使用 Object365 预训练权重初始化。
- **检测类别**：单类别 `fire` 检测，导出后 `label_list` 为 `["fire"]`。提交时通过后处理选出一个最可信的 `firebig` 框。
- **推理尺寸**：`infer_cfg.yml` 中推理预处理使用 `Resize 640x640`，保持长宽比，加 `PadStride(32)` 对齐。
- **训练数据**：采用混杂数据集方案，训练集为 `merged_fire_coco`（合并带环境因素的标注数据与纯净数据）。
- **最终阈值**：`predict.py` 中 `FIRE_MIN_OUTPUT_SCORE` 默认值 `0.1`，配合相对最高分比例 `0.5` 动态筛选。
- **输出策略**：每张图从可信候选中选择面积最大的框，以 `firebig` 标签提交。
- **最终提交 F1**：**`0.9269`**。

### 训练配置与超参数

最终模型对应的训练配置文件：

```text
dog/最后提交结果/B/B_fire_atmosphere.yml
```

核心模型配置：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| 模型 | `PP-YOLOE+ CRN M` | M 级别模型作为最终方案 |
| `pretrain_weights` | `ppyoloe_crn_m_obj365_pretrained.pdparams` | 使用 Object365 预训练权重（而非 COCO） |
| `num_classes` | `1` | 只检测 fire |
| `depth_mult` | `0.67` | M 模型深度系数 |
| `width_mult` | `0.75` | M 模型宽度系数 |
| `eval_size` | `~`（未指定） | 推理导出时固定为 640，训练使用动态多尺度 |
| `metric` | `COCO` | 使用 COCO 指标评估 |

训练与优化参数：

| 参数 | 值 |
| --- | --- |
| `epoch` | `100` |
| `batch_size` | `4` |
| `worker_num` | `8` |
| `base_lr` | `0.0005` |
| 学习率策略 | `CosineDecay(max_epochs=100)` + `LinearWarmup(start_factor=0.1, epochs=5)` |
| 优化器 | `Momentum(momentum=0.9)` |
| 正则化 | `L2(factor=0.0005)` |
| `snapshot_epoch` | `5` |
| `save_dir` | `output/B_fire_atmosphere` |
| `weights` | `output/B_fire_atmosphere/best_model` |

数据集配置：

| 数据集 | 路径/配置 |
| --- | --- |
| 训练集 | `dataset_dir: ../A_train/merged_fire_coco`，`anno_path: annotations/instance_train_21.json` |
| 验证集 | `dataset_dir: ../A_train/merged_fire_coco`，`anno_path: annotations/instance_val_21.json` |
| 测试配置 | `ImageFolder`，`anno_path: ../A_train/merged_fire_coco/annotations/instance_val_21.json` |
| 空样本 | `allow_empty: true` |
| 数据字段 | `image`，`gt_bbox`，`gt_class`，`is_crowd` |

训练阶段增强策略：

| 增强 | 参数 |
| --- | --- |
| `RandomDistort` | `brightness: [0.78, 1.32, 0.55]`，`contrast: [0.78, 1.28, 0.45]`，`saturation: [0.75, 1.35, 0.45]`，`hue: [-8, 8, 0.20]`，`random_apply: true`，`count: 4` |
| `GaussianBlur` | `k: [3, 5]`，`sigma: [0.2, 1.2]`，`prob: 0.08` |
| `MotionBlur` | `k: [3, 5]`，`angle: [-20, 20]`，`prob: 0.08` |
| `GaussianNoise` | `mean: 0`，`std: [2, 8]`，`prob: 0.08` |
| `RandomErasing` | `prob: 0.04`，`scale: [0.005, 0.035]`，`ratio: [0.5, 2.0]`，`value: [245, 235, 190]` |
| `RandomExpand` | `ratio: 1.12`，`prob: 0.75`，`fill_value: [123.675, 116.28, 103.53]` |
| `RandomFlip` | `{}`（默认随机水平翻转） |
| `BatchRandomResize` | `target_size: [704, 768, 832, 896]`，`random_size: True`，`random_interp: True`，`keep_ratio: True` |
| 归一化 | `NormalizeImage(mean=[0,0,0], std=[1,1,1], norm_type=none)` |
| 对齐与 GT 填充 | `PadBatch(pad_to_stride=32)` + `PadGT` |

Eval/Test 预处理：`Resize([832, 832], keep_ratio=True)`，随后 `NormalizeImage`、`Permute` 和 `PadBatch(pad_to_stride=32)`，`batch_size=4`。

检测头和 NMS 参数：

| 参数 | 值 |
| --- | --- |
| `fpn_strides` | `[32, 16, 8]` |
| `grid_cell_scale` | `5.0` |
| `grid_cell_offset` | `0.5` |
| `static_assigner_epoch` | `40` |
| `use_varifocal_loss` | `True` |
| `loss_weight` | `{class: 1.0, iou: 2.5, dfl: 0.5}` |
| `static_assigner` | `ATSSAssigner(topk=9)` |
| `assigner` | `TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)` |
| `nms` | `MultiClassNMS(nms_top_k=1000, keep_top_k=300, score_threshold=0.02, nms_threshold=0.65)` |

## F1 指标演进

| 阶段 | 关键调整 | F1 |
| --- | --- | --- |
| 初始方案 | 使用单一数据集、S 模型 | `0.63` |
| 数据策略修正 | 从单一数据集 → 混杂数据集（1:2 比例） | `0.78` |
| 模型修正 | 从 S 模型 → M 模型 | `0.90` |
| 推理阈值调整 | `FIRE_MIN_OUTPUT_SCORE` 调至 `0.1` | `0.9269` |

## 走过的弯路与关键错误决定

以下三个决策点均在实验早期被错误选择，后期通过实验验证逐步修正。

### 1. 类别策略：firebig 单类识别 → fire 识别 + 取最大框

**错误决定**：初期认为任务要求输出 `firebig`，应该直接训练模型检测 `firebig` 这一个类别。

**问题**：`firebig` 是评测定义的提交格式，而非物理可分的对象类别。训练数据中大量中等偏小的火焰被标注为 `fire` 而非 `firebig`。如果只检测 `firebig`，模型会漏掉大量中小火焰，导致召回极低。

**后期改正**：改为单类别 `fire` 检测，训练时统一学习所有火焰样本。推理后在候选框中通过 `select_largest_credible_fire` 筛选出最可信的大面积火焰框，以 `firebig` 格式提交。这样既保留了完整的火焰学习信号，又在输出时满足 `firebig` 评测格式。

### 2. 训练集选择：单一纯净训练集 → 混杂数据集（1:2 混合）

**错误决定**：早期只使用单一纯净标注数据集训练，认为数据越干净越好。

**问题**：纯净数据集虽然标注质量高、分布稳定，但场景变化有限。实际测试集中包含大量复杂光照、烟雾氛围、运动模糊、过曝等环境因素带来的视觉变化。只用纯净数据训练，模型泛化不足，在这些场景下频繁漏检或误检，F1 一直在 0.6~0.7 附近徘徊。

**后期改正**：引入带环境因素的数据增强策略（GaussianBlur、MotionBlur、GaussianNoise、RandomErasing 等），同时将包含环境氛围的标注数据（有烟雾、弱对比等的 "atmosphere" 数据）与纯净训练数据合并为 `merged_fire_coco`。为了让纯净数据的主分布不受干扰，控制混杂数据与纯净数据的比例为 `1:2`。这一改动直接将 F1 从约 0.63 提升到约 0.78。

### 3. 模型选择：PP-YOLOE+ S → PP-YOLOE+ CRN M

**错误决定**：项目初期选择了 S 级别模型，认为火焰检测任务相对简单，轻量模型足够。

**问题**：S 模型参数量少、训练快，但在复杂场景下的特征表达能力明显不足。火焰目标形态多变（小火焰、大面积过曝火焰、被烟雾遮挡的火焰），且背景干扰严重（红色建筑物、路灯、车灯等）。S 模型在这些情况下容易出现召回低或边框不稳定。

**后期改正**：升级到 M 级别模型（depth_mult=0.67，width_mult=0.75）。M 模型在保持可控训练时间和部署体积（约 90 MB）的同时，显著提升了复杂场景下的召回和框稳定性，F1 从约 0.78 提升到约 0.90。

## 思路演进

### 1. 模型从 S 到 M

项目初期优先选择 S 模型，是为了先建立一个轻量的 baseline，快速跑通数据读取、标注格式、训练配置、模型导出和 `predict.py` 提交格式。S 模型让整条链路暴露基础问题更快。

在流程稳定后，升级到 M 模型。M 模型更强的特征表达能力，更好地处理了火焰尺度变化、形态变化和背景干扰。最终导出模型约 90 MB，仍处于可接受的部署体积范围内。

### 2. 数据从单一到混杂

单一数据集训练虽然纯净、稳定，但易导致泛化不足。混杂数据集方案引入更丰富的场景变化（光照、噪声、模糊、遮挡等增强），同时将带环境因素的标注数据与纯净数据混合，让模型在训练阶段接触更多火焰形态和背景干扰。

为了避免混杂数据过多导致分布偏移，控制混杂数据与纯净数据比例为 `1:2`。纯净数据仍占主体，保证标注质量和主分布稳定；混杂数据作为补充，增强模型对复杂场景的适应能力。

### 3. 推理后处理中的阈值选择

训练完成后，最后的提升来自推理后处理的调优。`predict.py` 中默认最低输出阈值为 `0.1`，同时在每张图的候选框中采用绝对阈值 + 相对最高分的双重筛选逻辑：

```python
threshold = max(ABSOLUTE_SCORE_FLOOR, MIN_OUTPUT_SCORE, RELATIVE_SCORE_RATIO * max_score)
```

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `ABSOLUTE_SCORE_FLOOR` | `0.02` | 绝对最低保护阈值 |
| `MIN_OUTPUT_SCORE` | `0.1` | 手动调整后的最低输出阈值 |
| `RELATIVE_SCORE_RATIO` | `0.5` | 相对当前图最高分的筛选比例 |

这样做的好处是：当模型对某张图整体置信度较高时，弱候选被动态过滤；当整体置信度偏低时，`0.1` 的硬阈值维持一定召回空间，减少漏检。

最终每张图从可信候选中选择面积最大的火焰框：

```python
selected = select_largest_credible_fire(detections)
```

选择最大可信框的原因是最终输出要求 `firebig`，应关注主要火焰区域而非拆成多个小框。该策略让输出更稳定。

## 最终方案总结

1. **类别策略**：单类别 `fire` 检测，推理后取最大可信框作为 `firebig` 提交。
2. **模型**：PP-YOLOE+ CRN M，Object365 预训练权重初始化。
3. **数据**：混杂数据集（`merged_fire_coco`），混杂数据与纯净数据按 `1:2` 混合。
4. **训练增强**：适度退化增强（GaussianBlur、MotionBlur、GaussianNoise、RandomErasing）+ 随机色光变化 + 多尺度训练（704~896）。
5. **推理输入**：`Resize 832x832`（模型内部 NMS 后导出为 640x640）。
6. **后处理**：绝对阈值 0.1 + 相对最高分 0.5 筛选，选最大框。
7. **最终提交 F1**：**`0.9269`**。

### 运行方式

```bash
# 使用默认模型目录（先 cd 到 dog/最后提交结果/B/）
python predict.py <image_list.txt> <result.json>

# 显式指定模型目录
python predict.py <image_list.txt> <result.json> dog/最后提交结果/B/model/
```
