# PaddleDetection 工具脚本详解

## 目录

1. [整体架构](#整体架构)
2. [支持的模型](#支持的模型)
3. [训练流程](#训练流程)
4. [工具脚本详解](#工具脚本详解)
5. [部署能力](#部署能力)
6. [模型压缩](#模型压缩)

---

## 整体架构

```
PaddleDetection/
├── configs/          # 模型配置文件 (YAML 分层继承)
├── ppdet/            # 核心库
│   ├── core/         # 配置系统、模块注册
│   ├── data/         # 数据加载、增强
│   ├── engine/       # 训练器 (Trainer)
│   ├── modeling/     # 模型架构、骨干、检测头
│   ├── optimizer/    # 优化器、学习率调度
│   ├── metrics/      # 评估指标
│   ├── slim/         # 模型压缩
│   └── utils/        # 工具函数
├── tools/            # 命令行工具
└── deploy/           # 部署工具
```

---

## 支持的模型

| 类别 | 模型 | 说明 |
|------|------|------|
| **YOLO 系列** | PP-YOLOE, PP-YOLOv2, YOLOv3, YOLOX, YOLOF | 实时检测 |
| **Transformer** | DETR, Deformable DETR, DINO, RT-DETR | 高精度 |
| **两阶段** | Faster R-CNN, Mask R-CNN, Cascade R-CNN | 经典方法 |
| **单阶段** | FCOS, RetinaNet, GFL, TOOD, SSD | Anchor-Free |
| **轻量级** | PicoDet, CenterNet, TTFNet | 移动端 |
| **旋转检测** | S2ANet, FCOSR, PP-YOLOE-R | 任意方向 |
| **跟踪** | JDE, FairMOT, DeepSORT, ByteTrack | 多目标跟踪 |
| **关键点** | HRNet, ViTPose, PETR | 姿态估计 |

---

## 训练流程

```
tools/train.py
  ├── 加载配置 (YAML + _BASE_ 继承)
  ├── 创建 Trainer
  │   ├── 创建 Dataset (COCODataSet)
  │   ├── 创建 DataLoader (数据增强)
  │   ├── 创建 Model (骨干 + 颈部 + 检测头)
  │   ├── 创建 Optimizer (Adam/SGD)
  │   └── 创建 LR Scheduler (CosineDecay)
  ├── 加载预训练权重
  └── 开始训练循环
      ├── 前向传播 → 损失计算
      ├── 反向传播 → 参数更新
      ├── EMA 更新
      └── 定期评估 + 保存模型
```

---

## 工具脚本详解

### 1. `tools/train.py` - 训练入口

**功能**：训练目标检测模型

**核心流程**：
```python
# 1. 解析命令行参数
parse_args()  # --config, --eval, --resume, --amp, --fleet

# 2. 加载配置
cfg = load_config(FLAGS.config)

# 3. 创建训练器
trainer = Trainer(cfg, mode='train')

# 4. 加载预训练权重
trainer.load_weights(cfg.pretrain_weights)

# 5. 开始训练
trainer.train(FLAGS.eval)
```

**使用命令**：
```bash
python -m paddle.distributed.launch --gpus 0 tools/train.py \
    -c configs/custom/ppyoloe_fire.yml \
    --eval \
    --use_vdl=True \
    --output_dir=output/ppyoloe_fire
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `-c` | 配置文件路径 |
| `--eval` | 训练时是否评估 |
| `-r` | 恢复训练的权重路径 |
| `--amp` | 启用混合精度训练 |
| `--fleet` | 使用分布式训练 |
| `--use_vdl` | 使用 VisualDL 记录 |
| `--slim_config` | 模型压缩配置 |

---

### 2. `tools/eval.py` - 模型评估

**功能**：评估训练好的模型性能

**核心流程**：
```python
# 1. 创建训练器 (评估模式)
trainer = Trainer(cfg, mode='eval')

# 2. 加载权重
trainer.load_weights(cfg.weights)

# 3. 评估
trainer.evaluate()
```

**使用命令**：
```bash
python tools/eval.py \
    -c configs/custom/ppyoloe_fire.yml \
    -o weights=output/ppyoloe_fire/best_model.pdparams
```

**输出指标**：

| 指标 | 说明 |
|------|------|
| mAP | 平均精度 (IoU=0.5:0.95) |
| AP50 | IoU=0.5 时的精度 |
| AP75 | IoU=0.75 时的精度 |
| APs | 小目标精度 |
| APm | 中等目标精度 |
| APl | 大目标精度 |

**参数说明**：

| 参数 | 说明 |
|------|------|
| `-c` | 配置文件路径 |
| `-o weights` | 模型权重路径 |
| `--classwise` | 按类别显示 AP |
| `--json_eval` | 使用已有 JSON 结果评估 |

---

### 3. `tools/infer.py` - 动态图推理

**功能**：使用训练好的模型进行推理预测

**核心流程**：
```python
# 1. 创建训练器 (测试模式)
trainer = Trainer(cfg, mode='test')

# 2. 加载权重
trainer.load_weights(cfg.weights)

# 3. 推理
trainer.predict(
    images,
    draw_threshold=0.5,
    output_dir='output'
)
```

**使用命令**：
```bash
# 推理单张图片
python tools/infer.py \
    -c configs/custom/ppyoloe_fire.yml \
    --infer_img=demo/test.jpg \
    -o weights=output/ppyoloe_fire/best_model.pdparams

# 推理整个目录
python tools/infer.py \
    -c configs/custom/ppyoloe_fire.yml \
    --infer_dir=demo/ \
    -o weights=output/ppyoloe_fire/best_model.pdparams
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--infer_img` | 单张图片路径 |
| `--infer_dir` | 图片目录 |
| `--infer_list` | 图片列表文件 |
| `--output_dir` | 输出目录 |
| `--draw_threshold` | 可视化阈值 |
| `--save_threshold` | 保存阈值 |

---

### 4. `tools/export_model.py` - 导出静态图

**功能**：将动态图模型导出为静态图 (用于部署)

**核心流程**：
```python
# 1. 创建训练器 (测试模式)
trainer = Trainer(cfg, mode='test')

# 2. 加载权重
trainer.load_weights(cfg.weights)

# 3. 导出
trainer.export(
    output_dir='output_inference',
    for_fd=False  # 是否为 FastDeploy 格式
)
```

**使用命令**：
```bash
python tools/export_model.py \
    -c configs/custom/ppyoloe_fire.yml \
    --output_dir=./output_inference \
    -o weights=output/ppyoloe_fire/best_model.pdparams
```

**输出文件**：

| 文件 | 说明 |
|------|------|
| `model.pdmodel` | 模型结构文件 |
| `model.pdiparams` | 模型权重文件 |
| `infer_cfg.yml` | 推理配置文件 |

**参数说明**：

| 参数 | 说明 |
|------|------|
| `-c` | 配置文件路径 |
| `--output_dir` | 输出目录 |
| `--for_fd` | 导出 FastDeploy 格式 |
| `--export_serving_model` | 导出 Serving 格式 |

---

### 5. `tools/post_quant.py` - 训练后量化

**功能**：对训练好的模型进行量化压缩

**核心流程**：
```python
# 1. 创建训练器 (评估模式)
trainer = Trainer(cfg, mode='eval')

# 2. 加载权重
trainer.load_weights(cfg.weights)

# 3. 量化
trainer.post_quant(output_dir)
```

**使用命令**：
```bash
python tools/post_quant.py \
    -c configs/custom/ppyoloe_fire.yml \
    --slim_config=configs/slim/ptq/ppyoloe_ptq.yml \
    --output_dir=./output_quant \
    -o weights=output/ppyoloe_fire/best_model.pdparams
```

**量化配置示例** (`configs/slim/ptq/ppyoloe_ptq.yml`)：
```yaml
slim: PTQ
PTQ:
  quant_config: {
    'weight_quantize_type': 'channel_wise_abs_max',
    'activation_quantize_type': 'moving_average_abs_max',
    'weight_bits': 8,
    'activation_bits': 8
  }
  calib: {batch_size: 32, samples: 100}
```

**效果**：
- 模型大小减少约 4x
- 推理速度提升 2-3x
- 精度损失约 1-2%

---

### 6. `tools/x2coco.py` - 数据格式转换

**功能**：将各种标注格式转换为 COCO 格式

**支持的格式**：

| 格式 | 说明 |
|------|------|
| LabelMe | LabelMe JSON 标注 |
| VOC | Pascal VOC XML 标注 |
| Cityscape | 城市景观数据集 |
| Genet | 自定义格式 |

**使用命令**：
```bash
# LabelMe 转 COCO
python tools/x2coco.py \
    --dataset_type labelme \
    --json_input_dir ./labelme/ \
    --image_input_dir ./images/ \
    --output ./coco/annotations.json

# VOC 转 COCO
python tools/x2coco.py \
    --dataset_type voc \
    --voc_anno_dir ./VOC/Annotations/ \
    --voc_anno_list ./VOC/ImageSets/Main/train.txt \
    --voc_label_list ./VOC/label_list.txt \
    --output ./coco/annotations.json
```

**转换逻辑**：
```python
# LabelMe → COCO
{
    "images": [{"id": 1, "file_name": "xxx.jpg", "width": 1920, "height": 1080}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [x, y, w, h]}],
    "categories": [{"id": 0, "name": "battery"}]
}
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--dataset_type` | 数据集类型 (labelme/voc/cityscape) |
| `--json_input_dir` | LabelMe JSON 目录 |
| `--image_input_dir` | 图片目录 |
| `--output` | 输出文件路径 |
| `--voc_anno_dir` | VOC 标注目录 |
| `--voc_anno_list` | VOC 图片列表 |
| `--voc_label_list` | VOC 类别列表 |

---

### 7. `tools/anchor_cluster.py` - Anchor 聚类

**功能**：使用 K-Means 聚类为数据集生成最优 Anchor 尺寸

**核心流程**：
```python
# 1. 加载数据集
dataset = cfg['TrainDataset']
dataset.parse_dataset()

# 2. 提取所有边界框的宽高
for rec in dataset.roidbs:
    bbox = rec['gt_bbox']
    wh = bbox[:, 2:4] - bbox[:, 0:2] + 1

# 3. K-Means 聚类
centers = kmeans(whs, k=9)  # 9 个 anchor

# 4. 输出结果
print("Anchor sizes:", centers)
```

**使用命令**：
```bash
python tools/anchor_cluster.py \
    -c configs/yolov3/yolov3_darknet53.yml \
    --n 9 \
    --algorithm kmeans
```

**输出示例**：
```
Anchor sizes (width x height):
  [(10, 13), (16, 30), (33, 23),
   (30, 61), (62, 45), (59, 119),
   (116, 90), (156, 198), (373, 326)]
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `-c` | 配置文件路径 |
| `--n` | Anchor 数量 |
| `--algorithm` | 聚类算法 (kmeans/voc) |
| `--cache` | 缓存结果 |

---

### 8. `tools/slice_image.py` - 大图切片

**功能**：将大图片切分成小块，用于小目标检测

**核心流程**：
```python
# 使用 SAHI 库切片
from sahi.scripts.slice_coco import slice

slice(
    image_dir=image_dir,
    dataset_json_path=dataset_json_path,
    output_dir=output_dir,
    slice_size=500,
    overlap_ratio=0.25
)
```

**使用命令**：
```bash
python tools/slice_image.py \
    --image_dir ./images/ \
    --json_path ./annotations.json \
    --output_dir ./sliced/ \
    --slice_size 500 \
    --overlap_ratio 0.25
```

**效果示例**：
```
原始图片: 4000x3000
↓ 切片 (500x500, overlap=0.25)
输出: 48 张小图片 (带重叠区域)
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--image_dir` | 图片目录 | - |
| `--json_path` | COCO 标注路径 | - |
| `--output_dir` | 输出目录 | - |
| `--slice_size` | 切片大小 | 500 |
| `--overlap_ratio` | 重叠比例 | 0.25 |

**应用场景**：
- 航拍图像中的小目标检测
- 遥感图像分析
- 高分辨率图像中的小物体检测

---

## 工具脚本对比总结

| 脚本 | 功能 | 输入 | 输出 | 使用场景 |
|------|------|------|------|---------|
| `train.py` | 训练模型 | 配置文件 + 数据集 | 模型权重 | 模型训练 |
| `eval.py` | 评估模型 | 配置文件 + 权重 | mAP 指标 | 模型验证 |
| `infer.py` | 推理预测 | 配置文件 + 权重 + 图片 | 检测结果 | 模型测试 |
| `export_model.py` | 导出模型 | 配置文件 + 权重 | pdmodel + pdiparams | 模型部署 |
| `post_quant.py` | 量化压缩 | 配置文件 + 权重 | 量化模型 | 模型优化 |
| `x2coco.py` | 格式转换 | 各种标注格式 | COCO JSON | 数据准备 |
| `anchor_cluster.py` | Anchor 聚类 | 数据集 | Anchor 尺寸 | 模型调优 |
| `slice_image.py` | 大图切片 | 大图片 | 小图片 | 小目标检测 |

---

## 部署能力

| 部署方式 | 路径 | 说明 |
|---------|------|------|
| Python Paddle Inference | `deploy/python/` | CPU/GPU/TRT |
| C++ Paddle Inference | `deploy/cpp/` | 高性能 |
| Paddle Serving | `deploy/serving/` | 服务化 |
| Paddle Lite | `deploy/lite/` | 移动端 |
| ONNX | `deploy/EXPORT_ONNX_MODEL.md` | 跨框架 |
| TensorRT | `deploy/TENSOR_RT.md` | NVIDIA 加速 |

---

## 模型压缩

| 方法 | 说明 |
|------|------|
| 知识蒸馏 | 大模型 → 小模型 |
| 结构化剪枝 | 通道剪枝 |
| 非结构化剪枝 | 权重稀疏 |
| 量化感知训练 (QAT) | 训练时量化 |
| 训练后量化 (PTQ) | 免训练量化 |
| OFA | 神经架构搜索 |
