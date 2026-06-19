# 火焰检测项目

基于 PaddleDetection 的 PP-YOLOE-s 进行火焰/电池/指示牌目标检测。

## 环境要求

- Windows 10/11
- NVIDIA GPU（支持 CUDA）
- CUDA 12.x
- cuDNN 8.x+
- Anaconda 或 Miniconda

## 安装环境

### 1. 创建 Conda 环境

```bash
conda create -n dog python=3.10 -y
conda activate dog
```

### 2. 安装 PaddlePaddle GPU 版本

```bash
# 安装 PaddlePaddle GPU 版本（CUDA 12.3）
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

验证安装：
```bash
python -c "import paddle; print('CUDA:', paddle.is_compiled_with_cuda()); paddle.utils.run_check()"
```

应输出：
```
CUDA: True
PaddlePaddle works well on 1 GPU.
```

### 3. 克隆并安装 PaddleDetection

```bash
cd D:\work\Marchine Dog\dog
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip install -r requirements.txt
python setup.py install
```

### 4. 安装其他依赖

```bash
pip install opencv-python<=4.6.0 pyyaml pillow packaging>=21.0
pip install numpy<2.0 visualdl>=2.2.0 pycocotools==2.0.8 imgaug>=0.4.0
```

### 5. 验证环境

```bash
python -c "from ppdet.data.transform.operators import MotionBlur, GaussianNoise; print('Enhancement operators loaded successfully!')"
```

## 项目结构

```
dog/
├── A_train/                          # 训练数据
│   ├── Image/                        # 原始图片（405张）
│   ├── label/                        # LabelMe 格式标注
│   └── coco/                         # COCO 格式数据
│       ├── train/                    # 训练集图片（324张）
│       ├── val/                      # 验证集图片（81张）
│       └── annotations/              # COCO 标注文件
│           ├── instance_train.json
│           └── instance_val.json
├── model/                            # 导出的推理模型
│   ├── model.pdmodel
│   ├── model.pdiparams
│   └── infer_cfg.yml
├── mycode/                           # 自定义代码
│   ├── predict.py                    # 推理脚本（含 NMS + 扩框）
│   ├── calculate_f1.py               # F1 评估脚本
│   ├── find_best_thresholds.py       # 阈值优化脚本
│   ├── find_best_thresholds_v2.py    # 阈值优化脚本 v2
│   ├── test_fps.py                   # FPS 测试脚本
│   ├── hard_negative_mining.py       # Hard Negative 挖掘脚本
│   ├── analyze_fp.py                 # FP 分析脚本
│   ├── analyze_fp_features.py        # FP 特征分析脚本
│   ├── analyze_fp_stats.py           # FP 统计分析脚本
│   ├── add_hard_negative.py          # 添加 Hard Negative 脚本
│   ├── check_conf.py                 # 检查 conf 分布脚本
│   ├── copy_paste_augmentation.py    # Copy-Paste 数据增强脚本
│   ├── offline_augmentation.py       # 离线数据增强脚本
│   └── configs/                      # 训练配置备份
│       ├── ppyoloe_fire.yml          # PP-YOLOE-s 训练配置
│       ├── ppyoloe_fire_a1.yml       # PP-YOLOE-s A1 调参版
│       ├── ppyoloe_fire_hn.yml       # Hard Negative 训练配置
│       └── ppyoloe_crn.yml           # PP-YOLOE 模型架构配置
├── PaddleDetection/                  # PaddleDetection 框架
├── RT-DETR/                          # RT-DETR 方案
│   ├── configs/rtdetr_r18vd_fire.yml #   RT-DETR-R18 训练配置
│   ├── README.md                     #   方案文档
│   └── train.bat                     #   训练脚本
├── ppyolo/                           # 预训练权重
├── README.md                         # 本文件
├── AGENTS.md                         # Agent 配置
├── Agent.md                          # 配置变更历史
├── 训练结果改进调研报告.md            # 优化方案调研
└── requirements.txt                  # 依赖列表
```

## 类别说明

| 类别 ID | 类别名称 | 描述 |
|---------|----------|------|
| 1 | battery | 电池 |
| 2 | board | 指示牌 |
| 3 | fire | 火焰 |

## 训练验证指令

训练配置使用相对数据路径，默认从 `PaddleDetection/` 目录启动。移动项目到其他电脑时，只要保持 `dog/PaddleDetection` 和 `dog/A_train` 的相对位置不变，无需改配置中的数据路径。

### 1. 训练模型

```bash
cd PaddleDetection
python tools/train.py -c configs/custom/ppyoloe_fire.yml --eval
```

### 2. 导出模型

```bash
cd PaddleDetection
python tools/export_model.py -c configs/custom/ppyoloe_fire.yml --output_dir=../../model -o weights=output/best_model.pdparams
```

### 3. 推理预测

```bash
python predict.py <data_txt> <result_json>
```

示例：
```bash
python predict.py val.txt val_result.json
```

### 4. 评估 F1 值

```bash
python mycode/calculate_f1.py val_result.json dog/A_train/coco/annotations/instance_val.json
```

### 5. 测试 FPS

```bash
python mycode/test_fps.py
```

## 训练配置

### PP-YOLOE-s 当前配置（`mycode/configs/ppyoloe_fire.yml`）

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | PP-YOLOE-s | 预训练权重：COCO，depth_mult=0.33, width_mult=0.50 |
| Epoch | 200 | 小数据集训练轮数 |
| Batch Size | 4 | 受限于 RTX 4060 8GB VRAM |
| 学习率 | 0.01 | CosineDecay + LinearWarmup(5 epochs) |
| 优化器 | Momentum | 动量=0.9 |
| 权重衰减 | 0.0005 | L2 正则化 |
| close_mosaic | 30 | 最后 30 轮关闭 Mosaic/Mixup |
| NMS threshold | 0.55 | 降低以减少漏检 |
| Score threshold | 0.01 | 导出模型使用低阈值 |

### 推理后处理配置（`predict.py`）

| 参数 | 值 | 说明 |
|------|-----|------|
| class_thresholds | {1: 0.4, 2: 0.4, 3: 0.5} | 最优阈值（F1=0.9189） |
| nms_threshold | 0.55 | NMS IoU 阈值 |
| bbox_scales | {1: (1.55, 1.15), 2: (1.0, 1.0), 3: (1.18, 1.18)} | 扩框比例 |

处理顺序：阈值过滤 → class-wise NMS → battery/fire 扩框 → clip 到图片边界 → 输出

### 数据增强配置

| 增强操作 | 概率 | 说明 |
|----------|------|------|
| Mosaic | 0.5 | 四图拼接 + Mixup(0.3) |
| GridMask | 0.4 | 网格遮挡正则化 |
| RandomDistort | - | 颜色抖动（亮度/对比度/饱和度/色调） |
| MotionBlur | 0.3 | 运动模糊 |
| GaussianNoise | 0.2 | 高斯噪声 |
| RandomExpand | - | 随机扩展 |
| RandomCrop | - | 随机裁剪 |
| RandomFlip | - | 随机水平翻转 |
| BatchRandomResize | - | 多尺度训练（640/768/896） |

### 类别分布

| 类别 | 实例数 | 占比 |
|------|--------|------|
| battery | 151 | 15.4% |
| board | 112 | 11.4% |
| fire | 717 | 73.2% |

> **注意**：VarifocalLoss 不支持 per-class 权重，`class_weight` 配置会被静默忽略。
> 类别不平衡通过推理后处理（NMS + 扩框）解决。

## 性能指标

### 最终成绩（2026-06-12）

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

### 泛化验证

训练集/验证集 F1 差距仅 2.2%，无过拟合风险。

## 常见问题

### 1. CUDA 不可用

检查 CUDA 环境变量：
```bash
echo %CUDA_HOME%
echo %PATH%
```

如未设置，执行：
```bash
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3
set PATH=%CUDA_HOME%\bin;%PATH%
```

### 2. PaddleDetection 导入错误

确保已安装 PaddleDetection：
```bash
cd PaddleDetection
python setup.py install
```

### 3. 增强算子未找到

确保使用的是 dog 环境的 Python：
```bash
where.exe python
# 应显示 D:\Anaconda\envs\dog\python.exe
```

### 4. 内存不足

减小 batch_size：
```yaml
# 修改 ppyoloe_fire.yml
TrainReader:
  batch_size: 2  # 默认为 4
```

### 5. 配置不生效

`class_weight` 列表格式在 PPYOLOEHead 中不被识别（仅支持 `{class, iou, dfl}` dict）。
VarifocalLoss 不支持 per-class 权重，需通过推理后处理解决类别不平衡。
详见 `Agent.md` 和 `AGENTS.md`。

### 6. 训练时报错 `ValueError: all input arrays must have the same shape`

这是 `DataLoader` 组 batch 时的典型配置问题，不是图片文件本身损坏。

常见根因：
- batch 内不同图片的目标框数量不同
- `TrainReader.batch_transforms` 里缺少 `PadGT: {}`
- `default_collate_fn` 在拼接 `gt_bbox / gt_class / is_crowd` 时直接 `np.stack` 失败

修复方式：
```yaml
TrainReader:
  batch_transforms:
    - PadBatch: {pad_to_stride: 32}
    - PadGT: {}
```

快速校验：
```bash
python mycode/scripts/validate_reader_batch_transforms.py
```

如果你在自定义 `RandomResize`、`keep_ratio`、`PadBatch` 或 reader 流程后再次遇到这个报错，先检查 `PadGT` 是否还在。

### 7. 验证时报错 `custom_pan.py` 中 `paddle.concat` 尺寸不一致

典型报错类似：
- `input[0] shape = [1, 192, 28, 48]`
- `input[1] shape = [1, 256, 27, 48]`

这通常不是模型权重损坏，而是 `EvalReader/TestReader` 的输入尺寸没有对齐到稳定的 32 倍数。

常见根因：
- 使用了 `Resize(..., keep_ratio=True)`
- 但没有额外做 `PadBatch`
- `Resize(keep_ratio=True)` 只等比缩放，不会自动补边
- 结果某些图片在进入 PP-YOLOE 的 FPN/PAN 后，上采样分支和旁路分支会出现 `27/28` 这种错一格

修复方式：
```yaml
EvalReader:
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [768, 768], keep_ratio: True, interp: 2}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    - Permute: {}
  batch_transforms:
    - PadBatch: {pad_to_stride: 32}

TestReader:
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [768, 768], keep_ratio: True, interp: 2}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    - Permute: {}
  batch_transforms:
    - PadBatch: {pad_to_stride: 32}
```

快速校验：
```bash
python mycode/scripts/validate_reader_batch_transforms.py
```

如果你打算保留 `keep_ratio=True` 的验证/测试流程，就把 `PadBatch` 当成配套项，不要单独开启。

## 许可证

本项目基于 PaddleDetection 开发，遵循 Apache 2.0 许可证。
