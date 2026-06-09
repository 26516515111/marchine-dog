# 火焰检测项目

基于 PaddleDetection 的 PP-YOLOE-s 模型进行火焰/电池/指示牌目标检测。

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
│   ├── predict.py                    # 推理脚本
│   ├── calculate_f1.py               # F1 评估脚本
│   ├── find_best_thresholds.py       # 阈值优化脚本
│   ├── test_fps.py                   # FPS 测试脚本
│   ├── hard_negative_mining.py       # Hard Negative 挖掘脚本
│   ├── analyze_fp.py                 # FP 分析脚本
│   ├── analyze_fp_features.py        # FP 特征分析脚本
│   ├── analyze_fp_stats.py           # FP 统计分析脚本
│   ├── add_hard_negative.py          # 添加 Hard Negative 脚本
│   └── check_conf.py                 # 检查 conf 分布脚本
├── PaddleDetection/                  # PaddleDetection 框架
├── ppyolo/                           # 预训练权重
├── README.md                         # 本文件
└── requirements.txt                  # 依赖列表
```

## 类别说明

| 类别 ID | 类别名称 | 描述 |
|---------|----------|------|
| 1 | battery | 电池 |
| 2 | board | 指示牌 |
| 3 | fire | 火焰 |

## 训练验证指令

### 1. 训练模型

```bash
cd PaddleDetection
python tools/train.py -c configs/custom/ppyoloe_fire.yml --eval
```

### 2. 导出模型

```bash
cd PaddleDetection
python tools/export_model.py -c configs/custom/ppyoloe_fire.yml --output_dir=./output_inference -o weights=output/ppyoloe_fire/best_model.pdparams
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
python calculate_f1.py val_result.json val_ground_truth.json
```

### 5. 测试 FPS

```bash
python test_fps.py
```

## 训练配置

当前配置（`mycode/configs/ppyoloe_fire.yml`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | PP-YOLOE-s | 预训练权重：COCO |
| Epoch | 200 | 训练轮数 |
| Batch Size | 8 | 批次大小 |
| 学习率 | 0.0005 | 初始学习率 |
| 优化器 | Momentum | 动量=0.9 |
| 权重衰减 | 0.0005 | L2 正则化 |

### 数据增强

| 增强操作 | 概率 | 说明 |
|----------|------|------|
| Mosaic | 0.3 | 四图拼接 |
| Mixup | 0.2 | 两图混合 |
| RandomDistort | - | 颜色抖动（亮度/对比度/饱和度/色调） |
| MotionBlur | 0.3 | 运动模糊 |
| GaussianNoise | 0.2 | 高斯噪声 |
| RandomExpand | - | 随机扩展 |
| RandomCrop | - | 随机裁剪 |
| RandomFlip | - | 随机水平翻转 |
| BatchRandomResize | - | 多尺度训练（640/768/896） |

### 类别权重

针对类别不平衡问题，设置了不同的损失权重：

| 类别 | 权重 | 样本占比 |
|------|------|----------|
| battery | 1.5 | 13.5% |
| board | 2.0 | 9.9% |
| fire | 1.2 | 76.6% |

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
  batch_size: 4  # 原为 8
```

## 性能指标

| 指标 | 值 |
|------|-----|
| F1 分数 | 0.80614 |
| 推理速度 | ≥ 20 FPS |
| 模型大小 | ≤ 200MB |

## 许可证

本项目基于 PaddleDetection 开发，遵循 Apache 2.0 许可证。
