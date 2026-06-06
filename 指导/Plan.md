# 火焰检测目标检测比赛 - 执行计划

## 任务概述

基于 PaddlePaddle 框架训练并部署目标检测模型，对 3 类目标进行检测：
- **battery**（电池）- 类别编号 1
- **board**（指示牌）- 类别编号 2
- **fire**（火焰）- 类别编号 3

**核心约束**：
- FPS ≥ 20（推理速度门槛，否则得分为 0）
- 模型大小 ≤ 200MB
- 使用 PaddlePaddle 框架

---

## 当前项目结构

```
submission_template_firedetect/
├── predict.py                    ✅ 已完成（推理脚本模板）
├── model/
│   ├── infer_cfg.yml            ✅ 已完成（配置文件示例）
│   ├── model.pdmodel            ❌ 缺失（需要训练后导出）
│   └── model.pdiparams          ❌ 缺失（需要训练后导出）
├── mycode/
│   ├── configs/
│   │   └── ppyoloe_fire.yml     ✅ PaddleDetection 训练配置
│   ├── scripts/
│   │   ├── train.bat            ✅ Windows 训练启动脚本
│   │   ├── export_model.bat     ✅ Windows 模型导出脚本
│   │   └── train_paddledet.py   ✅ Python 训练脚本
│   ├── tools/
│   │   └── convert_labelme_to_coco.py  ✅ 数据格式转换
│   └── data/
│       ├── annotations_train.json      ✅ 训练集标注
│       └── annotations_val.json        ✅ 验证集标注
├── PaddleDetection/
│   └── deploy/python/           ✅ 官方部署代码
└── Agent.md                     ✅ 操作规范
```

---

## 执行计划（使用 PaddleDetection 官方脚本）

### 阶段一：环境准备（约 30 分钟）

#### 1.1 安装 PaddlePaddle
```bash
conda activate dog
pip install paddlepaddle-gpu==3.0.0
```

#### 1.2 安装 PaddleDetection
```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip install -r requirements.txt
python setup.py install
```

#### 1.3 验证安装
```bash
python -c "import paddle; print(paddle.__version__)"
python -c "import paddledet; print(paddledet.__version__)"
```

---

### 阶段二：数据准备（已完成 ✅）

数据已转换为 COCO 格式：
- 训练集：324 张图片，743 个标注
- 验证集：81 张图片，187 个标注
- 类别分布：battery: 126, board: 92, fire: 712

---

### 阶段三：模型配置（已完成 ✅）

已创建 PaddleDetection 配置文件：`mycode/configs/ppyoloe_fire.yml`

配置内容：
- 模型：PP-YOLOE-s（预训练权重）
- 输入尺寸：640×640
- 类别数量：3
- 训练轮数：100
- 批次大小：8
- 学习率：0.001（余弦退火）
- 优化器：Adam

---

### 阶段四：模型训练（约 2-4 小时）

#### 4.1 使用批处理脚本（推荐）
```batch
conda activate dog
mycode\scripts\train.bat
```

#### 4.2 或手动执行
```bash
cd PaddleDetection

python -m paddle.distributed.launch --gpus 0 tools/train.py \
    -c ../mycode/configs/ppyoloe_fire.yml \
    --eval \
    --use_vdl=True \
    --output_dir=output/ppyoloe_fire
```

#### 4.3 训练参数说明
| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 100 | 训练轮数 |
| batch_size | 8 | 批次大小 |
| learning_rate | 0.001 | 初始学习率 |
| lr_scheduler | CosineDecay | 余弦退火 |
| warmup_epochs | 5 | 预热轮数 |
| input_size | 640×640 | 输入尺寸 |

---

### 阶段五：模型导出（约 10 分钟）

#### 5.1 使用批处理脚本
```batch
mycode\scripts\export_model.bat
```

#### 5.2 或手动执行
```bash
cd PaddleDetection

python tools/export_model.py \
    -c ../mycode/configs/ppyoloe_fire.yml \
    --output_dir=./output_inference \
    -o weights=output/ppyoloe_fire/best_model.pdparams

# 复制到提交目录
cp output_inference/ppyoloe_fire/model.pdmodel ../model/
cp output_inference/ppyoloe_fire/model.pdiparams ../model/
cp output_inference/ppyoloe_fire/infer_cfg.yml ../model/
```

---

### 阶段六：本地测试（约 30 分钟）

#### 6.1 准备测试数据
```bash
# 创建测试图片列表文件
ls test/Image/*.jpg > test_data.txt
```

#### 6.2 运行推理脚本
```bash
python predict.py test_data.txt result.json
```

#### 6.3 检查输出格式
```bash
python -c "
import json
with open('result.json') as f:
    data = json.load(f)
    print(f'检测到 {len(data[\"result\"])} 个目标')
"
```

#### 6.4 验证 FPS
- 输出日志中会显示 `total time`
- 计算 FPS = 2026 / total_time
- 确保 FPS ≥ 20

---

### 阶段七：打包提交（约 10 分钟）

#### 7.1 检查文件结构
```bash
dir model\
# 应该包含：
# - infer_cfg.yml
# - model.pdmodel
# - model.pdiparams
```

#### 7.2 检查模型大小
```bash
# 确保 model/ 目录总大小 ≤ 200MB
```

#### 7.3 打包提交
```bash
# 在 submission_template_firedetect 目录下
zip -r submission.zip predict.py model/ PaddleDetection/deploy/
```

---

## 关键注意事项

### 1. 类别编号映射
- PaddleDetection 默认输出 0-indexed 类别
- 比赛要求 1-indexed（battery=1, board=2, fire=3）
- **已在 predict.py 中处理**：`int(id_results[idx]) + 1`

### 2. 置信度阈值
- 当前设置：`threshold = 0.3`
- 可根据精度/召回率权衡调整

### 3. FPS 优化建议（如果 FPS < 20）
- 减小输入尺寸：640 → 416
- 使用 TensorRT 加速
- 使用更轻量的模型（如 PicoDet-s）

### 4. 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| FPS < 20 | 推理太慢 | 减小输入尺寸或使用 TensorRT |
| 检测结果为空 | 置信度阈值太高 | 降低 threshold |
| 类别编号错误 | 映射问题 | 检查 label_list 顺序 |
| 模型文件缺失 | 导出失败 | 重新运行 export_model.py |

---

## 预期结果

| 指标 | 预期值 |
|------|--------|
| 模型大小 | ~30MB（远低于 200MB 限制） |
| FPS | 30-50（GPU 环境） |
| F1 Score | 0.6-0.8（取决于数据集质量） |
| 训练时间 | 2-4 小时（单 GPU） |

---

## 时间规划

| 阶段 | 任务 | 状态 | 预计时间 |
|------|------|------|---------|
| 1 | 环境准备 | ⏳ 待执行 | 30 分钟 |
| 2 | 数据准备 | ✅ 已完成 | - |
| 3 | 模型配置 | ✅ 已完成 | - |
| 4 | 模型训练 | ⏳ 待执行 | 2-4 小时 |
| 5 | 模型导出 | ⏳ 待执行 | 10 分钟 |
| 6 | 本地测试 | ⏳ 待执行 | 30 分钟 |
| 7 | 打包提交 | ⏳ 待执行 | 10 分钟 |
| **总计** | | | **3-5 小时** |

---

## 在实验室电脑上执行

将整个 `submission_template_firedetect` 目录复制到实验室电脑，然后执行：

```batch
# 1. 激活环境
conda activate dog

# 2. 运行训练脚本（自动安装 PaddleDetection）
mycode\scripts\train.bat

# 3. 打包提交
zip -r submission.zip predict.py model/ PaddleDetection/deploy/
```

---

## 备选方案

如果 PP-YOLOE-s 无法满足要求，可考虑：

### 方案 A：PicoDet-s
- 更轻量（约 20MB）
- 速度更快（FPS 可达 50+）
- 精度略低

### 方案 B：PP-YOLOE-l
- 更大模型（约 100MB）
- 精度更高
- 速度可能略慢

### 方案 C：使用 TensorRT 加速
- 可将 FPS 提升 2-3 倍
- 需要额外配置
