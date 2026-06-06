# 火焰检测目标检测项目

## 项目概述

基于 PaddlePaddle 框架的目标检测项目，使用 PP-YOLOE-s 模型检测 3 类目标：
- battery（电池）
- board（指示牌）
- fire（火焰）

## 项目结构

```
submission_template_firedetect/
├── predict.py                    # 推理脚本
├── requirements.txt              # 项目依赖
├── model/                        # 模型文件目录
│   ├── infer_cfg.yml            # 推理配置
│   ├── model.pdmodel            # 模型结构
│   └── model.pdiparams          # 模型权重
├── mycode/                       # 自定义代码
│   ├── configs/
│   │   └── ppyoloe_fire.yml     # 训练配置
│   ├── scripts/
│   │   ├── train.bat            # Windows 训练脚本
│   │   └── export_model.bat     # Windows 导出脚本
│   ├── tools/
│   │   └── convert_labelme_to_coco.py
│   └── data/
│       ├── annotations_train.json
│       └── annotations_val.json
├── PaddleDetection/              # PaddleDetection 官方代码
├── Plan.md                       # 执行计划
├── Agent.md                      # 操作规范
└── 提交说明.md                    # 比赛提交说明
```

## 环境配置

### 1. 创建 conda 环境

```bash
conda create -n dog python=3.10
conda activate dog
```

### 2. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装 PaddleDetection
cd PaddleDetection
pip install -r requirements.txt
python setup.py install
cd ..
```

### 3. 验证安装

```bash
python -c "import paddle; print('PaddlePaddle:', paddle.__version__)"
python -c "import ppdet; print('PaddleDetection OK')"
```

## 训练流程

### 方法一：使用批处理脚本（推荐）

```bash
conda activate dog
mycode\scripts\train.bat
```

### 方法二：手动执行

```bash
conda activate dog
cd PaddleDetection

# 训练
python -m paddle.distributed.launch --gpus 0 tools/train.py \
    -c configs/custom/ppyoloe_fire.yml \
    --eval \
    --use_vdl=True \
    --output_dir=output/ppyoloe_fire

# 评估
python tools/eval.py \
    -c configs/custom/ppyoloe_fire.yml \
    -o weights=output/ppyoloe_fire/best_model.pdparams

# 导出
python tools/export_model.py \
    -c configs/custom/ppyoloe_fire.yml \
    --output_dir=./output_inference \
    -o weights=output/ppyoloe_fire/best_model.pdparams
```

## 提交打包

```bash
# 复制模型文件到 model/ 目录
copy PaddleDetection\output_inference\ppyoloe_fire\model.pdmodel model\
copy PaddleDetection\output_inference\ppyoloe_fire\model.pdiparams model\
copy PaddleDetection\output_inference\ppyoloe_fire\infer_cfg.yml model\

# 打包
zip -r submission.zip predict.py model/ PaddleDetection/deploy/
```

## 模型性能

| 指标 | 预期值 |
|------|--------|
| 模型大小 | ~28MB |
| FPS | 30-50 |
| F1 Score | 0.6-0.8 |
| 训练时间 | 2-4 小时 |

## 常见问题

### 1. 显存不足

修改 `mycode/configs/ppyoloe_fire.yml` 中的 `batch_size`：
```yaml
TrainReader:
  batch_size: 4  # 从 8 减小到 4
```

### 2. FPS 不满足要求

- 使用 TensorRT 加速
- 减小输入尺寸
- 使用更轻量的模型

### 3. 类别编号问题

比赛要求 1-indexed，predict.py 已处理：
```python
"type": int(id_results[idx]) + 1
```
