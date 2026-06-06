# 火焰检测训练代码

本目录包含火焰检测目标检测任务的训练代码，使用 PaddleDetection 官方脚本。

## 目录结构

```
mycode/
├── configs/                        # 训练配置文件
│   └── ppyoloe_fire.yml            # PaddleDetection 训练配置
├── scripts/                        # 训练脚本
│   ├── train.bat                   # Windows 训练启动脚本
│   ├── export_model.bat            # Windows 模型导出脚本
│   ├── train_paddledet.py          # Python 训练脚本
│   └── install_paddledet.bat       # PaddleDetection 安装脚本
├── tools/                          # 工具脚本
│   └── convert_labelme_to_coco.py  # 数据格式转换
└── data/                           # 数据处理相关
    ├── annotations_all.json        # 全部数据 COCO 格式
    ├── annotations_train.json      # 训练集
    └── annotations_val.json        # 验证集
```

## 快速开始

### 方法一：Windows 批处理脚本（推荐）

```batch
# 激活环境
conda activate dog

# 运行训练脚本（自动安装 PaddleDetection）
mycode\scripts\train.bat

# 或者单独导出模型
mycode\scripts\export_model.bat
```

### 方法二：Python 脚本

```bash
conda activate dog
python mycode/scripts/train_paddledet.py
```

### 方法三：手动执行

```bash
# 1. 激活环境
conda activate dog

# 2. 克隆并安装 PaddleDetection
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip install -r requirements.txt
python setup.py install
cd ..

# 3. 复制配置文件
copy mycode\configs\ppyoloe_fire.yml PaddleDetection\configs\custom\

# 4. 训练模型
cd PaddleDetection
python -m paddle.distributed.launch --gpus 0 tools/train.py ^
    -c configs/custom/ppyoloe_fire.yml ^
    --eval ^
    --use_vdl=True ^
    --output_dir=output/ppyoloe_fire
cd ..

# 5. 导出模型
cd PaddleDetection
python tools/export_model.py ^
    -c configs/custom/ppyoloe_fire.yml ^
    --output_dir=./output_inference ^
    -o weights=output/ppyoloe_fire/best_model.pdparams
cd ..

# 6. 复制到提交目录
copy PaddleDetection\output_inference\ppyoloe_fire\model.pdmodel model\
copy PaddleDetection\output_inference\ppyoloe_fire\model.pdiparams model\
copy PaddleDetection\output_inference\ppyoloe_fire\infer_cfg.yml model\
```

## 配置说明

### ppyoloe_fire.yml 配置

```yaml
# 模型配置
pretrain_weights: https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams
num_classes: 3
input_size: [640, 640]

# 训练配置
epoch: 100
batch_size: 8
learning_rate: 0.001
lr_scheduler: CosineDecay
warmup_epochs: 5

# 数据集配置
TrainDataset:
  image_dir: D:/work/Marchine Dog/A_train/Image
  anno_path: mycode/data/annotations_train.json
```

### 数据集统计

| 类别 | 数量 |
|------|------|
| battery | 126 |
| board | 92 |
| fire | 712 |
| **总计** | **930** |

| 数据集 | 图片数 | 标注数 |
|--------|--------|--------|
| 训练集 | 324 | 743 |
| 验证集 | 81 | 187 |

## 模型导出

训练完成后，使用以下命令导出模型：

```bash
# 使用批处理脚本
mycode\scripts\export_model.bat

# 或手动执行
cd PaddleDetection
python tools/export_model.py ^
    -c ../mycode/configs/ppyoloe_fire.yml ^
    --output_dir=./output_inference ^
    -o weights=output/ppyoloe_fire/best_model.pdparams
```

导出的模型文件会保存到 `model/` 目录：
- `model.pdmodel` - 模型结构文件
- `model.pdiparams` - 模型权重文件
- `infer_cfg.yml` - 推理配置文件

## 打包提交

```bash
# 检查模型文件
dir model\

# 打包
zip -r submission.zip predict.py model/ PaddleDetection/deploy/
```

## 注意事项

1. **GPU 要求**: 训练需要 GPU，推荐使用实验室电脑
2. **显存要求**: batch_size=8 约需 8GB 显存
3. **训练时间**: 约 2-4 小时（取决于 GPU）
4. **模型大小**: PP-YOLOE-s 约 28MB，满足 200MB 限制
5. **FPS**: PP-YOLOE-s 可达 30+ FPS，满足 20 FPS 要求

## 常见问题

### 1. PaddleDetection 安装失败

```bash
# 尝试使用 pip 安装
pip install paddledet

# 或从源码安装
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip install -r requirements.txt
python setup.py install
```

### 2. 显存不足

修改配置文件中的 `batch_size`：
```yaml
TrainReader:
  batch_size: 4  # 从 8 减小到 4
```

### 3. 训练速度慢

- 使用多 GPU 训练：
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py ...
```

- 减小输入尺寸：
```yaml
input_size: [416, 416]  # 从 640 减小到 416
```

### 4. FPS 不满足要求

- 使用 TensorRT 加速
- 使用更轻量的模型（如 PicoDet-s）
- 减小输入尺寸
