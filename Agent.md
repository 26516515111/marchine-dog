# Agent 操作规范

## 环境要求

**所有操作必须在 `dog` conda 环境下执行！**

### 激活环境
```bash
conda activate dog
```

### 验证环境
```bash
python -c "import paddle; print(f'PaddlePaddle {paddle.__version__}')"
```

## 项目结构

```
dog/
├── mycode/                    # 所有自定义代码放在这里
│   ├── configs/               # 训练配置文件
│   │   └── ppyoloe_fire.yml   # PaddleDetection 训练配置
│   ├── scripts/               # 训练脚本
│   │   ├── train.bat          # Windows 训练启动脚本
│   │   ├── export_model.bat   # Windows 模型导出脚本
│   │   └── train_paddledet.py # Python 训练脚本
│   ├── tools/                 # 工具脚本
│   └── data/                  # 数据处理相关
├── model/                     # 模型文件（导出后）
├── PaddleDetection/           # 官方部署代码
├── predict.py                 # 推理脚本
└── Agent.md                   # 本文件
```

## 关键路径

- **数据集路径**: `D:\work\Marchine Dog\dog\A_train`
- **代码目录**: `D:\work\Marchine Dog\dog\mycode`
- **模型输出**: `D:\work\Marchine Dog\dog\model`

## 训练流程（使用 PaddleDetection 官方脚本）

### 方法一：Windows 批处理脚本（推荐）

```batch
# 激活环境
conda activate dog

# 进入项目目录
cd D:\work\Marchine Dog\dog

# 运行训练脚本
mycode\scripts\train.bat

# 或者单独导出模型
mycode\scripts\export_model.bat
```

### 方法二：Python 脚本

```bash
conda activate dog
cd D:\work\Marchine Dog\dog
python mycode/scripts/train_paddledet.py
```

### 方法三：手动执行

```bash
# 1. 激活环境
conda activate dog

# 2. 进入项目目录
cd D:\work\Marchine Dog\dog

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

## 打包提交

```bash
# 进入项目目录
cd D:\work\Marchine Dog\dog

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
