# 火焰检测项目 - firebig 单类目标检测

基于 PaddleDetection 的 PP-YOLOE+ CRN-M 进行单类别火焰检测，输出 firebig 格式提交。
最终 B 榜 F1 = **0.9269**，A 榜 F1 = **0.9016**。

## 环境要求

- Windows 10/11
- NVIDIA GPU（支持 CUDA 12.x）
- cuDNN 8.x+
- Anaconda 或 Miniconda
- RTX 4060 Laptop 8GB VRAM（batch_size=4）

## 安装环境

```bash
conda create -n dog python=3.10 -y
conda activate dog
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

```bash
cd PaddleDetection
pip install -r requirements.txt
python setup.py install
```

```bash
pip install opencv-python pyyaml pillow packaging>=21.0
pip install numpy<2.0 visualdl>=2.2.0 pycocotools==2.0.8 imgaug>=0.4.0
```

## 项目结构

```
dog/
├── A_train/                          # 训练数据
├── mycode/                           # 自定义代码
│   ├── evaluate_fire_predictions.py  # F1 评估（核心）
│   ├── firedetect_public.py          # 评测工具函数库
│   └── analysis/
│       ├── scheme_D_combined.py      # 方案 D 去雾算法
│       └── smoke_results/            # 烟雾方案对比图
├── 最后提交结果/
│   ├── A/                            # A 榜：PP-YOLOE+ full_hr
│   └── B/                            # B 榜：PP-YOLOE+ CRN M + atmosphere
├── A榜.md                            # A 榜结果与历程
├── B榜.md                            # B 榜结果与历程（含错误决定）
├── README.md
├── AGENTS.md
├── Agent.md
└── CLAUDE.md
```

## 最终方案（B 榜）

### 模型

- **架构**：PP-YOLOE+ CRN M（depth_mult=0.67, width_mult=0.75）
- **预训练**：Object365（ppyoloe_crn_m_obj365_pretrained.pdparams）
- **检测类别**：单类 fire（num_classes=1），推理后取最大可信框作为 firebig 提交
- **训练轮数**：100 epoch，batch_size=4
- **学习率**：base_lr=0.0005，CosineDecay + LinearWarmup(5 epochs)
- **优化器**：Momentum(0.9)，L2(0.0005)

### 数据

- **训练集**：merged_fire_coco（混杂数据:纯净数据 = 1:2）
- **多尺度训练**：BatchRandomResize 目标尺寸 [704, 768, 832, 896]
- **增强**：RandomDistort, GaussianBlur(0.08), MotionBlur(0.08), GaussianNoise(0.08), RandomErasing(0.04), RandomExpand, RandomFlip

### 推理

- **Eval/Test 尺寸**：Resize(832x832, keep_ratio=True) + PadStride(32)
- **导出尺寸**：Resize(640x640, keep_ratio=True) + PadStride(32)
- **后处理**：绝对阈值 0.1 + 相对最高分 0.5 筛选，选面积最大框

### 运行

```bash
# 训练
cd PaddleDetection
python tools/train.py -c ../最后提交结果/B/B_fire_atmosphere.yml --eval

# 推理提交
cd 最后提交结果/B
python predict.py <image_list.txt> <result.json>
```

## 关键错误决定（详见 B榜.md）

1. **类别策略**：初尝试 firebig 单类训练 → 修正为 fire 检测 + 取最大框
2. **训练集**：初使用单一纯净数据集 → 修正为混杂数据集（1:2 混合）
3. **模型**：初使用 PP-YOLOE+ S → 修正为 M

## F1 指标

| 方案 | F1 | 说明 |
|------|-----|------|
| B 榜（PP-YOLOE+ CRN M + atmosphere） | **0.9269** | Object365 预训练，混杂数据集 |
| A 榜（PP-YOLOE+ full_hr） | **0.9016** | 3 类检测（battery/board/fire） |

## 常见问题

### CUDA 不可用

```bash
python -c "import paddle; paddle.utils.run_check()"
```

### 训练报错 ValueError: all input arrays must have the same shape

检查 TrainReader.batch_transforms 中是否缺少 PadGT: {}。

### 验证报错 custom_pan.py 中 paddle.concat 尺寸不一致

检查 EvalReader/TestReader 中 Resize(keep_ratio=True) 是否配套了 PadBatch(pad_to_stride=32)。

## 许可证

本项目建设于 PaddleDetection 之上，遵循 Apache 2.0 许可证。