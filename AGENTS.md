# AGENTS.md - 火焰检测项目

## 项目结构（当前状态）

- `最后提交结果/A/` 和 `最后提交结果/B/` 是最终提交方案的权威来源
- `mycode/` 包含 F1 评估工具和方案分析代码
- `B榜.md` 记录 B 榜最终方案、配置参数和错误决定
- `A榜.md` 记录 A 榜历程（3 类检测：battery/board/fire）
- 训练在 `PaddleDetection/` 目录下执行，配置文件引用自 `最后提交结果/`

## 配置文件

| 方案 | 配置路径 | 说明 |
|------|----------|------|
| B 榜 | `最后提交结果/B/B_fire_atmosphere.yml` | PP-YOLOE+ CRN M，单类 fire |
| A 榜 | `最后提交结果/A/ppyoloe_plus_fire_full_hr.yml` | PP-YOLOE+ full_hr，3 类 |

## 性能指标

| 方案 | F1 | 关键策略 |
|------|-----|----------|
| B 榜 | **0.9269** | 单类 fire + 混杂数据集 + M 模型 + 阈值 0.1 |
| A 榜 | **0.9016** | 3 类检测 + high-res 训练 |

### F1 演进（B 榜）

| 阶段 | F1 | 调整 |
|------|-----|------|
| 初始（S + 单一数据集） | 0.63 | baseline |
| + 混杂数据集（1:2） | 0.78 | 数据策略修正 |
| + M 模型 | 0.90 | 模型升级 |
| + 阈值调整到 0.1 | **0.9269** | 推理后处理调优 |

## 推理后处理

`最后提交结果/B/predict.py` 的后处理逻辑：

1. 候选框按 FIRE_LABELS = {fire, firebig} 过滤
2. 动态阈值：max(0.02, 0.1, 0.5 * max_score)
3. 从可信候选中选择面积最大的框
4. 输出 firebig 格式 JSON

## 配置陷阱（PaddleDetection）

- `_BASE_` 继承的 `ppyoloe_crn.yml` 默认 `width_mult=1.0`，必须显式设置
- 缩放因子：s=(0.33,0.50), m=(0.67,0.75), l=(1.0,1.0), x=(1.33,1.25)
- `PPYOLOEHead.loss_weight` 仅接受 {class, iou, dfl} dict
- VarifocalLoss 不支持 per-class 权重
- `PadBatch` 必须配套 `PadGT: {}`
- `Resize(keep_ratio=True)` 必须配套 `PadBatch(pad_to_stride=32)`

## 环境

- PaddlePaddle GPU 3.0.0, CUDA 12.x, conda env `dog`
- RTX 4060 Laptop 8GB VRAM
- 路径使用 `D:/work/Marchine_Dog/dog/` 格式