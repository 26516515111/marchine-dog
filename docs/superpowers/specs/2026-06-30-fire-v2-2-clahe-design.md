# 火焰检测 V2.2 LAB-CLAHE 设计

## 目标

在 `fire/v2.1` 和 `fire/predict.py` 的基础上生成隔离的 V2.2：

- 取消模型输入端的 HSV 橙色区域增强。
- 使用方案 A 的 LAB-L 通道 CLAHE 去黑雾。
- 训练、评估、PaddleDetection 测试和导出后的独立预测使用相同算法及参数。
- 保留 V2.1 文件和行为，避免破坏旧模型。

本次只生成训练配置、算子、V2.2 推理脚本、测试和说明文档，不复制 V2.1 权重，也不声称已有 V2.2 模型。V2.2 必须重新训练并导出。

## 版本边界

新增目录 `fire/v2.2`，包含：

- `B_fire_atmosphere_v2_2.yml`
- `predict.py`
- `氛围火焰v2.2更新记录.md`
- 训练导出后由用户放入的 `model/`

现有 `fire/predict.py`、`fire/v2.1` 和 V2.1 模型保持不变。

PaddleDetection 的算子注册仍位于：

- `dog/PaddleDetection/ppdet/data/transform/operators.py`

新算子使用独立名称 `CLAHEDehaze`，原有 V2.1 算子代码继续保留，以兼容旧配置。

## CLAHE 算法

算子接收 `Decode` 输出的 RGB `uint8` 图像，执行：

1. `cv2.COLOR_RGB2LAB`
2. 对 LAB 的 L 通道执行 CLAHE
3. `cv2.COLOR_LAB2RGB`

参数为：

```yaml
clip_limit: 3.0
tile_grid_size: [8, 8]
```

推理脚本从导出模型的 `infer_cfg.yml` 读取同名算子和这两个参数，不在脚本中维护另一套可漂移的业务参数。推理实现与训练实现使用相同的 RGB 输入约定、OpenCV 转换和 CLAHE 调用顺序。

算子是确定性的，没有概率开关；同一输入和参数必须产生相同输出。

## 训练与导出数据流

V2.2 的三个 Reader 都在 `Decode` 后立即使用 `CLAHEDehaze`：

```text
Train: Decode -> CLAHEDehaze -> 常规随机增强 -> Resize/Normalize/Permute
Eval:  Decode -> CLAHEDehaze -> Resize/Normalize/Permute
Test:  Decode -> CLAHEDehaze -> Resize/Normalize/Permute
```

`TestReader` 中的算子会被 PaddleDetection 导出逻辑写入 `infer_cfg.yml`：

```yaml
Preprocess:
  - type: CLAHEDehaze
    clip_limit: 3.0
    tile_grid_size: [8, 8]
  - type: Resize
  - type: NormalizeImage
  - type: Permute
  - type: PadStride
```

因此，独立推理脚本可以根据模型元数据构建 CLAHE 预处理，参数来源与训练配置一致。

## V2.1 算子的处理

V2.2 配置移除以下 V2.1 氛围增强算子：

- `AtmosphereFireEnhance`
- `BackgroundDesaturate`
- `OrangeBloom`
- `UnsharpMask`

前三者依赖 HSV 特征，最后一个属于 V2.1 配套增强；整组移除可让 V2.2 的核心变化保持单一、可消融。原有通用增强如有界 `RandomDistort`、低概率 `GaussianBlur`、低概率 `GaussianNoise`、`RandomExpand`、`RandomFlip` 和 `BatchRandomResize` 继续保留。

CLAHE 必须位于 `RandomDistort` 之前，确保模型先看到与预测一致的确定性去雾结果，再叠加仅训练阶段使用的随机增强。

## 推理数据流

`fire/v2.2/predict.py` 基于现有 `fire/predict.py`，默认模型目录改为 `fire/v2.2/model`。

推理流程为：

```text
cv2.imread(BGR)
-> BGR 转 RGB
-> infer_cfg 中的 CLAHEDehaze
-> Resize/Normalize/Permute/PadStride
-> PP-YOLOE 推理
-> 现有框选择和提交格式
```

删除的内容仅为 `_hsv_enhance_atmosphere()` 及其输入增强参数和调用。C1 中对检测框橙色像素的 HSV 统计继续保留，因为它属于输出置信度修正，不改变模型输入图像。C2 阈值、强制选框、合框及 `firebig` 输出格式保持不变。

若 `infer_cfg.yml` 缺少 `CLAHEDehaze`，V2.2 推理脚本应明确报错，防止误用旧模型时静默跳过去雾预处理。对于其他未知预处理算子也应保持显式错误，避免不完整预处理。

## 错误处理

`CLAHEDehaze` 在构造时验证：

- `clip_limit` 必须大于 0。
- `tile_grid_size` 必须是两个正整数。

在执行时验证：

- 图像必须是三通道 NumPy 数组。
- 训练算子接收 RGB 图像；推理算子在 BGR 转 RGB 后调用。

图片读取失败沿用现有 `preprocess()` 的 `(None, None)` 返回行为。模型配置缺失、CLAHE 配置缺失或参数非法则尽早抛出带明确信息的异常。

## 测试

测试按以下层次覆盖：

1. 训练算子注册测试：`CLAHEDehaze` 可由 PaddleDetection 配置构造。
2. 算法测试：保持形状和 `uint8` 类型，确定性执行，并改变非均匀低对比度图像的亮度分布。
3. 参数校验测试：拒绝非正 `clip_limit` 和非法网格。
4. 训练/推理一致性测试：对同一合成 RGB 图像和相同参数，训练算子与 V2.2 推理算子输出逐像素相同。
5. 配置测试：Train/Eval/Test 均在 `Decode` 后包含 `CLAHEDehaze`，且不包含四个 V2.1 氛围算子。
6. 推理配置测试：缺少 `CLAHEDehaze` 时拒绝运行，存在时只执行一次。
7. 回归测试：现有 C1/C2、框选择和结果格式测试继续通过。

## 验收标准

- `fire/v2.2` 是完整、隔离的配置与推理版本。
- V2.2 不再对模型输入做 HSV 增强。
- Train/Eval/Test 与导出推理均使用 `clip_limit=3.0`、`tile_grid_size=[8,8]`。
- 训练算子和推理算子在测试输入上逐像素一致。
- V2.1 文件、旧推理脚本和旧算子不被删除或改变行为。
- 所有新增测试及相关现有回归测试通过。
