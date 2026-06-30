# Fire 原始推理、最大火焰选择与 F1/FP 分析设计

## 目标

为 `D:\work\Marchine_Dog\fire\model` 生成与导出模型配置一致的
`D:\work\Marchine_Dog\fire\predict.py`，并在
`D:\work\Marchine_Dog\dog\A_train\sample100_coco` 的 100 张图片上评估
最大火焰框 F1。另行导出所有 FP 图片，图片上同时显示真值框、预测框、
置信度和 IoU。

## 已确认口径

- 模型训练类别是 `fire`，任务输出是每张图片中的最大火焰 `firebig`。
- 推理预处理只执行导出目录 `infer_cfg.yml` 声明的算子。
- 不启用 HSV 增强、饱和度分数增益、框合并、坐标校准、强制输出等额外处理。
- 使用 `sample100_coco/annotations/instance_train_full.json` 的全部 100 张图片。
- IoU 严格大于 `0.5` 记为 TP，否则该预测记为 FP，相应未匹配真值记为 FN。
- 不在 sample100 上搜索选择规则参数，避免用测试集调参。

## 推理流程

`predict.py` 参考现有脚本保留以下能力：

1. 从文本文件读取待推理图片路径。
2. 从 `model/infer_cfg.yml` 动态构建 `Resize`、`NormalizeImage`、
   `Permute` 和 `PadStride`。
3. 使用 Paddle Inference 加载 `model.pdmodel` 与 `model.pdiparams`。
4. 使用模型已经执行 NMS 后的原始候选框，不进行二次 NMS 或框融合。
5. 将模型标签 `fire` 或导出配置中的兼容标签 `firebig` 都视为火焰候选。

导出模型的 `infer_cfg.yml` 是推理事实来源。当前文件声明 640×640
Resize；训练 YAML 中的 832×832 EvalReader 不覆盖已导出的推理配置。

## 最大火焰选择

对每张图片的原始候选框：

1. 令最高候选置信度为 `s_max`。
2. 保留 `score >= max(0.02, 0.5 * s_max)` 的候选框。
3. 在保留候选中选择面积最大的框。
4. 若面积相同，依次以更高置信度和模型原始顺序作为稳定决胜条件。
5. 将最终框输出为任务要求的 `firebig` JSON；结果中保留 `score`，
   便于评估和排错。

其中 `0.02` 来自训练配置 NMS 的最低分数，`0.5` 是预先固定的相对门控系数。
该规则先排除相对明显不可信的大框，再在可信候选中贯彻“最大火焰”目标。

## 输出

`predict.py` 继续支持：

```text
python fire/predict.py <image_list.txt> <result.json> [model_dir]
```

结果 JSON 保持现有 `{"result": [...]}` 外层结构，每条结果包含：
`image_id`、`type=1`、`x`、`y`、`width`、`height`、`segmentation=[]` 和
`score`。

评估产物写入 `sample100_coco` 下的新目录，避免覆盖已有历史结果：

- 原始预测 JSON；
- F1 汇总 JSON；
- FP 分析文件夹；
- FP 汇总 JSON。

FP 图片使用绿色绘制 GT、红色绘制预测，并标注预测分数与 IoU。无预测导致的
FN 不属于 FP 图片，但会记录在汇总 JSON 中。

## F1 计算

sample100 每张图片只有一个 `firebig` 真值，预测最多一个框：

- `IoU > 0.5`：TP 加一；
- 有预测但 `IoU <= 0.5`：FP 与 FN 各加一；
- 无预测：FN 加一。

据此计算：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

同时报告三个不参与调参的对照结果：纯最高置信度、纯最大面积、相对置信度门控
后最大面积。最终预测和 FP 文件夹使用已确认的相对门控方案。

## 错误处理

- 模型文件、配置、图片列表或 COCO 标注缺失时给出明确错误并非零退出。
- 无法读取的图片记录路径并跳过；评估时仍按缺失预测计 FN。
- 模型输入名与准备的输入不一致时立即报错。
- 输出坐标裁剪到原图范围，退化框不进入候选。

## 测试与验证

按 TDD 添加测试，先观察失败再实现：

- 相对置信度门控会剔除低可信大框；
- 门控后按面积选择最大框；
- 并列时稳定选择；
- `fire`/`firebig` 标签兼容；
- 配置驱动预处理不包含额外增强；
- IoU 边界 `0.5` 不算 TP；
- FP/FN/TP 与 F1 统计正确；
- FP 标注图和汇总信息正确生成。

完成后运行单元测试、100 张真实推理、三种固定策略对照评估，并核对预测数量、
FP 图片数量和汇总 JSON 一致。
