# select_thresholds_conservative.py

用于从验证集预测结果中选择更保守的 `conf/score_threshold`。

这个脚本会计算两套方案：

- 全局统一阈值：三类共用同一个阈值
- 逐类别阈值：`battery / board / fire` 分别取不同阈值

最终推荐时默认偏向全局统一阈值。只有当逐类别阈值满足以下条件时，才会被采用：

- 相比全局阈值，F1 提升超过 `--min-improvement`
- 被调整的类别 GT 数量不少于 `--min-gt-for-class-tuning`

这样可以避免 `battery`、`board` 样本很少时，因为验证集偶然性把阈值调得过拟合。

## 输入要求

预测 JSON 必须带有 `score` 字段，例如：

```json
{
  "image_id": "frame_00001",
  "type": 3,
  "score": 0.73,
  "x": 10,
  "y": 20,
  "width": 100,
  "height": 120
}
```

## 基本用法

```bash
python dog/mycode/scripts/select_thresholds_conservative.py ^
  --pred-json dog/mycode/threshold_sweeps/hr_val_raw_preds_with_score.json ^
  --gt dog/A_train/coco_dataset/annotations/instance_val.json ^
  --out dog/mycode/threshold_sweeps/hr_val_thresholds_conservative.json
```

## 常用参数

候选阈值：

```bash
--candidates 0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50
```

提高逐类别调参门槛：

```bash
--min-gt-for-class-tuning 30 --min-improvement 0.02
```

如果更关心总体样本表现，而不是类别平均表现：

```bash
--objective micro_f1
```

默认是 `macro_f1`，会让小类别也参与平均，但推荐阶段仍会用样本数门槛防止小类别过拟合。

## 输出

输出 JSON 包含：

- `global_best`：统一阈值最佳结果
- `per_class_best`：逐类别阈值最佳结果
- `recommendation`：最终建议采用的阈值
- `gt_counts`：各类别验证集 GT 数量

实际使用时优先看：

```json
"recommendation": {
  "thresholds": {
    "battery": 0.05,
    "board": 0.05,
    "fire": 0.05
  }
}
```

## 注意

如果预测 JSON 本身来自异常推理结果，脚本仍能运行，但阈值结果没有意义。应优先确认预测框和验证 AP/F1 大致一致，再使用推荐阈值。
