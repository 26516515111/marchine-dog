# sweep_hr_val_thresholds.py

用于在高分辨率验证集配置 `ppyoloe_plus_fire_hr_val.yml` 上扫描 `score_threshold` 和 `nms_threshold`。

## 作用

- 不训练模型
- 不修改配置文件
- 只调用 PaddleDetection 的 `tools/eval.py`
- 每组阈值都会生成单独日志和评估目录

## 前提

1. 已完成验证集模型训练
2. 模型权重存在于默认路径：
   `dog/PaddleDetection/output/ppyoloe_plus_fire_hr_val/best_model`
3. PaddleDetection 可正常运行

## 默认扫描范围

- `score_threshold`：`0.01,0.03,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50`
- `nms_threshold`：`0.45,0.50,0.55,0.60,0.65`

## 用法

在仓库根目录执行：

```bash
python dog/mycode/scripts/sweep_hr_val_thresholds.py
```

只看命令不执行：

```bash
python dog/mycode/scripts/sweep_hr_val_thresholds.py --dry-run
```

自定义阈值范围：

```bash
python dog/mycode/scripts/sweep_hr_val_thresholds.py ^
  --score-thresholds 0.05,0.10,0.15,0.20 ^
  --nms-thresholds 0.45,0.50,0.55
```

如果权重不在默认位置，可显式指定：

```bash
python dog/mycode/scripts/sweep_hr_val_thresholds.py --weights D:\path\to\best_model
```

## 输出

默认输出到：

`dog/mycode/threshold_sweeps/hr_val/<时间戳>/`

其中包含：

- `logs/`：每组阈值对应的原始评估日志
- `eval/`：每组阈值对应的 `output_eval` 目录
- `summary.json`：所有组合的汇总

## 结果读取

`summary.json` 里会记录：

- `score_threshold`
- `nms_threshold`
- `returncode`
- `metrics`

如果某次 eval 输出里带有 AP / mAP 字样，脚本会尝试自动提取。

## 推荐流程

1. 先用 `--dry-run` 看命令是否正确
2. 再正式执行扫描
3. 从 `summary.json` 和日志里选出验证集 F1 最优的阈值
4. 把选出的阈值迁移到最终全量模型
