# New Label 与 COCO Firebig 融合数据集设计

## 目标

合并：

- `dog/A_train/new_label_coco`
- `dog/A_train/coco_firebig`

输出到 `dog/A_train/new_label_coco_firebig_merged`，重新生成可复现的
train/val/full COCO 标注，并让 `firebig/B_firebig.yml` 使用融合数据训练。

## 数据事实

- `new_label_coco`：2827 张、2676 条标注、151 张负样本。
- `coco_firebig`：1200 张、1147 条标注、1144 张正样本、56 张负样本。
- 两套数据无同名文件、无相同文件哈希。
- 合并后：4027 张、3823 条标注、207 张负样本。

## 合并与划分

- 类别统一为 `id=1, name=firebig`。
- 重新分配连续 image ID 与 annotation ID。
- 图片复制到输出目录的 `images`。
- 文件名加来源前缀 `new_label__` 与 `coco_firebig__`，从机制上避免未来冲突。
- 保留所有空标注图片，以便模型学习不输出。
- 保留原始多标注，不在本步骤隐式清洗。
- 按“来源 × 是否有标注”四个分层分别随机划分。
- train/val 比例为 80%/20%，随机种子为 2026。
- 每层先确定 val 数量，再用固定 seed 洗牌，确保重复运行结果一致。
- 写出：
  - `annotations/instance_train.json`
  - `annotations/instance_val.json`
  - `annotations/instance_train_full.json`
  - `splits/train.txt`
  - `splits/val.txt`
  - `merge_summary.json`

## 训练配置

修改 `firebig/B_firebig.yml`：

- `dataset_dir: ../A_train/new_label_coco_firebig_merged`
- `TrainDataset.allow_empty: true`
- `BatchRandomResize.target_size: [576, 640, 704]`
- EvalReader 与 TestReader 的 Resize 固定为 `[640, 640]`

其余模型结构、优化器、epoch、增强和损失保持不变。

## 验证

- 先用 TDD 覆盖 ID 重映射、负样本保留、分层比例、确定性和文件名冲突。
- 验证输出图片为 4027 张、标注为 3823 条、负样本为 207 张。
- 验证 train/val 无交集且并集等于 full。
- 验证所有 annotation 的 image_id 均存在。
- 验证配置路径、`allow_empty` 和三个分辨率设置。
