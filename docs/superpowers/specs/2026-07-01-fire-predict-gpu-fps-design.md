# Fire Predict GPU FPS 修复设计

## 目标

修复 `D:\work\Marchine_Dog\fire\predict.py` 的性能与 GPU 使用问题，同时保持：

- 100 张 sample100 数据集 F1 为 0.96；
- 相对置信度门控与最大火焰选择规则不变；
- `result.json` 严格只有 `result` 顶层字段；
- 每条结果严格只有 `image_id`、`type`、`x`、`y`、`width`、`height`、
  `segmentation`；
- 在 Conda `dog` 环境、RTX 4060 Laptop GPU 上端到端 FPS 大于 20。

## 根因

1. Paddle 在脚本计时开始后才导入，约 1.54 秒初始化时间被计入脚本 FPS。
2. CUDA 不可用时脚本静默回退 CPU，可能以极低 FPS 继续生成结果。
3. GPU 显存池允许被环境变量设置得过小，可能触发推理过程中的内存重分配。
4. 脚本保存并复制全部原始候选框，但严格提交 JSON 不使用这些数据。
5. 提交 JSON 使用缩进序列化，存在少量无必要输出开销。

## 设计

- 在模块顶部导入 `paddle`、`Config` 和 `create_predictor`，与已验证脚本保持相同计时边界。
- `Detector` 初始化时要求 Paddle 为 CUDA 构建且至少检测到一张 GPU；否则抛出
  `RuntimeError`，禁止 CPU 回退。
- `PREDICT_GPU_POOL_MB` 默认 2000；小于 1000 时抛出 `ValueError`。
- 保持以下 Paddle 配置不变：
  - `enable_use_gpu(pool_mb, 0)`
  - `enable_memory_optim()`
  - `switch_use_feed_fetch_ops(False)`
  - `switch_ir_optim(False)`
- `predict_images()` 仅保留最终 `result` 与不可读图片计数；不保存
  `raw_candidates`，也不复制候选字典。
- 每张图仍先读取模型 NMS 后的候选框，再使用固定规则
  `score >= max(0.02, 0.5 * max_score)`，从可信候选中选面积最大框。
- 使用紧凑 `json.dump`/`json.dumps` 写入严格提交格式。
- 不通过隐藏 GPU 同步、删除预处理或缩小模型输入来伪造 FPS。

## 测试与验收

先写失败测试，再实现：

- CUDA 不可用时拒绝启动；
- GPU 显存池小于 1000 MB 时拒绝启动；
- `predict_images()` 不返回 `raw_candidates`；
- 源码不在 `main()` 或 `Detector` 内延迟导入 Paddle；
- 提交 JSON 字段严格匹配。

真实验证：

1. Conda `dog` 环境执行 100 张 GPU 推理三次；
2. 报告平均 FPS 与最低 FPS，最低必须大于 20；
3. 校验 100 条结果的精确 JSON schema；
4. 使用严格 `IoU > 0.5` 重新计算 F1，必须保持 0.96；
5. 运行预测与评估聚焦测试，必须全部通过。
