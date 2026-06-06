将以下三个文件放入本目录（model/）：

  model.pdmodel      — 模型结构文件
  model.pdiparams    — 模型权重文件
  infer_cfg.yml      — 推理配置文件（已提供示例，请按需修改）

导出命令参考（PaddleDetection）：
  python tools/export_model.py \
      -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
      --output_dir=./output_inference \
      -o weights=best_model.pdparams

模型大小限制：model/ 目录总大小不超过 200MB。
