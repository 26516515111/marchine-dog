# -*- coding: utf-8 -*-
"""
PP-YOLOE-s 火焰检测训练脚本
"""
import subprocess
import sys
import os

# 配置路径
PADDLEDET_DIR = r"D:\work\Marchine Dog\dog\PaddleDetection"
CONFIG_FILE = "configs/custom/ppyoloe_fire.yml"

# 构建训练命令
# 使用单 GPU 训练
cmd = [
    sys.executable,
    "-m", "paddle.distributed.launch",
    "--gpus", "0",
    "tools/train.py",
    "-c", CONFIG_FILE,
    "--eval",
    "--amp"  # 启用混合精度训练，加速训练
]

print("=" * 60)
print("开始训练 PP-YOLOE-s 火焰检测模型")
print("=" * 60)
print(f"配置文件: {CONFIG_FILE}")
print(f"工作目录: {PADDLEDET_DIR}")
print("=" * 60)
print("\n训练命令:")
print(" ".join(cmd))
print("\n" + "=" * 60)

# 运行训练
try:
    result = subprocess.run(cmd, cwd=PADDLEDET_DIR, check=True, text=True)
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n模型保存位置:")
    print(f"  {PADDLEDET_DIR}/output/ppyoloe_fire/")
    print(f"  - best_model.pdparams (最佳模型)")
    print(f"  - model_final.pdparams (最终模型)")
except subprocess.CalledProcessError as e:
    print(f"\n训练失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n发生错误: {e}")
    sys.exit(1)
