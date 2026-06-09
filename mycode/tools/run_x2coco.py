# -*- coding: utf-8 -*-
"""
使用本地 x2coco_custom.py 转换数据集（支持固定类别映射）
"""
import subprocess
import sys
import os

# 配置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
X2COCO_SCRIPT = os.path.join(SCRIPT_DIR, "x2coco_custom.py")
JSON_INPUT_DIR = r"D:\work\Marchine Dog\dog\A_train\label"
IMAGE_INPUT_DIR = r"D:\work\Marchine Dog\dog\A_train\Image"
OUTPUT_DIR = r"D:\work\Marchine Dog\dog\A_train\coco"

# 构建命令
cmd = [
    sys.executable,
    X2COCO_SCRIPT,
    "--dataset_type", "labelme",
    "--json_input_dir", JSON_INPUT_DIR,
    "--image_input_dir", IMAGE_INPUT_DIR,
    "--output_dir", OUTPUT_DIR,
    "--train_proportion", "0.8",
    "--val_proportion", "0.2",
    "--test_proportion", "0.0"
]

print("=" * 60)
print("开始转换 LabelMe -> COCO 格式")
print("=" * 60)
print(f"标注目录: {JSON_INPUT_DIR}")
print(f"图片目录: {IMAGE_INPUT_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"转换脚本: {X2COCO_SCRIPT}")
print("=" * 60)

# 运行转换
try:
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, check=True, text=True)
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  训练集: {OUTPUT_DIR}/annotations/instance_train.json")
    print(f"  验证集: {OUTPUT_DIR}/annotations/instance_val.json")
    print(f"  训练图片: {OUTPUT_DIR}/train/")
    print(f"  验证图片: {OUTPUT_DIR}/val/")
except subprocess.CalledProcessError as e:
    print(f"\n转换失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n发生错误: {e}")
    sys.exit(1)
