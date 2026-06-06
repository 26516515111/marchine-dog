# -*- coding: utf-8 -*-
"""
模型导出脚本
将训练好的模型导出为 Paddle Inference 格式

使用方法:
    conda activate dog
    python mycode/tools/export_model.py
"""
import os
import sys
import yaml
import paddle
import paddle.nn as nn
from paddle.static import InputSpec


# 导入训练脚本中的模型定义
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import SimpleDetector


def export_model(config_path, model_path, output_dir):
    """
    导出模型为 Paddle Inference 格式
    
    Args:
        config_path: 训练配置文件路径
        model_path: 训练好的模型权重路径
        output_dir: 输出目录
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    model = SimpleDetector(
        num_classes=config['data']['num_classes'],
        pretrained=False
    )
    
    # 加载训练好的权重
    if os.path.exists(model_path):
        state_dict = paddle.load(model_path)
        model.set_state_dict(state_dict)
        print(f"加载模型权重: {model_path}")
    else:
        print(f"警告: 模型权重文件不存在 {model_path}")
        print("将使用随机初始化的权重")
    
    # 设置为评估模式
    model.eval()
    
    # 定义输入规格
    input_size = config['model']['input_size'][0]
    input_spec = [
        InputSpec(shape=[None, 3, input_size, input_size], dtype='float32', name='image')
    ]
    
    # 导出为静态图
    static_model = paddle.jit.to_static(
        model,
        input_spec=input_spec
    )
    
    # 保存模型
    save_path = os.path.join(output_dir, "model")
    paddle.jit.save(static_model, save_path)
    
    print(f"\n模型导出成功!")
    print(f"  模型文件: {save_path}.pdmodel")
    print(f"  参数文件: {save_path}.pdiparams")
    
    # 创建推理配置文件
    create_infer_config(config, output_dir)
    
    return save_path


def create_infer_config(config, output_dir):
    """
    创建推理配置文件
    
    Args:
        config: 训练配置
        output_dir: 输出目录
    """
    infer_config = {
        "mode": "paddle",
        "draw_threshold": 0.3,
        "metric": "COCO",
        "use_dynamic_shape": False,
        "arch": "PPYOLOE",
        "min_subgraph_size": 3,
        "Preprocess": [
            {
                "type": "Resize",
                "target_size": config['model']['input_size'],
                "keep_ratio": False,
                "interp": 2
            },
            {
                "type": "NormalizeImage",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "is_scale": True
            },
            {
                "type": "Permute"
            }
        ],
        "label_list": config['data']['class_names'],
        "NMS": {
            "keep_top_k": 100,
            "name": "MultiClassNMS",
            "nms_threshold": 0.5,
            "nms_top_k": 1000,
            "score_threshold": 0.3
        }
    }
    
    config_path = os.path.join(output_dir, "infer_cfg.yml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(infer_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"  推理配置: {config_path}")


def main():
    # 路径配置
    config_path = "mycode/configs/train_config.yml"
    output_dir = "model"
    
    # 查找最新的模型权重
    save_dir = "mycode/output"
    if os.path.exists(save_dir):
        model_files = [f for f in os.listdir(save_dir) if f.endswith('.pdparams')]
        if model_files:
            # 按文件名排序，获取最新的
            model_files.sort()
            model_path = os.path.join(save_dir, model_files[-1])
            print(f"找到模型权重: {model_path}")
        else:
            print("警告: 未找到训练好的模型权重")
            model_path = None
    else:
        print("警告: 输出目录不存在")
        model_path = None
    
    # 导出模型
    export_model(config_path, model_path, output_dir)
    
    # 打印目录结构
    print("\n" + "=" * 60)
    print("导出完成! 目录结构:")
    print("=" * 60)
    print("model/")
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"  {file} ({file_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
