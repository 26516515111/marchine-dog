# -*- coding: utf-8 -*-
"""
测试模型推理速度
"""
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
import numpy as np
import time

def test_inference():
    model_dir = r"D:\work\Marchine Dog\dog\PaddleDetection\output_inference\ppyoloe_fire"
    
    # 加载模型
    model_file = model_dir + "/model.pdmodel"
    params_file = model_dir + "/model.pdiparams"
    
    # 创建配置
    config = Config(model_file, params_file)
    config.enable_use_gpu(200, 0)
    config.switch_ir_optim()
    
    # 创建预测器
    predictor = create_predictor(config)
    
    # 获取输入信息
    input_names = predictor.get_input_names()
    print(f"输入名称: {input_names}")
    
    # 直接使用正确的输入形状
    input_shape = [1, 3, 640, 640]
    scale_factor_shape = [1, 2]
    
    print(f"\n使用输入形状: {input_shape}")
    
    # 创建随机输入
    image_input = np.random.randn(*input_shape).astype(np.float32)
    scale_factor_input = np.array([[640.0/1080.0, 640.0/1920.0]]).astype(np.float32)
    
    # 设置输入
    image_handle = predictor.get_input_handle(input_names[0])
    image_handle.reshape(input_shape)
    image_handle.copy_from_cpu(image_input)
    
    if len(input_names) > 1:
        scale_handle = predictor.get_input_handle(input_names[1])
        scale_handle.reshape(scale_factor_shape)
        scale_handle.copy_from_cpu(scale_factor_input)
    
    # 预热
    print("预热中...")
    for i in range(10):
        predictor.run()
    
    # 测试推理速度
    print("测试推理速度...")
    num_runs = 100
    start_time = time.time()
    
    for i in range(num_runs):
        predictor.run()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time
    
    print(f"\n推理结果:")
    print(f"  平均推理时间: {avg_time*1000:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  竞赛要求: FPS >= 20")
    print(f"  是否满足: {'是' if fps >= 20 else '否'}")
    
    # 获取输出
    output_names = predictor.get_output_names()
    print(f"\n输出名称: {output_names}")
    
    for name in output_names:
        output_handle = predictor.get_output_handle(name)
        output_data = output_handle.copy_to_cpu()
        print(f"输出 '{name}' 形状: {output_data.shape}")
        if output_data.size > 0:
            print(f"输出 '{name}' 示例: {output_data.flatten()[:10]}")

if __name__ == "__main__":
    test_inference()
